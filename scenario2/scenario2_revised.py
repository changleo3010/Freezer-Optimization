#!/usr/bin/env python3
"""
Scenario 2: Freezer Optimization with Demand Charges and Geographical Variations.

Extends Scenario 1 by:
  1. Augmenting state to (temperature, peak_power_so_far) for demand charge tracking.
  2. Evaluating three climate zones: Phoenix AZ, San Francisco CA, Minneapolis MN.
  3. Total daily cost = energy_cost + (demand_charge_rate / 30) * peak_power.

Additions in this version:
  - Seed sweep over five RNG seeds to validate Monte Carlo stability.
  - Savings decomposition: energy savings vs. demand-charge savings per location.
  - Visualizations: trajectory comparison, cost breakdown, peak-power distribution.
  - Peak-grid floor-rounding fix: simulator uses math.floor for the peak index,
    which is the conservative direction and avoids snapping the current peak
    upward toward the next grid point during policy lookup.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Config:
    alpha: float = 0.10          # thermal leakage coefficient [1/h]
    beta: float = 0.65           # cooling efficiency [C/kWh]
    pmax: float = 12.0           # maximum compressor power [kW]
    t_min: float = -25.0         # lower temperature bound [C]
    t_max: float = -15.0         # upper temperature bound [C]
    t0: float = -20.0            # initial temperature [C]
    terminal_max: float = -20.0  # terminal temperature target [C]
    dt: float = 1.0              # time step [h]

    # Grid resolution
    temp_step: float = 0.1
    power_step: float = 0.5
    temp_padding: float = 5.0

    # Stochastic expectation
    expectation_samples_per_hour: int = 201
    expectation_seed: int = 271

    # Monte Carlo validation
    validation_trials: int = 5_000
    validation_seed: int = 20260519

    # Seed sweep (same five seeds as Scenario 1)
    seed_sweep_trials: int = 5_000
    seed_sweep_seeds: tuple[int, ...] = (11, 271, 1234, 20260519, 987654)

    # Penalty weights
    violation_penalty: float = 50_000.0
    terminal_penalty: float = 10_000.0

    thermostat_initial_on: bool = True

    # Demand charge: $18.50/kW per month; amortised to daily below
    demand_charge_rate: float = 18.50          # $/kW per 30-day billing period
    monthly_to_daily_ratio: float = 1.0 / 30.0


# ─── Data tables ──────────────────────────────────────────────────────────────

# Hourly ambient temperature profiles from the project handout (Table 1)
AMBIENT_PROFILES: dict[str, list[float]] = {
    "Phoenix": [
        27.5, 26.3, 25.2, 24.5, 23.8, 23.5, 24.2, 26.0,
        28.4, 31.2, 33.8, 35.9, 37.5, 38.7, 39.4, 39.8,
        39.5, 38.2, 36.5, 34.3, 32.1, 30.5, 29.2, 28.3,
    ],
    "San Francisco": [
        14.8, 14.5, 14.2, 13.9, 13.7, 13.5, 13.7, 14.2,
        15.0, 16.1, 17.3, 18.2, 19.0, 19.6, 20.0, 20.2,
        19.8, 19.0, 18.1, 17.0, 16.2, 15.7, 15.3, 15.0,
    ],
    "Minneapolis": [
        5.2, 4.6, 4.0, 3.5, 3.0, 2.8, 3.2, 4.5,
        6.3, 8.7, 11.2, 13.5, 15.3, 16.8, 17.5, 17.8,
        17.2, 16.0, 14.3, 12.5, 10.2, 8.4, 7.0, 6.0,
    ],
}

# Time-of-use electricity prices (Table 3 from handout)
PRICE_USD_PER_KWH: list[float] = [
    0.05, 0.05, 0.05, 0.05, 0.05, 0.05,   # 00-05
    0.10, 0.10,                             # 06-07
    0.20, 0.20, 0.20, 0.20, 0.20, 0.20,   # 08-13
    0.20, 0.20, 0.20, 0.20,                # 14-17
    0.10, 0.10,                             # 18-19
    0.05, 0.05, 0.05, 0.05,                # 20-23
]

# Time-varying disturbance model (Table 2 from handout)
NOISE_SCHEDULE: list[tuple[int, int, float, float]] = [
    (0,  5,  0.0, 0.3),
    (6,  7,  0.5, 0.5),
    (8,  9,  2.0, 1.0),
    (10, 11, 1.0, 0.8),
    (12, 13, 0.5, 0.5),
    (14, 15, 1.5, 0.8),
    (16, 17, 2.0, 1.0),
    (18, 23, 0.2, 0.4),
]


# ─── Helper utilities ─────────────────────────────────────────────────────────

def noise_params(hour: int) -> tuple[float, float]:
    """Return (mean, std) of the disturbance for a given hour."""
    for start, end, mean, std in NOISE_SCHEDULE:
        if start <= hour <= end:
            return mean, std
    raise ValueError(f"No disturbance parameters for hour {hour}")


def make_grid(start: float, stop: float, step: float) -> list[float]:
    """Uniformly spaced grid from start to stop (inclusive) with given step."""
    n = int(round((stop - start) / step))
    return [round(start + i * step, 10) for i in range(n + 1)]


def nearest_idx(grid_arr: list[float], val: float) -> int:
    """Index of the grid point nearest to val (clamped to grid bounds)."""
    if val <= grid_arr[0]:
        return 0
    if val >= grid_arr[-1]:
        return len(grid_arr) - 1
    step = grid_arr[1] - grid_arr[0]
    return max(0, min(len(grid_arr) - 1, int(round((val - grid_arr[0]) / step))))


def floor_idx(grid_arr: list[float], val: float) -> int:
    """Floor-rounded index into a uniformly-spaced grid (clamped to bounds).

    Used for the peak-power state lookup during simulation. Nearest-neighbor
    rounding can snap the current peak upward toward the next grid point,
    which causes the simulator to under-estimate the demand cost of crossing
    that grid point. Floor-rounding is conservative in the safer direction:
    it assumes the current peak is slightly lower than it really is, so any
    new power above floor(peak) is treated as a potential new peak.
    """
    if val <= grid_arr[0]:
        return 0
    if val >= grid_arr[-1]:
        return len(grid_arr) - 1
    step = grid_arr[1] - grid_arr[0]
    return max(0, min(len(grid_arr) - 1,
                      int(math.floor((val - grid_arr[0]) / step))))


def violation_penalty(temp: float, cfg: Config) -> float:
    """Quadratic penalty for being outside the safe temperature band."""
    high = max(0.0, temp - cfg.t_max)
    low  = max(0.0, cfg.t_min - temp)
    return cfg.violation_penalty * (high * high + low * low)


def final_penalty(temp: float, cfg: Config) -> float:
    """Penalty for finishing warmer than the repeatable-day target."""
    warm = max(0.0, temp - cfg.terminal_max)
    return cfg.terminal_penalty * warm * warm


def admissible_controls(temp: float, power_grid: list[float], cfg: Config) -> list[float]:
    """Safety override outside the safe band; full control grid inside it."""
    if temp < cfg.t_min:
        return [0.0]
    if temp > cfg.t_max:
        return [cfg.pmax]
    return power_grid


def interp_value(
    temp_grid: list[float], col: list[float], temp: float, cfg: Config
) -> float:
    """Linear interpolation of a 1-D value column with penalised extrapolation."""
    if temp <= temp_grid[0]:
        extra = max(0.0, violation_penalty(temp, cfg) - violation_penalty(temp_grid[0], cfg))
        return col[0] + extra
    if temp >= temp_grid[-1]:
        extra = max(0.0, violation_penalty(temp, cfg) - violation_penalty(temp_grid[-1], cfg))
        return col[-1] + extra
    step = temp_grid[1] - temp_grid[0]
    pos  = (temp - temp_grid[0]) / step
    lo   = int(math.floor(pos))
    frac = pos - lo
    return (1.0 - frac) * col[lo] + frac * col[lo + 1]


def make_expectation_samples(cfg: Config) -> dict[int, list[float]]:
    """Antithetic-variates disturbance samples for the Bellman expectation."""
    if cfg.expectation_samples_per_hour < 3:
        raise ValueError("expectation_samples_per_hour must be at least 3")
    rng = random.Random(cfg.expectation_seed)
    result: dict[int, list[float]] = {}
    pairs = cfg.expectation_samples_per_hour // 2
    include_mean = (cfg.expectation_samples_per_hour % 2 == 1)
    for hour in range(24):
        mu, sigma = noise_params(hour)
        samps: list[float] = [mu] if include_mean else []
        for _ in range(pairs):
            z = rng.gauss(0.0, 1.0)
            samps.append(mu + sigma * z)
            samps.append(mu - sigma * z)
        result[hour] = samps
    return result


# ─── 2-D Stochastic DP ────────────────────────────────────────────────────────

def solve_dp_2d(cfg: Config, ambient_profile: list[float]) -> dict:
    """Stochastic backward-induction DP with state (temperature, peak_power_so_far)."""
    temp_grid  = make_grid(cfg.t_min - cfg.temp_padding,
                           cfg.t_max + cfg.temp_padding, cfg.temp_step)
    power_grid = make_grid(0.0, cfg.pmax, cfg.power_step)
    nt  = len(temp_grid)
    np_ = len(power_grid)

    samples = make_expectation_samples(cfg)
    daily_demand_rate = cfg.demand_charge_rate * cfg.monthly_to_daily_ratio

    # Terminal value V[24](temp_idx, peak_idx)
    v_next: list[list[float]] = [
        [final_penalty(temp_grid[i], cfg) + daily_demand_rate * power_grid[j]
         for j in range(np_)]
        for i in range(nt)
    ]

    policy: list[list[list[float]]] = [
        [[0.0] * np_ for _ in range(nt)]
        for _ in range(24)
    ]

    # Precompute next_peak_idx[current_peak_idx][chosen_power_idx]
    next_peak_idx: list[list[int]] = [
        [nearest_idx(power_grid, max(power_grid[j], power_grid[k]))
         for k in range(np_)]
        for j in range(np_)
    ]

    for hour in range(23, -1, -1):
        amb         = ambient_profile[hour]
        hour_samps  = samples[hour]
        n_samp      = len(hour_samps)
        price       = PRICE_USD_PER_KWH[hour]

        v_cols: list[list[float]] = [
            [v_next[i][j] for i in range(nt)]
            for j in range(np_)
        ]

        v_curr: list[list[float]] = [[math.inf] * np_ for _ in range(nt)]

        for i, temp in enumerate(temp_grid):
            controls = admissible_controls(temp, power_grid, cfg)
            viol     = violation_penalty(temp, cfg)

            det_nexts = [
                temp + cfg.alpha * (amb - temp) * cfg.dt - cfg.beta * p * cfg.dt
                for p in controls
            ]

            for j in range(np_):
                best_cost = math.inf
                best_p    = controls[0]

                for ci, power in enumerate(controls):
                    p_idx = nearest_idx(power_grid, power)
                    nj    = next_peak_idx[j][p_idx]
                    col   = v_cols[nj]
                    det   = det_nexts[ci]

                    exp_future = sum(
                        interp_value(temp_grid, col, det + w, cfg)
                        for w in hour_samps
                    ) / n_samp

                    total = viol + price * power * cfg.dt + exp_future

                    if total < best_cost - 1e-12:
                        best_cost = total
                        best_p    = power

                v_curr[i][j] = best_cost
                policy[hour][i][j] = best_p

        v_next = v_curr

    t0_i      = nearest_idx(temp_grid, cfg.t0)
    objective = v_next[t0_i][0]

    return {
        "temp_grid":       temp_grid,
        "power_grid":      power_grid,
        "policy":          policy,
        "samples":         samples,
        "objective_at_t0": objective,
        "ambient_profile": ambient_profile,
    }


# ─── Simulation ───────────────────────────────────────────────────────────────

def mean_disturbance_path() -> list[float]:
    """Return the 24-hour path of mean disturbances."""
    return [noise_params(h)[0] for h in range(24)]


def simulate_policy_2d(
    disturbances: list[float],
    dp: dict,
    cfg: Config,
) -> dict:
    """Simulate the DP policy on one disturbance realisation.

    Uses floor_idx for the peak-power lookup so the simulator does not
    under-estimate demand cost by snapping the current peak upward.
    """
    temp_grid       = dp["temp_grid"]
    power_grid      = dp["power_grid"]
    policy          = dp["policy"]
    ambient_profile = dp["ambient_profile"]

    temps: list[float] = [cfg.t0]
    powers: list[float] = []
    costs: list[float]  = []
    peak_power  = 0.0
    forced_off  = forced_full = 0

    for hour, w in enumerate(disturbances):
        temp = temps[-1]

        if temp < cfg.t_min:
            power = 0.0
            forced_off += 1
        elif temp > cfg.t_max:
            power = cfg.pmax
            forced_full += 1
        else:
            ti    = nearest_idx(temp_grid, temp)
            pj    = floor_idx(power_grid, peak_power)   # floor, not nearest
            power = float(policy[hour][ti][pj])

        peak_power = max(peak_power, power)
        t_next = (temp
                  + cfg.alpha * (ambient_profile[hour] - temp) * cfg.dt
                  - cfg.beta * power * cfg.dt
                  + w)
        temps.append(t_next)
        powers.append(power)
        costs.append(PRICE_USD_PER_KWH[hour] * power * cfg.dt)

    demand_cost = cfg.demand_charge_rate * cfg.monthly_to_daily_ratio * peak_power
    return _summarize(temps, powers, costs, demand_cost, peak_power,
                      forced_off, forced_full, cfg)


def simulate_thermostat_2d(
    disturbances: list[float],
    ambient_profile: list[float],
    cfg: Config,
) -> dict:
    """Simulate the bang-bang thermostat baseline with demand charge tracking."""
    temps: list[float] = [cfg.t0]
    powers: list[float] = []
    costs: list[float]  = []
    on         = cfg.thermostat_initial_on
    peak_power = 0.0

    for hour, w in enumerate(disturbances):
        temp = temps[-1]
        if temp > -17.0:
            on = True
        elif temp < -23.0:
            on = False
        power = cfg.pmax if on else 0.0

        peak_power = max(peak_power, power)
        t_next = (temp
                  + cfg.alpha * (ambient_profile[hour] - temp) * cfg.dt
                  - cfg.beta * power * cfg.dt
                  + w)
        temps.append(t_next)
        powers.append(power)
        costs.append(PRICE_USD_PER_KWH[hour] * power * cfg.dt)

    demand_cost = cfg.demand_charge_rate * cfg.monthly_to_daily_ratio * peak_power
    return _summarize(temps, powers, costs, demand_cost, peak_power, 0, 0, cfg)


def _summarize(
    temps: list[float],
    powers: list[float],
    costs: list[float],
    demand_cost: float,
    peak_power: float,
    forced_off: int,
    forced_full: int,
    cfg: Config,
) -> dict:
    high_viol     = any(t > cfg.t_max for t in temps)
    low_viol      = any(t < cfg.t_min for t in temps)
    terminal_viol = temps[-1] > cfg.terminal_max
    return {
        "temps":                temps,
        "powers":               powers,
        "energy_cost":          sum(costs),
        "demand_cost":          demand_cost,
        "total_cost":           sum(costs) + demand_cost,
        "energy_kwh":           sum(p * cfg.dt for p in powers),
        "peak_power_kw":        peak_power,
        "min_temp":             min(temps),
        "max_temp":             max(temps),
        "final_temp":           temps[-1],
        "high_violation":       int(high_viol),
        "low_violation":        int(low_viol),
        "temperature_violation": int(high_viol or low_viol),
        "terminal_violation":   int(terminal_viol),
        "forced_off_actions":   forced_off,
        "forced_full_actions":  forced_full,
    }


# ─── Monte Carlo validation ───────────────────────────────────────────────────

def generate_disturbances(cfg: Config, seed: int, trials: int) -> list[list[float]]:
    rng = random.Random(seed)
    return [
        [rng.gauss(*noise_params(h)) for h in range(24)]
        for _ in range(trials)
    ]


def run_validation(
    cfg: Config,
    dp: dict,
    seed: int,
    trials: int,
    ambient_profile: list[float],
) -> tuple[list[dict], list[dict]]:
    """Run Monte Carlo for both DP and thermostat on the same disturbance paths."""
    dp_rows, th_rows = [], []
    for dist in generate_disturbances(cfg, seed, trials):
        dp_rows.append(_strip(simulate_policy_2d(dist, dp, cfg)))
        th_rows.append(_strip(simulate_thermostat_2d(dist, ambient_profile, cfg)))
    return dp_rows, th_rows


def run_seed_sweep(
    cfg: Config,
    dp: dict,
    ambient_profile: list[float],
) -> list[dict]:
    """Evaluate the (already-solved) DP across multiple RNG seeds.

    Returns one summary row per seed. The DP is NOT re-solved - we are
    testing the stability of the Monte Carlo evaluation, not the policy.
    """
    rows = []
    for seed in cfg.seed_sweep_seeds:
        dp_rows, _ = run_validation(cfg, dp, seed, cfg.seed_sweep_trials, ambient_profile)
        summary = summarize_trials("dp_with_demand_charge", dp_rows)
        rows.append({
            "seed":                       seed,
            "trials":                     cfg.seed_sweep_trials,
            "mean_total_cost":            summary["mean_total_cost"],
            "mean_energy_cost":           summary["mean_energy_cost"],
            "mean_demand_cost":           summary["mean_demand_cost"],
            "mean_peak_power_kw":         summary["mean_peak_power_kw"],
            "temperature_violation_rate": summary["temperature_violation_rate"],
            "terminal_violation_rate":    summary["terminal_violation_rate"],
        })
    return rows


def _strip(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in {"temps", "powers"}}


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def summarize_trials(label: str, rows: list[dict]) -> dict:
    n = len(rows)

    def avg(key: str) -> float:
        return sum(float(r[key]) for r in rows) / n

    def vrate(key: str) -> float:
        return sum(int(r[key]) for r in rows) / n

    totals = [float(r["total_cost"])    for r in rows]
    peaks  = [float(r["peak_power_kw"]) for r in rows]
    mtemps = [float(r["max_temp"])      for r in rows]

    return {
        "controller":                  label,
        "trials":                      n,
        "mean_energy_cost":            avg("energy_cost"),
        "mean_demand_cost":            avg("demand_cost"),
        "mean_total_cost":             avg("total_cost"),
        "p05_total_cost":              percentile(totals, 0.05),
        "p95_total_cost":              percentile(totals, 0.95),
        "mean_energy_kwh":             avg("energy_kwh"),
        "mean_peak_power_kw":          avg("peak_power_kw"),
        "p95_peak_power_kw":           percentile(peaks, 0.95),
        "temperature_violation_rate":  vrate("temperature_violation"),
        "high_violation_rate":         vrate("high_violation"),
        "low_violation_rate":          vrate("low_violation"),
        "terminal_violation_rate":     vrate("terminal_violation"),
        "mean_forced_off":             avg("forced_off_actions"),
        "mean_forced_full":            avg("forced_full_actions"),
        "p95_max_temp":                percentile(mtemps, 0.95),
    }


# ─── Output helpers ───────────────────────────────────────────────────────────

def hourly_rows(nominal: dict, ambient_profile: list[float]) -> list[dict]:
    rows = []
    for hour in range(24):
        mu, sigma = noise_params(hour)
        rows.append({
            "hour":               hour,
            "ambient_c":          ambient_profile[hour],
            "price_usd_per_kwh":  PRICE_USD_PER_KWH[hour],
            "disturbance_mean_c": mu,
            "disturbance_std_c":  sigma,
            "temp_start_c":       nominal["temps"][hour],
            "power_kw":           nominal["powers"][hour],
            "temp_end_c":         nominal["temps"][hour + 1],
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# ─── Visualizations ───────────────────────────────────────────────────────────

# Price tier background colors for the trajectory plot
PRICE_TIER_COLORS = {0.05: "#d4edda", 0.10: "#fff3cd", 0.20: "#f8d7da"}


def plot_trajectories(all_results: dict[str, dict], out_path: Path) -> None:
    """3x1 panel: temperature, power, and price tier per location.

    Each subplot shows the DP policy under the mean-disturbance rollout
    so the pre-cooling behavior is visible.
    """
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    hours = list(range(24))

    for ax, (loc, data) in zip(axes, all_results.items()):
        nominal = data["nominal"]
        temps   = nominal["temps"][:24]    # start-of-hour temperatures
        powers  = nominal["powers"]

        # Shade price tiers as background bands
        for h in hours:
            ax.axvspan(h - 0.5, h + 0.5,
                       color=PRICE_TIER_COLORS[PRICE_USD_PER_KWH[h]],
                       alpha=0.4, zorder=0)

        # Temperature on left axis
        ax.plot(hours, temps, "o-", color="#1f4e8c", linewidth=2,
                markersize=4, label="Temperature", zorder=3)
        ax.axhline(-15, color="red",   linestyle="--", alpha=0.6, label="T_max")
        ax.axhline(-25, color="blue",  linestyle="--", alpha=0.6, label="T_min")
        ax.set_ylim(-26, -13)
        ax.set_ylabel("Temperature [C]", color="#1f4e8c")
        ax.tick_params(axis="y", labelcolor="#1f4e8c")

        # Power on right axis (step)
        ax2 = ax.twinx()
        ax2.step(hours, powers, where="mid", color="#c0392b",
                 linewidth=1.8, label="Power", zorder=2)
        ax2.fill_between(hours, 0, powers, step="mid",
                         color="#c0392b", alpha=0.15, zorder=1)
        ax2.set_ylim(0, 13)
        ax2.set_ylabel("Power [kW]", color="#c0392b")
        ax2.tick_params(axis="y", labelcolor="#c0392b")

        # Title with cost info
        s = data["dp_summary"]
        ax.set_title(
            f"{loc}: total ${s['mean_total_cost']:.2f}/day  "
            f"(energy ${s['mean_energy_cost']:.2f} + demand ${s['mean_demand_cost']:.2f})  "
            f"viol {100*s['temperature_violation_rate']:.1f}%  "
            f"peak {s['mean_peak_power_kw']:.1f} kW",
            fontsize=10
        )
        ax.grid(True, alpha=0.3, zorder=0)

    axes[-1].set_xlabel("Hour of day")
    axes[-1].set_xticks(range(0, 24, 2))

    # Legend for price tier shading
    legend_elements = [
        Patch(facecolor=PRICE_TIER_COLORS[0.05], alpha=0.4, label="$0.05/kWh"),
        Patch(facecolor=PRICE_TIER_COLORS[0.10], alpha=0.4, label="$0.10/kWh"),
        Patch(facecolor=PRICE_TIER_COLORS[0.20], alpha=0.4, label="$0.20/kWh"),
    ]
    fig.legend(handles=legend_elements, loc="upper center",
               bbox_to_anchor=(0.5, 1.00), ncol=3, frameon=False)

    fig.suptitle("Scenario 2: DP trajectories under mean disturbance",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cost_breakdown(all_results: dict[str, dict], out_path: Path) -> None:
    """Stacked bar chart: energy + demand cost for DP and thermostat per location."""
    locs        = list(all_results.keys())
    controllers = ["DP", "Thermostat"]

    energy = {c: [] for c in controllers}
    demand = {c: [] for c in controllers}

    for loc in locs:
        dp_s = all_results[loc]["dp_summary"]
        th_s = all_results[loc]["thermostat_summary"]
        energy["DP"].append(dp_s["mean_energy_cost"])
        demand["DP"].append(dp_s["mean_demand_cost"])
        energy["Thermostat"].append(th_s["mean_energy_cost"])
        demand["Thermostat"].append(th_s["mean_demand_cost"])

    fig, ax = plt.subplots(figsize=(9, 5))
    x         = list(range(len(locs)))
    bar_width = 0.36
    offsets   = {"DP": -bar_width / 2, "Thermostat": +bar_width / 2}
    colors    = {"DP": ("#3a76c4", "#a8c8f0"),
                 "Thermostat": ("#c0392b", "#f5b7b1")}

    for c in controllers:
        pos = [xi + offsets[c] for xi in x]
        ax.bar(pos, energy[c], bar_width, color=colors[c][0],
               label=f"{c} energy")
        ax.bar(pos, demand[c], bar_width, bottom=energy[c],
               color=colors[c][1], label=f"{c} demand")
        for xi, (e, d) in enumerate(zip(energy[c], demand[c])):
            ax.text(x[xi] + offsets[c], e + d + 0.3,
                    f"${e + d:.2f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(locs)
    ax.set_ylabel("Daily cost [USD]")
    ax.set_title("Scenario 2: cost breakdown by controller and location")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_peak_distribution(all_results: dict[str, dict], out_path: Path) -> None:
    """Histogram of per-trial peak power for DP and thermostat, per location."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    bins = [i * 0.5 for i in range(25)]   # 0 to 12 kW, 0.5 kW bins

    for ax, (loc, data) in zip(axes, all_results.items()):
        dp_peaks = [r["peak_power_kw"] for r in data["dp_trial_rows"]]
        th_peaks = [r["peak_power_kw"] for r in data["th_trial_rows"]]

        ax.hist(dp_peaks, bins=bins, alpha=0.7, color="#3a76c4",
                label="DP", edgecolor="white", linewidth=0.5)
        ax.hist(th_peaks, bins=bins, alpha=0.7, color="#c0392b",
                label="Thermostat", edgecolor="white", linewidth=0.5)
        ax.set_title(loc)
        ax.set_xlabel("Peak power [kW]")
        ax.set_xlim(0, 12.5)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Trial count")
    fig.suptitle(
        "Scenario 2: peak power distribution over Monte Carlo trials",
        fontsize=12
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── Report ───────────────────────────────────────────────────────────────────

def write_report(
    path: Path,
    cfg: Config,
    all_results: dict[str, dict],
    sweep_results: dict[str, list[dict]],
) -> None:
    lines = [
        "# Scenario 2: Demand Charges and Geographical Variations\n\n",
        "## Model\n\n",
        "State: `(T_k, P_peak_k)` - freezer temperature and peak power seen so far.\n\n",
        f"Control: `P_k in [0, {cfg.pmax}]` kW\n\n",
        "Stage cost: `violation_penalty(T_k) + price_k * P_k * dt`\n\n",
        f"Terminal cost: `final_penalty(T_24) + (${cfg.demand_charge_rate:.2f}/30) * P_peak`\n\n",
        "Safety override: `T < Tmin -> P=0`, `T > Tmax -> P=Pmax`\n\n",
        "Peak-grid handling: simulator uses floor-rounding for the peak-power "
        "state index, which avoids snapping the current peak upward toward "
        "the next grid point during policy lookup.\n\n",
        "## Parameters\n\n",
        f"- alpha = {cfg.alpha}, beta = {cfg.beta}, Pmax = {cfg.pmax} kW\n",
        f"- Safe range: [{cfg.t_min}, {cfg.t_max}] C,  T0 = {cfg.t0} C\n",
        f"- temp_step = {cfg.temp_step} C, power_step = {cfg.power_step} kW\n",
        f"- violation_penalty = {cfg.violation_penalty:g}, "
        f"terminal_penalty = {cfg.terminal_penalty:g}\n",
        f"- Demand charge: ${cfg.demand_charge_rate}/kW/month "
        f"(= ${cfg.demand_charge_rate * cfg.monthly_to_daily_ratio:.4f}/kW/day)\n\n",
        "## Monte Carlo Results by Location\n\n",
        "| Location | Controller | Energy Cost | Demand Cost | **Total Cost** | "
        "5-95% Total | Peak Power | Temp Viol. Rate |\n",
        "|---|---|---:|---:|---:|---:|---:|---:|\n",
    ]

    for loc, data in all_results.items():
        for s in [data["dp_summary"], data["thermostat_summary"]]:
            lines.append(
                f"| {loc} | {s['controller']} "
                f"| ${s['mean_energy_cost']:.2f} "
                f"| ${s['mean_demand_cost']:.2f} "
                f"| **${s['mean_total_cost']:.2f}** "
                f"| ${s['p05_total_cost']:.2f}-${s['p95_total_cost']:.2f} "
                f"| {s['mean_peak_power_kw']:.2f} kW "
                f"| {100*s['temperature_violation_rate']:.2f}% |\n"
            )

    # Savings decomposition
    lines.append("\n## Savings Decomposition: Energy vs. Demand\n\n")
    lines.append(
        "| Location | Energy Saving | Demand Saving | **Total Saving** | "
        "Energy Share | Demand Share | % vs Thermostat |\n"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for loc, data in all_results.items():
        dp_s = data["dp_summary"]
        th_s = data["thermostat_summary"]
        e_sav = th_s["mean_energy_cost"] - dp_s["mean_energy_cost"]
        d_sav = th_s["mean_demand_cost"] - dp_s["mean_demand_cost"]
        t_sav = e_sav + d_sav
        e_share = 100.0 * e_sav / t_sav if abs(t_sav) > 1e-9 else 0.0
        d_share = 100.0 * d_sav / t_sav if abs(t_sav) > 1e-9 else 0.0
        pct = 100.0 * t_sav / th_s["mean_total_cost"]
        lines.append(
            f"| {loc} | ${e_sav:.2f} | ${d_sav:.2f} | **${t_sav:.2f}** "
            f"| {e_share:.1f}% | {d_share:.1f}% | {pct:.1f}% |\n"
        )

    # Seed sweep
    lines.append("\n## Seed Stability (DP only)\n\n")
    lines.append(
        f"Each row is an independent {cfg.seed_sweep_trials}-trial Monte Carlo "
        f"with a different RNG seed. The DP policy is solved once per location "
        f"and reused.\n\n"
    )
    for loc, sweep in sweep_results.items():
        lines.append(f"### {loc}\n\n")
        lines.append(
            "| Seed | Trials | Mean Total | Mean Energy | Mean Demand | "
            "Mean Peak | Temp Viol. | Terminal Viol. |\n"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in sweep:
            lines.append(
                f"| {r['seed']} | {r['trials']} "
                f"| ${r['mean_total_cost']:.2f} "
                f"| ${r['mean_energy_cost']:.2f} "
                f"| ${r['mean_demand_cost']:.2f} "
                f"| {r['mean_peak_power_kw']:.2f} kW "
                f"| {100*r['temperature_violation_rate']:.2f}% "
                f"| {100*r['terminal_violation_rate']:.2f}% |\n"
            )

        # Compact spread summary
        totals = [r["mean_total_cost"] for r in sweep]
        viols  = [r["temperature_violation_rate"] for r in sweep]
        lines.append(
            f"\nSpread across seeds: total cost ${min(totals):.2f}-${max(totals):.2f} "
            f"(range ${max(totals) - min(totals):.2f}), violation rate "
            f"{100*min(viols):.2f}%-{100*max(viols):.2f}% "
            f"(range {100*(max(viols) - min(viols)):.2f} pp).\n\n"
        )

    lines.append("\n## Figures\n\n")
    lines.append("- `scenario2_trajectories.png` - hourly temperature, power, "
                 "and price tier per location.\n")
    lines.append("- `scenario2_cost_breakdown.png` - stacked energy/demand "
                 "cost per controller per location.\n")
    lines.append("- `scenario2_peak_distribution.png` - histogram of "
                 "per-trial peak power, DP vs thermostat.\n")

    path.write_text("".join(lines), encoding="utf-8")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    cfg = Config()
    out = Path(__file__).resolve().parent / "outputs_scenario2"
    out.mkdir(exist_ok=True)

    all_results:   dict[str, dict] = {}
    sweep_results: dict[str, list[dict]] = {}

    for loc, ambient in AMBIENT_PROFILES.items():
        print(f"\n=== {loc} ===")

        print("  Solving DP ...", end=" ", flush=True)
        dp = solve_dp_2d(cfg, ambient)
        print(f"done  (V* = {dp['objective_at_t0']:.3f})")

        nominal = simulate_policy_2d(mean_disturbance_path(), dp, cfg)
        write_csv(out / f"{loc}_hourly.csv", hourly_rows(nominal, ambient))

        print(f"  Running {cfg.validation_trials} MC trials ...",
              end=" ", flush=True)
        dp_rows, th_rows = run_validation(
            cfg, dp, cfg.validation_seed, cfg.validation_trials, ambient
        )
        dp_sum = summarize_trials("dp_with_demand_charge", dp_rows)
        th_sum = summarize_trials("thermostat",            th_rows)
        print("done")

        write_csv(out / f"{loc}_validation.csv", [dp_sum, th_sum])

        # Seed sweep (DP only)
        print(f"  Running seed sweep ({len(cfg.seed_sweep_seeds)} seeds x "
              f"{cfg.seed_sweep_trials} trials) ...", end=" ", flush=True)
        sweep = run_seed_sweep(cfg, dp, ambient)
        print("done")
        write_csv(out / f"{loc}_seed_sweep.csv", sweep)
        sweep_results[loc] = sweep

        all_results[loc] = {
            "dp_summary":         dp_sum,
            "thermostat_summary": th_sum,
            "nominal":            nominal,
            "dp_trial_rows":      dp_rows,
            "th_trial_rows":      th_rows,
        }

        # Per-location console summary
        e_sav = th_sum["mean_energy_cost"] - dp_sum["mean_energy_cost"]
        d_sav = th_sum["mean_demand_cost"] - dp_sum["mean_demand_cost"]
        t_sav = e_sav + d_sav
        pct   = 100.0 * t_sav / th_sum["mean_total_cost"]
        print(f"  DP    : energy ${dp_sum['mean_energy_cost']:.2f} "
              f"+ demand ${dp_sum['mean_demand_cost']:.2f} "
              f"= total ${dp_sum['mean_total_cost']:.2f}/day")
        print(f"  Therm : energy ${th_sum['mean_energy_cost']:.2f} "
              f"+ demand ${th_sum['mean_demand_cost']:.2f} "
              f"= total ${th_sum['mean_total_cost']:.2f}/day")
        print(f"  Savings: energy ${e_sav:.2f} + demand ${d_sav:.2f} "
              f"= ${t_sav:.2f}/day ({pct:.1f}%)")
        print(f"  DP temp violation rate: "
              f"{100*dp_sum['temperature_violation_rate']:.2f}%")
        print(f"  DP mean peak power: {dp_sum['mean_peak_power_kw']:.2f} kW "
              f"(thermostat: {th_sum['mean_peak_power_kw']:.2f} kW)")

        sweep_totals = [r["mean_total_cost"] for r in sweep]
        print(f"  Seed sweep total cost range: "
              f"${min(sweep_totals):.2f}-${max(sweep_totals):.2f} "
              f"(spread ${max(sweep_totals) - min(sweep_totals):.2f})")

    # Plots
    print("\nGenerating figures ...", end=" ", flush=True)
    plot_trajectories(all_results,      out / "scenario2_trajectories.png")
    plot_cost_breakdown(all_results,    out / "scenario2_cost_breakdown.png")
    plot_peak_distribution(all_results, out / "scenario2_peak_distribution.png")
    print("done")

    write_report(out / "SCENARIO2_REPORT.md", cfg, all_results, sweep_results)
    print(f"\nAll outputs written to: {out}")


if __name__ == "__main__":
    main()
