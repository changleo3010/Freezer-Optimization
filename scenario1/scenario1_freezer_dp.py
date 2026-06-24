#!/usr/bin/env python3
"""
Scenario 1 freezer optimization with stochastic disturbances and safety overrides.

This script solves the Phoenix Scenario 1 problem as a stochastic expected-cost
dynamic program. It uses the time-varying disturbance model from the project
handout and adds a realistic emergency override:

    T < Tmin  -> compressor forced OFF
    T > Tmax  -> compressor forced to Pmax

The override is a recovery action, not a magic eraser. If the random
disturbance pushes the freezer outside [-25, -15] C, the violation is still
counted and penalized.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Main model and numerical parameters. Edit these first."""

    alpha: float = 0.10
    beta: float = 0.65
    pmax: float = 12.0
    t_min: float = -25.0
    t_max: float = -15.0
    t0: float = -20.0
    terminal_max: float = -20.0
    dt: float = 1.0

    temp_step: float = 0.10
    power_step: float = 0.05
    temp_padding: float = 5.0

    expectation_samples_per_hour: int = 401
    expectation_seed: int = 271
    validation_trials: int = 10_000
    validation_seed: int = 20260519
    seed_sweep_trials: int = 5_000
    seed_sweep_seeds: tuple[int, ...] = (11, 271, 1234, 20260519, 987654)

    violation_penalty: float = 10_000.0
    terminal_penalty: float = 10_000.0
    thermostat_initial_on: bool = True


PHOENIX_AMBIENT_C = [
    27.5, 26.3, 25.2, 24.5, 23.8, 23.5, 24.2, 26.0,
    28.4, 31.2, 33.8, 35.9, 37.5, 38.7, 39.4, 39.8,
    39.5, 38.2, 36.5, 34.3, 32.1, 30.5, 29.2, 28.3,
]

PRICE_USD_PER_KWH = [
    0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
    0.10, 0.10,
    0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20,
    0.10, 0.10,
    0.05, 0.05, 0.05, 0.05,
]

# start hour, end hour, mean disturbance C, standard deviation C
NOISE_SCHEDULE = [
    (0, 5, 0.0, 0.3),
    (6, 7, 0.5, 0.5),
    (8, 9, 2.0, 1.0),
    (10, 11, 1.0, 0.8),
    (12, 13, 0.5, 0.5),
    (14, 15, 1.5, 0.8),
    (16, 17, 2.0, 1.0),
    (18, 23, 0.2, 0.4),
]


def noise_params(hour: int) -> tuple[float, float]:
    for start, end, mean, std in NOISE_SCHEDULE:
        if start <= hour <= end:
            return mean, std
    raise ValueError(f"No disturbance parameters for hour {hour}")


def grid(start: float, stop: float, step: float) -> list[float]:
    n = int(round((stop - start) / step))
    return [round(start + i * step, 10) for i in range(n + 1)]


def violation_penalty(temp: float, cfg: Config) -> float:
    """Quadratic penalty for being outside the safe temperature range."""

    high = max(0.0, temp - cfg.t_max)
    low = max(0.0, cfg.t_min - temp)
    return cfg.violation_penalty * (high * high + low * low)


def final_penalty(temp: float, cfg: Config) -> float:
    """Penalty for ending warmer than the repeatable-day target."""

    warm = max(0.0, temp - cfg.terminal_max)
    return cfg.terminal_penalty * warm * warm


def next_temperature(temp: float, power: float, hour: int, disturbance: float, cfg: Config) -> float:
    return (
        temp
        + cfg.alpha * (PHOENIX_AMBIENT_C[hour] - temp) * cfg.dt
        - cfg.beta * power * cfg.dt
        + disturbance
    )


def admissible_controls(temp: float, power_grid: list[float], cfg: Config) -> list[float]:
    """Recovery override outside the safe band; normal DP control inside it."""

    if temp < cfg.t_min:
        return [0.0]
    if temp > cfg.t_max:
        return [cfg.pmax]
    return power_grid


def interpolate_value(temp_grid: list[float], values: list[float], temp: float, cfg: Config) -> float:
    """Linear interpolation with conservative tail penalties outside the padded grid."""

    if temp <= temp_grid[0]:
        extra = max(0.0, violation_penalty(temp, cfg) - violation_penalty(temp_grid[0], cfg))
        return values[0] + extra
    if temp >= temp_grid[-1]:
        extra = max(0.0, violation_penalty(temp, cfg) - violation_penalty(temp_grid[-1], cfg))
        return values[-1] + extra

    step = temp_grid[1] - temp_grid[0]
    pos = (temp - temp_grid[0]) / step
    left = int(math.floor(pos))
    frac = pos - left
    return (1.0 - frac) * values[left] + frac * values[left + 1]


def nearest_index(temp_grid: list[float], temp: float) -> int:
    if temp <= temp_grid[0]:
        return 0
    if temp >= temp_grid[-1]:
        return len(temp_grid) - 1
    step = temp_grid[1] - temp_grid[0]
    return max(0, min(len(temp_grid) - 1, int(round((temp - temp_grid[0]) / step))))


def expectation_samples(cfg: Config) -> dict[int, list[float]]:
    """Fixed antithetic disturbance samples for the Bellman expectation."""

    if cfg.expectation_samples_per_hour < 3:
        raise ValueError("expectation_samples_per_hour must be at least 3")

    rng = random.Random(cfg.expectation_seed)
    samples_by_hour: dict[int, list[float]] = {}
    pairs = cfg.expectation_samples_per_hour // 2
    include_mean = cfg.expectation_samples_per_hour % 2 == 1

    for hour in range(24):
        mean, std = noise_params(hour)
        samples: list[float] = [mean] if include_mean else []
        for _ in range(pairs):
            z = rng.gauss(0.0, 1.0)
            samples.append(mean + std * z)
            samples.append(mean - std * z)
        samples_by_hour[hour] = samples

    return samples_by_hour


def best_action(
    hour: int,
    temp: float,
    next_values: list[float],
    temp_grid: list[float],
    power_grid: list[float],
    samples: list[float],
    cfg: Config,
) -> tuple[float, float]:
    """Minimize energy plus violation penalty plus expected future value."""

    best_power = 0.0
    best_cost = math.inf
    current_penalty = violation_penalty(temp, cfg)

    for power in admissible_controls(temp, power_grid, cfg):
        deterministic_next = (
            temp
            + cfg.alpha * (PHOENIX_AMBIENT_C[hour] - temp) * cfg.dt
            - cfg.beta * power * cfg.dt
        )
        expected_future = 0.0
        for disturbance in samples:
            temp_next = deterministic_next + disturbance
            expected_future += interpolate_value(temp_grid, next_values, temp_next, cfg)

        stage_cost = PRICE_USD_PER_KWH[hour] * power * cfg.dt
        total_cost = current_penalty + stage_cost + expected_future / len(samples)
        if total_cost < best_cost - 1e-12:
            best_cost = total_cost
            best_power = power

    return best_power, best_cost


def solve_dp(cfg: Config) -> dict[str, object]:
    temp_grid = grid(cfg.t_min - cfg.temp_padding, cfg.t_max + cfg.temp_padding, cfg.temp_step)
    power_grid = grid(0.0, cfg.pmax, cfg.power_step)
    samples = expectation_samples(cfg)

    values = [[0.0 for _ in temp_grid] for _ in range(25)]
    policy = [[0.0 for _ in temp_grid] for _ in range(24)]
    values[24] = [violation_penalty(t, cfg) + final_penalty(t, cfg) for t in temp_grid]

    for hour in range(23, -1, -1):
        for i, temp in enumerate(temp_grid):
            power, value = best_action(
                hour, temp, values[hour + 1], temp_grid, power_grid, samples[hour], cfg
            )
            policy[hour][i] = power
            values[hour][i] = value

    return {
        "temp_grid": temp_grid,
        "power_grid": power_grid,
        "samples": samples,
        "values": values,
        "policy": policy,
        "objective_at_t0": interpolate_value(temp_grid, values[0], cfg.t0, cfg),
    }


def policy_power(temp: float, hour: int, dp: dict[str, object], cfg: Config) -> float:
    """Apply hard recovery override first, then use the DP policy inside bounds."""

    if temp < cfg.t_min:
        return 0.0
    if temp > cfg.t_max:
        return cfg.pmax
    temp_grid = dp["temp_grid"]
    policy = dp["policy"]
    return float(policy[hour][nearest_index(temp_grid, temp)])


def simulate_policy(disturbances: list[float], dp: dict[str, object], cfg: Config) -> dict[str, object]:
    temps = [cfg.t0]
    powers = []
    costs = []
    forced_off = 0
    forced_full = 0

    for hour, disturbance in enumerate(disturbances):
        temp = temps[-1]
        if temp < cfg.t_min:
            forced_off += 1
        elif temp > cfg.t_max:
            forced_full += 1

        power = policy_power(temp, hour, dp, cfg)
        temps.append(next_temperature(temp, power, hour, disturbance, cfg))
        powers.append(power)
        costs.append(PRICE_USD_PER_KWH[hour] * power * cfg.dt)

    return summarize_path(temps, powers, costs, forced_off, forced_full, cfg)


def simulate_thermostat(disturbances: list[float], cfg: Config) -> dict[str, object]:
    temps = [cfg.t0]
    powers = []
    costs = []
    on = cfg.thermostat_initial_on

    for hour, disturbance in enumerate(disturbances):
        temp = temps[-1]
        if temp > -17.0:
            on = True
        elif temp < -23.0:
            on = False
        power = cfg.pmax if on else 0.0
        temps.append(next_temperature(temp, power, hour, disturbance, cfg))
        powers.append(power)
        costs.append(PRICE_USD_PER_KWH[hour] * power * cfg.dt)

    return summarize_path(temps, powers, costs, 0, 0, cfg)


def summarize_path(
    temps: list[float],
    powers: list[float],
    costs: list[float],
    forced_off: int,
    forced_full: int,
    cfg: Config,
) -> dict[str, object]:
    high_violation = any(t > cfg.t_max for t in temps)
    low_violation = any(t < cfg.t_min for t in temps)
    terminal_violation = temps[-1] > cfg.terminal_max
    return {
        "temps": temps,
        "powers": powers,
        "energy_cost": sum(costs),
        "energy_kwh": sum(p * cfg.dt for p in powers),
        "min_temp": min(temps),
        "max_temp": max(temps),
        "final_temp": temps[-1],
        "high_violation": int(high_violation),
        "low_violation": int(low_violation),
        "temperature_violation": int(high_violation or low_violation),
        "terminal_violation": int(terminal_violation),
        "forced_off_actions": forced_off,
        "forced_full_actions": forced_full,
    }


def validation_disturbances(cfg: Config, seed: int, trials: int) -> list[list[float]]:
    rng = random.Random(seed)
    paths = []
    for _ in range(trials):
        path = []
        for hour in range(24):
            mean, std = noise_params(hour)
            path.append(rng.gauss(mean, std))
        paths.append(path)
    return paths


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def summarize_trials(label: str, rows: list[dict[str, object]]) -> dict[str, object]:
    n = len(rows)
    costs = [float(r["energy_cost"]) for r in rows]
    max_temps = [float(r["max_temp"]) for r in rows]
    return {
        "controller": label,
        "trials": n,
        "mean_cost": sum(costs) / n,
        "p05_cost": percentile(costs, 0.05),
        "p95_cost": percentile(costs, 0.95),
        "mean_kwh": sum(float(r["energy_kwh"]) for r in rows) / n,
        "temperature_violation_rate": sum(int(r["temperature_violation"]) for r in rows) / n,
        "high_violation_rate": sum(int(r["high_violation"]) for r in rows) / n,
        "low_violation_rate": sum(int(r["low_violation"]) for r in rows) / n,
        "terminal_violation_rate": sum(int(r["terminal_violation"]) for r in rows) / n,
        "mean_forced_off_actions": sum(int(r["forced_off_actions"]) for r in rows) / n,
        "mean_forced_full_actions": sum(int(r["forced_full_actions"]) for r in rows) / n,
        "p95_max_temp": percentile(max_temps, 0.95),
    }


def run_validation(cfg: Config, dp: dict[str, object], seed: int, trials: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    dp_rows = []
    thermostat_rows = []
    for disturbances in validation_disturbances(cfg, seed, trials):
        dp_rows.append(strip_path(simulate_policy(disturbances, dp, cfg)))
        thermostat_rows.append(strip_path(simulate_thermostat(disturbances, cfg)))
    return dp_rows, thermostat_rows


def strip_path(row: dict[str, object]) -> dict[str, object]:
    return {k: v for k, v in row.items() if k not in {"temps", "powers"}}


def mean_disturbance_path() -> list[float]:
    return [noise_params(hour)[0] for hour in range(24)]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def hourly_rows(nominal: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for hour in range(24):
        mean, std = noise_params(hour)
        rows.append(
            {
                "hour": hour,
                "ambient_c": PHOENIX_AMBIENT_C[hour],
                "price_usd_per_kwh": PRICE_USD_PER_KWH[hour],
                "disturbance_mean_c": mean,
                "disturbance_std_c": std,
                "temp_start_c": nominal["temps"][hour],
                "power_kw": nominal["powers"][hour],
                "temp_end_c": nominal["temps"][hour + 1],
            }
        )
    return rows


def seed_sweep(cfg: Config, dp: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for seed in cfg.seed_sweep_seeds:
        dp_rows, _ = run_validation(replace(cfg, validation_seed=seed), dp, seed, cfg.seed_sweep_trials)
        summary = summarize_trials("dp", dp_rows)
        rows.append(
            {
                "seed": seed,
                "trials": cfg.seed_sweep_trials,
                "mean_cost": summary["mean_cost"],
                "temperature_violation_rate": summary["temperature_violation_rate"],
                "terminal_violation_rate": summary["terminal_violation_rate"],
                "mean_forced_off_actions": summary["mean_forced_off_actions"],
                "mean_forced_full_actions": summary["mean_forced_full_actions"],
            }
        )
    return rows


def validate(cfg: Config, dp: dict[str, object], dp_summary: dict[str, object], thermostat_summary: dict[str, object], sweep_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    samples = dp["samples"]
    checks = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    mean_errors = []
    for hour in range(24):
        target_mean, _ = noise_params(hour)
        sample_mean = sum(samples[hour]) / len(samples[hour])
        mean_errors.append(abs(sample_mean - target_mean))

    off_ok = admissible_controls(cfg.t_min - 0.1, dp["power_grid"], cfg) == [0.0]
    full_ok = admissible_controls(cfg.t_max + 0.1, dp["power_grid"], cfg) == [cfg.pmax]
    inside_ok = len(admissible_controls((cfg.t_min + cfg.t_max) / 2, dp["power_grid"], cfg)) > 1

    add("expectation_sample_count", all(len(samples[h]) == cfg.expectation_samples_per_hour for h in range(24)), f"{cfg.expectation_samples_per_hour} samples per hour.")
    add("expectation_sample_means", max(mean_errors) <= 1e-12, f"Worst sample mean error: {max(mean_errors):.3e} C.")
    add("override_control_sets", off_ok and full_ok and inside_ok, "Below Tmin -> OFF, above Tmax -> full blast, inside bounds -> normal grid.")
    add("policy_power_bounds", all(0.0 <= p <= cfg.pmax for row in dp["policy"] for p in row), "All policy actions satisfy compressor limits.")
    add("dp_safer_than_thermostat", dp_summary["temperature_violation_rate"] < thermostat_summary["temperature_violation_rate"], f"DP {100*dp_summary['temperature_violation_rate']:.2f}% vs thermostat {100*thermostat_summary['temperature_violation_rate']:.2f}%.")
    add("dp_mean_cost_no_more_than_thermostat", dp_summary["mean_cost"] <= thermostat_summary["mean_cost"], f"DP ${dp_summary['mean_cost']:.2f} vs thermostat ${thermostat_summary['mean_cost']:.2f}.")
    add("seed_sweep_stable", max(r["temperature_violation_rate"] for r in sweep_rows) <= 0.05, "All seed-sweep violation rates are below 5%.")
    return checks


def write_report(path: Path, cfg: Config, nominal: dict[str, object], summaries: list[dict[str, object]], sweep_rows: list[dict[str, object]]) -> None:
    dp = next(s for s in summaries if s["controller"] == "dp_with_safety_override")
    thermostat = next(s for s in summaries if s["controller"] == "thermostat")
    savings = thermostat["mean_cost"] - dp["mean_cost"]
    pct = 100.0 * savings / thermostat["mean_cost"]
    path.write_text(
        f"""# Scenario 1 Stochastic DP with Safety Override

## Model

State: freezer temperature `T_k`.

Control: compressor power `P_k`.

Disturbance: `w_k ~ N(mu_k, sigma_k^2)` using the project handout's time-varying disturbance table.

Safety override:

```text
T_k < {cfg.t_min:.1f} C  -> P_k = 0
T_k > {cfg.t_max:.1f} C  -> P_k = Pmax
inside bounds -> DP chooses P_k from the normal power grid
```

Violations still count and receive a quadratic penalty. The override is a recovery rule, not a claim that the freezer remained safe.

## Main Parameters

- `alpha = {cfg.alpha}`
- `beta = {cfg.beta}`
- `Pmax = {cfg.pmax} kW`
- `T0 = {cfg.t0} C`
- safe range: `[{cfg.t_min}, {cfg.t_max}] C`
- terminal target: `T_24 <= {cfg.terminal_max} C`
- temperature penalty weight: `{cfg.violation_penalty:g}`
- terminal penalty weight: `{cfg.terminal_penalty:g}`

## Mean-Disturbance Rollout

- cost: `${nominal['energy_cost']:.2f}` per day
- energy use: `{nominal['energy_kwh']:.2f}` kWh
- min temperature: `{nominal['min_temp']:.2f} C`
- max temperature: `{nominal['max_temp']:.2f} C`
- final temperature: `{nominal['final_temp']:.2f} C`

## Monte Carlo Validation

| Controller | Mean Cost | 5-95% Cost Range | Temp Violation Rate | Terminal Violation Rate | Mean Forced OFF | Mean Forced Full |
|---|---:|---:|---:|---:|---:|---:|
| DP + safety override | `${dp['mean_cost']:.2f}` | `${dp['p05_cost']:.2f}`-`${dp['p95_cost']:.2f}` | `{100*dp['temperature_violation_rate']:.2f}%` | `{100*dp['terminal_violation_rate']:.2f}%` | `{dp['mean_forced_off_actions']:.3f}` | `{dp['mean_forced_full_actions']:.3f}` |
| Thermostat | `${thermostat['mean_cost']:.2f}` | `${thermostat['p05_cost']:.2f}`-`${thermostat['p95_cost']:.2f}` | `{100*thermostat['temperature_violation_rate']:.2f}%` | `{100*thermostat['terminal_violation_rate']:.2f}%` | `{thermostat['mean_forced_off_actions']:.3f}` | `{thermostat['mean_forced_full_actions']:.3f}` |

Mean savings versus thermostat: `${savings:.2f}` per day, or `{pct:.1f}%`.

## Seed Sweep

| Seed | Trials | Mean Cost | Temp Violation Rate | Terminal Violation Rate | Mean Forced OFF | Mean Forced Full |
|---:|---:|---:|---:|---:|---:|---:|
"""
        + "\n".join(
            f"| {r['seed']} | {r['trials']} | `${r['mean_cost']:.2f}` | {100*r['temperature_violation_rate']:.2f}% | {100*r['terminal_violation_rate']:.2f}% | {r['mean_forced_off_actions']:.3f} | {r['mean_forced_full_actions']:.3f} |"
            for r in sweep_rows
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    cfg = Config()
    out = Path(__file__).resolve().parent / "outputs"
    dp = solve_dp(cfg)

    nominal = simulate_policy(mean_disturbance_path(), dp, cfg)
    dp_rows, thermostat_rows = run_validation(cfg, dp, cfg.validation_seed, cfg.validation_trials)
    summaries = [
        summarize_trials("dp_with_safety_override", dp_rows),
        summarize_trials("thermostat", thermostat_rows),
    ]
    sweep_rows = seed_sweep(cfg, dp)
    checks = validate(cfg, dp, summaries[0], summaries[1], sweep_rows)

    write_csv(out / "scenario1_hourly_mean_profile.csv", hourly_rows(nominal))
    write_csv(out / "scenario1_validation_summary.csv", summaries)
    write_csv(out / "scenario1_seed_sweep.csv", sweep_rows)
    write_csv(out / "scenario1_validation_checks.csv", checks)
    write_report(out / "SCENARIO1_REPORT.md", cfg, nominal, summaries, sweep_rows)

    for check in checks:
        print(f"{check['status']}: {check['check']} - {check['detail']}")
    if not all(c["status"] == "PASS" for c in checks):
        raise SystemExit(1)

    dp_summary, thermostat_summary = summaries
    print("\nScenario 1 stochastic DP with safety override")
    print(f"Objective at T0: {dp['objective_at_t0']:.3f}")
    print(f"Mean-disturbance cost: ${nominal['energy_cost']:.2f}")
    print(f"MC mean cost: DP ${dp_summary['mean_cost']:.2f}, thermostat ${thermostat_summary['mean_cost']:.2f}")
    print(f"MC temp violation rate: DP {100*dp_summary['temperature_violation_rate']:.2f}%, thermostat {100*thermostat_summary['temperature_violation_rate']:.2f}%")
    print(f"Report: {out / 'SCENARIO1_REPORT.md'}")


if __name__ == "__main__":
    main()
