# Freezer Optimization via Dynamic Programming

Cost-minimizing control of an industrial freezer over a 24-hour horizon under time-varying electricity pricing, demand charges, and stochastic thermal disturbances. The freezer's thermal mass lets it "store cold": pre-cool when electricity is cheap and coast through expensive hours, all without leaving the food-safe temperature band.

ECE 271C course project. Authors: Nitin, Mayand, Neel, Ryan, Chunyu, Zhou.

## Problem

A freezer must stay inside the safe band **[−25, −15] °C** while minimizing its electricity bill. Prices vary up to fourfold across the day ($0.05–$0.20/kWh), and the utility also applies a **demand charge** of $18.50/kW billed monthly on the single highest power draw. This is a sequential decision problem under uncertainty, solved with dynamic programming.

The freezer temperature evolves as:

```
T_{k+1} = T_k + α(T_amb − T_k)Δt − β·P_k·Δt + ω_k
```

where `α` is thermal leakage, `β` is compressor cooling efficiency, `P_k` is the applied cooling power, and `ω_k` is a time-varying Gaussian thermal disturbance.

## Scenarios

The project is organized into three scenarios that build on each other.

### Scenario 1 — Finite-horizon DP

One-dimensional DP over freezer temperature for a Phoenix-only profile. The controller decides *when* to buy electricity, pre-cooling during cheap hours and coasting through expensive ones. Key modeling choices:

- A large but **finite** out-of-band penalty (100,000) instead of an infinite one. An infinite penalty breaks the Bellman expectation, since a single out-of-bounds disturbance sample makes the whole averaged cost-to-go infinite. A quadratic finite penalty also scales with the severity of the violation.
- **Importance sampling** in the Bellman backup. The original estimator used 401 antithetic samples drawn directly from the disturbance distribution, which only reached about ±3σ and never represented the rare, large disturbances that dominate the large-penalty value function. This produced a non-monotone failure-vs-penalty curve. Drawing from a widened proposal `q = N(μ, (3σ)²)` and reweighting by the likelihood ratio fixes the bias while keeping the same sample count.

Validation uses 10,000 Monte Carlo trials against a full-on/full-off thermostat baseline on identical disturbance paths. The DP's main advantage over the thermostat is reliability: orders of magnitude fewer temperature violations, because it anticipates future heat load instead of reacting after a threshold is crossed.

### Scenario 2 — Demand charges and geographic variation

Adds a demand charge and evaluates across three climates: **Phoenix** (hot desert), **San Francisco** (temperate coastal), and **Minneapolis** (cold northern).

The demand charge depends on the maximum power over the whole horizon, so it is not stage-decomposable. To keep the problem Markov, the state is augmented with the running peak power:

```
x_k = (T_k, M_k),   M_k = max_{0≤j<k} P_j
```

The charge enters through the **terminal cost** `(c_d/30)·M_N` rather than any stage cost. Augmenting the state turns the 1-D grid into a 2-D grid, which is expensive; tractability is recovered by tying the peak grid to the control grid (`ΔM = ΔP = 0.5 kW`) and halving the disturbance samples, bringing the per-stage cost back to the Scenario 1 order.

**Results (5,000-trial Monte Carlo, daily USD):**

| Location | Controller | Energy | Demand | Total | Viol. rate |
|---|---|---|---|---|---|
| Phoenix | DP | $27.32 | $7.07 | **$34.39** | 3.06% |
| Phoenix | Thermostat | $28.16 | $7.40 | $35.56 | 100% |
| San Francisco | DP | $19.51 | $5.36 | **$24.87** | 2.84% |
| San Francisco | Thermostat | $21.35 | $7.40 | $28.75 | 99.9% |
| Minneapolis | DP | $16.98 | $5.05 | **$22.03** | 3.28% |
| Minneapolis | Thermostat | $18.39 | $7.40 | $25.79 | 100% |

The DP wins on both cost components in every climate and holds the safe band on roughly 97% of trials versus the baseline's near-total violation. Where the saving comes from rotates with climate: in hot Phoenix the load pins the peak near the cap so the saving is mostly energy timing (3.3% total), while in mild Minneapolis the controller has room to flatten its profile and the demand charge supplies most of the saving (14.6% total). The energy saving is load-shifting, not consumption reduction; in Phoenix the DP actually uses more total energy but buys it during cheap hours.

### Scenario 3 — Compressor selection and lifecycle cost

Shifts from "how to run a freezer" to "which compressor to buy." The objective becomes the annual total cost of ownership:

```
J(q, ℓ) = C_op,yr(q, ℓ) + C_cap(q)/L(q) + C_maint(q)
```

over compressors A, B, C and locations ℓ. Only real billed dollars (energy + demand) count toward operating cost; safety penalties shape the policy but never enter the reported bill.

An honest annual model cannot just multiply one day by 365, because a demand charge is set by the single worst hour in a 30-day month. The year is broken into **D-day chunks** with the demand meter resetting every 30 days. To avoid the terminal-cost artifact of solving a chunk in isolation, **K chunks are chained back-to-back in one backward pass** (checkpointing), the artificial terminal is placed only after the last chunk, and only the first chunk's policy is kept. The carried ending-state converges by K = 2 to a stable, repeatable annual cycle.

**Recommendation:** Compressor C in all three climates, for two different reasons.

- In Phoenix it is the only feasible option (A and B cannot hold the band under the hot-desert load).
- In San Francisco and Minneapolis, B is also feasible, but C wins on lifecycle cost; its higher cooling coefficient repays the larger capital outlay, with a roughly two-year payback (1.83 yr in SF, 1.97 yr in Minneapolis).

Feasibility is a hard gate applied before any cost comparison. The recommendation holds across all 33 location-level decisions under 11 exogenous shock settings (energy price, demand charge, and usage intensity).

## Methods at a glance

- Backward-recursion finite-horizon DP with a discretized state and control grid (temperature 0.1 °C steps over [−30, −10] °C; power in 0.5 kW steps).
- Stochastic disturbances modeled with hour-specific Gaussian means and variances.
- Antithetic sampling for the Bellman expectation, with importance sampling in Scenario 1 to correct tail-event bias.
- State augmentation with running peak power to fold the demand charge into the DP without breaking Bellman's principle.
- Out-of-sample validation by Monte Carlo rollout against a full-on/full-off thermostat baseline on shared disturbance paths.

## Parameter studies

The report includes sweeps over the violation/terminal penalty weights, temperature and power discretization steps, peak-power grid resolution, temperature padding, expectation sample count, and Monte Carlo sample count. Practical takeaways: a penalty weight around 10–100 balances cost against safety, a temperature step of 0.05–0.1 °C and a power step of 0.5–1.0 kW are good operating points, and roughly 500 Monte Carlo trials are enough for stable cost/violation estimates.

Several sweeps showed non-monotone or otherwise unexpected behavior (e.g., coarser grids occasionally beating finer ones, validated costs exceeding J values, J values rising with more samples). These are flagged as likely code bugs rather than real effects. See the Limitations section.

## Limitations and future work

- **Sampling bias in Scenarios 2 and 3.** The same tail-event sampling problem that affected Scenario 1 also applies to the optimization expectations in Scenarios 2 and 3, but importance sampling has not yet been retrofitted there. The plan is to swap importance sampling into every sampled expectation inside the optimization and re-validate against a large brute-force Monte Carlo. Conclusions that do not depend on the tail (cost rankings, hard infeasibility) are not expected to change.
- **Repeatable-day assumption.** The `(c_d/30)·M_N` amortization treats every day as identical. A real billing month is set by its single worst day. The cleaner fix is a rolling-horizon DP that carries the month-to-date peak so the controller can reserve high peaks for genuinely hot days.
- **Override feature.** The safety override (force compressor off below T_min, full power above T_max) makes temperature padding irrelevant, since out-of-band J values are never consulted. Removing the override and relying on padded J values to steer the policy back is a cleaner design.

## Contributions

- Scenario 1: Nitin & Neel
- Scenario 2: Chunyu & Zhou
- Scenario 3: Mayand
- Parameter testing: Ryan
- Slides and final report: all members for their respective contributions

---

> Note: adjust the repository structure, setup instructions, and run commands below to match the actual code layout before publishing.

## Repository structure

```
.
├── scenario1/          # Finite-horizon DP, importance sampling
├── scenario2/          # Demand charges, 2-D (temperature, peak power) DP
├── scenario3/          # Compressor selection, chunked/checkpointed annual DP
├── parameter_testing/  # Penalty, discretization, and sampling sweeps
├── figures/            # Generated plots
└── README.md
```

## Running

```bash
# example placeholders — replace with the project's actual entry points
pip install -r requirements.txt
python scenario1/run.py
python scenario2/run.py
python scenario3/run.py
```
