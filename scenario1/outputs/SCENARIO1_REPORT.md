# Scenario 1 Stochastic DP with Safety Override

## Model

State: freezer temperature `T_k`.

Control: compressor power `P_k`.

Disturbance: `w_k ~ N(mu_k, sigma_k^2)` using the project handout's time-varying disturbance table.

Safety override:

```text
T_k < -25.0 C  -> P_k = 0
T_k > -15.0 C  -> P_k = Pmax
inside bounds -> DP chooses P_k from the normal power grid
```

Violations still count and receive a quadratic penalty. The override is a recovery rule, not a claim that the freezer remained safe.

## Main Parameters

- `alpha = 0.1`
- `beta = 0.5`
- `Pmax = 10.0 kW`
- `T0 = -20.0 C`
- safe range: `[-25.0, -15.0] C`
- terminal target: `T_24 <= -20.0 C`
- temperature penalty weight: `10000`
- terminal penalty weight: `10000`

## Mean-Disturbance Rollout

- cost: `$29.00` per day
- energy use: `240.00` kWh
- min temperature: `-22.39 C`
- max temperature: `-6.64 C`
- final temperature: `-11.34 C`

## Monte Carlo Validation

| Controller | Mean Cost | 5-95% Cost Range | Temp Violation Rate | Terminal Violation Rate | Mean Forced OFF | Mean Forced Full |
|---|---:|---:|---:|---:|---:|---:|
| DP + safety override | `$29.00` | `$29.00`-`$29.00` | `100.00%` | `100.00%` | `0.000` | `10.977` |
| Thermostat | `$28.49` | `$26.00`-`$29.00` | `100.00%` | `100.00%` | `0.000` | `0.000` |

Mean savings versus thermostat: `$-0.51` per day, or `-1.8%`.

## Seed Sweep

| Seed | Trials | Mean Cost | Temp Violation Rate | Terminal Violation Rate | Mean Forced OFF | Mean Forced Full |
|---:|---:|---:|---:|---:|---:|---:|
| 11 | 5000 | `$29.00` | 100.00% | 100.00% | 0.000 | 10.954 |
| 271 | 5000 | `$29.00` | 100.00% | 100.00% | 0.000 | 10.992 |
| 1234 | 5000 | `$29.00` | 100.00% | 100.00% | 0.000 | 10.984 |
| 20260519 | 5000 | `$29.00` | 100.00% | 100.00% | 0.000 | 10.962 |
| 987654 | 5000 | `$29.00` | 100.00% | 100.00% | 0.000 | 10.951 |
