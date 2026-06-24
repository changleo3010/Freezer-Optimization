# Scenario 2: Demand Charges and Geographical Variations

## Model

State: `(T_k, P_peak_k)` - freezer temperature and peak power seen so far.

Control: `P_k in [0, 12.0]` kW

Stage cost: `violation_penalty(T_k) + price_k * P_k * dt`

Terminal cost: `final_penalty(T_24) + ($18.50/30) * P_peak`

Safety override: `T < Tmin -> P=0`, `T > Tmax -> P=Pmax`

Peak-grid handling: simulator uses floor-rounding for the peak-power state index, which avoids snapping the current peak upward toward the next grid point during policy lookup.

## Parameters

- alpha = 0.1, beta = 0.65, Pmax = 12.0 kW
- Safe range: [-25.0, -15.0] C,  T0 = -20.0 C
- temp_step = 0.1 C, power_step = 0.5 kW
- violation_penalty = 50000, terminal_penalty = 10000
- Demand charge: $18.5/kW/month (= $0.6167/kW/day)

## Monte Carlo Results by Location

| Location | Controller | Energy Cost | Demand Cost | **Total Cost** | 5-95% Total | Peak Power | Temp Viol. Rate |
|---|---|---:|---:|---:|---:|---:|---:|
| Phoenix | dp_with_demand_charge | $27.32 | $7.07 | **$34.39** | $33.11-$35.70 | 11.46 kW | 3.06% |
| Phoenix | thermostat | $28.16 | $7.40 | **$35.56** | $33.20-$37.40 | 12.00 kW | 100.00% |
| San Francisco | dp_with_demand_charge | $19.51 | $5.36 | **$24.87** | $23.46-$26.73 | 8.69 kW | 2.84% |
| San Francisco | thermostat | $21.35 | $7.40 | **$28.75** | $26.60-$30.80 | 12.00 kW | 99.92% |
| Minneapolis | dp_with_demand_charge | $16.98 | $5.05 | **$22.03** | $20.62-$24.01 | 8.19 kW | 3.28% |
| Minneapolis | thermostat | $18.39 | $7.40 | **$25.79** | $24.20-$27.20 | 12.00 kW | 100.00% |

## Savings Decomposition: Energy vs. Demand

| Location | Energy Saving | Demand Saving | **Total Saving** | Energy Share | Demand Share | % vs Thermostat |
|---|---:|---:|---:|---:|---:|---:|
| Phoenix | $0.84 | $0.33 | **$1.17** | 71.5% | 28.5% | 3.3% |
| San Francisco | $1.84 | $2.04 | **$3.88** | 47.4% | 52.6% | 13.5% |
| Minneapolis | $1.42 | $2.35 | **$3.77** | 37.6% | 62.4% | 14.6% |

## Seed Stability (DP only)

Each row is an independent 5000-trial Monte Carlo with a different RNG seed. The DP policy is solved once per location and reused.

### Phoenix

| Seed | Trials | Mean Total | Mean Energy | Mean Demand | Mean Peak | Temp Viol. | Terminal Viol. |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 5000 | $34.41 | $27.34 | $7.07 | 11.47 kW | 2.80% | 0.00% |
| 271 | 5000 | $34.42 | $27.35 | $7.07 | 11.47 kW | 2.62% | 0.04% |
| 1234 | 5000 | $34.41 | $27.33 | $7.07 | 11.47 kW | 2.90% | 0.00% |
| 20260519 | 5000 | $34.39 | $27.32 | $7.07 | 11.46 kW | 3.06% | 0.00% |
| 987654 | 5000 | $34.40 | $27.33 | $7.07 | 11.47 kW | 3.16% | 0.00% |

Spread across seeds: total cost $34.39-$34.42 (range $0.04), violation rate 2.62%-3.16% (range 0.54 pp).

### San Francisco

| Seed | Trials | Mean Total | Mean Energy | Mean Demand | Mean Peak | Temp Viol. | Terminal Viol. |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 5000 | $24.90 | $19.53 | $5.37 | 8.71 kW | 2.56% | 0.02% |
| 271 | 5000 | $24.91 | $19.53 | $5.38 | 8.72 kW | 2.56% | 0.04% |
| 1234 | 5000 | $24.89 | $19.52 | $5.37 | 8.71 kW | 2.66% | 0.04% |
| 20260519 | 5000 | $24.87 | $19.51 | $5.36 | 8.69 kW | 2.84% | 0.00% |
| 987654 | 5000 | $24.89 | $19.51 | $5.38 | 8.72 kW | 2.82% | 0.02% |

Spread across seeds: total cost $24.87-$24.91 (range $0.05), violation rate 2.56%-2.84% (range 0.28 pp).

### Minneapolis

| Seed | Trials | Mean Total | Mean Energy | Mean Demand | Mean Peak | Temp Viol. | Terminal Viol. |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 5000 | $22.06 | $17.00 | $5.06 | 8.21 kW | 2.90% | 0.00% |
| 271 | 5000 | $22.07 | $17.00 | $5.06 | 8.21 kW | 3.02% | 0.06% |
| 1234 | 5000 | $22.05 | $16.99 | $5.06 | 8.21 kW | 3.16% | 0.04% |
| 20260519 | 5000 | $22.03 | $16.98 | $5.05 | 8.19 kW | 3.28% | 0.00% |
| 987654 | 5000 | $22.05 | $16.98 | $5.07 | 8.22 kW | 3.16% | 0.02% |

Spread across seeds: total cost $22.03-$22.07 (range $0.04), violation rate 2.90%-3.28% (range 0.38 pp).


## Figures

- `scenario2_trajectories.png` - hourly temperature, power, and price tier per location.
- `scenario2_cost_breakdown.png` - stacked energy/demand cost per controller per location.
- `scenario2_peak_distribution.png` - histogram of per-trial peak power, DP vs thermostat.
