# sumo_pilot

Seeds 10000–10019, 4 units, odd seeds use the held-out back-streets driver. Wall time 1227 s. Raw rows: `reports/sumo_pilot.jsonl`.

| Demand | Coverage | Method | Intercepted | 95% CI | Median escape s | Units parked / ep |
|---|---|---|---|---|---|---|
| normal | 30% | **Proposed** | 7/20 = 35% | 18%–57% | 485 | 0.00 |
| normal | 30% | B2 plan once | 3/20 = 15% | 5%–36% | 514 | 0.00 |
| normal | 30% | B1 checkpoints | 3/20 = 15% | 5%–36% | 538 | 0.00 |
| normal | 30% | B0 chase | 1/20 = 5% | 1%–24% | 538 | 0.05 |
| peak | 30% | **Proposed** | 6/20 = 30% | 15%–52% | 505 | 0.05 |
| peak | 30% | B2 plan once | 5/20 = 25% | 11%–47% | 513 | 0.00 |
| peak | 30% | B1 checkpoints | 0/20 = 0% | 0%–16% | 611 | 0.00 |
| peak | 30% | B0 chase | 0/20 = 0% | 0%–16% | 550 | 0.00 |

Failed episodes: 0 of 160.
