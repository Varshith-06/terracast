# Vehicle trafficability on the edge: test bundle

Forecast-aware off-road route planning for military vehicles, running on the vehicle.
From a daily weather log, the model forecasts ground firmness (rating cone index, RCI,
in psi) 1 to 7 days ahead. It then marks each vehicle band GO, SLOW-GO or NO-GO and plans
the fastest route that avoids NO-GO ground on a 90 m terrain grid of the Pachmarhi
plateau (76,218 cells).

```
weather log (400 days) -> features -> GRU x3 + ridge -> RCI forecast, days 1-7   [forecaster.pkl]
   -> minus safety margin -> soil score per cell -> min(slope, cover, soil)        [planner.pkl]
   -> graph -> Dijkstra -> route
```

SLOW-GO is also given in numbers:

- **Passes:** how many vehicles one track carries before it fails: `50 ** ((RCI - VCI1) / (VCI50 - VCI1))`. This gives 1 at VCI1 and 50 at VCI50.
- **Speed:** km/h from the NATO Reference Mobility Model's fine-grained soil relations: drawbar pull, motion resistance, soil-limited traction and slip, against the slope in the direction of travel.

```
fc.outlook(f, band="D")                     # per day: status, RCI after margin, passes, km/h
pl.plan(rci, "D", trips, by="time")         # routes by travel time; times in hours
```

`by="passability"` (the default) is the published planner. Sources and assumptions are in
`traffic/mobility.py` in the repo. The rim pull is constant power with an assumed 80%
driveline efficiency, and the top speed is the game template value. Vegetation and ride
limits are not modelled, so speeds are upper bounds.

## Files

| File | What it is |
|---|---|
| `vehicle_trafficability_forecaster.pkl` | 3 GRUs (128x2) and 7 ridge regressions with their scalers, feature lists, RCI constants, safety margins, and the VCI table of 1,145 vehicles. 531,228 parameters |
| `vehicle_trafficability_planner.pkl` | 90 m terrain grid: slope and land-cover scores, roads, elevation, wetness index, and the planning vehicle for each band (with the power-to-weight, contact pressure and top speed the speed model needs) |
| `trafficability_edge.py` | The runtime. `Forecaster`, `Planner`, `end_to_end()` |
| `train_trafficability.py` | Retrains the forecaster from a weather log and writes a new forecaster pickle in the same format. Needs torch |
| `benchmark.py` | Size, load time, memory, per-stage latency, scaling for 10–50 requests, accuracy, correctness |
| `edge_node.py` | Three-node TCP test bed (source → inference → sink), like a fetch / inference / end-to-end table |
| `sample_weather_log.csv` | One forecast input: the last 400 days of the log |
| `weather_log_2015_2025.csv` | The full log, used by `--accuracy` |
| `sample_trips.json` | 30 trips of 8–25 km (grid cell ids, plus lat/lon) |
| `expected_outputs.json` | Outputs the bundle must reproduce (checked by `benchmark.py`) |
| `reference_results.json` | Our own results on a laptop, for comparison |

The pickles hold only plain Python types. Arrays are stored as dtype, shape and raw bytes,
so they unpickle with any Python 3.8+ and need no numpy, torch or scikit-learn to load.
`trafficability_edge.py` turns them back into arrays.

## Setup (Raspberry Pi OS, Ubuntu, Windows)

```
python3 -m venv venv && . venv/bin/activate
pip install -r requirements.txt          # numpy, pandas, scipy, psutil
python benchmark.py --backend numpy      # last line should say all four checks are True
```

PyTorch is optional. With `--backend numpy` the GRU runs in plain numpy and gives the same
forecasts as PyTorch (identical to 0.0001 psi). On a Pi, numpy avoids loading torch, which
adds about 450 MB of memory and 1–2 s of import time. Use `--backend torch` to measure the
PyTorch path.

## Benchmark

```
python benchmark.py [--backend numpy|torch] [--threads 1] [--repeats 30] [--accuracy]
```

It measures:
- **Size:** parameters, pickle file sizes, float32 weight size.
- **Cold start:** import time, plus load time for each pickle.
- **Memory:** RSS after imports, after each load, after forecasting and after planning, and the process peak.
- **Latency per stage:** mean, p50, p95 and max over 30 runs after 3 warm-up runs.
  - Forecaster stages: `read_log`, `features`, `ridge`, `gru_x3`, `forecast_total`, `assess_all_bands`, `outlook_all_bands`.
  - Planner stages: `soil_map`, `fuzzy_and`, `graph_build`, `dijkstra_1_trip`, `replan_1_trip`, `time_graph_build`, `replan_by_time_1_trip`.
  - Whole pipeline: `end_to_end_1_trip` (log to route) and `convoy_10_wear_aware` (10 vehicles, each re-planning on the shared pass-count map).
- **Scaling with 10, 20, 30, 40 and 50 requests**, in two modes:
  - *independent*: each request runs the whole pipeline.
  - *batched*: one forecast and one map serve all the trips.
- **Message sizes:** a forecast broadcast (18 B), one vehicle's track, a convoy's tracks, and a full passability map.
- **Accuracy (`--accuracy`):**
  - RMSE against persistence for 2024–25.
  - Per band at 7 days: NO-GO recall, precision and false alarms. Margins are tuned on the other year.
- **Correctness:** forecast, routes, routes by travel time and convoy must match `expected_outputs.json`.

It defaults to one CPU thread, like a single-core budget. Results go to
`benchmark_<host>_<backend>.json`.

## Three-node test bed

| Node | Role | Suggested machine |
|---|---|---|
| source | Holds jobs: a 400-day weather log and trip requests | Raspberry Pi |
| infer | Fetches a job, forecasts, plans, sends the routes | The machine under test |
| sink | Receives the routes and acknowledges them to the source | Raspberry Pi |

Start them in this order:
```
# sink
python edge_node.py sink   --listen 0.0.0.0:6000 --source SOURCE_IP:7000
# source
python edge_node.py source --listen 0.0.0.0:5001 --ack 0.0.0.0:7000 --requests 10,20,30,40,50 --repeats 5 --mode independent
# infer
python edge_node.py infer  --source SOURCE_IP:5001 --sink SINK_IP:6000 --backend numpy
```
Or run `python edge_node.py demo [--mode batched]` to start all three on one machine.

What gets measured:
- **FT:** the time from the infer node's fetch request to receiving the job (about 41 kB per request).
- **Inference:** the infer node's compute time, from weather log to routes.
- **E2E:** measured by the source, from when the jobs are queued until every acknowledgement arrives. One clock, so the nodes need no clock sync.

The source prints a table per request count and writes `source_results_<mode>.csv`.

## Reference numbers (our laptop: AMD Ryzen 7 Zen 4, one thread, numpy backend)

Laptop timings vary by about ±30% with power state. The full figures are in `reference_results.json`.

| | |
|---|---|
| Parameters | 531,228 (3 × 176,775 GRU + 903 ridge), 2.03 MB float32 |
| Pickles | forecaster 2.1 MB, planner 3.2 MB |
| Load | ~6–10 ms per pickle |
| Forecast (log → 7-day RCI) | ~25–45 ms. Features ~15–20 ms, GRU ×3 ~6 ms, ridge < 0.1 ms |
| Re-plan one route | ~25–30 ms. Graph ~15–20 ms, Dijkstra ~10 ms |
| End to end, one trip | ~50–80 ms |
| Convoy of 10, wear-aware | ~240–300 ms |
| Peak memory | ~160 MB with numpy, ~620 MB with torch |
| Forecast message | 18 B (4 B day number + 7 × float16) |
| One passability map | 76,218 B at 1 B per cell |

Test bed on localhost, numpy, mean of 5 runs:

| Requests | Independent: FT (s) | Inference (s) | E2E (ms) | Batched: E2E (ms) |
|---|---|---|---|---|
| 10 | 0.005 | 0.74 | 744 | 70 |
| 20 | 0.010 | 1.55 | 1,569 | 93 |
| 30 | 0.014 | 2.30 | 2,317 | 112 |
| 40 | 0.017 | 2.86 | 2,889 | 115 |
| 50 | 0.023 | 3.79 | 3,825 | 148 |

Accuracy, 2024–25 test years:
- **Firmness forecast:** 7-day RMSE 20.76 psi against 26.02 for persistence. At 1 day it is 6.05 against 6.71, and at 3 days 13.29 against 15.16.
- **NO-GO calls at 7 days:** 90% recall in every band. Precision runs from 0.43 (light tracked) to 0.94 (wheeled).

## Retraining

```
python train_trafficability.py --log weather_log_2015_2025.csv [--train-end 2021-12-31 --val-end 2023-12-31]
```

The script refits the ridge regressions and the 3 GRUs. It then re-tunes the safety margins on the test period: everything after `--val-end`, or the validation period if there is no test period. The vehicle table, bands and constants are copied from the current forecaster pickle, so the planner pickle stays valid.

The output is `vehicle_trafficability_forecaster_retrained.pkl`. Rename it to replace the current forecaster pickle. To train on newer data, extend the log and move both dates forward.

## Notes

- The forecaster needs the last 400 days of the log. Columns as in `sample_weather_log.csv`,
  with `Date` as dd-mm-yyyy and no missing days. Fewer days raises an error.
- Bands A–E group vehicles by VCI1, the firmness needed for one pass:
  A < 18, B 18–21, C 21–24, D 24–30, E ≥ 30 psi. Each band plans with its median vehicle.
- Deployed safety margins are tuned on 2024 and 2025 pooled, at 90% NO-GO recall.
  The cross-year margins behind the reported accuracy are also in the pickle, as
  `margins_cross_year`.
- Inputs are ERA5-Land soil moisture and IMD rainfall for one 9 km cell. Firmness is
  computed from modelled soil moisture, not measured in the field.
