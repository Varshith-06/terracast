"""
Latency, memory and size benchmark for the vehicle trafficability edge bundle.

    python benchmark.py                    # all measurements, 1 CPU thread
    python benchmark.py --threads 4        # let numpy / torch use 4 threads
    python benchmark.py --backend numpy    # GRU in plain numpy (no torch needed)
    python benchmark.py --accuracy         # also forecast accuracy on 2024-25

Writes benchmark_<host>_<backend>.json next to this file and prints a summary.
"""

from __future__ import annotations

import argparse
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--backend", choices=["auto", "torch", "numpy"], default="auto")
ap.add_argument("--repeats", type=int, default=30)
ap.add_argument("--accuracy", action="store_true")
args = ap.parse_args()
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = str(args.threads)

import json, platform, socket, time  # noqa: E401,E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

try:
    import psutil
    _proc = psutil.Process()
except ImportError:
    psutil = None


def rss_mb():
    if psutil:
        return _proc.memory_info().rss / 2**20
    try:                                            # Linux without psutil
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        return float("nan")


def peak_mb():
    if psutil and hasattr(_proc.memory_info(), "peak_wset"):          # Windows
        return _proc.memory_info().peak_wset / 2**20
    try:
        import resource                                               # Linux / macOS
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 2**20 if sys.platform == "darwin" else r / 1024
    except ImportError:
        return float("nan")


mem = {"start": rss_mb()}
t0 = time.perf_counter()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import trafficability_edge as E  # noqa: E402
imports_s = time.perf_counter() - t0
if args.backend != "numpy":
    t0 = time.perf_counter()
    try:
        import torch
        torch.set_num_threads(args.threads)
    except ImportError:
        pass
    torch_import_s = time.perf_counter() - t0
else:
    torch_import_s = 0.0
mem["after_imports"] = rss_mb()


def timeit(fn, repeats=args.repeats, warmup=3):
    for _ in range(warmup):
        out = fn()
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        out = fn()
        ts.append((time.perf_counter() - t) * 1000)
    ts = np.array(ts)
    return dict(mean_ms=float(ts.mean()), p50_ms=float(np.percentile(ts, 50)),
                p95_ms=float(np.percentile(ts, 95)), max_ms=float(ts.max()), n=len(ts)), out


F_PKL = HERE / "vehicle_trafficability_forecaster.pkl"
P_PKL = HERE / "vehicle_trafficability_planner.pkl"
LOG = HERE / "sample_weather_log.csv"
trips = json.load(open(HERE / "sample_trips.json"))
pairs = [(t["src"]["cell"], t["dst"]["cell"]) for t in trips]
expected = json.load(open(HERE / "expected_outputs.json"))
R = {}

# ------------------------------------------------------------------ load (cold start)
t = time.perf_counter(); fc_model = E.Forecaster(F_PKL, backend=args.backend)
R["load_forecaster_ms"] = (time.perf_counter() - t) * 1000
mem["after_forecaster_load"] = rss_mb()
t = time.perf_counter(); pl = E.Planner(P_PKL)
R["load_planner_ms"] = (time.perf_counter() - t) * 1000
mem["after_planner_load"] = rss_mb()
p = fc_model.p

# ------------------------------------------------------------------ forecaster stages
S = {}
S["read_log"], raw = timeit(lambda: E.read_log(LOG, fc_model.window))
S["features"], _ = timeit(lambda: E.build_features(E.soil_strength(raw, p["rci"]), p["lat_deg"]))
_, _, Xtab, win = fc_model.features(LOG)
S["ridge"], _ = timeit(lambda: fc_model.ridge(Xtab))
S["gru_x3"], _ = timeit(lambda: fc_model.gru(win))
S["forecast_total"], fc = timeit(lambda: fc_model.forecast(LOG))
S["assess_all_bands"], _ = timeit(lambda: [fc_model.assess(fc, band=b) for b in "ABCDE"])
mem["after_forecast"] = rss_mb()

# ------------------------------------------------------------------ planner stages
band, h = expected["routes"]["band"], expected["routes"]["h"]
rci = float(fc["rci"][h - 1]) - fc_model.margin(band, h)
v = p["band_vehicles"][band]
S["soil_map"], soil = timeit(lambda: pl.soil(rci, v["vci1"], v["vci50"]))
S["fuzzy_and"], mu = timeit(lambda: pl.passability(v["kind"], soil))
S["graph_build"], G = timeit(lambda: pl.graph(mu))
S["dijkstra_1_trip"], _ = timeit(lambda: pl.routes(mu, pairs[:1], G))
S["replan_1_trip"], _ = timeit(lambda: pl.plan(rci, band, pairs[:1]))
S["end_to_end_1_trip"], _ = timeit(lambda: E.end_to_end(fc_model, pl, LOG, band, pairs[:1], h))
cx = expected["convoy"]
S["convoy_10_wear_aware"], cv = timeit(lambda: pl.convoy(cx["rci"], cx["band"], cx["src"], cx["dst"]),
                                       repeats=max(3, args.repeats // 10))
mem["after_planning"] = rss_mb()
R["stages"] = S

# ------------------------------------------------------- scaling with request count
rng = np.random.default_rng(0)
scale = []
for n in (10, 20, 30, 40, 50):
    req = [pairs[i] for i in rng.integers(0, len(pairs), n)]
    t = time.perf_counter()
    for r in req:                                   # independent: each request does it all
        E.end_to_end(fc_model, pl, LOG, band, [r], h)
    indep = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    E.end_to_end(fc_model, pl, LOG, band, req, h)   # batched: one forecast, one map
    batched = (time.perf_counter() - t) * 1000
    scale.append(dict(requests=n, independent_total_ms=indep, independent_per_req_ms=indep / n,
                      batched_total_ms=batched, batched_per_req_ms=batched / n))
R["scaling"] = scale
mem["peak"] = peak_mb()
mem["after_scaling"] = rss_mb()
R["memory_mb"] = mem

# ------------------------------------------------------------------ sizes and messages
lens = [len(q) for q, _ in pl.plan(rci, band, pairs) if q is not None]
R["model"] = dict(params_total=p["params"]["total"], params_gru_each=p["params"]["gru_each"],
                  gru_count=p["params"]["gru_count"], params_ridge=p["params"]["ridge"],
                  forecaster_pkl_mb=F_PKL.stat().st_size / 2**20,
                  planner_pkl_mb=P_PKL.stat().st_size / 2**20,
                  weights_float32_mb=p["params"]["total"] * 4 / 2**20,
                  grid_cells=int(pl.p["valid"].sum()), grid_shape=pl.shape)
R["messages_bytes"] = dict(forecast=len(E.Forecaster.message(fc)),
                           track_per_vehicle_mean=4 * float(np.mean(lens)) if lens else None,
                           convoy_10_tracks=cv[1],
                           passability_map_1B_per_cell=int(pl.p["valid"].sum()))

# ------------------------------------------------------------------ correctness
ok_fc = np.allclose(fc["rci"], expected["forecast"]["rci"], atol=0.05)
got = pl.plan(rci, band, pairs)
ok_rt = all((q is None) == (e["cells"] is None) and (q is None or len(q) == e["cells"])
            for (q, _), e in zip(got, expected["routes"]["trips"]))
R["correctness"] = dict(forecast_matches=bool(ok_fc), routes_match=bool(ok_rt),
                        convoy_matches=list(cv) == [cx["through"], cx["bytes"]])

# ------------------------------------------------------------------ accuracy (optional)
if args.accuracy:
    allf = fc_model.forecast_all(HERE / "weather_log_2015_2025.csv")
    y = allf["rci_today"].values
    test = (allf["date"] >= "2024-01-01").values
    idx = np.where(test)[0]
    idx = idx[idx + 7 < len(allf)]
    acc = {}
    for hh in (1, 3, 7):
        act = y[idx + hh]
        acc[f"rmse_h{hh}"] = float(np.sqrt(np.mean((allf[f"rci_h{hh}"].values[idx] - act) ** 2)))
        acc[f"rmse_persistence_h{hh}"] = float(np.sqrt(np.mean((y[idx] - act) ** 2)))
    veh = fc_model.vehicles
    years = pd.to_datetime(allf["date"].values[idx]).year
    nogo = {}
    for b, grp in veh.groupby("band"):
        v1 = grp.vci1.values
        tp = fp = fn = 0
        for yr, other in ((2024, 2025), (2025, 2024)):
            sel = idx[years == yr]
            m = p["margins_cross_year"][b][7][f"tuned_{other}"]
            a = y[sel + 7][None, :] < v1[:, None]
            c = (allf["rci_h7"].values[sel] - m)[None, :] < v1[:, None]
            tp += int((a & c).sum()); fp += int((~a & c).sum()); fn += int((a & ~c).sum())
        nogo[b] = dict(recall=tp / max(tp + fn, 1), precision=tp / max(tp + fp, 1),
                       false_alarms_per_vehicle_year=fp / (len(v1) * len(idx)) * 365)
    acc["nogo_h7_cross_year"] = nogo
    R["accuracy"] = acc

R["env"] = dict(host=socket.gethostname(), platform=platform.platform(), machine=platform.machine(),
                processor=platform.processor(), python=platform.python_version(),
                numpy=np.__version__, pandas=pd.__version__, backend=fc_model.backend,
                threads=args.threads, cpu_count=os.cpu_count(),
                imports_s=imports_s, torch_import_s=torch_import_s)

out = HERE / f"benchmark_{socket.gethostname()}_{fc_model.backend}.json"
json.dump(R, open(out, "w"), indent=1, default=float)

# ------------------------------------------------------------------ summary
print(f"\n{R['env']['platform']} | {R['env']['machine']} | python {R['env']['python']} | "
      f"backend {fc_model.backend} | {args.threads} thread(s)")
m = R["model"]
print(f"\nModel: {m['params_total']:,} parameters ({m['gru_count']} GRU x {m['params_gru_each']:,} + "
      f"ridge {m['params_ridge']:,}), {m['weights_float32_mb']:.2f} MB float32")
print(f"Files: forecaster {m['forecaster_pkl_mb']:.2f} MB, planner {m['planner_pkl_mb']:.2f} MB "
      f"({m['grid_cells']:,} cells)")
print(f"Load: forecaster {R['load_forecaster_ms']:.0f} ms, planner {R['load_planner_ms']:.0f} ms "
      f"(imports {imports_s:.2f} s, torch {torch_import_s:.2f} s)")
print("\nStage                     mean ms    p50    p95    max")
for k, s in S.items():
    print(f"  {k:<22} {s['mean_ms']:8.2f} {s['p50_ms']:7.2f} {s['p95_ms']:7.2f} {s['max_ms']:7.2f}")
print("\nRequests  independent total / per req (ms)   batched total / per req (ms)")
for r in scale:
    print(f"  {r['requests']:>3}     {r['independent_total_ms']:9.1f} / {r['independent_per_req_ms']:6.1f}"
          f"            {r['batched_total_ms']:8.1f} / {r['batched_per_req_ms']:5.1f}")
print("\nMemory (RSS, MB): " + ", ".join(f"{k} {v:.0f}" for k, v in mem.items()))
mb = R["messages_bytes"]
print(f"Messages (bytes): forecast {mb['forecast']}, one track ~{mb['track_per_vehicle_mean']:.0f}, "
      f"10-vehicle convoy {mb['convoy_10_tracks']}, one map {mb['passability_map_1B_per_cell']:,}")
if args.accuracy:
    a = R["accuracy"]
    print("RMSE psi (ours / persistence): " + ", ".join(
        f"h{hh} {a[f'rmse_h{hh}']:.2f} / {a[f'rmse_persistence_h{hh}']:.2f}" for hh in (1, 3, 7)))
    for b, d in a["nogo_h7_cross_year"].items():
        print(f"  band {b} NO-GO at 7 days: recall {d['recall']:.2f}, precision {d['precision']:.2f}, "
              f"false alarms {d['false_alarms_per_vehicle_year']:.1f}/vehicle-year")
print(f"Correctness: {R['correctness']}")
print(f"-> {out.name}")
