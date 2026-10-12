"""
Builds the edge bundle in edge/: the two pickles a test bench needs, sample inputs, and
reference outputs, then checks the standalone runtime (edge/trafficability_edge.py)
against this repo's published forecasts and routes.

    edge/vehicle_trafficability_forecaster.pkl   GRU x3 + ridge, scalers, margins, VCI table
    edge/vehicle_trafficability_planner.pkl      90 m terrain grid, memberships, band vehicles
    edge/sample_weather_log.csv                  last 400 days of the log (one forecast input)
    edge/weather_log_2015_2025.csv               full log (accuracy check)
    edge/sample_trips.json                       30 trips (cell ids, row/col, lat/lon)
    edge/expected_outputs.json                   what the bundle must reproduce

The pickles hold only plain Python types (arrays as dtype, shape, bytes) and carry a
sample weather log and sample trips. dist/vehicle_trafficability_edge.zip holds just the
two pickles and trafficability_edge.py, which runs them (python trafficability_edge.py).

Usage:  python -m traffic.edge_bundle
"""

from __future__ import annotations

import json
import pickle
import shutil
import sys
import zipfile

import numpy as np
import pandas as pd
import torch

from traffic import data as D
from traffic import forecast as FC
from traffic import paths
from traffic import planner as PL
from traffic import run as R
from traffic import terrain as T
from traffic import vehicles as V

EDGE = paths.ROOT / "edge"
DIST = paths.ROOT / "dist"
F_PKL = EDGE / "vehicle_trafficability_forecaster.pkl"
P_PKL = EDGE / "vehicle_trafficability_planner.pkl"
WINDOW = 400           # days of log per forecast; API memory 0.95^340 is negligible
BUNDLE = [F_PKL.name, P_PKL.name, "trafficability_edge.py"]     # what gets sent: nothing else


def _needs_numpy(pkls):
    """Unpickle each file in a fresh interpreter with numpy imports blocked."""
    import subprocess
    code = ("import sys, pickle; sys.modules['numpy'] = None\n"
            "for p in sys.argv[1:]: pickle.load(open(p, 'rb'))")
    r = subprocess.run([sys.executable, "-c", code, *map(str, pkls)], capture_output=True)
    return r.returncode != 0


def forecaster_payload(ds, fleet, reps):
    _, _, (sc, ridges) = FC.fit_ridge(ds)              # deterministic: as in traffic.forecast
    g = torch.load(paths.FORECASTER_OUT / "gru.pt", weights_only=False)
    states = [{k: v.cpu().numpy().astype(np.float32) for k, v in sd.items()} for sd in g["nets"]]

    fc = np.load(paths.FORECASTER_OUT / "test_forecasts.npz")
    y, p = fc["actual"], fc["GRU_Ridge"]
    margins = {b: [FC.tune_margin(y[:, h], p[:, h], grp.vci1.values) for h in range(R.H)]
               for b, grp in fleet.groupby("band")}
    dec = pd.read_csv(paths.FORECASTER_OUT / "decisions_by_band.csv")
    dec = dec[dec.rule == "GRU+Ridge"]
    cross = {b: {int(r.h): dict(tuned_2024=float(r.margin_tuned_2024), tuned_2025=float(r.margin_tuned_2025))
                 for r in grp.itertuples()} for b, grp in dec.groupby("band")}
    reg = pd.read_csv(paths.FORECASTER_OUT / "regression_test.csv")
    n_gru = sum(v.size for v in states[0].values())
    n_ridge = sum(m.coef_.size + 1 for m in ridges)
    return dict(
        format="vehicle_trafficability_forecaster/1",
        L=R.L, H=R.H, window_days=WINDOW, lat_deg=D.LAT_DEG,
        rci=dict(rci_dry=D.RCI_DRY, rci_sat=D.RCI_SAT, k=D.K_DECAY, w_surf=D.W_SURF,
                 w_deep=D.W_DEEP, theta_dry=D.THETA_DRY, theta_sat=D.THETA_SAT,
                 rain_max=D.RAIN_PENALTY_MAX, rain_scale=D.RAIN_PENALTY_SCALE),
        tab_cols=list(ds["tab_cols"]), seq_cols=list(ds["seq_cols"]),
        ridge_mean=sc.mean_, ridge_scale=sc.scale_,
        ridge_coef=np.stack([m.coef_ for m in ridges]), ridge_intercept=np.array([m.intercept_ for m in ridges]),
        gru_cfg=dict(n_in=len(ds["seq_cols"]), H=R.H, hidden=128, layers=2, dropout=0.2),
        gru_states=states, seq_mu=g["mu"], seq_sd=g["sd"], dmu=g["dmu"], dsd=g["dsd"],
        margins=margins, margins_cross_year=cross,
        bands=[(b, float(lo), float(hi)) for b, lo, hi in V.BANDS],
        band_vehicles=reps,
        vehicles={c: fleet[c].tolist() for c in ("name", "type", "mass_t", "vci1", "vci50", "band")},
        params=dict(gru_each=int(n_gru), gru_count=len(states), ridge=int(n_ridge),
                    total=int(len(states) * n_gru + n_ridge)),
        info=dict(site="Pachmarhi plateau, Satpura range, 1,067 m", units="psi",
                  train="2015-2021 (ridge refit on 2015-2023)", validation="2022-2023",
                  test="2024-2025", recall_target=FC.RECALL_TARGET,
                  margins="deployed: tuned on 2024+2025 pooled at 90% NO-GO recall; "
                          "margins_cross_year: those behind the reported results",
                  test_rmse_psi={f"h{int(r.h)}": round(float(r.rmse), 2)
                                 for r in reg[reg.model == "GRU+Ridge"].itertuples()},
                  test_rmse_persistence_psi={f"h{int(r.h)}": round(float(r.rmse), 2)
                                             for r in reg[reg.model == "Persistence"].itertuples()}),
    )


def planner_payload(t, g, reps):
    return dict(
        format="vehicle_trafficability_planner/1",
        shape=list(g.shape), cell_m=float(T.CELL_M), slow_floor=PL.SLOW_FLOOR,
        valid=t["valid"], paved=g.paved, decay=g.decay,
        static={k: tuple(np.asarray(a, np.float64) for a in v) for k, v in g.static.items()},
        band_vehicles=reps, crs=str(t["crs"]), transform=tuple(t["transform"])[:6],
        info="fuzzy AND (min) of slope, land cover and soil; speed proportional to membership",
    )


def trips_payload(t, sources, dests):
    from rasterio.transform import xy
    from rasterio.warp import transform as warp
    cols = t["valid"].shape[1]
    out = []
    for s in sources:
        for d in dests:
            row = {}
            for tag, cell in (("src", int(s)), ("dst", int(d))):
                r, c = divmod(cell, cols)
                x, y = xy(t["transform"], r, c)
                lon, lat = warp(t["crs"], "EPSG:4326", [x], [y])
                row[tag] = dict(cell=cell, row=r, col=c, lat=round(lat[0], 5), lon=round(lon[0], 5))
            out.append(row)
    return out


def main():
    EDGE.mkdir(exist_ok=True)
    ds = R.build_arrays()
    fleet = V.add_vci(V.load_vehicles())
    reps = PL.representatives(fleet)
    reps = {b: dict(kind=v["kind"], vci1=v["vci1"], vci50=v["vci50"], n=int(v["n"])) for b, v in reps.items()}
    t = T.load()
    g = PL.Grid(t)
    setup = json.load(open(PL.OUT / "setup.json"))

    sys.path.insert(0, str(EDGE))
    import trafficability_edge as E
    raw = pd.read_csv(paths.WEATHER_CSV)
    fpay, ppay = forecaster_payload(ds, fleet, reps), planner_payload(t, g, reps)
    # sample inputs travel inside the pickles, so the runtime file alone can run them
    fpay["sample_log"] = {c: raw[c].iloc[-WINDOW:].tolist() for c in raw.columns}
    ppay["sample_trips"] = [[int(s), int(d)] for s in setup["sources"] for d in setup["dests"]]
    for path, obj in ((F_PKL, fpay), (P_PKL, ppay)):
        with open(path, "wb") as f:
            pickle.dump(E.encode(obj), f, protocol=4)
    raw.to_csv(EDGE / "weather_log_2015_2025.csv", index=False)
    raw.iloc[-WINDOW:].to_csv(EDGE / "sample_weather_log.csv", index=False)
    trips = trips_payload(t, setup["sources"], setup["dests"])
    json.dump(trips, open(EDGE / "sample_trips.json", "w"), indent=1)

    # ---------------------------------------------------------------- checks
    # the pickles must unpickle with no numpy at all (plain Python types only)
    report = {"pickles_need_numpy_to_load": _needs_numpy([F_PKL, P_PKL])}
    ref = np.load(paths.FORECASTER_OUT / "test_forecasts.npz")
    ref_dates = pd.to_datetime(ref["dates"])
    for backend in ("torch", "numpy"):
        m = E.Forecaster(F_PKL, backend=backend)
        allf = m.forecast_all(EDGE / "weather_log_2015_2025.csv").set_index("date")
        got = allf.loc[ref_dates, [f"rci_h{h}" for h in range(1, 8)]].values
        report[f"max_diff_vs_published_{backend}_psi"] = float(np.abs(got - ref["GRU_Ridge"]).max())
    # one forecast from a 400-day window equals the full-history forecast for that day
    m = E.Forecaster(F_PKL)
    one = m.forecast(EDGE / "sample_weather_log.csv")
    report["max_diff_window_vs_full_psi"] = float(np.abs(one["rci"] - allf.iloc[-1, 1:].values).max())

    pl = E.Planner(P_PKL)
    pairs = [(r["src"]["cell"], r["dst"]["cell"]) for r in trips]
    same = 0
    for band in "ABCDE":
        rep = reps[band]
        for rci in (25.0, 35.0, 60.0, 150.0):
            mine = pl.plan(rci, band, pairs)
            theirs = g.routes(g.passability(rep["kind"], g.soil(rci, rep["vci1"], rep["vci50"])),
                              setup["sources"], setup["dests"])
            for (p, _), (s, d) in zip(mine, pairs):
                q = theirs[(s, d)]
                same += (p is None and q is None) or (p is not None and q is not None and np.array_equal(p, q))
    report["routes_identical_to_repo_planner"] = f"{same}/{5 * 4 * len(pairs)}"

    h, band = 3, "D"
    fc, rci, res = E.end_to_end(m, pl, EDGE / "sample_weather_log.csv", band, pairs, h)
    # a convoy case where sharing tracks matters: wear-aware gets more through than follow
    cv = None
    for s, d in pairs:
        for r in np.arange(28.0, 80.0, 1.0):
            w, f = pl.convoy(r, "D", s, d), pl.convoy(r, "D", s, d, wear_aware=False)
            if w[0] > f[0] + 2:
                cv = dict(rci=float(r), band="D", src=int(s), dst=int(d), through=w[0], bytes=w[1],
                          through_follow_leader=f[0])
                break
        if cv:
            break
    expected = dict(
        forecast=dict(date=str(fc["date"].date()), rci_today=fc["rci_today"], rci=fc["rci"].tolist(),
                      message_hex=E.Forecaster.message(fc).hex()),
        assess={b: m.assess(fc, band=b) for b in "ABCDE"},
        routes=dict(band=band, h=h, rci_after_margin=rci,
                    trips=[dict(src=s, dst=d, cells=None if p is None else len(p), time=tt)
                           for (s, d), (p, tt) in zip(pairs, res)]),
        convoy=cv,
        checks=report,
    )
    json.dump(expected, open(EDGE / "expected_outputs.json", "w"), indent=1, default=float)

    DIST.mkdir(exist_ok=True)
    zpath = DIST / "vehicle_trafficability_edge.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for name in BUNDLE:
            if (EDGE / name).exists():
                z.write(EDGE / name, f"vehicle_trafficability_edge/{name}")
            else:
                print(f"  (not in zip yet: {name})")
    for k, v in report.items():
        print(f"  {k}: {v}")
    for f in (F_PKL, P_PKL, zpath):
        print(f"-> {f.relative_to(paths.ROOT)}  {f.stat().st_size / 2**20:.2f} MB")


if __name__ == "__main__":
    main()
