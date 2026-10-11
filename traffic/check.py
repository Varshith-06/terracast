"""
Setup check: packages, input files, loaders and terrain grids, in one pass.

Usage:  python -m traffic.check
"""

from __future__ import annotations

import importlib
import sys

from traffic import paths

PACKAGES = ["numpy", "pandas", "scipy", "sklearn", "statsmodels", "lightgbm", "torch",
            "matplotlib", "pyarrow", "rasterio", "shapely", "networkx"]
MODULES = ["data", "dl", "run", "report", "artifact", "calibration", "vehicles",
           "forecast", "terrain", "planner", "poster_figs", "poster_build"]

_fails = 0


def _report(ok: bool, what: str, detail: str = ""):
    global _fails
    _fails += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}" + (f"  ({detail})" if detail else ""))


def _try(what, fn):
    try:
        _report(True, what, fn() or "")
    except Exception as e:                       # a setup check reports, it never stops
        _report(False, what, f"{type(e).__name__}: {e}")


def check_packages():
    print("packages")
    for name in PACKAGES:
        _try(name, lambda n=name: getattr(importlib.import_module(n), "__version__", "?"))
    import torch
    _report(torch.cuda.is_available(), "CUDA",
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")


def check_inputs():
    print("input files")
    files = [paths.WEATHER_CSV, paths.EXTREME_EVENTS_CSV, paths.WEATHER_METADATA_CSV,
             paths.VEHICLES_CSV]
    files += [f(s) for s in paths.SEASONS for f in (paths.terrain_stack, paths.terrain_table)]
    for f in files:
        _report(f.exists(), str(f.relative_to(paths.ROOT)))


def check_loaders():
    print("loaders")
    for m in MODULES:
        _try(f"import traffic.{m}", lambda m=m: importlib.import_module(f"traffic.{m}") and "")

    from traffic import data as D, vehicles as V

    def weather():
        raw, feats = D.get_dataset()
        assert len(raw) == 4018, len(raw)
        return f"{len(raw)} days, {feats.shape[1]} feature columns, RCI {raw.rci.min():.1f}-{raw.rci.max():.1f}"

    def vehicles():
        mi, v1, v50 = V.m1_check()
        assert abs(mi - 98.59) < 0.1 and abs(v1 - 26.34) < 0.05, (mi, v1)
        v = V.load_vehicles()
        return f"{len(v)} vehicles {v['type'].value_counts().to_dict()}; M1 check MI {mi:.2f}"

    _try("weather + RCI chain", weather)
    _try("vehicles + VCI (M1 check)", vehicles)


def check_terrain():
    print("terrain")
    import pyarrow.parquet as pq
    import rasterio

    grids = {}
    for s in paths.SEASONS:
        def one(s=s):
            with rasterio.open(paths.terrain_stack(s)) as r:
                grids[s] = (r.crs.to_string(), r.transform, r.width, r.height, r.descriptions)
                n_rows = pq.ParquetFile(paths.terrain_table(s)).metadata.num_rows
                assert n_rows == r.width * r.height, (n_rows, r.width * r.height)
                return f"{r.count} bands, {r.width}x{r.height} @ {r.res[0]:.0f} m, {r.crs}"
        _try(s, one)
    if len(grids) == len(paths.SEASONS):
        _report(len(set(grids.values())) == 1, "all seasons share one grid and band order")


def check_outputs():
    print("existing results")
    for f in ["results_full.csv", "test_predictions.npz", "RESULTS.md"]:
        p = paths.FORECAST_OUT / f
        _report(p.exists(), str(p.relative_to(paths.ROOT)))
    _report(paths.VEHICLES_VCI_CSV.exists(), str(paths.VEHICLES_VCI_CSV.relative_to(paths.ROOT)))


if __name__ == "__main__":
    print(f"python {sys.version.split()[0]}   root {paths.ROOT}\n")
    for step in (check_packages, check_inputs, check_loaders, check_terrain, check_outputs):
        step()
        print()
    print("all checks passed" if not _fails else f"{_fails} check(s) failed")
    sys.exit(1 if _fails else 0)
