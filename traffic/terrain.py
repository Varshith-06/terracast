"""
Static terrain for the route planner: a 90 m grid over the Pachmarhi GEE export with
slope, land cover, canopy density, OSM roads and a topographic wetness index, plus
the fuzzy passability memberships each vehicle type gets from them.

Only soil strength changes from day to day (traffic.planner); everything here is
fixed. The three seasonal stacks share one grid; WorldCover, DEM, slope are the same
in all three, and NDVI is reduced to its seasonal peak as a canopy-density proxy
(the plateau forest is deciduous, so single-season NDVI tracks leaf-out, not stems).

Membership tables follow Liu et al. (2025), Sci. Rep. 15:9957, Tables 1 and 3.

Usage:  python -m traffic.terrain
"""

from __future__ import annotations

import json

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.warp import transform_geom
from scipy.interpolate import PchipInterpolator

from traffic import paths

AGG = 3                                   # 30 m -> 90 m
CELL_M = 30.0 * AGG
ROADS_JSON = paths.TERRAIN / "osm_roads_pachmarhi.json"
TERRAIN_NPZ = paths.OUTPUTS / "terrain" / "terrain_90m.npz"

# OSM highway classes a military vehicle can use; footways, paths and steps are not
VEHICLE_ROADS = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
                 "residential", "service", "track"}
GRADED = {"motorway", "trunk", "primary", "secondary", "tertiary", "residential"}
PAVED_SURFACES = {"asphalt", "concrete", "paved", "paving_stones", "sett"}

# ESA WorldCover classes
TREE, SHRUB, GRASS, CROP, BUILT, BARE, WATER, WETLAND = 10, 20, 30, 40, 50, 60, 80, 90
DENSE_NDVI, OPEN_NDVI = 0.75, 0.60       # peak-NDVI split of tree cover


def _block(a, fn):
    h, w = (a.shape[0] // AGG) * AGG, (a.shape[1] // AGG) * AGG
    b = a[:h, :w].reshape(h // AGG, AGG, w // AGG, AGG).swapaxes(1, 2).reshape(h // AGG, w // AGG, -1)
    return fn(b, axis=-1)


def _mode(b, axis=-1):
    vals = np.sort(b, axis=axis)
    out = vals[..., 0].copy()
    best = np.zeros(out.shape, int)
    for v in np.unique(vals):
        c = (b == v).sum(axis=axis)
        upd = c > best
        out[upd], best[upd] = v, c[upd]
    return out


def _roads(shape30, transform30, crs):
    """Rasterise OSM vehicle roads at 30 m: 2 = graded/paved, 1 = unpaved, 0 = none."""
    els = json.load(open(ROADS_JSON))["elements"]
    shapes = []
    for e in els:
        tags = e.get("tags", {})
        hw = tags.get("highway")
        if hw not in VEHICLE_ROADS or "geometry" not in e:
            continue
        paved = hw in GRADED or tags.get("surface") in PAVED_SURFACES
        if tags.get("surface") in {"unpaved", "ground", "dirt", "grass", "gravel", "fine_gravel"}:
            paved = False
        geom = {"type": "LineString", "coordinates": [[p["lon"], p["lat"]] for p in e["geometry"]]}
        shapes.append((transform_geom("EPSG:4326", crs, geom), 2 if paved else 1))
    shapes.sort(key=lambda s: s[1])                     # paved burns last, wins
    return rasterize(shapes, out_shape=shape30, transform=transform30, fill=0,
                     all_touched=True, dtype="uint8")


def _twi(dem, valid):
    """Topographic wetness index ln(a / tan b) with D8 flow accumulation."""
    gy, gx = np.gradient(np.where(valid, dem, np.nan), CELL_M)
    slope = np.arctan(np.hypot(gx, gy))
    h, w = dem.shape
    z = np.where(valid, dem, np.inf)
    acc = np.where(valid, 1.0, 0.0)
    order = np.argsort(np.where(valid, -dem, np.inf), axis=None)
    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    dist = [np.hypot(a, b) for a, b in nbrs]
    for k in order:
        r, c = divmod(int(k), w)
        if not valid[r, c]:
            continue
        best, tgt = 0.0, None
        for (dr, dc), d in zip(nbrs, dist):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and valid[rr, cc]:
                drop = (dem[r, c] - z[rr, cc]) / d
                if drop > best:
                    best, tgt = drop, (rr, cc)
        if tgt is not None:
            acc[tgt] += acc[r, c]
    twi = np.log(acc * CELL_M / np.maximum(np.tan(np.nan_to_num(slope)), 0.001))
    return np.where(valid, twi, np.nan)


def build():
    bands = {}
    peak_ndvi = None
    for s in paths.SEASONS:
        with rasterio.open(paths.terrain_stack(s)) as r:
            b = {n: r.read(i + 1) for i, n in enumerate(r.descriptions)}
            crs, tr30, shape30 = r.crs, r.transform, (r.height, r.width)
        peak_ndvi = b["s2_ndvi"] if peak_ndvi is None else np.fmax(peak_ndvi, b["s2_ndvi"])
        if s == "monsoon":
            bands = b
    valid30 = (bands["esa_worldcover"] > 0) & (bands["dem_elevation_m"] > 0) & ~np.isnan(peak_ndvi)

    roads30 = _roads(shape30, tr30, crs)
    valid = _block(valid30, np.mean) > 0.5
    out = dict(
        valid=valid,
        elevation=_block(np.where(valid30, bands["dem_elevation_m"], np.nan), np.nanmean),
        slope=_block(np.where(valid30, bands["dem_slope_deg"], np.nan), np.nanmean),
        cover=_block(np.where(valid30, bands["esa_worldcover"], 0), _mode).astype(np.int16),
        ndvi=_block(np.where(valid30, peak_ndvi, np.nan), np.nanmean),
        water=_block(np.where(valid30, bands["esa_worldcover"] == WATER, False), np.mean) > 0.5,
        road=_block(roads30, np.max).astype(np.int8),
    )
    out["twi"] = _twi(np.nan_to_num(out["elevation"]), valid)
    tr90 = tr30 * Affine.scale(AGG, AGG)
    TERRAIN_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(TERRAIN_NPZ, transform=np.array(tr90)[:6], crs=str(crs), **out)
    return load()


def load():
    z = np.load(TERRAIN_NPZ, allow_pickle=False)
    t = {k: z[k] for k in z.files}
    t["transform"] = Affine(*t["transform"])
    t["crs"] = str(t["crs"])
    return t


# ------------------------------------------------------------------ memberships
# Liu et al. Table 1 slope classes, through bin midpoints, made smooth and monotone
_SLOPE_X = [0.0, 4.5, 8.0, 12.5, 17.5, 25.0, 32.5, 37.5]
_SLOPE_Y = {"wheeled": [1.0, 0.9, 0.675, 0.54, 0.36, 0.225, 0.0, 0.0],
            "tracked": [1.0, 0.9, 0.72, 0.60, 0.36, 0.24, 0.12, 0.0]}
# Liu et al. Table 3 land cover; tree cover split by canopy into their
# forest / bush-wood / grassland rows
_COVER = {"wheeled": {"dense": 0.0, "open": 0.0, "sparse": 0.4, SHRUB: 0.0, GRASS: 0.4,
                      CROP: 0.3, BUILT: 0.5, BARE: 0.8, WATER: 0.0, WETLAND: 0.3},
          "tracked": {"dense": 0.0, "open": 0.2, "sparse": 0.6, SHRUB: 0.2, GRASS: 0.6,
                      CROP: 0.5, BUILT: 0.5, BARE: 1.0, WATER: 0.0, WETLAND: 0.6}}


def slope_membership(slope_deg, kind):
    f = PchipInterpolator(_SLOPE_X, _SLOPE_Y[kind])
    return np.clip(f(np.clip(np.nan_to_num(slope_deg, nan=90.0), 0, 37.5)), 0, 1)


def cover_membership(t, kind):
    tab = _COVER[kind]
    mu = np.zeros(t["cover"].shape)
    for cls in (SHRUB, GRASS, CROP, BUILT, BARE, WATER, WETLAND):
        mu[t["cover"] == cls] = tab[cls]
    tree = t["cover"] == TREE
    nd = np.nan_to_num(t["ndvi"])
    mu[tree & (nd >= DENSE_NDVI)] = tab["dense"]
    mu[tree & (nd >= OPEN_NDVI) & (nd < DENSE_NDVI)] = tab["open"]
    mu[tree & (nd < OPEN_NDVI)] = tab["sparse"]
    return mu


def static_memberships(t, kind):
    """kind: 'wheeled' or 'tracked' (half-tracks use the wheeled tables: steered axle).
    Road cells are graded: slope and cover are 1 on them. Water is always 0."""
    k = "wheeled" if kind == "halftrack" else kind
    on_road = t["road"] > 0
    mu_slope = np.where(on_road, 1.0, slope_membership(t["slope"], k))
    mu_cover = np.where(on_road, 1.0, cover_membership(t, k))
    blocked = ~t["valid"] | t["water"]
    mu_slope[blocked] = 0.0
    mu_cover[blocked] = 0.0
    return mu_slope, mu_cover


def wetness_shift(t, scale=0.10, clip=2.0):
    """Per-cell shift of the soil wetness index from the TWI z-score: low, convergent
    cells wetter, ridges drier, area mean unchanged. A heuristic with no field data."""
    v = t["valid"]
    z = (t["twi"] - np.nanmean(t["twi"][v])) / np.nanstd(t["twi"][v])
    return np.where(v, scale * np.clip(np.nan_to_num(z), -clip, clip), 0.0)


if __name__ == "__main__":
    t = build()
    v = t["valid"]
    print("grid", v.shape, "valid cells", int(v.sum()), "roads", np.bincount(t["road"][v].ravel()).tolist())
    for kind in ("wheeled", "tracked"):
        s, c = static_memberships(t, kind)
        m = np.minimum(s, c)[v]
        print(f"{kind:8s} static passable (mu>0) {100*(m>0).mean():.1f}%   mean mu {m.mean():.2f}")
    print("TWI z range", np.nanpercentile(wetness_shift(t)[v], [1, 50, 99]).round(3))
    print("->", TERRAIN_NPZ.relative_to(paths.ROOT))
