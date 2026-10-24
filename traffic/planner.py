"""
Forecast-aware route planning on the 90 m terrain grid, its backtest over the
2024-2025 test years, and the convoy wear experiment.

Cell passability is a fuzzy AND of slope, land cover and soil memberships. Soil
comes from one plateau-wide RCI value spread per cell by the wetness index:
    RCI_cell = RCI * exp(-K_DECAY * shift_cell)
which moves the cell's soil wetness index by shift_cell in the same exponential model
that defines RCI. Graded (paved) roads ignore soil; unpaved tracks do not.

Travel cost is time at a speed proportional to passability, so a SLOW-GO cell is
slow and a NO-GO cell is impassable. Shortest paths use Dijkstra (SciPy), which
returns the same optimal paths as A*.

Every trip starts and ends on the largest connected piece of the road network (paved
and unpaved). Unpaved roads soften like any soil, so on very wet days a trip can be
impossible for a heavy vehicle. Each planner's trip is scored as: stuck (sent along a
route with a NO-GO cell on the travel day), held back (no route planned although one
existed), or completed, with its time against the best possible route that day.

Usage:  python -m traffic.planner
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy.ndimage import label
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from traffic import data as D
from traffic import mobility as M
from traffic import paths
from traffic import terrain as T
from traffic import vehicles as V

OUT = paths.OUTPUTS / "planner"
HORIZONS = [1, 3, 7]
N_SOURCES, N_DESTS, MIN_TRIP_KM = 6, 5, 8.0
CONVOY = 10
SLOW_FLOOR = 0.25                 # soil membership just above VCI1 (SLOW-GO, slow)
_NBRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# ------------------------------------------------------------------ vehicles
def representatives(fleet):
    """One planning vehicle per band: majority type, median VCI1 / VCI50; for the speed
    model, the median power-to-weight, contact pressure factor and top speed of the
    band's vehicles of that type."""
    reps = {}
    for band, g in fleet.groupby("band"):
        kind = g["type"].mode().iloc[0]
        k = g[g["type"] == kind]
        vmax = k["vmax_kmh"][k["vmax_kmh"] > 0]
        reps[band] = dict(kind=kind, vci1=float(g["vci1"].median()),
                          vci50=float(g["vci50"].median()), n=len(g),
                          hp_per_t=float((k["hp"] / k["mass_t"]).median()),
                          cpf=float(k["cpf"].median()), vmax_kmh=float(vmax.median()))
    return reps


# ------------------------------------------------------------------- maps
class Grid:
    def __init__(self, t):
        self.t = t
        self.shape = t["valid"].shape
        self.n = t["valid"].size
        self.paved = (t["road"] == 2) & t["valid"]
        self.decay = np.exp(-D.K_DECAY * T.wetness_shift(t))   # RCI_cell = RCI * decay
        self.static = {k: T.static_memberships(t, k) for k in ("wheeled", "tracked")}
        r, c = np.indices(self.shape)
        self._edges = []
        for dr, dc in _NBRS:
            ok = (r + dr >= 0) & (r + dr < self.shape[0]) & (c + dc >= 0) & (c + dc < self.shape[1])
            u = (r * self.shape[1] + c)[ok]
            v = ((r + dr) * self.shape[1] + (c + dc))[ok]
            self._edges.append((u, v, T.CELL_M / 1000 * np.hypot(dr, dc)))
        # sine of the slope along each edge, from u to v (+ uphill), for the speed model
        z = np.nan_to_num(t["elevation"]).ravel()
        self._sin = [(z[v] - z[u]) / np.hypot(z[v] - z[u], step * 1000) for u, v, step in self._edges]

    def soil(self, rci, vci1, vci50, passes=None):
        """Soil membership per cell. passes: prior vehicle passes per cell (convoy wear),
        which raise the one-pass threshold towards VCI50 as ln(n)/ln(50)."""
        rc = rci * self.decay
        need = np.full(self.shape, vci1)
        if passes is not None:
            n = passes + 1.0
            need = vci1 + (vci50 - vci1) * np.log(n) / np.log(50.0)
        mu = SLOW_FLOOR + (1 - SLOW_FLOOR) * np.clip((rc - need) / np.maximum(vci50 - need, 1e-6), 0, 1)
        mu = np.where(rc < need, 0.0, mu)
        return np.where(self.paved, 1.0, mu)

    def passability(self, kind, mu_soil, method="min"):
        k = "wheeled" if kind == "halftrack" else kind
        s, c = self.static[k]
        if method == "min":
            return np.minimum(np.minimum(s, c), mu_soil)
        if method == "product":
            return s * c * mu_soil
        mean = (s + c + mu_soil) / 3.0                  # Liu: equal AHP weights here
        if method == "liu_weighted":                    # with Liu's veto on any zero
            return np.where((s == 0) | (c == 0) | (mu_soil == 0), 0.0, mean)
        if method == "weighted_no_veto":
            return np.where((s == 0) | (c == 0), 0.0, mean)
        raise ValueError(method)

    def graph(self, mu):
        f = mu.ravel()
        us, vs, ws = [], [], []
        for u, v, step in self._edges:
            ok = (f[u] > 0) & (f[v] > 0)
            us.append(u[ok]); vs.append(v[ok])
            ws.append(step * 0.5 * (1 / f[u[ok]] + 1 / f[v[ok]]))
        return csr_matrix((np.concatenate(ws), (np.concatenate(us), np.concatenate(vs))),
                          shape=(self.n, self.n))

    def routes(self, mu, sources, dests):
        dist, pred = dijkstra(self.graph(mu), directed=True, indices=sources,
                              return_predecessors=True)
        out = {}
        for i, s in enumerate(sources):
            for d in dests:
                if not np.isfinite(dist[i, d]):
                    out[(s, d)] = None
                    continue
                p, k = [d], d
                while k != s:
                    k = pred[i, k]
                    p.append(k)
                out[(s, d)] = np.array(p[::-1])
        return out

    def path_time(self, path, mu):
        f = mu.ravel()[path]
        if (f <= 0).any():
            return np.inf
        r, c = np.divmod(path, self.shape[1])
        step = T.CELL_M / 1000 * np.hypot(np.diff(r), np.diff(c))
        return float(np.sum(step * 0.5 * (1 / f[:-1] + 1 / f[1:])))

    # ---------------------------------------------------- travel time (traffic.mobility)
    def _cell_speed(self, cells, sin_t, rci, rep, passes):
        """km/h in each cell along a slope: NRMM soil/slope speed (paved roads ignore the
        soil) times the land-cover score off road; 0 where slope or cover rule it out."""
        k = "wheeled" if rep["kind"] == "halftrack" else rep["kind"]
        s, c = (a.ravel()[cells] for a in self.static[k])
        need = None
        if passes is not None:
            need = rep["vci1"] + (rep["vci50"] - rep["vci1"]) * np.log(passes.ravel()[cells] + 1.0) / np.log(50.0)
        v = M.speed(rci * self.decay.ravel()[cells], sin_t, rep, self.paved.ravel()[cells], need)
        return np.where(s > 0, v * c, 0.0)

    def hours_graph(self, rci, rep, passes=None):
        """Edge weights in hours: half of each edge at the speed of either end cell."""
        us, vs, ws = [], [], []
        for (u, v, step), sin_t in zip(self._edges, self._sin):
            a = self._cell_speed(u, sin_t, rci, rep, passes)
            b = self._cell_speed(v, sin_t, rci, rep, passes)
            ok = (a > 0) & (b > 0)
            us.append(u[ok]); vs.append(v[ok])
            ws.append(step * 0.5 * (1 / a[ok] + 1 / b[ok]))
        return csr_matrix((np.concatenate(ws), (np.concatenate(us), np.concatenate(vs))),
                          shape=(self.n, self.n))

    def path_hours(self, path, rci, rep, passes=None):
        """Travel time of a path in hours on ground of plateau strength `rci`; inf if any
        stretch of it is impassable."""
        r, c = np.divmod(path, self.shape[1])
        step = T.CELL_M / 1000 * np.hypot(np.diff(r), np.diff(c))
        z = np.nan_to_num(self.t["elevation"]).ravel()[path]
        sin_t = np.diff(z) / np.hypot(np.diff(z), step * 1000)
        a = self._cell_speed(path[:-1], sin_t, rci, rep, passes)
        b = self._cell_speed(path[1:], sin_t, rci, rep, passes)
        if (a <= 0).any() or (b <= 0).any():
            return np.inf
        return float(np.sum(step * 0.5 * (1 / a + 1 / b)))

    def fastest(self, rci, rep, sources, dests, passes=None):
        """Routes that minimise travel time (hours) instead of the passability cost."""
        G = self.hours_graph(rci, rep, passes)
        dist, pred = dijkstra(G, directed=True, indices=sources, return_predecessors=True)
        out = {}
        for i, s in enumerate(sources):
            for d in dests:
                if not np.isfinite(dist[i, d]):
                    out[(s, d)] = None
                    continue
                p, k = [d], d
                while k != s:
                    k = pred[i, k]
                    p.append(k)
                out[(s, d)] = np.array(p[::-1])
        return out


def _spread(xy, k, start):
    """Farthest-point sample of k rows of xy, starting from row `start`."""
    pick = [start]
    d = np.hypot(*(xy - xy[start]).T)
    for _ in range(k - 1):
        pick.append(int(d.argmax()))
        d = np.minimum(d, np.hypot(*(xy - xy[pick[-1]]).T))
    return np.array(pick)


def trip_endpoints(g, seed=0):
    """Sources spread over one end of the largest road network, destinations spread
    over the far end, every trip at least MIN_TRIP_KM in a straight line."""
    lab, _ = label((g.t["road"] > 0) & g.t["valid"], structure=np.ones((3, 3)))
    big = np.bincount(lab.ravel())[1:].argmax() + 1
    cells = np.flatnonzero(lab.ravel() == big)
    rc = np.column_stack(np.divmod(cells, g.shape[1])) * T.CELL_M / 1000
    axis = np.linalg.svd(rc - rc.mean(0), full_matrices=False)[2][0]
    proj = (rc - rc.mean(0)) @ axis
    lo = np.flatnonzero(proj < np.quantile(proj, 0.3))
    src = lo[_spread(rc[lo], N_SOURCES, int(np.argmin(proj[lo])))]
    far = np.hypot(*(rc[:, None, :] - rc[src][None, :, :]).transpose(2, 0, 1)).min(1)
    hi = np.flatnonzero((proj > np.quantile(proj, 0.5)) & (far >= MIN_TRIP_KM))
    dst = hi[_spread(rc[hi], N_DESTS, int(np.argmax(proj[hi])))]
    return cells[src], cells[dst]


# ------------------------------------------------------------------ backtest
def _cross_year_margin(dec, rule, band, h, year):
    r = dec[(dec.rule == rule) & (dec.band == band) & (dec.h == h)].iloc[0]
    return r["margin_tuned_2025"] if year == 2024 else r["margin_tuned_2024"]


def backtest(g, reps, fc, dec, sources, dests, methods=("min",)):
    actual, dates = fc["actual"], pd.to_datetime(fc["dates"])
    anchor = fc["Persistence"][:, 0]
    rows = []
    cache = {}

    def plan(band, rci, method):
        rep = reps[band]
        key = (band, method, None if rci is None else round(float(rci) * 2) / 2)
        if key not in cache:
            soil = np.ones(g.shape) if rci is None else g.soil(key[2], rep["vci1"], rep["vci50"])
            cache[key] = g.routes(g.passability(rep["kind"], soil, method), sources, dests)
        return cache[key]

    for i, day in enumerate(dates):
        yr = day.year
        for h in HORIZONS:
            r_act = actual[i, h - 1]
            for band, rep in reps.items():
                mu_act = g.passability(rep["kind"], g.soil(r_act, rep["vci1"], rep["vci50"]))
                oracle = plan(band, r_act, "min")
                plans = {
                    "Static (terrain only)": ("min", None),
                    "Today (Liu et al.)": ("min", anchor[i]),
                    "Today + margin": ("min", anchor[i] - _cross_year_margin(dec, "Persistence", band, h, yr)),
                }
                m_fc = _cross_year_margin(dec, "GRU+Ridge", band, h, yr)
                for meth in methods:
                    tag = "Forecast + margin (ours)" if meth == "min" else f"Forecast + margin [{meth}]"
                    plans[tag] = (meth, fc["GRU_Ridge"][i, h - 1] - m_fc)
                for name, (meth, rci) in plans.items():
                    routes = plan(band, rci, meth)
                    stuck = held = done = possible = 0
                    extra = 0.0
                    for od, path in routes.items():
                        best = g.path_time(oracle[od], mu_act) if oracle[od] is not None else np.inf
                        possible += np.isfinite(best)
                        if path is None:
                            held += np.isfinite(best)
                            continue
                        tt = g.path_time(path, mu_act)
                        if not np.isfinite(tt):
                            stuck += 1
                        else:
                            done += 1
                            # the oracle comes from the map cached at the rounded RCI, so on
                            # rare days its route fails at the exact value: no extra time then
                            extra += tt / best - 1 if np.isfinite(best) else 0.0
                    rows.append(dict(date=day, h=h, band=band, planner=name, trips=len(routes),
                                     possible=possible, stuck=stuck, held=held, done=done,
                                     extra_time_sum=extra, rci_actual=r_act, rci_today=anchor[i]))
        if i % 60 == 0:
            print(f"  {day.date()}  maps cached {len(cache)}")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ trade-off
MARGIN_SCALES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]


def tradeoff(g, reps, fc, dec, sources, dests, horizons=(3, 7)):
    """Stuck vs held for today's value and the forecast, each with its cross-year margin
    scaled from 0 (no margin) to 3x: the caution curve of each information source."""
    actual, dates = fc["actual"], pd.to_datetime(fc["dates"])
    anchor = fc["Persistence"][:, 0]
    cache, rows = {}, []

    def plan(band, rci):
        rep, key = reps[band], (band, round(float(rci) * 2) / 2)
        if key not in cache:
            mu = g.passability(rep["kind"], g.soil(key[1], rep["vci1"], rep["vci50"]))
            cache[key] = g.routes(mu, sources, dests)
        return cache[key]

    for i, day in enumerate(dates):
        for h in horizons:
            r_act = actual[i, h - 1]
            for band, rep in reps.items():
                mu_act = g.passability(rep["kind"], g.soil(r_act, rep["vci1"], rep["vci50"]))
                oracle = plan(band, r_act)
                best = {od: (g.path_time(p, mu_act) if p is not None else np.inf)
                        for od, p in oracle.items()}
                for rule, base in [("Persistence", anchor[i]), ("GRU+Ridge", fc["GRU_Ridge"][i, h - 1])]:
                    m = _cross_year_margin(dec, rule, band, h, day.year)
                    for k in MARGIN_SCALES:
                        stuck = held = 0
                        for od, path in plan(band, base - k * m).items():
                            if path is None:
                                held += np.isfinite(best[od])
                            elif not np.isfinite(g.path_time(path, mu_act)):
                                stuck += 1
                        rows.append(dict(date=day, h=h, band=band, rule=rule, scale=k,
                                         trips=len(oracle), possible=int(np.isfinite(list(best.values())).sum()),
                                         stuck=stuck, held=held))
        if i % 120 == 0:
            print(f"  tradeoff {day.date()}  maps cached {len(cache)}", flush=True)
    return pd.DataFrame(rows)


def main_tradeoff():
    g = Grid(T.load())
    reps = representatives(V.add_vci(V.load_vehicles()))
    fc = dict(np.load(paths.FORECASTER_OUT / "test_forecasts.npz", allow_pickle=False))
    dec = pd.read_csv(paths.FORECASTER_OUT / "decisions_by_band.csv")
    st = json.load(open(OUT / "setup.json"))
    tr = tradeoff(g, reps, fc, dec, np.array(st["sources"]), np.array(st["dests"]))
    tr.to_csv(OUT / "tradeoff.csv", index=False)
    s = tr.groupby(["h", "rule", "scale"])[["trips", "possible", "stuck", "held"]].sum()
    s["stuck_pct"] = 100 * s.stuck / s.trips
    s["held_pct"] = 100 * s.held / s.possible
    print(s[["stuck_pct", "held_pct"]].round(2))


# ---------------------------------------------------------------------- speed
def speed_backtest(g, reps, fc, dec, sources, dests, horizons=(1, 3, 7)):
    """Travel times from the speed model (traffic.mobility) on the forecast + margin map.
    Two route choices: the passability routes the main backtest scores, and routes that
    minimise travel time. Each trip records its predicted time (on the forecast map), its
    actual time on the travel day, and the fastest time possible that day. A trip is
    stuck if any stretch of it is impassable on the travel day: a NO-GO cell, or a slope
    the soil cannot carry the vehicle up."""
    actual, dates = fc["actual"], pd.to_datetime(fc["dates"])
    q = lambda r: round(float(r) * 2) / 2                     # maps cached on a 0.5 psi grid
    fast, pas = {}, {}
    rows = []

    def fastest(band, r):
        k = (band, q(r))
        if k not in fast:
            fast[k] = g.fastest(k[1], reps[band], sources, dests)
        return fast[k]

    def passable(band, r):
        k = (band, q(r))
        if k not in pas:
            rep = reps[band]
            pas[k] = g.routes(g.passability(rep["kind"], g.soil(k[1], rep["vci1"], rep["vci50"])),
                              sources, dests)
        return pas[k]

    for i, day in enumerate(dates):
        for h in horizons:
            r_act = actual[i, h - 1]
            for band, rep in reps.items():
                # the oracle routes come from the map cached at the rounded value, as in the
                # main backtest; their times, like every other, use the exact travel-day RCI
                best = {od: (np.inf if p is None else g.path_hours(p, r_act, rep))
                        for od, p in fastest(band, r_act).items()}
                r_fc = fc["GRU_Ridge"][i, h - 1] - _cross_year_margin(dec, "GRU+Ridge", band, h, day.year)
                for name, routes in (("passability", passable(band, r_fc)), ("fastest", fastest(band, r_fc))):
                    stuck = held = done = n_best = n_pred = no_pred = 0
                    pred = pred_act = act = best_act = opt = 0.0
                    for od, path in routes.items():
                        b = best[od]
                        if path is None:
                            held += np.isfinite(b)
                            continue
                        a = g.path_hours(path, r_act, rep)
                        if not np.isfinite(a):
                            stuck += 1
                            continue
                        done += 1
                        act += a
                        if np.isfinite(b):
                            n_best += 1
                            best_act += a
                            opt += b
                        # a passability route can hold a climb the speed model rules out
                        # on the forecast map itself: no predicted time for that trip
                        p = g.path_hours(path, q(r_fc), rep)
                        if np.isfinite(p):
                            n_pred += 1
                            pred += p
                            pred_act += a
                        else:
                            no_pred += 1
                    rows.append(dict(date=day, h=h, band=band, routing=name, trips=len(routes),
                                     possible=int(np.isfinite(list(best.values())).sum()),
                                     stuck=stuck, held=held, done=done, actual_h=act,
                                     n_best=n_best, best_actual_h=best_act, best_h=opt,
                                     n_pred=n_pred, no_pred=no_pred, pred_h=pred, pred_actual_h=pred_act))
        if i % 60 == 0:
            print(f"  speed {day.date()}  maps {len(fast) + len(pas)}", flush=True)
    return pd.DataFrame(rows)


def main_speed():
    g = Grid(T.load())
    reps = representatives(V.add_vci(V.load_vehicles()))
    fc = dict(np.load(paths.FORECASTER_OUT / "test_forecasts.npz", allow_pickle=False))
    dec = pd.read_csv(paths.FORECASTER_OUT / "decisions_by_band.csv")
    st = json.load(open(OUT / "setup.json"))
    sp = speed_backtest(g, reps, fc, dec, np.array(st["sources"]), np.array(st["dests"]))
    sp.to_csv(OUT / "speed.csv", index=False)
    s = sp.groupby(["h", "routing"])[["trips", "possible", "stuck", "held", "done", "actual_h",
                                       "best_actual_h", "best_h", "n_pred", "no_pred", "pred_h",
                                       "pred_actual_h"]].sum()
    s["stuck_pct"] = 100 * s.stuck / s.trips
    s["held_pct"] = 100 * s.held / s.possible
    s["min_per_trip"] = 60 * s.actual_h / s.done
    s["over_best_pct"] = 100 * (s.best_actual_h / s.best_h - 1)
    s["pred_error_pct"] = 100 * (s.pred_h / s.pred_actual_h - 1)
    s["no_pred_pct"] = 100 * s.no_pred / s.done
    print(s[["stuck_pct", "held_pct", "min_per_trip", "over_best_pct", "pred_error_pct",
             "no_pred_pct"]].round(2))


# --------------------------------------------------------------------- convoy
def convoy(g, rep, rci, src, dst, mode):
    """Run CONVOY vehicles from src to dst on one day. mode: 'follow' (all take the
    leader's route) or 'wear_aware' (each plans on the shared pass-count map).
    Returns vehicles through, total time of those through, bytes shared."""
    passes = np.zeros(g.shape)
    flat = passes.ravel()
    # staging areas at both ends (3 x 3 cells) are not churned: every vehicle uses them
    stage = np.zeros(g.shape, bool)
    for p in (src, dst):
        r, c = divmod(int(p), g.shape[1])
        stage[max(r - 1, 0): r + 2, max(c - 1, 0): c + 2] = True
    stage = stage.ravel()
    through, total, shared = 0, 0.0, 0
    leader = None
    for k in range(CONVOY):
        mu = g.passability(rep["kind"], g.soil(rci, rep["vci1"], rep["vci50"], passes))
        if mode == "follow" and leader is not None:
            path = leader
        else:
            path = g.routes(mu, [src], [dst])[(src, dst)]
            if path is None:
                break
            if leader is None:
                leader = path
            if mode == "wear_aware":
                shared += 4 * len(path)                   # uint32 cell ids broadcast
        tt = g.path_time(path, mu)
        if not np.isfinite(tt):
            if mode == "follow":
                break                                     # column blocked behind it
            continue
        through += 1
        total += tt
        np.add.at(flat, path[~stage[path]], 1)
    return through, total, shared


def run_convoys(g, reps, fc, sources, dests, bands=("D", "E")):
    actual, dates = fc["actual"], pd.to_datetime(fc["dates"])
    rows = []
    for band in bands:
        rep = reps[band]
        today = actual[:, 0]
        # days where the soil is firm enough for one pass but not for fifty
        slow = np.flatnonzero((today > rep["vci1"] * 1.05) & (today < rep["vci50"] * 1.5))
        pick = slow[np.linspace(0, len(slow) - 1, min(30, len(slow))).astype(int)]
        for i in pick:
            for s, d in zip(sources[:3], dests[:3]):
                mu = g.passability(rep["kind"], g.soil(today[i], rep["vci1"], rep["vci50"]))
                if g.routes(mu, [s], [d])[(s, d)] is None:
                    continue                              # not even one vehicle can go
                for mode in ("follow", "wear_aware"):
                    n, tt, b = convoy(g, rep, today[i], s, d, mode)
                    rows.append(dict(band=band, date=dates[i], rci=today[i], src=s, dst=d,
                                     mode=mode, through=n, time=tt, bytes=b))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------- main
def plan_timing(g, reps, sources, dests):
    rep = reps["D"]
    t0 = time.perf_counter()
    for _ in range(5):
        mu = g.passability(rep["kind"], g.soil(40.0, rep["vci1"], rep["vci50"]))
        g.routes(mu, [sources[0]], [dests[0]])
    return dict(replan_ms=round((time.perf_counter() - t0) / 5 * 1000, 1),
                grid_cells=int(g.t["valid"].sum()), map_bytes_uint8=int(g.t["valid"].sum()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    g = Grid(T.load())
    fleet = V.add_vci(V.load_vehicles())
    reps = representatives(fleet)
    fc = dict(np.load(paths.FORECASTER_OUT / "test_forecasts.npz", allow_pickle=False))
    dec = pd.read_csv(paths.FORECASTER_OUT / "decisions_by_band.csv")
    sources, dests = trip_endpoints(g)
    json.dump(dict(reps=reps, sources=sources.tolist(), dests=dests.tolist()),
              open(OUT / "setup.json", "w"), indent=2)

    bt = backtest(g, reps, fc, dec, sources, dests,
                  methods=("min", "product", "liu_weighted", "weighted_no_veto"))
    bt.to_csv(OUT / "backtest.csv", index=False)
    print(f"backtest {time.time()-t0:.0f}s")

    cv = run_convoys(g, reps, fc, sources, dests)
    cv.to_csv(OUT / "convoy.csv", index=False)
    timing = plan_timing(g, reps, sources, dests)
    json.dump(timing, open(OUT / "timing.json", "w"), indent=2)

    s = bt.groupby(["planner", "h"])[["trips", "possible", "stuck", "held", "done",
                                      "extra_time_sum"]].sum()
    s["stuck_rate"] = s.stuck / s.trips
    s["held_rate"] = s.held / s.possible
    s["extra_time_pct"] = 100 * s.extra_time_sum / s.done
    print(s[["stuck_rate", "held_rate", "extra_time_pct"]].round(4))
    print(cv.groupby(["band", "mode"])[["through", "time", "bytes"]].mean().round(2))
    print(timing, f"total {time.time()-t0:.0f}s")


def main_convoy():
    g = Grid(T.load())
    reps = representatives(V.add_vci(V.load_vehicles()))
    fc = dict(np.load(paths.FORECASTER_OUT / "test_forecasts.npz", allow_pickle=False))
    st = json.load(open(OUT / "setup.json"))
    cv = run_convoys(g, reps, fc, np.array(st["sources"]), np.array(st["dests"]))
    cv.to_csv(OUT / "convoy.csv", index=False)
    print(cv.groupby(["band", "mode"])[["through", "time", "bytes"]].mean().round(2))


if __name__ == "__main__":
    import sys
    if "--convoy-only" in sys.argv:
        main_convoy()
    elif "--tradeoff" in sys.argv:
        main_tradeoff()
    elif "--speed" in sys.argv:
        main_speed()
    else:
        main()
