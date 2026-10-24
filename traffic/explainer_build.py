"""
The project explainer (A4 PDF): docs/explainer_template.html filled with numbers and
tables from outputs/, printed with headless Edge to docs/Project_Explainer_ICDCN_Poster.pdf.

Placeholders in the template are written <<name>>; every one comes from outputs/ via
traffic.poster_build.numbers() or the tables built here, so the explainer, poster and
abstract cannot disagree.

Usage:  python -m traffic.explainer_build
"""

from __future__ import annotations

import datetime as dt
import json
import re
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from traffic import data as D
from traffic import mobility as M
from traffic import paths
from traffic import planner as PL
from traffic import poster_build as B
from traffic import poster_figs as F
from traffic import poster_page as PP
from traffic import terrain as T
from traffic import vehicles as V

DOCS = paths.ROOT / "docs"
OUT_PDF = DOCS / "Project_Explainer_ICDCN_Poster.pdf"
FIGS = DOCS / "explainer_figs"


# ------------------------------------------------------------- two new figures
def fig_slope():
    """Liu et al.'s slope steps against the smooth curve the planner uses."""
    F.setup()
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10,
                         "ytick.labelsize": 10, "legend.fontsize": 10, "axes.titlesize": 11})
    edges = [0, 3, 6, 10, 15, 20, 30, 35, 40]
    fig, ax = plt.subplots(figsize=(6.4, 2.9), layout="constrained")
    for kind, col, steps in [("wheeled", F.FOIL, [1, .9, .675, .54, .36, .225, 0, 0]),
                             ("tracked", F.OURS, [1, .9, .72, .6, .36, .24, .12, 0])]:
        for i, v in enumerate(steps):
            ax.plot(edges[i:i + 2], [v, v], color=col, lw=1.4, alpha=0.45)
        x = np.linspace(0, 40, 400)
        ax.plot(x, T.slope_membership(x, kind), color=col, lw=2.2, label=f"{kind}: smooth curve")
    ax.set_xlabel("slope (degrees)")
    ax.set_ylabel("passability (0 to 1)")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.text(21, 0.47, "faint steps: Liu et al. Table 1", fontsize=10, color=F.INK2)
    fig.savefig(FIGS / "slope_curve.png", dpi=200)
    plt.close(fig)


def fig_wear(rep):
    """Firmness needed on pass n for the heavy tracked band (convoy wear model)."""
    F.setup()
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10,
                         "ytick.labelsize": 10, "axes.titlesize": 11})
    n = np.arange(1, 51)
    need = rep["vci1"] + (rep["vci50"] - rep["vci1"]) * np.log(n) / np.log(50)
    ground = 40.0
    first = int(n[need > ground][0])
    fig, ax = plt.subplots(figsize=(6.4, 2.6), layout="constrained")
    ax.plot(n, need, color=F.NOGO, lw=2.2)
    ax.axhline(ground, color=F.OURS, lw=1.6)
    ax.axvline(first, color=F.MUTED, lw=1.0)
    ax.text(48, ground + 1.0, f"ground firmness on the day = {ground:.0f} psi", ha="right",
            va="bottom", fontsize=10, color=F.OURS)
    ax.text(first + 0.8, rep["vci1"] + 1, f"vehicle {first} is the first\nthat would get stuck",
            fontsize=10, color=F.INK2, va="bottom")
    ax.set_xlabel("pass number (1 = first vehicle over this cell)")
    ax.set_ylabel("firmness needed (psi)")
    ax.set_xlim(1, 50)
    fig.savefig(FIGS / "wear_curve.png", dpi=200)
    plt.close(fig)
    return first


def fig_speed(reps):
    """Level-ground speed and passes per track against firmness above VCI1, per band."""
    F.setup()
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10,
                         "ytick.labelsize": 10, "legend.fontsize": 9.5, "axes.titlesize": 11})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.9), layout="constrained")
    x = np.linspace(0, 60, 241)
    cols = dict(zip("ABCDE", ["#7a5ad1", "#2a78d6", "#1f9d55", "#eb6834", "#16191d"]))
    for b, v in reps.items():
        a1.plot(x, M.speed(v["vci1"] + x, 0.0, v), color=cols[b], lw=1.8, label=f"{b} ({v['kind']})")
        a2.plot(x, M.passes(v["vci1"] + x, v["vci1"], v["vci50"]), color=cols[b], lw=1.8)
    a1.set_xlabel("firmness above VCI₁ (psi)")
    a1.set_ylabel("speed on level ground (km/h)")
    a1.set_xlim(0, 60)
    a1.set_ylim(0, None)
    a1.legend(loc="lower right")
    a2.set_yscale("log")
    a2.axhline(50, color=F.MUTED, lw=1.0)
    a2.text(1, 55, "50 passes = VCI₅₀ = GO", fontsize=9.5, color=F.INK2)
    a2.set_xlabel("firmness above VCI₁ (psi)")
    a2.set_ylabel("vehicles per track")
    a2.set_xlim(0, 60)
    fig.savefig(FIGS / "speed_curves.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------- worked example
def _f(x, d=2):
    return f"{x:,.{d}f}"


def worked_example(reps):
    """One planning day, one band, one trip, every formula with its real numbers.
    Returns the HTML rows of the stage table and a few values for the text."""
    import sys
    sys.path.insert(0, str(paths.ROOT / "edge"))
    import trafficability_edge as E

    day, h, band = pd.Timestamp("2024-10-24"), 3, "D"
    travel = day + pd.Timedelta(days=h)
    rep = reps[band]

    # 1-2. weather -> wetness -> firmness, planning day
    raw = D.add_soil_strength(D.load_raw())
    r = raw.set_index("Date").loc[day]
    span = D.THETA_SAT - D.THETA_DRY
    s1 = min(max((r.sm_surf - D.THETA_DRY) / span, 0), 1)
    s2 = min(max((r.sm_deep - D.THETA_DRY) / span, 0), 1)
    swi = D.W_SURF * s1 + D.W_DEEP * s2
    i = raw.index[raw.Date == day][0]
    r3 = raw.rain.iloc[i - 2: i + 1].sum()
    rain_f = 1 - D.RAIN_PENALTY_MAX * min(r3 / D.RAIN_PENALTY_SCALE, 1)
    rci_t = D.RCI_DRY * np.exp(-D.K_DECAY * swi) * rain_f
    assert abs(rci_t - r.rci) < 1e-6

    # 3. forecast: today + mean of the GRU and ridge changes (the edge runtime)
    fm = E.Forecaster(paths.ROOT / "edge" / "vehicle_trafficability_forecaster.pkl", backend="numpy")
    log = pd.read_csv(paths.WEATHER_CSV)
    log = log[pd.to_datetime(log["Date"], format="%d-%m-%Y") <= day]
    _, today, Xtab, win = fm.features(log)
    d_gru, d_ridge = float(fm.gru(win)[0, h - 1]), float(fm.ridge(Xtab)[0, h - 1])
    rci_fc = float(today[0]) + (d_gru + d_ridge) / 2
    dec = pd.read_csv(paths.FORECASTER_OUT / "decisions_by_band.csv")
    margin = PL._cross_year_margin(dec, "GRU+Ridge", band, h, day.year)
    safe = rci_fc - margin
    fcz = np.load(paths.FORECASTER_OUT / "test_forecasts.npz")
    k = list(pd.to_datetime(fcz["dates"])).index(day)
    rci_act = float(fcz["actual"][k, h - 1])

    # 4. one real band-D tracked vehicle through the Mobility Index
    fleet = V.add_vci(V.load_vehicles())
    g_d = fleet[(fleet.band == band) & (fleet.type == "tracked")]
    vr = g_d.iloc[(g_d.vci1 - rep["vci1"]).abs().argmin()]
    m_t, hp = vr.mass_t, vr.hp
    w = m_t * 1000 * V.KG_TO_LB
    wid, ln = np.interp(m_t, V._TR_MASS, V._TR_WIDTH), np.interp(m_t, V._TR_MASS, V._TR_LENGTH)
    wheels = 2 * round(np.interp(m_t, V._TR_MASS, V._TR_WHEELS) / 2)
    clr = np.interp(m_t, V._TR_MASS, V._TR_CLEAR)
    cpf, wf, tef, bf, cf = (w / (2 * ln * wid), V._wf_tracked(w), wid / 100,
                            w / (10 * wheels * wid * V.SHOE_PITCH_IN), clr / 10)
    ef = V._ef(m_t, hp)
    mi = (cpf * wf / tef + bf - cf) * ef
    assert abs(mi - vr.mi) < 1e-6
    v1, v50 = V.vci_tracked(mi)

    # 5. decision, passes, level-ground speed for the band's planning vehicle
    status = "NO-GO" if safe < rep["vci1"] else ("SLOW-GO" if safe < rep["vci50"] else "GO")
    n_pass = float(M.passes(safe, rep["vci1"], rep["vci50"]))
    key = M.gear(rep["kind"], rep["cpf"])
    x = safe - rep["vci1"]
    dw, rw = float(M.pull(x, key)), float(M.resistance(x, key))
    k1, k2 = M._TFOR[key]
    cfac = dw - k1 + rw
    tfor = cfac + k2
    slip0 = float(M.slip(rw - cfac, key))
    v_pow = M.ETA * rep["hp_per_t"] * M.HP_W / (1000 * M.G * rw) * 3.6
    v_lvl = float(M.speed(safe, 0.0, rep))
    assert abs(min(v_pow * (1 - slip0), rep["vmax_kmh"]) - v_lvl) < 1e-9

    # 6. map and route: the first sample trip with a route that day
    t = T.load()
    g = PL.Grid(t)
    st = json.load(open(PL.OUT / "setup.json"))
    mu = g.passability(rep["kind"], g.soil(safe, rep["vci1"], rep["vci50"]))
    routes = g.routes(mu, st["sources"], st["dests"])
    (src, dst), path = next((od, p) for od, p in routes.items() if p is not None)
    km = float(np.sum(T.CELL_M / 1000 * np.hypot(*np.diff(np.divmod(path, g.shape[1]), axis=1))))
    cost = g.path_time(path, mu)
    # the route's most limiting cell off paved road
    flat = mu.ravel()
    cand = [c for c in path if not g.paved.ravel()[c]]
    cell = min(cand, key=lambda c: flat[c]) if cand else path[len(path) // 2]
    v_ok = t["valid"]
    twi_z = (t["twi"].ravel()[cell] - np.nanmean(t["twi"][v_ok])) / np.nanstd(t["twi"][v_ok])
    shift = float(T.wetness_shift(t).ravel()[cell])
    rci_cell = safe * g.decay.ravel()[cell]
    kk = "wheeled" if rep["kind"] == "halftrack" else rep["kind"]
    mu_s, mu_c = (a.ravel()[cell] for a in g.static[kk])
    road = int(t["road"].ravel()[cell])
    soil_c = float(g.soil(safe, rep["vci1"], rep["vci50"]).ravel()[cell])
    # the step into that cell along the route, and its speed
    j = list(path).index(cell)
    a_, b_ = path[max(j - 1, 0)], path[max(j, 1)]
    z = np.nan_to_num(t["elevation"]).ravel()
    step_m = T.CELL_M * np.hypot(*np.subtract(divmod(int(b_), g.shape[1]), divmod(int(a_), g.shape[1])))
    sin_t = (z[b_] - z[a_]) / np.hypot(z[b_] - z[a_], step_m)
    v_cell = float(M.speed(rci_cell, sin_t, rep, False)) * mu_c
    eta = g.path_hours(path, safe, rep)
    fast = g.fastest(safe, rep, [src], [dst])[(src, dst)]
    eta_fast = g.path_hours(fast, safe, rep)
    # 7. the travel day
    mu_act = g.passability(rep["kind"], g.soil(rci_act, rep["vci1"], rep["vci50"]))
    ok = np.isfinite(g.path_time(path, mu_act))
    eta_act = g.path_hours(path, rci_act, rep)
    best = g.path_hours(g.fastest(rci_act, rep, [src], [dst])[(src, dst)], rci_act, rep)
    cover_name = {10: "tree cover", 20: "shrub", 30: "grass", 40: "cropland", 50: "built-up", 60: "bare",
                  80: "water", 90: "wetland"}.get(int(t["cover"].ravel()[cell]), "other")

    S = []   # (stage, formula, with the numbers, result)
    S.append(("1. Soil moisture → wetness",
              "s = (θ − 20) / (49 − 20), clipped to 0–1<br>SWI = 0.35·s<sub>0–7</sub> + 0.65·s<sub>7–28</sub>",
              f"θ = {_f(r.sm_surf, 1)} and {_f(r.sm_deep, 1)} vol% → s = {_f(s1, 3)}, {_f(s2, 3)}<br>"
              f"SWI = 0.35·{_f(s1, 3)} + 0.65·{_f(s2, 3)}",
              f"SWI = {_f(swi, 3)}"))
    S.append(("2. Wetness → firmness today",
              f"RCI = 300·e<sup>−{D.K_DECAY:.3f}·SWI</sup>·(1 − 0.15·min(R<sub>3d</sub>/50, 1))",
              f"R<sub>3d</sub> = {_f(r3, 1)} mm → rain factor {_f(rain_f, 3)}<br>"
              f"300·e<sup>−{D.K_DECAY:.3f}·{_f(swi, 3)}</sup>·{_f(rain_f, 3)}",
              f"RCI<sub>today</sub> = {_f(rci_t, 1)} psi"))
    S.append(("3. Forecast for the travel day",
              "RCI<sub>t+3</sub> = RCI<sub>t</sub> + ½·(ΔGRU + Δridge)",
              f"{_f(today[0], 1)} + ½·({d_gru:+.2f} {'+' if d_ridge >= 0 else '−'} {abs(d_ridge):.2f})"
              f"<br>3 GRUs averaged: ΔGRU; 1 ridge regression: Δridge",
              f"RCI<sub>forecast</sub> = {_f(rci_fc, 1)} psi"))
    S.append(("4. Safety margin",
              "RCI<sub>safe</sub> = forecast − m(band, lead)",
              f"m(D, 3 days) = {_f(margin, 2)} psi, tuned on {2025 if day.year == 2024 else 2024}<br>"
              f"{_f(rci_fc, 1)} − {_f(margin, 2)}",
              f"RCI<sub>safe</sub> = {_f(safe, 1)} psi"))
    S.append(("5. A band-D vehicle → MI",
              "MI = (CPF·WF / (TEF·GF) + BF − CF)·EF·TF",
              f"{_f(m_t, 1)} t = {_f(w, 0)} lb, {hp:.0f} hp; tracks {_f(wid, 1)} × {_f(ln, 0)} in, {wheels} road wheels, "
              f"clearance {_f(clr, 1)} in<br>CPF = {_f(cpf, 2)}, WF = {wf:.1f}, TEF = {_f(tef, 3)}, GF = 1, "
              f"BF = {_f(bf, 2)}, CF = {_f(cf, 2)}, EF = {ef:.2f}, TF = 1",
              f"MI = {_f(mi, 2)}"))
    S.append(("6. MI → vehicle cone index",
              "VCI₁ = 7.0 + 0.2·MI − 39.2/(MI + 5.6)<br>VCI₅₀ = 19.27 + 0.43·MI − 125.79/(MI + 7.08)",
              f"MI = {_f(mi, 2)}",
              f"VCI₁ = {_f(v1, 1)}, VCI₅₀ = {_f(v50, 1)} psi"))
    S.append(("7. Band planning vehicle",
              "median VCI₁, VCI₅₀, power-to-weight and contact pressure of band D's tracked vehicles",
              f"{rep['n']} vehicles in band D",
              f"VCI₁ {_f(rep['vci1'], 1)}, VCI₅₀ {_f(rep['vci50'], 1)} psi; {_f(rep['hp_per_t'], 1)} hp/t; "
              f"CPF {_f(rep['cpf'], 1)} psi"))
    S.append(("8. Decision",
              "NO-GO if RCI<sub>safe</sub> < VCI₁; SLOW-GO if < VCI₅₀; else GO",
              f"{_f(rep['vci1'], 1)} ≤ {_f(safe, 1)} &lt; {_f(rep['vci50'], 1)}",
              f"<b>{status}</b>"))
    S.append(("9. Passes per track",
              "n = 50<sup>(RCI − VCI₁)/(VCI₅₀ − VCI₁)</sup>",
              f"50<sup>({_f(safe, 1)} − {_f(rep['vci1'], 1)})/({_f(rep['vci50'], 1)} − {_f(rep['vci1'], 1)})</sup>",
              f"{n_pass:.1f} → <b>{int(n_pass)} vehicles</b> per track"))
    S.append(("10. Soil traction (NRMM)",
              "x = RCI − VCI₁<br>D/W = a + bx − √((a + bx)² − cx)<br>R/W = p + q/(x + r)",
              f"x = {_f(x, 1)} psi; tracked, CPF {'≥' if key[1] else '&lt;'} 4 psi<br>"
              f"D/W = {_f(dw, 3)}, R/W = {_f(rw, 3)}<br>TFOR/W = D/W − {k1} + R/W + {k2} = {_f(tfor, 3)}",
              f"moves if TFOR/W ≥ R/W ({_f(tfor, 3)} ≥ {_f(rw, 3)})"))
    S.append(("11. Speed on level ground",
              "v = η·P / (W·R/W) · (1 − S), capped at top speed",
              f"η = {M.ETA}, P/W = {_f(rep['hp_per_t'], 1)} hp/t → power-limited {_f(v_pow, 1)} km/h<br>"
              f"slip S at y = R/W − CF = {_f(rw - cfac, 3)}: {_f(100 * slip0, 1)}%",
              f"<b>{_f(v_lvl, 1)} km/h</b>"))
    S.append(("12. Firmness in one map cell",
              f"shift = 0.10·clip(z<sub>TWI</sub>, −2, 2); RCI<sub>cell</sub> = RCI·e<sup>−{D.K_DECAY:.3f}·shift</sup>",
              f"the route's weakest {'unpaved-track' if road else 'off-road'} cell: z = {twi_z:+.2f} → shift {shift:+.3f}",
              f"RCI<sub>cell</sub> = {_f(rci_cell, 1)} psi"))
    S.append(("13. Cell passability",
              "min(slope score, cover score, soil score)<br>soil = 0.25 + 0.75·(RCI − VCI₁)/(VCI₅₀ − VCI₁)",
              (f"unpaved track: slope and cover count as 1 on roads; soil "
               f"0.25 + 0.75·({_f(rci_cell, 1)} − {_f(rep['vci1'], 1)})/({_f(rep['vci50'], 1)} − {_f(rep['vci1'], 1)})"
               f" = {_f(soil_c, 2)}" if road else
               f"slope {_f(t['slope'].ravel()[cell], 1)}° → {_f(mu_s, 2)}; {cover_name} → {_f(mu_c, 2)}; "
               f"soil {_f(soil_c, 2)}"),
              f"μ = {_f(flat[cell], 2)}"))
    S.append(("14. Route (published planner)",
              "Dijkstra on cost = step·½(1/μ<sub>u</sub> + 1/μ<sub>v</sub>)",
              f"trip of {_f(km, 1)} km on the grid, {len(path)} cells",
              f"cost {_f(cost, 2)} (km ÷ speed fraction)"))
    S.append(("15. Speed in that cell",
              "same as 10–11 with load = sin θ + R/W·cos θ, × cover score",
              f"step into the cell: grade {100 * sin_t:+.1f}%, cell RCI {_f(rci_cell, 1)} psi",
              f"{_f(v_cell, 1)} km/h"))
    S.append(("16. Travel time",
              "Σ step·½(1/v<sub>u</sub> + 1/v<sub>v</sub>)",
              f"this route; the route chosen by travel time instead",
              f"{60 * eta:.0f} min; {60 * eta_fast:.0f} min"))
    S.append(("17. Check on the travel day",
              "real RCI on the travel day, same map rules",
              f"{travel.date():%d %b %Y}: RCI {_f(rci_act, 1)} psi",
              f"{'route passable' if ok else 'STUCK'}; {60 * eta_act:.0f} min (best possible {60 * best:.0f})"))
    S.append(("18. What is radioed",
              "4-byte day number + 7 × 16-bit firmness values",
              "4 + 7 × 2",
              "18 bytes"))
    rows = "\n".join(f"<tr><td><b>{a}</b></td><td class='math2'>{b}</td><td>{c}</td><td>{d}</td></tr>"
                     for a, b, c, d in S)
    return dict(t_worked=rows, we_day=f"{day:%d %B %Y}", we_travel=f"{travel:%d %B %Y}",
                we_band=band, we_today=_f(rci_t, 1), we_status=status, we_actual=_f(rci_act, 1))


def speed_tables(reps):
    t = {}
    rows = []
    xs = [0, 5, 10, 20, 40]
    for b, v in reps.items():
        sp = M.speed(v["vci1"] + np.array(xs, float), 0.0, v)
        n = M.passes(v["vci1"] + np.array([5.0, 10.0, 20.0]), v["vci1"], v["vci50"])
        rows.append(_tr([b, v["kind"], f"{v['hp_per_t']:.1f}", f"{v['cpf']:.1f}"]
                        + [f"{s:.0f}" for s in sp] + [" / ".join(f"{p:.0f}" for p in n)]))
    t["t_speed_bands"] = "\n".join(rows)

    sp = pd.read_csv(PL.OUT / "speed.csv")
    s = sp.groupby(["h", "routing"])[["trips", "possible", "stuck", "held", "done", "actual_h",
                                       "best_actual_h", "best_h", "pred_h", "pred_actual_h"]].sum()
    rows = []
    for routing, lbl, cls in [("passability", "Published planner (passability cost)", ""),
                              ("fastest", "Routed by travel time", "us")]:
        cells = [lbl]
        for h in (1, 3, 7):
            r = s.loc[(h, routing)]
            cells.append(f"{100 * r.stuck / r.trips:.1f}% / {100 * r.held / r.possible:.1f}% / "
                         f"{60 * r.actual_h / r.done:.1f} min")
        rows.append(_tr(cells, cls))
    t["t_speed_bt"] = "\n".join(rows)
    r3, r7 = s.loc[(3, "fastest")], s.loc[(7, "fastest")]
    p3 = s.loc[(3, "passability")]
    t.update(sp_faster3=f"{100 * (1 - (r3.actual_h / r3.done) / (p3.actual_h / p3.done)):.0f}",
             sp_pred3=f"{100 * (r3.pred_h / r3.pred_actual_h - 1):+.1f}",
             sp_pred7=f"{100 * (r7.pred_h / r7.pred_actual_h - 1):+.1f}")
    r = sp[(sp.routing == "fastest") & (sp.h == 3) & (sp.pred_actual_h > 0)]
    e = 100 * (r.pred_h / r.pred_actual_h - 1)
    t.update(sp_p05=f"{e.quantile(0.05):+.1f}", sp_p95=f"{e.quantile(0.95):+.1f}")
    # timing of a re-plan by travel time, one trip
    import time
    g = PL.Grid(T.load())
    st = json.load(open(PL.OUT / "setup.json"))
    rep = reps["D"]
    g.fastest(40.0, rep, [st["sources"][0]], [st["dests"][0]])
    t0 = time.perf_counter()
    for _ in range(5):
        g.fastest(40.0, rep, [st["sources"][0]], [st["dests"][0]])
    t["replan_time_ms"] = f"{(time.perf_counter() - t0) / 5 * 1000:.0f}"
    m1 = 120000 / V.KG_TO_LB / 1000
    m1v = dict(kind="tracked", vci1=26.34, cpf=V.cpf_tracked(m1, 25, 180), hp_per_t=1500 / m1, vmax_kmh=67.0)
    t.update(m1_v0=f"{float(M.speed(26.34, 0.0, m1v)):.0f}", m1_v20=f"{float(M.speed(46.34, 0.0, m1v)):.0f}")
    t.update(eta_pct=f"{100 * M.ETA:.0f}", eta_lo=f"{100 * 0.7 / M.ETA - 100:.0f}")
    return t


# ------------------------------------------------------------------- tables
def _tr(cells, cls=""):
    c = f' class="{cls}"' if cls else ""
    return f"<tr{c}>" + "".join(f"<td>{x}</td>" for x in cells) + "</tr>"


def tables(n, reps):
    t = {}
    reg = pd.read_csv(paths.FORECASTER_OUT / "regression_test.csv")
    piv = reg[reg.h.isin([1, 3, 7])].pivot(index="model", columns="h", values="rmse")
    skill = reg[reg.h == 7].set_index("model")["skill_vs_persistence"]
    rows = []
    for m, lbl, cls in [("GRU+Ridge", "GRU + ridge, averaged (chosen)", "us"), ("GRU", "GRU alone", ""),
                        ("Ridge", "Ridge alone", ""), ("Persistence", "Persistence ('same as today')", "")]:
        rows.append(_tr([lbl] + [f"{piv.loc[m, h]:.2f}" for h in (1, 3, 7)] + [f"{100 * skill[m]:+.0f}%"], cls))
    t["t_rmse"] = "\n".join(rows)

    dec = pd.read_csv(paths.FORECASTER_OUT / "decisions_by_band.csv")
    rows = []
    for b, _, _ in V.BANDS:
        p = dec[(dec.rule == "Persistence") & (dec.h == 7) & (dec.band == b)].iloc[0]
        f = dec[(dec.rule == "GRU+Ridge") & (dec.h == 7) & (dec.band == b)].iloc[0]
        rows.append(_tr([b, f"{100 * p.recall:.0f}% / {p.false_alarms_per_year:.0f}",
                         f"{100 * f.recall:.0f}% / {f.false_alarms_per_year:.0f}",
                         f"{f.margin_tuned_2024:.1f} / {f.margin_tuned_2025:.1f}"]))
    t["t_decisions"] = "\n".join(rows)

    fleet = V.add_vci(V.load_vehicles())
    rows = []
    names = {"A": "light tracked", "B": "light-medium tracked", "C": "medium tracked, some half-tracks",
             "D": "heavy tracked, some wheeled", "E": "medium and heavy wheeled"}
    for b, lo, hi in V.BANDS:
        g = fleet[fleet.band == b]
        rng = f"{lo:.0f}–{hi:.0f}" if np.isfinite(hi) else f"{lo:.0f}+"
        rows.append(_tr([b, rng, names[b], f"{len(g)}", f"{reps[b]['vci1']:.1f} / {reps[b]['vci50']:.1f}",
                         f"{n['band_nogo'][b]:.0f}%"]))
    t["t_bands"] = "\n".join(rows)

    bt = n["bt"]
    rows = []
    for p, lbl, cls in [("Static (terrain only)", "Terrain only (no soil)", ""),
                        ("Today (Liu et al.)", "Today's map (what Liu et al. do)", "foil"),
                        ("Today + margin", "Today's map + safety margin", ""),
                        ("Forecast + margin (ours)", "Forecast + safety margin (ours)", "us")]:
        cells = [lbl]
        for h in (1, 3, 7):
            r = bt.loc[(p, h)]
            cells.append(f"{r.stuck_pct:.1f}% / {r.held_pct:.1f}%")
        rows.append(_tr(cells, cls))
    t["t_backtest"] = "\n".join(rows)

    rows = []
    for p, lbl, cls in [("Forecast + margin (ours)", "Fuzzy AND, minimum (ours)", "us"),
                        ("Forecast + margin [product]", "Fuzzy AND, product", ""),
                        ("Forecast + margin [liu_weighted]", "Weighted mean + veto (Liu et al.)", ""),
                        ("Forecast + margin [weighted_no_veto]", "Weighted mean, no veto", "")]:
        rows.append(_tr([lbl] + [f"{bt.loc[(p, h)].stuck_pct:.1f}%" for h in (1, 3, 7)], cls))
    t["t_methods"] = "\n".join(rows)
    return t


# --------------------------------------------------------------------- build
def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    n = B.numbers()
    fleet = V.add_vci(V.load_vehicles())
    reps = PL.representatives(fleet)
    fig_slope()
    fig_speed(reps)
    first_stuck = fig_wear(reps["D"])
    for f in ("f1_season", "f2_fleet", "f3_decisions", "f4_map", "f5_tradeoff", "f6_convoy", "f7_onset"):
        shutil.copy(B.FIG / f"{f}.png", FIGS / f"{f}.png")

    raw = pd.read_csv(paths.VEHICLES_CSV)
    mass = pd.to_numeric(raw.mass_kg, errors="coerce")
    hp = pd.to_numeric(raw.horsePowers_hp, errors="coerce")
    plausible = raw[mass.between(1000, 80000) & hp.between(20, 3000)]
    vc = pd.to_numeric(plausible.mass_kg).value_counts()
    t = T.load()
    v = t["valid"]
    pass_w = np.minimum(*T.static_memberships(t, "wheeled"))[v]
    pass_t = np.minimum(*T.static_memberships(t, "tracked"))[v]
    edge = json.load(open(paths.FORECASTER_OUT / "edge.json"))
    reg = pd.read_csv(paths.FORECASTER_OUT / "regression_test.csv")

    vals = {k: val for k, val in n.items() if not hasattr(val, "loc") and not isinstance(val, dict)}
    vals.update(tables(n, reps))
    vals.update(speed_tables(reps))
    vals.update(worked_example(reps))
    vals.update(
        today=dt.date.today().strftime("%d %B %Y").lstrip("0"),
        days_left=(dt.date(2026, 10, 15) - dt.date.today()).days,
        n_days=f"{n['n_days']:,}", n_raw_rows=f"{len(raw):,}", n_plausible=f"{len(plausible):,}", n_mass_values=len(vc),
        top15_share=f"{100 * vc.head(15).sum() / len(plausible):.0f}",
        n_placeholder=int((mass == 5500).sum()),
        n_public=int((fleet.mass_source == "public_approx").sum()),
        n_vehicles=f"{len(fleet):,}", n_tracked=f"{n['n_tracked']:,}",
        bt_trips=f"{n['bt_trips']:,}", map_ratio=f"{n['map_ratio']:,}",
        tree_pct=f"{100 * (t['cover'][v] == T.TREE).mean():.0f}",
        pass_w=f"{100 * (pass_w > 0).mean():.0f}", pass_t=f"{100 * (pass_t > 0).mean():.0f}",
        grid_rows=t["valid"].shape[0], grid_cols=t["valid"].shape[1],
        gru_each=f"{edge['gru_params_each']:,}", feat_ms=f"{edge['cpu_feature_build_ms']:.0f}",
        rmse1=f"{reg[(reg.model == 'GRU+Ridge') & (reg.h == 1)].rmse.iloc[0]:.2f}",
        rmse1_p=f"{reg[(reg.model == 'Persistence') & (reg.h == 1)].rmse.iloc[0]:.2f}",
        first_stuck=first_stuck, d_vci1=f"{reps['D']['vci1']:.1f}", d_vci50=f"{reps['D']['vci50']:.1f}",
        e_vci1=f"{reps['E']['vci1']:.1f}", a_vci1=f"{reps['A']['vci1']:.1f}",
        k_decay=f"{D.K_DECAY:.3f}", rci_sat=f"{D.RCI_SAT:.0f}",
        nogo_lo=f"{min(n['band_nogo'].values()):.0f}", nogo_hi=f"{max(n['band_nogo'].values()):.0f}",
        pipeline=PP.pipeline_svg(),
    )
    tpl = (DOCS / "explainer_template.html").read_text(encoding="utf-8")
    missing = sorted(set(re.findall(r"<<(\w+)>>", tpl)) - set(vals))
    if missing:
        raise SystemExit(f"template fields with no value: {missing}")
    html = re.sub(r"<<(\w+)>>", lambda m: str(vals[m.group(1)]), tpl)
    page = DOCS / "explainer.html"
    page.write_text(html, encoding="utf-8")
    subprocess.run([B.EDGE, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--virtual-time-budget=15000", f"--print-to-pdf={OUT_PDF}", page.as_uri()],
                   check=True, capture_output=True, timeout=240)
    import pymupdf
    print(f"-> {OUT_PDF.relative_to(paths.ROOT)}  {len(pymupdf.open(OUT_PDF))} pages")


if __name__ == "__main__":
    main()
