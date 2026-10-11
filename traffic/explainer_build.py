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
