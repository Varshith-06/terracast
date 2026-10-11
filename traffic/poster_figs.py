"""
Figures for the ICDCN 2027 poster, drawn at print size for an A0 portrait sheet.

SVG with live text (svg.fonttype = none) so the poster page sets every label in the
same IBM Plex face as its body text. Every figure is exactly one poster column wide
(constrained layout, no tight-bbox growth), so point sizes here are point sizes on
paper.

Usage:  python -m traffic.poster_figs
"""

from __future__ import annotations

import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from traffic import data as D
from traffic import paths
from traffic import planner as PL
from traffic import terrain as T
from traffic import vehicles as V

POSTER = paths.ROOT / "poster"
FIG = POSTER / "figures"
COL_IN = (841 - 2 * 30 - 2 * 20) / 3 / 25.4      # one poster column: 247 mm

# ---- palette: chart ink + reserved status + one emphasis hue (dataviz reference)
INK, INK2, MUTED, GRID, AXIS = "#16191d", "#4b4f55", "#8a8d91", "#e3e2dc", "#c3c2b7"
OURS = "#2a78d6"                      # the method: the only saturated series colour
FOIL = "#eb6834"                      # the baseline the poster argues against
GREY, GREY_L = "#7d7f83", "#b9bbbe"
GO, SLOW, NOGO = "#0ca30c", "#fab219", "#d03b3b"
TYPE_COL = {"tracked": "#2a78d6", "wheeled": "#eb6834", "halftrack": "#1baf7a"}


def setup():
    for f in glob.glob(str(POSTER / "fonts" / "ttf" / "*.ttf")):
        fm.fontManager.addfont(f)
    plt.rcParams.update({
        "svg.fonttype": "none", "font.family": "IBM Plex Sans", "font.size": 23,
        "axes.edgecolor": AXIS, "axes.linewidth": 1.2, "axes.labelcolor": INK2,
        "axes.labelsize": 22, "axes.titlesize": 23, "axes.titleweight": 600,
        "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.titlepad": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0,
        "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "xtick.labelsize": 21,
        "ytick.labelsize": 21, "xtick.major.size": 0, "ytick.major.size": 0,
        "xtick.major.pad": 7, "ytick.major.pad": 7, "legend.frameon": False,
        "legend.fontsize": 19, "text.color": INK,
        "figure.constrained_layout.h_pad": 0.04, "figure.constrained_layout.w_pad": 0.04,
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.transparent": True,
    })
    FIG.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(FIG / f"{name}.svg")
    fig.savefig(FIG / f"{name}.png", dpi=110)
    with plt.rc_context({"pdf.fonttype": 42}):       # embedded TrueType, for the ACM abstract
        fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


def _log_rci_axis(ax):
    ax.set_yscale("log")
    ax.set_yticks([15, 30, 60, 120, 240])
    ax.set_yticklabels(["15", "30", "60", "120", "240"])
    ax.minorticks_off()


# ------------------------------------------------------------- 1. the season
def fig_season(reps):
    raw = D.add_soil_strength(D.load_raw())
    d = raw[(raw.Date >= "2024-01-01") & (raw.Date <= "2025-12-31")]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL_IN, 6.9), sharex=True, layout="constrained",
                                 gridspec_kw=dict(height_ratios=[1, 2.8]))
    a1.bar(d.Date, d.rain, width=1.0, color=GREY, linewidth=0)
    a1.set_ylabel("rain (mm)")
    a1.set_yticks([0, 100])
    a1.grid(axis="x", visible=False)
    a2.plot(d.Date, d.rci, color=INK, lw=2.2)
    start = d.Date.iloc[0] + pd.Timedelta(days=8)
    for band, lbl in [("A", "light tracked, A"), ("D", "heavy tracked, D"), ("E", "wheeled, E")]:
        v = reps[band]["vci1"]
        a2.axhline(v, color=NOGO, lw=1.3)
        a2.text(start, v * 1.035, lbl, va="bottom", ha="left", fontsize=18, color=NOGO)
    _log_rci_axis(a2)
    a2.set_ylim(12, 300)
    a2.set_ylabel("ground firmness, RCI (psi)")
    a2.grid(axis="x", visible=False)
    a2.set_xlim(d.Date.iloc[0], d.Date.iloc[-1])
    a2.xaxis.set_major_locator(mdates.MonthLocator([1, 7]))
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    save(fig, "f1_season")


# ------------------------------------------------------------- 2. the fleet
def fig_fleet(fleet, rci):
    fig, ax = plt.subplots(figsize=(COL_IN, 5.1), layout="constrained")
    bins = np.arange(14, 50.5, 0.75)
    bottom = np.zeros(len(bins) - 1)
    for kind, lbl in [("tracked", "tracked"), ("wheeled", "wheeled"), ("halftrack", "half-track")]:
        h, _ = np.histogram(fleet.loc[fleet.type == kind, "vci1"].clip(upper=49.9), bins)
        ax.bar(bins[:-1], h, width=0.75 * 0.86, bottom=bottom, align="edge",
               color=TYPE_COL[kind], linewidth=0, label=f"{lbl} ({(fleet.type == kind).sum():,})")
        bottom += h
    ymax = bottom.max() * 1.5
    for band, lo, hi in V.BANDS:
        lo, hi = max(lo, 14), min(hi, 50)
        share = 100 * np.mean(rci < fleet.loc[fleet.band == band, "vci1"].median())
        if lo > 14:
            ax.axvline(lo, color=AXIS, lw=1.0, zorder=0)
        ax.text((lo + hi) / 2, ymax * 0.99, band, ha="center", va="top", fontsize=24, weight=600)
        ax.text((lo + hi) / 2, ymax * 0.88, f"{share:.0f}%", ha="center", va="top",
                fontsize=18, color=NOGO)
    ax.text(49.6, ymax * 0.77, "of days NO-GO, 2015–25", fontsize=17, color=NOGO,
            va="top", ha="right")
    ax.set_ylim(0, ymax)
    ax.set_xlim(14, 50)
    ax.set_xlabel("one-pass vehicle cone index, VCI₁ (psi)")
    ax.set_ylabel("vehicles")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="center right", handlelength=1.0, bbox_to_anchor=(1.0, 0.40))
    save(fig, "f2_fleet")


# --------------------------------------------------- 3. forecast vs today
def fig_decisions():
    dec = pd.read_csv(paths.FORECASTER_OUT / "decisions_by_band.csv")
    fig, axes = plt.subplots(1, 2, figsize=(COL_IN, 5.4), sharey=True, layout="constrained")
    bands = [b for b, _, _ in V.BANDS]
    y = np.arange(len(bands))[::-1]
    for ax, h in zip(axes, [3, 7]):
        p = dec[(dec.rule == "Persistence") & (dec.h == h)].set_index("band").loc[bands]
        f = dec[(dec.rule == "GRU+Ridge") & (dec.h == h)].set_index("band").loc[bands]
        for yi, a, b in zip(y, p.false_alarms_per_year, f.false_alarms_per_year):
            ax.plot([a, b], [yi, yi], color=GREY_L, lw=3, solid_capstyle="round", zorder=1)
            ax.text(max(a, b) + 3, yi, f"−{100 * (1 - b / a):.0f}%", va="center",
                    fontsize=18, color=OURS, weight=600)
        ax.scatter(p.false_alarms_per_year, y, s=140, color=GREY, zorder=2,
                   edgecolor="white", linewidth=2, label="today's value + margin")
        ax.scatter(f.false_alarms_per_year, y, s=140, color=OURS, zorder=3,
                   edgecolor="white", linewidth=2, label="forecast + margin")
        ax.set_title(f"{h} days ahead")
        ax.set_xlim(0, 95)
        ax.set_xticks([0, 25, 50, 75])
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"band {b}" for b in bands])
    fig.supxlabel("false NO-GO calls per vehicle-year (90% of real NO-GO days caught)",
                  fontsize=20, color=INK2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2,
               handletextpad=0.2, columnspacing=1.6)
    save(fig, "f3_decisions")


# ------------------------------------------------------------ 4. one real day
def candidate_days(bt, band="D", h=3):
    """Planning days ranked by how many more of today's routes than ours got stuck."""
    a = bt[(bt.band == band) & (bt.h == h)].pivot_table(index="date", columns="planner",
                                                         values="stuck", aggfunc="sum")
    gain = (a["Today (Liu et al.)"] - a["Forecast + margin (ours)"])
    return [pd.Timestamp(d) for d in gain[gain > 0].sort_values(ascending=False).index]


def _hillshade(z):
    gy, gx = np.gradient(np.nan_to_num(z, nan=np.nanmean(z)), T.CELL_M)
    slope = np.pi / 2 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(315), np.radians(45)
    return np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)


def _state_rgba(g, rep, rci):
    """Map colours: hillshade, terrain-blocked grey, soil SLOW-GO amber, NO-GO red."""
    hs = _hillshade(g.t["elevation"])
    base = 0.80 + 0.17 * (hs - hs.min()) / (np.ptp(hs) + 1e-9)
    img = np.dstack([base, base, base * 0.985, np.ones_like(base)])
    s, c = g.static["wheeled" if rep["kind"] == "wheeled" else "tracked"]
    blocked = (np.minimum(s, c) == 0) & g.t["valid"]
    img[blocked, :3] = img[blocked, :3] * 0.72
    soil = g.soil(rci, rep["vci1"], rep["vci50"])
    open_ = ~blocked & g.t["valid"]
    for mask, col, a in [((soil == 0) & open_, NOGO, 0.62),
                         ((soil > 0) & (soil < 0.999) & open_, SLOW, 0.32)]:
        rgb = np.array(matplotlib.colors.to_rgb(col))
        img[mask, :3] = (1 - a) * img[mask, :3] + a * rgb
    img[~g.t["valid"]] = [1, 1, 1, 0]
    return img


MAP_PREFS = [("D", 3), ("C", 3), ("B", 3), ("D", 7), ("A", 3), ("E", 3), ("D", 1)]


def fig_map(bt, fc, reps):
    g = PL.Grid(T.load())
    st = json.load(open(paths.OUTPUTS / "planner" / "setup.json"))
    src, dst = np.array(st["sources"]), np.array(st["dests"])
    dec = pd.read_csv(paths.FORECASTER_OUT / "decisions_by_band.csv")
    dates = pd.to_datetime(fc["dates"])
    rcx = lambda p: np.column_stack(np.divmod(p, g.shape[1])) * T.CELL_M / 1000
    best = None
    for band, h in MAP_PREFS[:4]:
        rep = reps[band]
        for plan_day in candidate_days(bt, band, h)[:25]:
            i = int(np.flatnonzero(dates == plan_day)[0])
            r_today, r_act = fc["Persistence"][i, 0], fc["actual"][i, h - 1]
            m_fc = PL._cross_year_margin(dec, "GRU+Ridge", band, h, plan_day.year)
            r_fc = fc["GRU_Ridge"][i, h - 1] - m_fc
            soil = lambda r: g.soil(r, rep["vci1"], rep["vci50"])
            mu_act = g.passability(rep["kind"], soil(r_act))
            R_today = g.routes(g.passability(rep["kind"], soil(round(r_today * 2) / 2)), src, dst)
            R_fc = g.routes(g.passability(rep["kind"], soil(round(r_fc * 2) / 2)), src, dst)
            for od, pt in R_today.items():
                pf = R_fc[od]
                if pt is None or pf is None or not np.isfinite(g.path_time(pf, mu_act)):
                    continue
                bad = np.flatnonzero(mu_act.ravel()[pt] == 0)
                if not len(bad):
                    continue
                # how far the stuck point lies from the forecast route, km
                sep = np.hypot(*(rcx(pf) - rcx(pt[bad[0]])).T).min()
                if best is None or sep > best[0]:
                    best = (sep, band, h, plan_day, r_today, r_act, m_fc, mu_act, od, pt, pf, bad[0])
    sep, band, h, plan_day, r_today, r_act, m_fc, mu_act, od, pt, pf, bad = best
    rep = reps[band]
    kind_name = {"tracked": "tracked", "wheeled": "wheeled", "halftrack": "half-track"}[rep["kind"]]

    fig, axes = plt.subplots(1, 2, figsize=(COL_IN, 6.0), layout="constrained")
    rc = lambda p: np.divmod(p, g.shape[1])
    for ax, rci, title in zip(axes, [r_today, r_act],
                              [f"Planning day, {plan_day:%d %b %Y}", f"Travel day, {h} days later"]):
        ax.imshow(_state_rgba(g, rep, rci), interpolation="nearest")
        rr, cc = np.nonzero(g.t["road"] > 0)
        ax.scatter(cc, rr, s=2.5, color="#3c4046", linewidths=0, alpha=0.55)
        ax.set_title(title, fontsize=21, pad=8)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    r1, c1 = rc(pt)
    r2, c2 = rc(pf)
    for ax in axes:                     # blue wide underneath, orange thin on top
        ax.plot(c2, r2, color=OURS, lw=5.5, solid_capstyle="round")
        ax.plot(c1, r1, color=FOIL, lw=2.6, solid_capstyle="round")
    br, bc = rc(pt[bad])
    axes[1].scatter([bc], [br], marker="X", s=420, color=INK, edgecolor="white", linewidth=2, zorder=5)
    axes[1].annotate("stuck", (bc, br), xytext=(14, -26), textcoords="offset points",
                     fontsize=20, weight=600, color=INK)
    for ax in axes:
        for p, mk in [(od[0], "o"), (od[1], "s")]:
            r, c = rc(p)
            ax.scatter([c], [r], marker=mk, s=170, color=INK, edgecolor="white", linewidth=2, zorder=6)
    allr, allc = np.concatenate([r1, r2]), np.concatenate([c1, c2])
    pad = 22
    r0, r_1 = allr.min() - pad, allr.max() + pad
    c0, c_1 = allc.min() - pad, allc.max() + pad
    # match the panel's aspect (about 1 : 1.12) so cells stay square
    hgt, wid = r_1 - r0, c_1 - c0
    if wid < hgt * 1.12:
        c0, c_1 = (c0 + c_1) / 2 - hgt * 0.56, (c0 + c_1) / 2 + hgt * 0.56
    else:
        r0, r_1 = (r0 + r_1) / 2 - wid / 2.24, (r0 + r_1) / 2 + wid / 2.24
    for ax in axes:
        ax.set_xlim(c0, c_1)
        ax.set_ylim(r_1, r0)
    km = 2 / (T.CELL_M / 1000)
    y0, x0 = r_1 - 0.06 * (r_1 - r0), c0 + 0.05 * (c_1 - c0)
    axes[0].plot([x0, x0 + km], [y0, y0], color=INK, lw=3)
    axes[0].text(x0 + km / 2, y0 - 0.025 * (r_1 - r0), "2 km", ha="center", va="bottom", fontsize=17)
    handles = [Patch(color=NOGO, alpha=0.62, label="NO-GO soil"),
               Patch(color=SLOW, alpha=0.32, label="SLOW-GO soil"),
               Patch(color="#9a9a96", label="dense forest, water, too steep"),
               Line2D([], [], color=FOIL, lw=2.6, label="planned on today's map"),
               Line2D([], [], color=OURS, lw=5.5, label="planned on the forecast")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=17,
               handlelength=1.2, columnspacing=1.2)
    save(fig, "f4_map")
    return (f"A {kind_name} vehicle from band {band} (VCI₁ {rep['vci1']:.0f} psi) plans on "
            f"{plan_day:%d %B %Y} for a trip {h} days later. On the planning day the ground "
            f"reads {r_today:.0f} psi and the direct route (orange) looks open. On the travel "
            f"day it reads {r_act:.0f} psi and that route crosses NO-GO soil (×). The route "
            f"planned on the forecast (blue, margin {m_fc:.1f} psi) avoids it. "
            f"Circle: start; square: end; grey lines: OpenStreetMap roads.")


# --------------------------------------------------- 5. caution trade-off
def fig_tradeoff(tr, bt):
    s = tr.groupby(["h", "rule", "scale"])[["trips", "possible", "stuck", "held"]].sum()
    s["stuck"] = 100 * s.stuck / s.trips
    s["held"] = 100 * s.held / s.possible
    static = bt[bt.planner == "Static (terrain only)"]
    st_stuck = 100 * static.stuck.sum() / static.trips.sum()
    fig, axes = plt.subplots(1, 2, figsize=(COL_IN, 5.1), layout="constrained")
    for ax, h in zip(axes, [3, 7]):
        for rule, col, lbl in [("Persistence", FOIL, "today's ground"),
                               ("GRU+Ridge", OURS, "forecast (ours)")]:
            d = s.loc[(h, rule)]
            ax.plot(d.held, d.stuck, color=col, lw=2.6, marker="o", ms=8, mec="white",
                    mew=1.8, zorder=3, label=f"{lbl}, margin ×0 to ×3")
            ax.scatter([d.loc[1.0, "held"]], [d.loc[1.0, "stuck"]], s=230, facecolor="none",
                       edgecolor=col, linewidth=2.2, zorder=4)
        z = s.loc[(h, "Persistence", 0.0)]
        ax.annotate("today's map,\nno margin [1]", (z.held, z.stuck), xytext=(12, 6),
                    textcoords="offset points", fontsize=17, color=FOIL)
        ax.set_title(f"leaving {h} days after planning")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("held back needlessly, %")
    axes[0].set_ylabel("sent into mud, % of trips")
    axes[1].text(0.97, 0.04, f"terrain only: {st_stuck:.0f}% stuck\n(off this scale)",
                 transform=axes[1].transAxes, ha="right", va="bottom", fontsize=17, color=GREY)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2,
               handletextpad=0.3, columnspacing=1.4, fontsize=18)
    save(fig, "f5_tradeoff")


# --------------------------------------------------------------- 6. convoy
def fig_convoy(cv):
    m = cv.groupby(["band", "mode"])["through"].mean()
    fig, ax = plt.subplots(figsize=(COL_IN, 3.1), layout="constrained")
    rows = [("D", "follow"), ("D", "wear_aware"), ("E", "follow"), ("E", "wear_aware")]
    labels = ["heavy tracked, follow the leader", "heavy tracked, shared wear map",
              "wheeled, follow the leader", "wheeled, shared wear map"]
    y = np.array([3.3, 2.5, 1.0, 0.2])
    for yi, (b, mode) in zip(y, rows):
        v = m.loc[(b, mode)]
        ax.barh(yi, v, height=0.62, color=OURS if mode == "wear_aware" else FOIL, linewidth=0)
        ax.text(v + 0.15, yi, f"{v:.1f}", va="center", fontsize=20, weight=600)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=18)
    ax.set_xlim(0, 10.5)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_xlabel("vehicles of 10 that get through (mean over soft days)", fontsize=18)
    ax.grid(axis="y", visible=False)
    save(fig, "f6_convoy")


# ------------------------------------------------------- 7. what it misses
def fig_onset(fc, h=3):
    dates = pd.to_datetime(fc["dates"])
    valid = dates + pd.Timedelta(days=h)
    sel = (valid >= "2024-05-25") & (valid <= "2024-08-31")
    fig, ax = plt.subplots(figsize=(COL_IN, 4.5), layout="constrained")
    ax.plot(valid[sel], fc["actual"][sel, h - 1], color=INK, lw=2.4, label="real")
    ax.plot(valid[sel], fc["Persistence"][sel, 0], color=FOIL, lw=2.0,
            label=f"today's value, {h} days earlier")
    ax.plot(valid[sel], fc["GRU_Ridge"][sel, h - 1], color=OURS, lw=2.4,
            label=f"forecast made {h} days earlier")
    _log_rci_axis(ax)
    ax.set_ylabel("RCI (psi)")
    ax.grid(axis="x", visible=False)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(loc="upper right", fontsize=17, handlelength=1.4)
    save(fig, "f7_onset")


def main():
    setup()
    fleet = V.add_vci(V.load_vehicles())
    reps = PL.representatives(fleet)
    rci = D.add_soil_strength(D.load_raw()).rci.values
    fc = dict(np.load(paths.FORECASTER_OUT / "test_forecasts.npz", allow_pickle=False))
    fig_season(reps)
    fig_fleet(fleet, rci)
    fig_decisions()
    fig_onset(fc)
    pdir = paths.OUTPUTS / "planner"
    bt = pd.read_csv(pdir / "backtest.csv", parse_dates=["date"])
    cap = fig_map(bt, fc, reps)
    (FIG / "f4_map_caption.txt").write_text(cap, encoding="utf-8")
    if (pdir / "tradeoff.csv").exists():
        fig_tradeoff(pd.read_csv(pdir / "tradeoff.csv"), bt)
    fig_convoy(pd.read_csv(pdir / "convoy.csv"))
    print("->", FIG.relative_to(paths.ROOT))


if __name__ == "__main__":
    main()
