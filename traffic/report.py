"""
Figures and written comparison for the trafficability forecasting experiment.

Reads outputs/forecasting/results_full.csv + test_predictions.npz and writes
outputs/forecasting/figures/*.png plus outputs/forecasting/RESULTS.md.

Usage:  python -m traffic.report
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from traffic import data as D
from traffic import paths

OUT = str(paths.FORECAST_OUT)
FIG = f"{OUT}/figures"

# validated categorical palette (light surface), used in fixed slot order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
STATUS = {"GO": "#0ca30c", "SLOW-GO": "#fab219", "NO-GO": "#d03b3b"}
SEQ = "#2a78d6"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": BASE,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 12, "figure.dpi": 140,
    # leaves room for the standfirst line each figure draws just above the axes
    "axes.titlepad": 26,
})


def _clean(ax, ygrid=True, xgrid=False):
    ax.grid(axis="y", visible=ygrid)
    ax.grid(axis="x", visible=xgrid)
    ax.set_axisbelow(True)


def load():
    res = pd.read_csv(f"{OUT}/results_full.csv")
    npz = np.load(f"{OUT}/test_predictions.npz", allow_pickle=True)
    dates = pd.to_datetime(npz["dates"])
    Y = npz["Y"]
    preds = {k: npz[k] for k in npz.files if k not in ("dates", "origins", "Y")}
    return res, dates, Y, preds


# ------------------------------------------------------------------------ figure 1
def fig_rmse_by_horizon(res, top_n=6):
    order = (res[res.horizon == 7].sort_values("rmse")["model"].tolist())
    keep = [m for m in order if m != "SeasonalNaive365"][:top_n]
    if "Persistence" not in keep:
        keep.append("Persistence")

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for i, m in enumerate(keep):
        s = res[res.model == m].sort_values("horizon")
        is_base = m in ("Persistence", "Climatology")
        ax.plot(s.horizon, s.rmse, lw=2.0,
                color=MUTED if is_base else SERIES[i % len(SERIES)],
                ls="--" if is_base else "-",
                marker="o", ms=5, mec=SURFACE, mew=1.5, label=m, zorder=3 - is_base)
        ax.annotate(m, (s.horizon.iloc[-1] + 0.12, s.rmse.iloc[-1]),
                    color=MUTED if is_base else SERIES[i % len(SERIES)],
                    fontsize=9, va="center", fontweight=500)
    ax.set_xlim(0.7, 9.9)
    ax.set_xticks(range(1, 8))
    ax.set_xlabel("Forecast horizon (days ahead)")
    ax.set_ylabel("Test RMSE  (kPa RCI)")
    ax.set_title("Forecast error grows with lead time", loc="left", fontweight=600)
    ax.text(0, 1.012, "Hold-out 2024-2025, 725 rolling origins", transform=ax.transAxes,
            color=MUTED, fontsize=9)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(f"{FIG}/f1_rmse_by_horizon.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------ figure 2
def fig_leaderboard(res, h=7):
    s = res[res.horizon == h].sort_values("skill_vs_persistence")
    fig, ax = plt.subplots(figsize=(8.0, 0.38 * len(s) + 1.6))
    colors = [STATUS["NO-GO"] if v < 0 else SEQ for v in s.skill_vs_persistence]
    ax.barh(s.model, 100 * s.skill_vs_persistence, color=colors, height=0.62)
    for y, (v, r) in enumerate(zip(s.skill_vs_persistence, s.rmse)):
        off = 1.2 if v >= 0 else -1.2
        ax.text(100 * v + off, y, f"{100*v:+.1f}%   ({r:.1f} kPa)",
                va="center", ha="left" if v >= 0 else "right",
                fontsize=9, color=INK2)
    ax.axvline(0, color=BASE, lw=1.2)
    ax.set_xlabel("RMSE skill score vs. persistence  (%)")
    ax.set_title(f"Model leaderboard, {h}-day-ahead RCI", loc="left", fontweight=600)
    ax.text(0, 1.012, "Higher is better; 0% = no better than yesterday's value",
            transform=ax.transAxes, color=MUTED, fontsize=9)
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo - 6, hi + 14)
    _clean(ax, ygrid=False, xgrid=True)
    fig.tight_layout()
    fig.savefig(f"{FIG}/f2_leaderboard_h{h}.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------ figure 3
def fig_timeseries(dates, Y, preds, best, h=7, year=2025):
    spec = D.VEHICLES["tracked_heavy"]
    m = (dates.year == year)
    d, obs, prd = dates[m], Y[m, h - 1], preds[best][m, h - 1]
    dp = pd.to_datetime(d) + pd.Timedelta(days=h)

    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    ax.axhspan(0, spec["vci1"], color=STATUS["NO-GO"], alpha=0.10, zorder=0)
    ax.axhspan(spec["vci1"], spec["vci50"], color=STATUS["SLOW-GO"], alpha=0.12, zorder=0)
    ax.axhspan(spec["vci50"], 320, color=STATUS["GO"], alpha=0.09, zorder=0)
    for v, lab in [(spec["vci1"], "VCI₁"), (spec["vci50"], "VCI₅₀")]:
        ax.axhline(v, color=BASE, lw=1, ls=":")
        ax.text(dp[-1], v + 3, lab, color=MUTED, fontsize=8, ha="right")

    ax.plot(dp, obs, color=INK, lw=1.8, label="Observed RCI", zorder=3)
    ax.plot(dp, prd, color=SERIES[0], lw=1.8, label=f"{best}, {h}-day forecast", zorder=4)
    ax.set_ylim(0, 320)
    ax.set_ylabel("Rating Cone Index  (kPa)")
    ax.set_title(f"{h}-day-ahead trafficability forecast, {year}", loc="left",
                 fontweight=600)
    ax.text(0, 1.012, f"Bands: {spec['label']} — red NO-GO, amber SLOW-GO, green GO",
            transform=ax.transAxes, color=MUTED, fontsize=9)
    leg = ax.legend(frameon=False, loc="lower left", ncol=2)
    for t in leg.get_texts():
        t.set_color(INK2)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(f"{FIG}/f3_timeseries_{year}_h{h}.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------ figure 4
def fig_nogo(res, h=7, vehicle="tracked_heavy"):
    col = f"{vehicle}_nogo_recall"
    s = res[(res.horizon == h)].dropna(subset=[col]).sort_values(col)
    s = s[~s.model.isin(["SeasonalNaive365"])]
    fig, ax = plt.subplots(figsize=(8.0, 0.38 * len(s) + 1.6))
    # the P10 entry is a decision rule read off the TFT, not a separate model
    is_rule = s.model.str.contains("P10").values
    ax.barh(s.model, 100 * s[col], height=0.62,
            color=[STATUS["SLOW-GO"] if r else SEQ for r in is_rule])
    for y, (v, r) in enumerate(zip(s[col], is_rule)):
        ax.text(100 * v + 1.2, y, f"{100*v:.1f}%" + ("   decision rule" if r else ""),
                va="center", fontsize=9, color=INK2)
    ax.set_xlim(0, 108)
    ax.set_xlabel("Recall on NO-GO days  (%)")
    ax.set_title(f"Catching the days an MBT cannot move, {h} days out",
                 loc="left", fontweight=600)
    ax.text(0, 1.012, "Missed NO-GO days are the operationally costly error",
            transform=ax.transAxes, color=MUTED, fontsize=9)
    _clean(ax, ygrid=False, xgrid=True)
    fig.tight_layout()
    fig.savefig(f"{FIG}/f4_nogo_recall_h{h}.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------ figure 5
def fig_monthly_error(res, dates, Y, preds, h=7, top_n=4):
    order = res[res.horizon == h].sort_values("rmse")["model"].tolist()
    keep = [m for m in order if m not in ("SeasonalNaive365",)][:top_n]
    if "Persistence" not in keep:
        keep.append("Persistence")
    months = np.arange(1, 13)
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    w = 0.8 / len(keep)
    for i, m in enumerate(keep):
        e = (preds[m][:, h - 1] - Y[:, h - 1]) ** 2
        rm = [np.sqrt(e[dates.month == mo].mean()) for mo in months]
        c = MUTED if m == "Persistence" else SERIES[i % len(SERIES)]
        ax.bar(months + (i - (len(keep) - 1) / 2) * w, rm, width=w * 0.9,
               color=c, label=m)
    ax.set_xticks(months)
    ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_ylabel(f"RMSE at h={h}  (kPa RCI)")
    ax.set_title("Error concentrates in the monsoon onset and withdrawal",
                 loc="left", fontweight=600)
    ax.text(0, 1.012, "June and October are the transition months that decide GO / NO-GO",
            transform=ax.transAxes, color=MUTED, fontsize=9)
    leg = ax.legend(frameon=False, ncol=len(keep), loc="upper left",
                    bbox_to_anchor=(0, -0.12))
    for t in leg.get_texts():
        t.set_color(INK2)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(f"{FIG}/f5_monthly_error_h{h}.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------ figure 6
def fig_feature_gain(h=7, top=18):
    path = f"{OUT}/lgbm_feature_gain.json"
    if not os.path.exists(path):
        return
    g = json.load(open(path))[str(h)]
    s = pd.Series(g).sort_values(ascending=False).head(top)[::-1]
    s = 100 * s / sum(g.values())
    fig, ax = plt.subplots(figsize=(7.6, 0.30 * len(s) + 1.6))
    ax.barh(s.index, s.values, color=SEQ, height=0.66)
    for y, v in enumerate(s.values):
        ax.text(v + 0.3, y, f"{v:.1f}%", va="center", fontsize=8.5, color=INK2)
    ax.set_xlabel("Share of LightGBM split gain  (%)")
    ax.set_title(f"What drives the {h}-day RCI forecast", loc="left", fontweight=600)
    _clean(ax, ygrid=False, xgrid=True)
    fig.tight_layout()
    fig.savefig(f"{FIG}/f6_feature_gain_h{h}.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------ figure 7
def fig_tft_quantiles(dates, Y, h=7, year=2025):
    path = f"{OUT}/tft_quantiles.npy"
    if not os.path.exists(path):
        return
    q = np.load(path)                      # (N, H, 3) in RCI units
    m = (dates.year == year) & dates.month.isin([5, 6, 7, 8, 9, 10, 11])
    d = pd.to_datetime(dates[m]) + pd.Timedelta(days=h)
    lo, mid, hi = q[m, h - 1, 0], q[m, h - 1, 1], q[m, h - 1, 2]
    spec = D.VEHICLES["tracked_heavy"]

    fig, ax = plt.subplots(figsize=(10.0, 4.4))
    ax.axhspan(0, spec["vci1"], color=STATUS["NO-GO"], alpha=0.10, zorder=0)
    ax.axhspan(spec["vci1"], spec["vci50"], color=STATUS["SLOW-GO"], alpha=0.12, zorder=0)
    ax.axhspan(spec["vci50"], 320, color=STATUS["GO"], alpha=0.09, zorder=0)
    ax.fill_between(d, lo, hi, color=SERIES[0], alpha=0.22, lw=0,
                    label="TFT P10–P90")
    ax.plot(d, mid, color=SERIES[0], lw=1.8, label="TFT P50")
    ax.plot(d, Y[m, h - 1], color=INK, lw=1.6, label="Observed")
    ax.set_ylim(0, 320)
    ax.set_ylabel("Rating Cone Index  (kPa)")
    ax.set_title(f"TFT gives a calibrated band, not just a number ({year} monsoon)",
                 loc="left", fontweight=600)
    ax.text(0, 1.012, f"{h}-day-ahead quantile forecast; a planner reads risk off the band width",
            transform=ax.transAxes, color=MUTED, fontsize=9)
    leg = ax.legend(frameon=False, loc="lower left", ncol=3)
    for t in leg.get_texts():
        t.set_color(INK2)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(f"{FIG}/f7_tft_quantiles_h{h}.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------ figure 8
def fig_confusion(Y, preds, best, h=7, vehicle="tracked_heavy"):
    yc = D.rci_to_class(Y[:, h - 1], vehicle)
    pc = D.rci_to_class(preds[best][:, h - 1], vehicle)
    cm = np.zeros((3, 3))
    for a, b in zip(yc, pc):
        cm[a, b] += 1
    pct = 100 * cm / cm.sum(1, keepdims=True).clip(1)

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    ax.grid(False)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{int(cm[i,j])}\n{pct[i,j]:.0f}%", ha="center", va="center",
                    color="#ffffff" if pct[i, j] > 55 else INK, fontsize=10)
    ax.set_xticks(range(3), D.CLASS_NAMES)
    ax.set_yticks(range(3), D.CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Observed")
    ax.set_title(f"{best}, h={h} — {D.VEHICLES[vehicle]['label']}",
                 loc="left", fontweight=600)
    fig.tight_layout()
    fig.savefig(f"{FIG}/f8_confusion_h{h}.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------ figure 9
def fig_target_construction(raw):
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 6.0), sharex=True,
                             height_ratios=[1, 1.4])
    m = raw["Date"].dt.year.isin([2024, 2025])
    d = raw.loc[m, "Date"]
    axes[0].bar(d, raw.loc[m, "rain"], color=SERIES[0], width=1.0, lw=0)
    axes[0].set_ylabel("Rain (mm/d)")
    axes[0].set_title("From rainfall to soil moisture to soil strength",
                      loc="left", fontweight=600, pad=10)
    _clean(axes[0])

    spec = D.VEHICLES["tracked_heavy"]
    axes[1].axhspan(0, spec["vci1"], color=STATUS["NO-GO"], alpha=0.10, zorder=0)
    axes[1].axhspan(spec["vci1"], spec["vci50"], color=STATUS["SLOW-GO"], alpha=0.12, zorder=0)
    axes[1].axhspan(spec["vci50"], 320, color=STATUS["GO"], alpha=0.09, zorder=0)
    axes[1].plot(d, raw.loc[m, "rci"], color=INK, lw=1.6)
    axes[1].set_ylim(0, 320)
    axes[1].set_ylabel("RCI (kPa)")
    axes[1].legend(handles=[Patch(color=STATUS[k], alpha=0.5, label=k)
                            for k in ["GO", "SLOW-GO", "NO-GO"]],
                   frameon=False, ncol=3, loc="upper left")
    _clean(axes[1])
    fig.tight_layout()
    fig.savefig(f"{FIG}/f9_target_construction.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------- write-up
def md_table(df, index_name="model", dec=2):
    """Minimal GitHub-flavoured markdown table (avoids a tabulate dependency)."""
    cols = [index_name] + [str(c) for c in df.columns]
    rows = [[str(i)] + ["" if pd.isna(v) else
                        f"{v:.{dec}f}" if isinstance(v, (float, np.floating)) else str(v)
                        for v in r] for i, r in zip(df.index, df.values)]
    w = [max(len(c), *(len(r[k]) for r in rows)) if rows else len(c)
         for k, c in enumerate(cols)]
    out = ["| " + " | ".join(c.ljust(w[k]) for k, c in enumerate(cols)) + " |",
           "|" + "|".join("-" * (w[k] + 2) if k == 0 else "-" * (w[k] + 1) + ":"
                          for k in range(len(cols))) + "|"]
    for r in rows:
        out.append("| " + " | ".join(v.rjust(w[k]) if k else v.ljust(w[k])
                                     for k, v in enumerate(r)) + " |")
    return "\n".join(out)


def write_markdown(res, dates, Y, preds, best):
    lines = []
    A = lines.append
    A("# Trafficability forecasting - model comparison\n")
    A(f"Test period **{dates.min().date()} - {dates.max().date()}** "
      f"({len(dates)} rolling forecast origins), never seen during training or "
      "model selection.\n")

    A("\n## Headline: test RMSE (kPa RCI)\n")
    piv = res[res.horizon.isin([1, 3, 7])].pivot_table(
        index="model", columns="horizon", values="rmse").round(2)
    sk = res[res.horizon == 7].set_index("model")["skill_vs_persistence"] * 100
    piv["skill@7 (%)"] = sk.round(1)
    piv = piv.sort_values(7)
    piv.columns = [f"h={c}" if isinstance(c, (int, np.integer)) else c for c in piv.columns]
    A(md_table(piv))

    A("\n\n## Trafficability decision quality (tracked heavy / MBT)\n")
    cols = ["tracked_heavy_acc", "tracked_heavy_macro_f1", "tracked_heavy_nogo_recall"]
    t = res[res.horizon == 7].set_index("model")[cols].mul(100).round(1)
    t.columns = ["GO/SLOW-GO/NO-GO accuracy %", "macro F1 %", "NO-GO recall %"]
    A(md_table(t.sort_values("macro F1 %", ascending=False), dec=1))

    A("\n\n## Best model per horizon\n")
    for h in [1, 3, 7]:
        s = res[res.horizon == h].sort_values("rmse").iloc[0]
        A(f"- **h={h}**: {s.model} - RMSE {s.rmse:.2f} kPa, MAE {s.mae:.2f}, "
          f"R2 {s.r2:.4f}, skill vs persistence {100*s.skill_vs_persistence:+.1f}%")

    with open(f"{OUT}/RESULTS.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    os.makedirs(FIG, exist_ok=True)
    res, dates, Y, preds = load()
    ranked = res[res.horizon == 7].sort_values("rmse")["model"].tolist()
    best = ranked[0]
    print("best @ h=7:", best)

    raw, _ = D.get_dataset()
    fig_target_construction(raw)
    fig_rmse_by_horizon(res)
    fig_leaderboard(res, 7)
    fig_leaderboard(res, 1)
    fig_timeseries(dates, Y, preds, best, 7, 2025)
    fig_nogo(res, 7)
    fig_monthly_error(res, dates, Y, preds, 7)
    fig_feature_gain(7)
    fig_tft_quantiles(dates, Y, 7)
    fig_confusion(Y, preds, best, 7)
    write_markdown(res, dates, Y, preds, best)
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
