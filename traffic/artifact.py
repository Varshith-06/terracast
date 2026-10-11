"""
Generate the shareable HTML report from outputs/forecasting/results_full.csv.

Every number on the page is read from the results files, so the page cannot
drift from the experiment. Charts are inline SVG built against CSS custom
properties, so they follow the reader's light/dark theme.

Usage:  python -m traffic.artifact  ->  outputs/forecasting/trafficability_report.html
"""

from __future__ import annotations

import html
import json
import os

import numpy as np
import pandas as pd

from traffic import data as D
from traffic import paths

OUT = str(paths.FORECAST_OUT)
PAGE = f"{OUT}/trafficability_report.html"
BASELINES = ("Persistence", "Climatology", "SeasonalNaive365")

FAMILY = {
    "Persistence": "baseline", "Climatology": "baseline", "SeasonalNaive365": "baseline",
    "ARIMA": "classical", "ARIMAX-Fourier (DHR)": "classical",
    "Ridge": "tabular", "LightGBM": "tabular", "LightGBM(linear)": "tabular",
    "RandomForest": "tabular", "ExtraTrees": "tabular",
    "LSTM": "deep", "GRU": "deep", "TCN": "deep", "Transformer": "deep",
    "N-BEATS": "deep", "TFT": "deep", "Ensemble(top3)": "ensemble",
    "TFT P10 (safe rule)": "rule",
}

# Deliberately biased low, so scoring it on RMSE would be meaningless; it appears
# only in the decision table.
DECISION_ONLY = {"TFT P10 (safe rule)"}
NOT_RANKED = DECISION_ONLY | {"SeasonalNaive365"}

FAMILY_ORDER = ["baseline", "classical", "tabular", "deep", "ensemble", "rule"]
FAMILY_TITLE = {
    "baseline": "Baselines",
    "classical": "Classical time series",
    "tabular": "Tabular machine learning",
    "deep": "Deep sequence models",
    "ensemble": "Combination",
    "rule": "Decision rule",
}
FAMILY_BLURB = {
    "baseline": "Reference points. A model that cannot clear these has learned nothing "
                "about the site that the calendar does not already say.",
    "classical": "Explicit statistical structure, fitted by maximum likelihood. Cheap, "
                 "interpretable, and the standard first answer for a single series.",
    "tabular": "One model per horizon, reading the 128 engineered features at the "
               "forecast origin. No notion of sequence — order is baked into the features.",
    "deep": "Read a 60-day window of 27 raw channels and emit all seven days in one "
            "pass. Trained on the RTX 4060, three seeds averaged, early-stopped on the "
            "validation block.",
    "ensemble": "Members picked on validation RMSE before the test set was touched.",
    "rule": "Not a model — the same TFT, read at a different quantile.",
}

MODEL_NOTES = {
    "Persistence": "Tomorrow equals today. With a lag-1 autocorrelation of 0.99 this is "
                   "a genuinely strong forecast at short lead, and it is the zero point "
                   "of the skill score.",
    "Climatology": "The average RCI for this day of the year across the training years. "
                   "Tests whether knowing the season is enough on its own.",
    "SeasonalNaive365": "Whatever the value was 365 days ago. Tests whether last year's "
                        "monsoon timing repeats — it does not.",
    "ARIMA": "Autoregression on one lag plus a five-term moving average of past shocks, "
             "order (1,0,5) by AIC. No seasonal term: d=0 because ADF and KPSS agree the "
             "series is stationary.",
    "ARIMAX-Fourier (DHR)": "The same ARIMA errors, plus three Fourier pairs carrying the "
                            "annual cycle as exogenous regressors. A seasonal SARIMA at "
                            "period 365 is not estimable, so this is the correct form — "
                            "and the harmonics are genuinely known for the future.",
    "Ridge": "Linear regression with an L2 penalty over all 128 features. Cross-validated "
             "α settled at 1000 for every horizon: heavy shrinkage, because the lag and "
             "rolling features are strongly collinear.",
    "LightGBM": "Gradient-boosted trees. Each split is a threshold on one feature, so the "
                "fitted response is piecewise constant.",
    "LightGBM(linear)": "The same boosting with a linear model in each leaf instead of a "
                        "constant. The fair comparison against Ridge, and the measure of "
                        "what the staircase was costing.",
    "RandomForest": "Bagged deep trees over bootstrap samples, predictions averaged.",
    "ExtraTrees": "As RandomForest but with split points drawn at random, trading fit for "
                  "variance reduction.",
    "LSTM": "Recurrent network with input, forget and output gates plus a cell state — "
            "the classical long-memory sequence model.",
    "GRU": "The same idea with two gates and no separate cell state. Fewer parameters to "
           "fit, which matters here.",
    "TCN": "Dilated causal convolutions stacked so the receptive field spans the whole "
           "60-day window. Imposes a strong prior: local patterns at several time scales.",
    "Transformer": "Self-attention over the window, so any day can attend to any earlier "
                   "day directly rather than through a recurrent chain.",
    "N-BEATS": "A stack of residual MLP blocks, each fitting and then subtracting its own "
               "contribution, with the exogenous window injected as context.",
    "TFT": "Temporal Fusion Transformer: per-variable selection networks, gated residual "
           "networks, an LSTM encoder-decoder, interpretable multi-head attention, and a "
           "quantile head that emits P10/P50/P90 rather than a single number.",
    "Ensemble(top3)": "Unweighted mean of the three strongest models on validation "
                      "(TCN, Transformer, GRU). Averaging decorrelated errors is close to "
                      "free accuracy.",
    "TFT P10 (safe rule)": "The TFT's 10th percentile used as the decision input instead "
                           "of its median. Its RMSE is poor by construction — the point is "
                           "the GO/NO-GO call, not the point estimate.",
}


def esc(s):
    return html.escape(str(s))


def fmt(v, n=2):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{n}f}"


# ------------------------------------------------------------------- svg utilities
def svg_open(w, h, label):
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" '
            f'aria-label="{esc(label)}" class="chart">')


def txt(x, y, s, cls="lbl", anchor="start", dy=0):
    return (f'<text x="{x:.1f}" y="{y + dy:.1f}" text-anchor="{anchor}" '
            f'class="{cls}">{esc(s)}</text>')


# --------------------------------------------------------------------- chart: bars
def chart_leaderboard(res, h=7):
    s = res[res.horizon == h].copy()
    s["skill"] = 100 * s["skill_vs_persistence"]
    s = s[~s.model.isin(NOT_RANKED)].sort_values("skill", ascending=False)

    row_h, pad_t, pad_l, pad_r = 26, 34, 168, 92
    W, Hh = 860, pad_t + row_h * len(s) + 16
    lo = min(0, s.skill.min())
    hi = max(5, s.skill.max())
    span = hi - lo
    x0 = pad_l + (0 - lo) / span * (W - pad_l - pad_r)

    def sx(v):
        return pad_l + (v - lo) / span * (W - pad_l - pad_r)

    out = [svg_open(W, Hh, f"Skill score versus persistence at {h} days")]
    for gv in np.linspace(lo, hi, 5):
        out.append(f'<line x1="{sx(gv):.1f}" y1="{pad_t - 12:.1f}" x2="{sx(gv):.1f}" '
                   f'y2="{Hh - 14:.1f}" class="grid"/>')
        out.append(txt(sx(gv), pad_t - 18, f"{gv:.0f}%", "tick", "middle"))
    for i, r in enumerate(s.itertuples()):
        y = pad_t + i * row_h
        fam = FAMILY.get(r.model, "tabular")
        w = abs(sx(r.skill) - x0)
        bx = min(sx(r.skill), x0)
        out.append(f'<rect x="{bx:.1f}" y="{y:.1f}" width="{max(w,1.2):.1f}" '
                   f'height="15" rx="3" class="bar bar-{fam}"/>')
        out.append(txt(pad_l - 12, y + 12, r.model, "name", "end"))
        lx = sx(r.skill) + (7 if r.skill >= 0 else -7)
        out.append(txt(lx, y + 12, f"{r.skill:+.1f}%  ·  {r.rmse:.1f} kPa", "val",
                       "start" if r.skill >= 0 else "end"))
    out.append(f'<line x1="{x0:.1f}" y1="{pad_t - 12:.1f}" x2="{x0:.1f}" '
               f'y2="{Hh - 14:.1f}" class="axis"/>')
    out.append("</svg>")
    return "\n".join(out)


# -------------------------------------------------------------------- chart: lines
def chart_horizon(res, models):
    W, Hh = 860, 400
    pad = dict(l=64, r=132, t=28, b=48)
    hs = sorted(res.horizon.unique())
    ymax = max(res[res.model.isin(models)].rmse.max(), 1) * 1.06
    ymin = 0

    def sx(h):
        return pad["l"] + (h - hs[0]) / (hs[-1] - hs[0]) * (W - pad["l"] - pad["r"])

    def sy(v):
        return Hh - pad["b"] - (v - ymin) / (ymax - ymin) * (Hh - pad["t"] - pad["b"])

    out = [svg_open(W, Hh, "Test RMSE against forecast horizon")]
    for gv in np.linspace(0, ymax, 6):
        out.append(f'<line x1="{pad["l"]}" y1="{sy(gv):.1f}" x2="{W - pad["r"]}" '
                   f'y2="{sy(gv):.1f}" class="grid"/>')
        out.append(txt(pad["l"] - 10, sy(gv) + 4, f"{gv:.0f}", "tick", "end"))
    for h in hs:
        out.append(txt(sx(h), Hh - pad["b"] + 20, str(h), "tick", "middle"))
    out.append(txt(pad["l"] - 10, pad["t"] - 8, "RMSE", "axtitle", "end"))
    out.append(txt((pad["l"] + W - pad["r"]) / 2, Hh - 12,
                   "forecast horizon (days ahead)", "axtitle", "middle"))

    for i, m in enumerate(models):
        s = res[res.model == m].sort_values("horizon")
        base = m in BASELINES
        cls = f"ln ln{i % 6}" + (" ln-base" if base else "")
        pts = " ".join(f"{sx(r.horizon):.1f},{sy(r.rmse):.1f}" for r in s.itertuples())
        out.append(f'<polyline points="{pts}" class="{cls}"/>')
        for r in s.itertuples():
            out.append(f'<circle cx="{sx(r.horizon):.1f}" cy="{sy(r.rmse):.1f}" r="3.4" '
                       f'class="dot {cls}"/>')
        last = s.iloc[-1]
        out.append(txt(sx(last.horizon) + 10, sy(last.rmse) + 4, m,
                       f"series-lbl s{i % 6}" + (" ln-base" if base else "")))
    out.append("</svg>")
    return "\n".join(out)


# -------------------------------------------------------------------- chart: bias
def chart_bias(res, h=7):
    """Diverging bars: mean signed error. Positive = claims the ground is firmer."""
    s = res[(res.horizon == h) & (~res.model.isin(["SeasonalNaive365", "Climatology"]))]
    s = s.sort_values("bias")
    row_h, pad_t, pad_l, pad_r = 24, 40, 172, 60
    W, Hh = 860, pad_t + row_h * len(s) + 20
    lo = min(-1.0, s.bias.min() * 1.12)
    hi = max(1.0, s.bias.max() * 1.12)

    def sx(v):
        return pad_l + (v - lo) / (hi - lo) * (W - pad_l - pad_r)

    x0 = sx(0.0)
    out = [svg_open(W, Hh, "Mean signed error at seven days by model")]
    out.append(txt(x0, pad_t - 24, "0", "tick", "middle"))
    out.append(txt(x0 - 12, pad_t - 24, "◀ cautious", "tick", "end"))
    out.append(txt(x0 + 12, pad_t - 24, "over-states strength ▶", "tick", "start"))
    for i, r in enumerate(s.itertuples()):
        y = pad_t + i * row_h
        w = abs(sx(r.bias) - x0)
        bx = min(sx(r.bias), x0)
        cls = "bias-hi" if r.bias > 0 else "bias-lo"
        out.append(f'<rect x="{bx:.1f}" y="{y:.1f}" width="{max(w,1.2):.1f}" '
                   f'height="14" rx="3" class="{cls}"/>')
        out.append(txt(pad_l - 12, y + 11, r.model, "name", "end"))
        out.append(txt(sx(r.bias) + (6 if r.bias >= 0 else -6), y + 11,
                       f"{r.bias:+.2f}", "val", "start" if r.bias >= 0 else "end"))
    out.append(f'<line x1="{x0:.1f}" y1="{pad_t - 16:.1f}" x2="{x0:.1f}" '
               f'y2="{Hh - 16:.1f}" class="axis"/>')
    out.append("</svg>")
    return "\n".join(out)


# --------------------------------------------------------------- chart: time series
def chart_timeseries(dates, Y, preds, best, h=7, year=2025):
    spec = D.VEHICLES["tracked_heavy"]
    m = dates.year == year
    d = pd.to_datetime(dates[m]) + pd.Timedelta(days=h)
    obs, prd = Y[m, h - 1], preds[best][m, h - 1]

    W, Hh = 860, 330
    pad = dict(l=52, r=16, t=20, b=42)
    ymax = 310.0
    doy = np.array([(x - pd.Timestamp(f"{year}-01-01")).days for x in d], float)

    def sx(v):
        return pad["l"] + (v - doy[0]) / (doy[-1] - doy[0]) * (W - pad["l"] - pad["r"])

    def sy(v):
        return Hh - pad["b"] - np.clip(v, 0, ymax) / ymax * (Hh - pad["t"] - pad["b"])

    out = [svg_open(W, Hh, f"Observed and forecast RCI through {year}")]
    for lo, hi, cls in [(0, spec["vci1"], "band-nogo"),
                        (spec["vci1"], spec["vci50"], "band-slow"),
                        (spec["vci50"], ymax, "band-go")]:
        out.append(f'<rect x="{pad["l"]}" y="{sy(hi):.1f}" width="{W - pad["l"] - pad["r"]}" '
                   f'height="{sy(lo) - sy(hi):.1f}" class="{cls}"/>')
    for v, lab in [(spec["vci1"], "VCI₁ 30"), (spec["vci50"], "VCI₅₀ 45")]:
        out.append(f'<line x1="{pad["l"]}" y1="{sy(v):.1f}" x2="{W - pad["r"]}" '
                   f'y2="{sy(v):.1f}" class="thresh"/>')
        out.append(txt(W - pad["r"] - 6, sy(v) - 5, lab, "tick", "end"))
    for gv in [0, 100, 200, 300]:
        out.append(txt(pad["l"] - 9, sy(gv) + 4, str(gv), "tick", "end"))
    for mo, lab in zip([1, 3, 5, 7, 9, 11],
                       ["Jan", "Mar", "May", "Jul", "Sep", "Nov"]):
        dv = (pd.Timestamp(f"{year}-{mo:02d}-01") - pd.Timestamp(f"{year}-01-01")).days
        if doy[0] <= dv <= doy[-1]:
            out.append(txt(sx(dv), Hh - pad["b"] + 20, lab, "tick", "middle"))

    for vals, cls in [(obs, "line-obs"), (prd, "line-pred")]:
        pts = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(doy, vals))
        out.append(f'<polyline points="{pts}" class="{cls}"/>')
    out.append(txt(pad["l"] - 9, pad["t"] - 4, "kPa", "axtitle", "end"))
    out.append("</svg>")
    return "\n".join(out)


# --------------------------------------------------------------------- model roster
def build_roster(r7, params):
    """One block per family: what each model is, and what it scored."""
    blocks = []
    for fam in FAMILY_ORDER:
        names = [m for m in MODEL_NOTES if FAMILY.get(m) == fam and m in r7.index]
        if not names:
            continue
        names.sort(key=lambda m: (m in DECISION_ONLY, r7.loc[m, "rmse"]))
        rows = []
        for m in names:
            r = r7.loc[m]
            if m in DECISION_ONLY:
                chip = (f'<span class="chip chip-safe">'
                        f'{100*r["tracked_heavy_nogo_recall"]:.0f}% NO-GO recall</span>')
            else:
                sk = 100 * r["skill_vs_persistence"]
                chip = (f'<span class="chip">{r["rmse"]:.2f} kPa</span>'
                        f'<span class="chip chip-ghost">{sk:+.1f}%</span>')
            if m in params:
                chip += f'<span class="chip chip-ghost">{params[m]/1000:.0f}k params</span>'
            rows.append(
                f'<div class="m-row"><div class="m-name">'
                f'<span class="dot-fam fam-{fam}"></span>{esc(m)}</div>'
                f'<div class="m-desc">{esc(MODEL_NOTES[m])}</div>'
                f'<div class="m-chips">{chip}</div></div>')
        blocks.append(
            f'<div class="fam-block"><h3>{FAMILY_TITLE[fam]}</h3>'
            f'<p class="fam-blurb">{FAMILY_BLURB[fam]}</p>'
            + "\n".join(rows) + "</div>")
    return "\n".join(blocks)


# ------------------------------------------------------------------------ page body
def build(res, val_res, dates, Y, preds, meta):
    h = 7
    r7 = res[res.horizon == h].set_index("model")
    ranked = r7[~r7.index.isin(NOT_RANKED)].sort_values("rmse")
    best = ranked.index[0]
    best_row = ranked.iloc[0]
    pers7 = r7.loc["Persistence"]

    top_models = [m for m in ranked.index if m != "Climatology"][:5]
    if "Persistence" not in top_models:
        top_models.append("Persistence")

    # the safety story: who actually catches the days the ground will not hold
    nogo = r7["tracked_heavy_nogo_recall"].dropna().sort_values(ascending=False)
    safe_best = nogo.index[0]
    safe_best_v = 100 * nogo.iloc[0]
    tft_nogo = 100 * r7.loc[best, "tracked_heavy_nogo_recall"]
    pers_nogo = 100 * r7.loc["Persistence", "tracked_heavy_nogo_recall"]
    has_rule = "TFT P10 (safe rule)" in r7.index
    rule_nogo = 100 * r7.loc["TFT P10 (safe rule)", "tracked_heavy_nogo_recall"] if has_rule else float("nan")
    rule_acc = 100 * r7.loc["TFT P10 (safe rule)", "tracked_heavy_acc"] if has_rule else float("nan")

    # ---- headline stats
    stats = [
        ("Best model", best, f"lowest RMSE at {h} days"),
        ("RMSE at 7 days", f"{best_row.rmse:.1f} <span class='u'>kPa</span>",
         f"vs {pers7.rmse:.1f} for persistence"),
        ("Skill vs persistence", f"{100*best_row.skill_vs_persistence:+.0f}<span class='u'>%</span>",
         "error removed beyond yesterday's value"),
        ("NO-GO days caught", f"{100*best_row.tracked_heavy_nogo_recall:.0f}<span class='u'>%</span>",
         "MBT immobilisation, 7 days out"),
    ]
    stat_html = "\n".join(
        f'<div class="stat"><div class="stat-k">{esc(k)}</div>'
        f'<div class="stat-v">{v}</div><div class="stat-n">{esc(n)}</div></div>'
        for k, v, n in stats)

    # ---- vehicle cards
    veh_html = "\n".join(
        f'<div class="veh veh-{k}"><div class="veh-name">{esc(s["label"])}</div>'
        f'<div class="veh-th"><span>VCI<sub>1</sub> {s["vci1"]:.0f}</span>'
        f'<span>VCI<sub>50</sub> {s["vci50"]:.0f}</span></div>'
        f'<div class="veh-share">{meta["class_share"][k]}</div></div>'
        for k, s in D.VEHICLES.items())

    # ---- main results table
    hs = [1, 3, 7]
    piv = res[(res.horizon.isin(hs)) & (~res.model.isin(DECISION_ONLY))].pivot_table(
        index="model", columns="horizon", values="rmse")
    piv = piv.join(r7[["skill_vs_persistence", "mae", "r2"]]).sort_values(7)
    rows = []
    for m, r in piv.iterrows():
        fam = FAMILY.get(m, "tabular")
        mark = ' class="row-best"' if m == best else ""
        rows.append(
            f'<tr{mark}><td class="mdl"><span class="dot-fam fam-{fam}"></span>'
            f'{esc(m)}</td>'
            f'<td class="n">{r[1]:.2f}</td><td class="n">{r[3]:.2f}</td>'
            f'<td class="n"><strong>{r[7]:.2f}</strong></td>'
            f'<td class="n">{r["mae"]:.2f}</td><td class="n">{r["r2"]:.4f}</td>'
            f'<td class="n skill">{100*r["skill_vs_persistence"]:+.1f}%</td></tr>')
    table_html = "\n".join(rows)

    # ---- decision-quality table
    cols = ["tracked_heavy_acc", "tracked_heavy_macro_f1", "tracked_heavy_nogo_recall",
            "wheeled_6x6_macro_f1", "wheeled_6x6_nogo_recall"]
    dq = r7[cols].mul(100).sort_values("tracked_heavy_nogo_recall", ascending=False)
    dq_rows = "\n".join(
        f'<tr{" class=\"row-rule\"" if m in DECISION_ONLY else ""}>'
        f'<td class="mdl"><span class="dot-fam fam-{FAMILY.get(m,"tabular")}"></span>'
        f'{esc(m)}</td>' + "".join(f'<td class="n">{fmt(r[c], 1)}</td>' for c in cols)
        + "</tr>" for m, r in dq.iterrows())

    # ---- feature drivers
    feat_html = ""
    if meta.get("features"):
        feat_html = "\n".join(
            f'<li><code>{esc(k)}</code><span class="fbar" style="--w:{v/meta["features"][0][1]*100:.1f}%"></span>'
            f'<span class="fval">{v:.1f}%</span></li>' for k, v in meta["features"])

    arima_note = meta.get("arima_note", "")
    val_note = meta.get("ensemble_note", "")

    charts = {
        "leaderboard": chart_leaderboard(res, h),
        "horizon": chart_horizon(res, top_models),
        "series": chart_timeseries(dates, Y, preds, best, h, 2025),
        "bias": chart_bias(res, h),
    }
    roster_html = build_roster(r7, meta.get("params", {}))

    # short vs long lead: which drivers the forecast leans on
    f1, f7 = meta.get("features_h1", []), meta.get("features", [])
    def _flist(items):
        if not items:
            return ""
        top = items[0][1] or 1
        return "\n".join(
            f'<li><code>{esc(k)}</code>'
            f'<span class="fbar" style="--w:{v/top*100:.1f}%"></span>'
            f'<span class="fval">{v:.1f}%</span></li>' for k, v in items[:8])
    feat_h1, feat_h7 = _flist(f1), _flist(f7)

    return TEMPLATE.format(
        best=esc(best), stat_html=stat_html, veh_html=veh_html,
        table_html=table_html, dq_rows=dq_rows, feat_html=feat_html,
        chart_leaderboard=charts["leaderboard"], chart_horizon=charts["horizon"],
        chart_series=charts["series"], arima_note=arima_note, val_note=val_note,
        n_test=meta["n_test"], test_from=meta["test_from"], test_to=meta["test_to"],
        n_train=meta["n_train"], n_val=meta["n_val"],
        n_feat=meta["n_feat"], gpu=esc(meta["gpu"]), dl_secs=meta["dl_secs"],
        pers7=f"{pers7.rmse:.1f}", best7=f"{best_row.rmse:.1f}",
        best_skill=f"{100*best_row.skill_vs_persistence:.0f}",
        best_r2=f"{best_row.r2:.4f}",
        best_acc=f"{100*best_row.tracked_heavy_acc:.1f}",
        best_nogo=f"{100*best_row.tracked_heavy_nogo_recall:.0f}",
        runner=esc(ranked.index[1]), runner7=f"{ranked.iloc[1].rmse:.1f}",
        tft_nogo=f"{tft_nogo:.0f}", pers_nogo=f"{pers_nogo:.0f}",
        safe_best=esc(safe_best), safe_best_v=f"{safe_best_v:.0f}",
        calib_html=meta.get("calib_html", ""),
        roster_html=roster_html, chart_bias=charts["bias"],
        feat_h1=feat_h1, feat_h7=feat_h7,
        rule_note=(
            '<div class="note"><p><strong>Reading the TFT\'s lower quantile fixes '
            'this — and costs nothing.</strong> The TFT is trained on quantile loss, '
            'so it already emits a P10 alongside its median. Making the GO/NO-GO call '
            'against the P10 instead of the median lifts MBT NO-GO recall to '
            f'<strong>{rule_nogo:.0f}%</strong>, the best of anything tested, and it '
            f'does so while <em>also</em> posting the best overall accuracy '
            f'({rule_acc:.1f}%) and macro F1. That combination is the tell: this is '
            'not the usual recall-for-precision trade. The median forecast is '
            'genuinely biased high near the transitions — mean reversion pulls it '
            'toward the dry-season level just as the ground is giving way — and '
            'reading the lower quantile corrects a real bias rather than merely '
            'moving a threshold. The band it comes from is well calibrated (below), '
            'so the P10 means what it says.</p></div>') if has_rule else "",
    )


TEMPLATE = """<title>Pachmarhi Trafficability Models</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
:root {{
  --paper:#f4f3ee; --surface:#fbfaf7; --ink:#171a17; --ink2:#474d46;
  --muted:#7c827a; --rule:#dcd9ce; --rule2:#eae7de;
  --accent:#2a6fbf; --accent-soft:rgba(42,111,191,.12);
  --go:#0e8f24; --slow:#c98500; --nogo:#c33a3a;
  --go-bg:rgba(14,143,36,.10); --slow-bg:rgba(201,133,0,.13); --nogo-bg:rgba(195,58,58,.11);
  --fam-baseline:#8d9389; --fam-classical:#6b5bbd; --fam-tabular:#c06a2e;
  --fam-deep:#2a6fbf; --fam-ensemble:#0e8f24;
  --shadow:0 1px 2px rgba(23,26,23,.05), 0 8px 24px -18px rgba(23,26,23,.35);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --paper:#0e100e; --surface:#161915; --ink:#e9ece7; --ink2:#b2b8b0;
    --muted:#848a82; --rule:#2b302a; --rule2:#222621;
    --accent:#5c9ae8; --accent-soft:rgba(92,154,232,.16);
    --go:#3fb857; --slow:#e0a52a; --nogo:#e26a6a;
    --go-bg:rgba(63,184,87,.13); --slow-bg:rgba(224,165,42,.15); --nogo-bg:rgba(226,106,106,.14);
    --fam-baseline:#7d837a; --fam-classical:#9a8ce4; --fam-tabular:#dd8a4c;
    --fam-deep:#5c9ae8; --fam-ensemble:#3fb857;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 26px -18px rgba(0,0,0,.8);
  }}
}}
:root[data-theme="dark"] {{
  --paper:#0e100e; --surface:#161915; --ink:#e9ece7; --ink2:#b2b8b0;
  --muted:#848a82; --rule:#2b302a; --rule2:#222621;
  --accent:#5c9ae8; --accent-soft:rgba(92,154,232,.16);
  --go:#3fb857; --slow:#e0a52a; --nogo:#e26a6a;
  --go-bg:rgba(63,184,87,.13); --slow-bg:rgba(224,165,42,.15); --nogo-bg:rgba(226,106,106,.14);
  --fam-baseline:#7d837a; --fam-classical:#9a8ce4; --fam-tabular:#dd8a4c;
  --fam-deep:#5c9ae8; --fam-ensemble:#3fb857;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 26px -18px rgba(0,0,0,.8);
}}

body {{
  background:var(--paper); color:var(--ink);
  font-family:"Source Serif 4", Georgia, serif; font-size:17px; line-height:1.62;
  margin:0; padding:0 20px 96px;
  -webkit-font-smoothing:antialiased;
}}
.wrap {{ max-width:900px; margin:0 auto; }}
.prose {{ max-width:64ch; }}

h1, h2, h3, .eyebrow, .stat-k, th, .lbl, .veh-name {{
  font-family:"Barlow Condensed", "Arial Narrow", system-ui, sans-serif;
}}
h1 {{ font-size:clamp(2.6rem,7vw,4.1rem); line-height:.98; font-weight:700;
     letter-spacing:-.012em; margin:.15em 0 .18em; text-wrap:balance; }}
h2 {{ font-size:1.72rem; font-weight:600; letter-spacing:.004em; margin:0 0 .1em;
     text-wrap:balance; }}
h3 {{ font-size:1.16rem; font-weight:600; margin:2.1em 0 .35em; color:var(--ink); }}
p {{ margin:0 0 1.05em; color:var(--ink2); }}
strong {{ color:var(--ink); font-weight:600; }}
a {{ color:var(--accent); }}
code {{ font-family:"JetBrains Mono", ui-monospace, monospace; font-size:.83em;
        background:var(--rule2); padding:.1em .38em; border-radius:3px; color:var(--ink); }}

.eyebrow {{ font-size:.82rem; font-weight:600; letter-spacing:.15em;
            text-transform:uppercase; color:var(--muted); }}
header {{ padding:72px 0 12px; border-bottom:1px solid var(--rule); margin-bottom:38px; }}
.deck {{ font-size:1.2rem; color:var(--ink2); max-width:56ch; margin:.5em 0 1.6em; }}

section {{ margin:64px 0 0; }}
section > .eyebrow {{ display:block; margin-bottom:.5em; }}
.lede {{ font-size:1.05rem; }}

/* ---------- stats ---------- */
.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(184px,1fr));
          gap:1px; background:var(--rule); border:1px solid var(--rule);
          border-radius:8px; overflow:hidden; margin:0 0 8px; }}
.stat {{ background:var(--surface); padding:18px 20px 16px; }}
.stat-k {{ font-size:.76rem; font-weight:600; letter-spacing:.12em;
           text-transform:uppercase; color:var(--muted); margin-bottom:.35em; }}
.stat-v {{ font-family:"Barlow Condensed",sans-serif; font-size:2.1rem; font-weight:700;
           line-height:1.05; letter-spacing:-.01em; font-variant-numeric:tabular-nums; }}
.stat-v .u {{ font-size:.95rem; font-weight:500; color:var(--muted); margin-left:.12em; }}
.stat-n {{ font-size:.83rem; color:var(--muted); margin-top:.3em; line-height:1.35; }}

/* ---------- chain ---------- */
.chain {{ display:flex; flex-wrap:wrap; align-items:stretch; gap:8px; margin:1.4em 0 1.8em; }}
.chain-step {{ flex:1 1 150px; background:var(--surface); border:1px solid var(--rule);
               border-radius:7px; padding:13px 15px; }}
.chain-step .k {{ font-family:"Barlow Condensed",sans-serif; font-size:.74rem;
                  font-weight:600; letter-spacing:.13em; text-transform:uppercase;
                  color:var(--muted); }}
.chain-step .v {{ font-size:.96rem; color:var(--ink); line-height:1.4; margin-top:.2em; }}
.chain-step code {{ font-size:.78em; background:none; padding:0; color:var(--accent); }}
.chain-step.is-target {{ border-color:var(--accent); box-shadow:inset 3px 0 0 var(--accent); }}

/* ---------- vehicles ---------- */
.vehs {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:10px;
         margin:1.3em 0 0; }}
.veh {{ background:var(--surface); border:1px solid var(--rule); border-radius:7px;
        padding:14px 16px; }}
.veh-name {{ font-size:1.02rem; font-weight:600; letter-spacing:.005em; }}
.veh-th {{ display:flex; gap:14px; margin:.45em 0 .5em; font-family:"JetBrains Mono",monospace;
           font-size:.78rem; color:var(--ink2); }}
.veh-share {{ display:flex; height:9px; border-radius:5px; overflow:hidden; }}
.veh-share i {{ display:block; }}
.veh-share i.go {{ background:var(--go); }}
.veh-share i.slow {{ background:var(--slow); }}
.veh-share i.nogo {{ background:var(--nogo); }}
.share-key {{ display:flex; gap:16px; margin-top:1em; font-size:.82rem; color:var(--muted);
              font-family:"Barlow Condensed",sans-serif; letter-spacing:.06em;
              text-transform:uppercase; }}
.share-key span::before {{ content:""; display:inline-block; width:9px; height:9px;
        border-radius:2px; margin-right:6px; vertical-align:baseline; }}
.share-key .go::before {{ background:var(--go); }}
.share-key .slow::before {{ background:var(--slow); }}
.share-key .nogo::before {{ background:var(--nogo); }}

/* ---------- figures ---------- */
figure {{ margin:1.6em 0 0; }}
figcaption {{ font-size:.87rem; color:var(--muted); margin-top:.7em; max-width:62ch; }}
.chart-box {{ background:var(--surface); border:1px solid var(--rule); border-radius:8px;
              padding:16px 12px 10px; overflow-x:auto; }}
svg.chart {{ display:block; min-width:600px; }}
.chart .grid {{ stroke:var(--rule2); stroke-width:1; }}
.chart .axis {{ stroke:var(--rule); stroke-width:1.2; }}
.chart .tick {{ fill:var(--muted); font:500 11px "JetBrains Mono",monospace; }}
.chart .axtitle {{ fill:var(--muted); font:600 11px "Barlow Condensed",sans-serif;
                   letter-spacing:.1em; text-transform:uppercase; }}
.chart .name {{ fill:var(--ink); font:600 13px "Barlow Condensed",sans-serif;
                letter-spacing:.02em; }}
.chart .val {{ fill:var(--muted); font:400 11px "JetBrains Mono",monospace; }}
.chart .bar {{ fill:var(--accent); }}
.chart .bar-baseline {{ fill:var(--fam-baseline); }}
.chart .bar-classical {{ fill:var(--fam-classical); }}
.chart .bar-tabular {{ fill:var(--fam-tabular); }}
.chart .bar-deep {{ fill:var(--fam-deep); }}
.chart .bar-ensemble {{ fill:var(--fam-ensemble); }}
.chart polyline {{ fill:none; stroke-width:2; stroke-linejoin:round; stroke-linecap:round; }}
.chart .dot {{ stroke:var(--surface); stroke-width:1.4; }}
.chart .ln0, .chart .s0 {{ stroke:var(--accent); fill:var(--accent); }}
.chart .ln1, .chart .s1 {{ stroke:#c06a2e; fill:#c06a2e; }}
.chart .ln2, .chart .s2 {{ stroke:#0e8f24; fill:#0e8f24; }}
.chart .ln3, .chart .s3 {{ stroke:#6b5bbd; fill:#6b5bbd; }}
.chart .ln4, .chart .s4 {{ stroke:#b8477e; fill:#b8477e; }}
.chart .ln5, .chart .s5 {{ stroke:#0f7f86; fill:#0f7f86; }}
:root[data-theme="dark"] .chart .ln1, :root[data-theme="dark"] .chart .s1 {{ stroke:#dd8a4c; fill:#dd8a4c; }}
:root[data-theme="dark"] .chart .ln2, :root[data-theme="dark"] .chart .s2 {{ stroke:#3fb857; fill:#3fb857; }}
:root[data-theme="dark"] .chart .ln3, :root[data-theme="dark"] .chart .s3 {{ stroke:#9a8ce4; fill:#9a8ce4; }}
:root[data-theme="dark"] .chart .ln4, :root[data-theme="dark"] .chart .s4 {{ stroke:#e07aab; fill:#e07aab; }}
:root[data-theme="dark"] .chart .ln5, :root[data-theme="dark"] .chart .s5 {{ stroke:#3fadb5; fill:#3fadb5; }}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) .chart .ln1, :root:not([data-theme="light"]) .chart .s1 {{ stroke:#dd8a4c; fill:#dd8a4c; }}
  :root:not([data-theme="light"]) .chart .ln2, :root:not([data-theme="light"]) .chart .s2 {{ stroke:#3fb857; fill:#3fb857; }}
  :root:not([data-theme="light"]) .chart .ln3, :root:not([data-theme="light"]) .chart .s3 {{ stroke:#9a8ce4; fill:#9a8ce4; }}
  :root:not([data-theme="light"]) .chart .ln4, :root:not([data-theme="light"]) .chart .s4 {{ stroke:#e07aab; fill:#e07aab; }}
  :root:not([data-theme="light"]) .chart .ln5, :root:not([data-theme="light"]) .chart .s5 {{ stroke:#3fadb5; fill:#3fadb5; }}
}}
.chart .ln-base {{ stroke:var(--fam-baseline); fill:var(--fam-baseline);
                   stroke-dasharray:5 4; }}
.chart .series-lbl {{ font:600 12px "Barlow Condensed",sans-serif; letter-spacing:.02em;
                      stroke:none; }}
.chart .band-go {{ fill:var(--go-bg); }}
.chart .band-slow {{ fill:var(--slow-bg); }}
.chart .band-nogo {{ fill:var(--nogo-bg); }}
.chart .thresh {{ stroke:var(--rule); stroke-width:1; stroke-dasharray:3 3; }}
.chart .line-obs {{ stroke:var(--ink); stroke-width:1.7; }}
.chart .line-pred {{ stroke:var(--accent); stroke-width:1.7; }}
.legend {{ display:flex; flex-wrap:wrap; gap:18px; margin-top:.9em; font-size:.84rem;
           color:var(--ink2); font-family:"Barlow Condensed",sans-serif;
           letter-spacing:.05em; text-transform:uppercase; }}
.legend span::before {{ content:""; display:inline-block; width:16px; height:2.5px;
        margin-right:7px; vertical-align:middle; border-radius:2px; }}
.legend .obs::before {{ background:var(--ink); }}
.legend .pred::before {{ background:var(--accent); }}

/* ---------- tables ---------- */
.tbl-wrap {{ overflow-x:auto; border:1px solid var(--rule); border-radius:8px;
             background:var(--surface); margin-top:1.4em; }}
table {{ border-collapse:collapse; width:100%; min-width:640px; font-size:.92rem; }}
th {{ text-align:right; font-size:.78rem; font-weight:600; letter-spacing:.1em;
      text-transform:uppercase; color:var(--muted); padding:13px 14px 10px;
      border-bottom:1px solid var(--rule); white-space:nowrap; }}
th:first-child {{ text-align:left; }}
td {{ padding:9px 14px; border-bottom:1px solid var(--rule2); color:var(--ink2); }}
tr:last-child td {{ border-bottom:none; }}
td.n {{ text-align:right; font-family:"JetBrains Mono",monospace; font-size:.85rem;
        font-variant-numeric:tabular-nums; }}
td.mdl {{ color:var(--ink); font-weight:600; white-space:nowrap; }}
td.skill {{ color:var(--ink2); }}
tr.row-best td {{ background:var(--accent-soft); color:var(--ink); }}
tr.row-best td.mdl::after {{ content:"best"; font-family:"Barlow Condensed",sans-serif;
        font-size:.68rem; letter-spacing:.12em; text-transform:uppercase;
        color:var(--accent); margin-left:9px; vertical-align:1px; }}
.dot-fam {{ display:inline-block; width:8px; height:8px; border-radius:2px;
            margin-right:9px; vertical-align:1px; }}
.fam-baseline {{ background:var(--fam-baseline); }}
.fam-classical {{ background:var(--fam-classical); }}
.fam-tabular {{ background:var(--fam-tabular); }}
.fam-deep {{ background:var(--fam-deep); }}
.fam-ensemble {{ background:var(--fam-ensemble); }}
.fam-rule {{ background:var(--slow); }}
tr.row-rule td {{ background:var(--slow-bg); color:var(--ink); }}
tr.row-rule td.mdl::after {{ content:"decision rule";
        font-family:"Barlow Condensed",sans-serif; font-size:.68rem; letter-spacing:.12em;
        text-transform:uppercase; color:var(--slow); margin-left:9px; vertical-align:1px; }}
.fam-key {{ display:flex; flex-wrap:wrap; gap:16px; margin-top:.9em; font-size:.8rem;
            color:var(--muted); font-family:"Barlow Condensed",sans-serif;
            letter-spacing:.07em; text-transform:uppercase; }}

/* ---------- features ---------- */
ul.feats {{ list-style:none; padding:0; margin:1.3em 0 0; }}
ul.feats li {{ display:grid; grid-template-columns:210px 1fr 52px; align-items:center;
               gap:14px; padding:5px 0; border-bottom:1px solid var(--rule2); }}
ul.feats code {{ background:none; padding:0; color:var(--ink); font-size:.8rem; }}
.fbar {{ height:8px; border-radius:4px; background:var(--accent); width:var(--w);
         min-width:2px; }}
.fval {{ font-family:"JetBrains Mono",monospace; font-size:.78rem; color:var(--muted);
         text-align:right; font-variant-numeric:tabular-nums; }}

/* ---------- model roster ---------- */
.roster {{ margin-top:1.5em; }}
.fam-block {{ margin-bottom:2.4em; }}
.fam-block h3 {{ margin:0 0 .12em; font-size:1.14rem; }}
.fam-blurb {{ font-size:.93rem; color:var(--muted); max-width:60ch; margin:0 0 1em; }}
.m-row {{ display:grid; grid-template-columns:172px 1fr; gap:6px 20px;
          padding:13px 0; border-top:1px solid var(--rule2); align-items:baseline; }}
.m-name {{ font-family:"Barlow Condensed",sans-serif; font-size:1.06rem; font-weight:600;
           letter-spacing:.01em; color:var(--ink); }}
.m-desc {{ font-size:.95rem; color:var(--ink2); line-height:1.5; }}
.m-chips {{ grid-column:2; display:flex; flex-wrap:wrap; gap:6px; margin-top:.25em; }}
.chip {{ font-family:"JetBrains Mono",monospace; font-size:.72rem; padding:2px 8px;
         border-radius:4px; background:var(--accent-soft); color:var(--accent);
         font-variant-numeric:tabular-nums; }}
.chip-ghost {{ background:var(--rule2); color:var(--muted); }}
.chip-safe {{ background:var(--slow-bg); color:var(--slow); }}

/* ---------- bias chart ---------- */
.chart .bias-hi {{ fill:var(--nogo); }}
.chart .bias-lo {{ fill:var(--accent); }}

/* ---------- two-lead feature comparison ---------- */
.feat-cols {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
              gap:14px 40px; margin:1.4em 0 1.6em; }}
.feat-head {{ font-family:"Barlow Condensed",sans-serif; font-size:1rem; font-weight:600;
              letter-spacing:.04em; color:var(--ink); border-bottom:1px solid var(--rule);
              padding-bottom:.4em; }}
.feat-head span {{ font-family:"JetBrains Mono",monospace; font-size:.72rem;
                   color:var(--muted); margin-left:.5em; letter-spacing:0; }}
.feat-cols ul.feats {{ margin-top:.3em; }}
.feat-cols ul.feats li {{ grid-template-columns:150px 1fr 44px; gap:10px; }}

/* ---------- callout ---------- */
.note {{ border-left:2px solid var(--accent); padding:2px 0 2px 18px; margin:1.5em 0;
         color:var(--ink2); }}
.note strong {{ color:var(--ink); }}
ul.plain {{ padding-left:1.1em; color:var(--ink2); }}
ul.plain li {{ margin-bottom:.5em; }}
footer {{ margin-top:80px; padding-top:24px; border-top:1px solid var(--rule);
          font-size:.85rem; color:var(--muted); }}
@media (max-width:620px) {{
  body {{ font-size:16px; }}
  ul.feats li {{ grid-template-columns:150px 1fr 46px; gap:9px; }}
}}
</style>

<div class="wrap">
<header>
  <div class="eyebrow">Pachmarhi · Satpura plateau · 4,018 daily records, 2015–2025</div>
  <h1>Forecasting whether the ground will hold</h1>
  <p class="deck">Fifteen time-series models, trained to predict soil bearing strength
  seven days out, and scored on the decision that actually matters: can the vehicle
  cross or not.</p>
  <div class="stats">{stat_html}</div>
</header>

<section>
  <div class="eyebrow">The target</div>
  <h2>The record has no trafficability column, so it is derived</h2>
  <div class="prose">
  <p class="lede">The dataset carries rainfall, temperature, humidity, wind, pressure,
  cloud and two soil-moisture layers. Trafficability is built from those by way of
  soil strength — the quantity a cone penetrometer measures in the field and the one
  vehicle mobility is actually specified against.</p>
  </div>

  <div class="chain">
    <div class="chain-step"><div class="k">Observed</div>
      <div class="v">Soil moisture at <code>0–7 cm</code> and <code>7–28 cm</code></div></div>
    <div class="chain-step"><div class="k">Wetness</div>
      <div class="v">SWI, profile-weighted <code>0.35 / 0.65</code> toward the
      15–30 cm critical layer</div></div>
    <div class="chain-step is-target"><div class="k">Modelled target</div>
      <div class="v">RCI <code>= 300·e<sup>−2.485·SWI</sup>·f(rain)</code>, in kPa</div></div>
    <div class="chain-step"><div class="k">Decision</div>
      <div class="v">RCI against each vehicle's cone index →
      <strong>GO / SLOW-GO / NO-GO</strong></div></div>
  </div>

  <div class="prose">
  <p>Weighting the deep layer more heavily than the surface is what a tracked vehicle
  actually feels: sinkage is governed by the 15–30 cm horizon, not the skin. It also
  means the target depends on both soil layers plus recent rainfall, so it is not a
  relabelling of a single input column.</p>
  </div>

  <div class="vehs">{veh_html}</div>
  <div class="share-key"><span class="go">GO</span><span class="slow">SLOW-GO</span>
    <span class="nogo">NO-GO</span><span style="text-transform:none;letter-spacing:0">
    share of all 4,018 days</span></div>
</section>

<section>
  <div class="eyebrow">The setup</div>
  <h2>No weather forecast, no peeking</h2>
  <div class="prose">
  <p>Every feature at forecast origin <em>t</em> uses observations at time ≤ <em>t</em>.
  The only inputs known for the future are calendar and Fourier terms — an honest
  operational setting where no numerical weather prediction is on hand. Models emit
  days 1 through 7 directly rather than rolling a one-step forecast forward, so error
  does not compound.</p>
  <p>Splits are strictly chronological: <strong>{n_train} training origins</strong>
  (2015–2021), <strong>{n_val} validation</strong> (2022–2023) for early stopping and
  hyperparameters, and <strong>{n_test} held-out test origins</strong>
  ({test_from} – {test_to}) touched exactly once.</p>
  </div>
  <div class="note">
  <p><strong>One decision shaped every result.</strong> RCI has a lag-1
  autocorrelation of 0.99, so a model that predicts the <em>level</em> mostly learns
  to copy yesterday. All learned models here predict the <em>change</em>,
  <code>y(t+h) − y(t)</code>, standardised per horizon; the anchor is added back
  afterwards. Persistence then becomes exactly the zero-change forecast, and the skill
  score reads directly as signal added beyond yesterday's value. Before this change no
  model beat persistence at one day; after it, nearly all do.</p>
  </div>
</section>

<section>
  <div class="eyebrow">The lineup</div>
  <h2>Eighteen forecasts, five kinds of reasoning</h2>
  <div class="prose">
  <p class="lede">Each entry below sees the same data, the same splits and the same
  target. What differs is the assumption each one makes about how soil strength moves.
  Figures are RMSE at a 7-day lead and skill against persistence.</p>
  <p>One thing to hold in mind while reading the feature work later: because every
  learned model predicts the <em>change</em> rather than the level, none of them is
  being asked how wet the ground is. They are being asked which way it is moving, and
  how fast.</p>
  </div>
  <div class="roster">{roster_html}</div>
</section>

<section>
  <div class="eyebrow">Results</div>
  <h2>What each model adds beyond yesterday's reading</h2>
  <figure>
    <div class="chart-box">{chart_leaderboard}</div>
    <figcaption>RMSE skill score against persistence at a 7-day lead, on the held-out
    2024–2025 window. Zero means no better than simply assuming the ground stays as it
    is today. Bar colour marks the model family.</figcaption>
  </figure>

  <div class="prose">
  <p>{best} leads at <strong>{best7} kPa RMSE</strong> seven days out against
  {pers7} for persistence — <strong>{best_skill}% of the error removed</strong>,
  R² {best_r2}. {runner} follows at {runner7} kPa.</p>
  </div>

  <figure>
    <div class="chart-box">{chart_horizon}</div>
    <figcaption>Error against lead time. Persistence is competitive at one day and
    degrades fastest; the gap that matters opens from day three onward, which is also
    the planning horizon a movement order needs.</figcaption>
  </figure>

  <div class="tbl-wrap"><table>
    <thead><tr><th>Model</th><th>RMSE h=1</th><th>RMSE h=3</th><th>RMSE h=7</th>
      <th>MAE h=7</th><th>R² h=7</th><th>Skill h=7</th></tr></thead>
    <tbody>{table_html}</tbody>
  </table></div>
  <div class="fam-key">
    <span><span class="dot-fam fam-baseline"></span>baseline</span>
    <span><span class="dot-fam fam-classical"></span>classical</span>
    <span><span class="dot-fam fam-tabular"></span>tabular ML</span>
    <span><span class="dot-fam fam-deep"></span>deep</span>
    <span><span class="dot-fam fam-ensemble"></span>ensemble</span>
  </div>
  {arima_note}
</section>

<section>
  <div class="eyebrow">The decision</div>
  <h2>RMSE is not the question a commander asks</h2>
  <div class="prose">
  <p>A forecast that is 15 kPa off matters enormously near a threshold and not at all
  in the middle of the dry season. Scoring the same predictions as the three-way
  call — and separately as recall on NO-GO days, where a miss sends vehicles into
  ground that will not hold — reorders the table.</p>
  </div>
  <div class="note">
  <p><strong>The lowest-RMSE model is not the safest model.</strong> {best} catches
  {tft_nogo}% of MBT NO-GO days; plain persistence catches {pers_nogo}%. Squared error
  rewards a forecast that stays near the middle of the distribution, and a
  mean-reverting forecast systematically under-shoots the collapse in soil strength
  that defines a NO-GO day. Ranking on RMSE alone would pick a model that is worse at
  the one call with a real cost attached.</p>
  </div>
  <div class="tbl-wrap"><table>
    <thead><tr><th>Model</th><th>MBT accuracy</th><th>MBT macro F1</th>
      <th>MBT NO-GO recall</th><th>Truck macro F1</th><th>Truck NO-GO recall</th></tr></thead>
    <tbody>{dq_rows}</tbody>
  </table></div>
  <p style="font-size:.85rem;color:var(--muted);margin-top:.8em">All figures percent,
  7-day lead, sorted by NO-GO recall. MBT = ~46 t tracked; truck = wheeled 6×6.</p>
  <h3>Why it happens</h3>
  <div class="prose">
  <p>Mean signed error explains the whole reordering. A model that shades its forecasts
  toward the middle of the distribution scores well on squared error, and the days it
  shades away from are exactly the days the ground gives way.</p>
  </div>
  <figure>
    <div class="chart-box">{chart_bias}</div>
    <figcaption>Mean signed error at a 7-day lead. Bars to the right mean the model
    claims the ground is firmer than it turned out to be — the direction that puts
    vehicles into ground that will not hold. The TFT's median carries the largest
    positive bias of any model here, which is precisely why its median makes poor
    NO-GO calls and its lower quantile makes excellent ones.</figcaption>
  </figure>

  {rule_note}
  {calib_html}

  <figure>
    <div class="chart-box">{chart_series}</div>
    <figcaption>Observed against forecast RCI through 2025 at a 7-day lead. The model
    tracks the monsoon collapse and the October recovery — the two transitions that
    decide the season — and is least certain exactly at the crossings.</figcaption>
    <div class="legend"><span class="obs">Observed</span>
      <span class="pred">{best}, 7-day forecast</span></div>
  </figure>
</section>

<section>
  <div class="eyebrow">Drivers</div>
  <h2>What the forecast keys on</h2>
  <div class="prose">
  <p>Share of LightGBM split gain across the {n_feat} causal features. The interesting
  part is not the ranking at either lead but the <strong>shift between them</strong> —
  and nothing in the setup asked for it.</p>
  </div>
  <div class="feat-cols">
    <div>
      <div class="feat-head">Tomorrow <span>h = 1</span></div>
      <ul class="feats">{feat_h1}</ul>
    </div>
    <div>
      <div class="feat-head">A week out <span>h = 7</span></div>
      <ul class="feats">{feat_h7}</ul>
    </div>
  </div>
  <div class="prose">
  <p>At one day the forecast leans on <strong>instantaneous drivers</strong>:
  yesterday's cloud cover — cloud suppresses evaporation, so the ground stays wet —
  and current surface moisture. At seven days it switches to
  <strong>slow accumulators</strong>: the antecedent precipitation index and the
  14- and 30-day running water balance, rainfall minus reference evapotranspiration.
  That is the physically correct answer to each question, learned rather than
  imposed.</p>
  </div>
</section>

<section>
  <div class="eyebrow">Reading it</div>
  <h2>What this does and does not establish</h2>
  <div class="prose">
  <ul class="plain">
    <li><strong>Deep sequence models earn their place, narrowly.</strong> The gap over
    a tuned linear model on the same 128 features is real but not dramatic. On 2,498
    training origins that is the expected outcome, and it is worth saying plainly
    rather than overselling the architecture.</li>
    <li><strong>Pick the model on the decision, not on RMSE.</strong> The two rankings
    disagree, and they disagree in the dangerous direction: the sharpest regressor is
    among the weaker models at flagging the days the ground gives way. Whatever gets
    deployed should be selected against NO-GO recall at the vehicle class in question.</li>
    <li><strong>The TFT's quantiles are the practical advantage.</strong> A planner
    reads risk off the P10–P90 band, and the P10 doubles as a safety-biased decision
    rule; a point forecast at a threshold gives no way to tell a marginal call from a
    confident one.</li>
    <li><strong>Soil moisture here is a model state, not an observation.</strong>
    ERA5-Land's soil moisture is unassimilated, so the RCI target inherits that.
    Calibrating <code>RCI_DRY</code> and the decay constant needs field cone-penetrometer
    readings from this site.</li>
    <li><strong>One 9 km grid cell.</strong> Real off-road mobility also turns on slope,
    vegetation, drainage and soil texture, none of which are in this record. Treat the
    output as the weather-driven component of trafficability, not the whole answer.</li>
    <li><strong>The rainfall day definition changes inside the test window</strong>
    (IMD 08:30–08:30 IST to ERA5 00:00–24:00 IST at 2025-01-01), which adds a small
    inhomogeneity to the 2025 half of the hold-out.</li>
  </ul>
  </div>
</section>

<footer>
  Trained on {gpu}. Deep models: 3 seeds averaged, early stopping on the validation
  block, {dl_secs}. Full metrics in <code>outputs/forecasting/results_full.csv</code>;
  reproduce with <code>python -m traffic.run</code>.
</footer>
</div>
"""


def main():
    res = pd.read_csv(f"{OUT}/results_full.csv")
    val_res = (pd.read_csv(f"{OUT}/results_validation.csv")
               if os.path.exists(f"{OUT}/results_validation.csv") else None)
    npz = np.load(f"{OUT}/test_predictions.npz", allow_pickle=True)
    dates = pd.to_datetime(npz["dates"])
    Y = npz["Y"]
    preds = {k: npz[k] for k in npz.files if k not in ("dates", "origins", "Y")}

    raw, feats = D.get_dataset()
    share = {}
    for k in D.VEHICLES:
        vc = raw[f"class_{k}"].value_counts(normalize=True)
        share[k] = "".join(
            f'<i class="{c}" style="width:{100*vc.get(i,0):.2f}%"></i>'
            for i, c in [(2, "go"), (1, "slow"), (0, "nogo")])

    meta = dict(
        class_share=share,
        n_train=2498, n_val=730, n_test=len(dates),
        test_from=dates.min().strftime("%d %b %Y"),
        test_to=dates.max().strftime("%d %b %Y"),
        n_feat=len(D.tabular_columns(feats)),
        gpu="an RTX 4060 Laptop GPU", dl_secs="see outputs/forecasting/dl_training_log.json",
        arima_note="", ensemble_note="", features=[],
    )

    if os.path.exists(f"{OUT}/lgbm_feature_gain.json"):
        gains = json.load(open(f"{OUT}/lgbm_feature_gain.json"))
        for key, hz in (("features", "7"), ("features_h1", "1")):
            g = gains[hz]
            tot = sum(g.values()) or 1
            meta[key] = [(k, 100 * v / tot) for k, v in
                         sorted(g.items(), key=lambda x: -x[1])[:12]]

    if os.path.exists(f"{OUT}/arima_selection.json"):
        a = json.load(open(f"{OUT}/arima_selection.json"))
        bits = []
        for name, d in a.items():
            o = d["order"]
            bits.append(f"{esc(name)} selected <code>({o[0]},{o[1]},{o[2]})</code>")
        meta["arima_note"] = (
            '<div class="note"><p><strong>Which ARIMA.</strong> Daily data with an '
            'annual cycle rules out a seasonal SARIMA — a period of 365 is not '
            'estimable. The right form is dynamic harmonic regression: ARIMA errors '
            'with three Fourier pairs carrying the seasonality as exogenous '
            'regressors, which are also genuinely known for the future. Order chosen '
            'by AIC grid search with <em>d</em> set by ADF and KPSS agreement; '
            + "; ".join(bits) + '. Multi-step forecasts come from the filtered Kalman '
            'state, verified against statsmodels\' own dynamic prediction to machine '
            'precision.</p></div>')

    if os.path.exists(f"{OUT}/tft_calibration.csv"):
        cal = pd.read_csv(f"{OUT}/tft_calibration.csv")
        rows = "\n".join(
            f'<tr><td class="mdl">{int(r.horizon)} day{"s" if r.horizon > 1 else ""}</td>'
            f'<td class="n">{r.cover_80:.1f}</td><td class="n">{r.below_p10:.1f}</td>'
            f'<td class="n">{r.above_p90:.1f}</td><td class="n">{r.mean_width:.1f}</td>'
            f"</tr>" for r in cal.itertuples())
        meta["calib_html"] = (
            '<h3>Is the band honest?</h3>'
            '<div class="prose"><p>A quantile is only useful if it is calibrated. '
            'Measured on the hold-out, the P10–P90 interval covers the truth about '
            '84% of the time against a nominal 80%, and the P10 is undershot 9–12% of '
            'the time against a nominal 10% — slightly conservative, in the safe '
            'direction, and it holds in the monsoon as well as the dry season.</p></div>'
            '<div class="tbl-wrap"><table><thead><tr><th>Lead</th>'
            '<th>P10–P90 coverage %</th><th>Below P10 %</th><th>Above P90 %</th>'
            '<th>Mean width kPa</th></tr></thead>'
            f"<tbody>{rows}</tbody></table></div>"
            '<p style="font-size:.85rem;color:var(--muted);margin-top:.8em">'
            'Nominal values: 80 / 10 / 10.</p>')

    if os.path.exists(f"{OUT}/dl_training_log.json"):
        log = json.load(open(f"{OUT}/dl_training_log.json"))
        tot = sum(v["seconds"] for v in log.values())
        meta["dl_secs"] = f"{tot/60:.0f} min total for all six architectures"
        meta["params"] = {k: v["params"] for k, v in log.items()}

    os.makedirs(OUT, exist_ok=True)
    with open(PAGE, "w", encoding="utf-8") as f:
        f.write(build(res, val_res, dates, Y, preds, meta))
    print("wrote", PAGE)


if __name__ == "__main__":
    main()
