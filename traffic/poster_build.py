"""
Assemble the A0 poster: poster/poster.html (figures inlined as SVG, fonts local),
then poster/poster.pdf and a PNG preview through headless Edge.

Every number on the poster is read from outputs/ here; nothing is typed in by hand.

Usage:  python -m traffic.poster_build
"""

from __future__ import annotations

import json
import re
import subprocess

import numpy as np
import pandas as pd

from traffic import paths
from traffic import vehicles as V

POSTER = paths.ROOT / "poster"
FIG = POSTER / "figures"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


def svg(name):
    s = (FIG / f"{name}.svg").read_text(encoding="utf-8")
    s = re.sub(r"<\?xml.*?\?>|<!DOCTYPE.*?>|<metadata>.*?</metadata>", "", s, flags=re.S)
    # matplotlib writes width/height in pt; let CSS size it to the column instead
    s = re.sub(r'(<svg[^>]*?)\swidth="[^"]+"\sheight="[^"]+"', r"\1", s, count=1)
    return s


def numbers():
    n = {}
    from traffic import data as D
    fleet = V.add_vci(V.load_vehicles())
    rci = D.add_soil_strength(D.load_raw()).rci.values
    n["n_vehicles"] = len(fleet)
    n["n_days"] = len(rci)
    from traffic import terrain as T
    els = json.load(open(T.ROADS_JSON))["elements"]
    n["n_roads"] = sum(e.get("tags", {}).get("highway") in T.VEHICLE_ROADS for e in els)
    n["n_tracked"] = int((fleet.type == "tracked").sum())
    n["n_wheeled"] = int((fleet.type == "wheeled").sum())
    n["n_half"] = int((fleet.type == "halftrack").sum())

    edge = json.load(open(paths.FORECASTER_OUT / "edge.json"))
    n["params_k"] = f"{edge['params'] / 1000:.0f}k"
    n["size_mb"] = f"{edge['size_mb_float32']:.1f}"
    n["cpu_ms"] = f"{edge['cpu_forecast_ms']:.0f}"
    n["msg_bytes"] = edge["forecast_message_bytes"]

    reg = pd.read_csv(paths.FORECASTER_OUT / "regression_test.csv")
    r7 = reg[(reg.model == "GRU+Ridge") & (reg.h == 7)].iloc[0]
    n["skill7"] = f"{100 * r7.skill_vs_persistence:.0f}"
    n["rmse7"] = f"{r7.rmse:.1f}"
    n["rmse7_p"] = f"{reg[(reg.model == 'Persistence') & (reg.h == 7)].rmse.iloc[0]:.1f}"

    dec = pd.read_csv(paths.FORECASTER_OUT / "decisions_by_band.csv")
    d7 = dec[dec.h == 7].pivot(index="band", columns="rule", values="false_alarms_per_year")
    red = 1 - d7["GRU+Ridge"] / d7["Persistence"]
    n["fa_red_lo"], n["fa_red_hi"] = f"{100 * red.min():.0f}", f"{100 * red.max():.0f}"
    m = dec[(dec.rule == "GRU+Ridge")]
    n["margin_gap_max"] = f"{(m.margin_tuned_2024 - m.margin_tuned_2025).abs().max():.1f}"

    bt = pd.read_csv(paths.OUTPUTS / "planner" / "backtest.csv")
    s = bt.groupby(["planner", "h"])[["trips", "possible", "stuck", "held", "done", "extra_time_sum"]].sum()
    s["stuck_pct"] = 100 * s.stuck / s.trips
    s["held_pct"] = 100 * s.held / s.possible
    s["extra_pct"] = 100 * s.extra_time_sum / s.done
    n["bt"] = s
    n["bt_trips"] = int(bt[bt.planner == "Today (Liu et al.)"].trips.sum())
    n["bt_days"] = bt.date.nunique()
    n["extra3_ours"] = f"{s.loc[('Forecast + margin (ours)', 3), 'extra_pct']:.2f}"
    n["static_stuck"] = f"{s.loc[('Static (terrain only)', 3), 'stuck_pct']:.0f}"

    # caution trade-off: forecast vs today's value at matched stuck rate
    tr = pd.read_csv(paths.OUTPUTS / "planner" / "tradeoff.csv")
    t = tr.groupby(["h", "rule", "scale"])[["trips", "possible", "stuck", "held"]].sum()
    t["stuck_pct"] = 100 * t.stuck / t.trips
    t["held_pct"] = 100 * t.held / t.possible
    n["to"] = t
    for h in (3, 7):
        p1 = t.loc[(h, "Persistence", 1.0)]
        g = t.loc[(h, "GRU+Ridge")].sort_values("stuck_pct")
        held_at = float(np.interp(p1.stuck_pct, g.stuck_pct, g.held_pct))
        n[f"held_cut{h}"] = f"{100 * (1 - held_at / p1.held_pct):.0f}"
        p0, g1 = t.loc[(h, "Persistence", 0.0)], t.loc[(h, "GRU+Ridge", 1.0)]
        n[f"stuck_cut{h}"] = f"{100 * (1 - g1.stuck_pct / p0.stuck_pct):.0f}"
        n[f"stuck_today{h}"] = f"{p0.stuck_pct:.1f}"
        n[f"stuck_ours{h}"] = f"{g1.stuck_pct:.1f}"
        n[f"held_today{h}"] = f"{p0.held_pct:.1f}"
        n[f"held_ours{h}"] = f"{g1.held_pct:.1f}"

    cv = pd.read_csv(paths.OUTPUTS / "planner" / "convoy.csv")
    c = cv.groupby(["band", "mode"])[["through", "bytes"]].mean()
    n["cv_d_follow"] = f"{c.loc[('D', 'follow'), 'through']:.1f}"
    n["cv_d_wear"] = f"{c.loc[('D', 'wear_aware'), 'through']:.1f}"
    n["cv_e_follow"] = f"{c.loc[('E', 'follow'), 'through']:.1f}"
    n["cv_e_wear"] = f"{c.loc[('E', 'wear_aware'), 'through']:.1f}"
    n["cv_bytes"] = f"{cv[cv['mode'] == 'wear_aware'].bytes.mean() / 1000:.1f}"
    n["cv_runs"] = len(cv) // 2
    n["cv_days"] = cv.groupby(["band", "date"]).ngroups
    n["band_nogo"] = {b: 100 * np.mean(rci < fleet.loc[fleet.band == b, "vci1"].median())
                      for b, _, _ in V.BANDS}
    n["map_caption"] = (FIG / "f4_map_caption.txt").read_text(encoding="utf-8")
    timing = json.load(open(paths.OUTPUTS / "planner" / "timing.json"))
    n["replan_ms"] = f"{timing['replan_ms']:.0f}"
    n["map_kb"] = f"{timing['map_bytes_uint8'] / 1000:.0f}"
    n["maps_mb"] = f"{timing['map_bytes_uint8'] * 5 * 7 / 1e6:.1f}"
    n["map_ratio"] = int(round(timing["map_bytes_uint8"] / edge["forecast_message_bytes"], -2))
    n["cv_bytes_vehicle"] = f"{cv[cv['mode'] == 'wear_aware'].bytes.mean() / 1000 / 10:.1f}"
    return n


def render(html_path):
    pdf = POSTER / "poster.pdf"
    png = POSTER / "poster_preview.png"
    url = html_path.as_uri()
    common = [EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars",
              "--virtual-time-budget=15000", "--run-all-compositor-stages-before-draw"]
    subprocess.run(common + ["--no-pdf-header-footer", f"--print-to-pdf={pdf}", url],
                   check=True, capture_output=True, timeout=180)
    subprocess.run(common + ["--window-size=3179,4494", f"--screenshot={png}", url],
                   check=True, capture_output=True, timeout=180)
    return pdf, png


if __name__ == "__main__":
    from traffic.poster_page import page          # the page template and copy
    html = page(numbers(), svg)
    out = POSTER / "poster.html"
    out.write_text(html, encoding="utf-8")
    pdf, png = render(out)
    print("->", out.relative_to(paths.ROOT), pdf.relative_to(paths.ROOT), png.relative_to(paths.ROOT))
