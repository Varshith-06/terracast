"""
The poster page: layout, type and copy. Numbers come in through `n` from
traffic.poster_build; figures through `svg(name)`.
"""

from __future__ import annotations

AUTHORS = "[Author One]<sup>1</sup>, [Author Two]<sup>1</sup>, [Author Three]<sup>2</sup>"
AFFIL = "<sup>1</sup>[Department, Institution, City] &nbsp;&nbsp; <sup>2</sup>[Institution, City]"
CONTACT = "[corresponding.author@institution.edu]"

CSS = """
@import url("fonts/fonts.css");
@page { size: 841mm 1189mm; margin: 0; }
:root {
  --paper: #fcfbf7; --ink: #16191d; --ink2: #4b4f55; --muted: #8a8d91;
  --rule: #d8d6cd; --ours: #2a78d6; --foil: #eb6834; --nogo: #d03b3b; --band: #18201b;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { width: 841mm; height: 1189mm; background: var(--paper); }
body { font-family: "IBM Plex Sans", sans-serif; color: var(--ink); font-size: 29pt;
       line-height: 1.32; -webkit-print-color-adjust: exact; print-color-adjust: exact;
       position: relative; overflow: hidden; }

header { background: var(--band); color: #f3f1ea; padding: 18mm 30mm 14mm; position: relative;
         display: grid; grid-template-columns: 82mm 1fr; column-gap: 16mm; align-items: start; }
.logo { width: 82mm; height: 82mm; border: 1.5pt dashed #6f7a6a; border-radius: 2mm; display: flex;
        align-items: center; justify-content: center; text-align: center; font-size: 17pt;
        color: #8f9a89; line-height: 1.25; margin-top: 3mm; }
header .venue { position: absolute; right: 30mm; top: 25mm; text-align: right;
       font-size: 20pt; line-height: 1.35; color: #b9c0b4; }
header .venue b { color: #f3f1ea; font-weight: 600; letter-spacing: 0.04em; }
h1 { font-family: "IBM Plex Sans Condensed", sans-serif; font-weight: 700; font-size: 86pt;
     line-height: 1.0; letter-spacing: -0.01em; max-width: 560mm; }
.sub { font-size: 30pt; line-height: 1.25; margin-top: 7mm; color: #d5d9cf; max-width: 620mm; }
.authors { font-size: 27pt; margin-top: 10mm; font-weight: 500; }
.affil { font-size: 20pt; color: #b9c0b4; margin-top: 2mm; }

.claim { padding: 10mm 30mm 9mm; border-bottom: 1.2pt solid var(--rule);
         display: grid; grid-template-columns: 1fr auto; gap: 20mm; align-items: center; }
.claim p { font-size: 36pt; line-height: 1.24; }
.claim b { font-weight: 700; }
.claim .ours { color: var(--ours); font-weight: 600; } .claim .foil { color: var(--foil); font-weight: 600; }
.tiles { display: grid; grid-template-columns: repeat(2, auto); gap: 5mm 13mm; }
.tile .v { font-size: 44pt; font-weight: 600; line-height: 1.0; }
.tile .v small { font-size: 24pt; font-weight: 500; margin-left: 1mm; }
.tile .k { font-size: 16.5pt; color: var(--ink2); line-height: 1.22; margin-top: 1.5mm; }

.pipe { padding: 8mm 30mm 4mm; border-bottom: 1.2pt solid var(--rule); }
.pipe svg { width: 100%; height: auto; display: block; }

main { display: grid; grid-template-columns: repeat(3, 1fr); column-gap: 20mm;
       padding: 10mm 30mm 0; }
section { margin-bottom: 8.5mm; }
.col > section:last-child { margin-bottom: 0; }
h2 { font-size: 35pt; line-height: 1.12; font-weight: 600; margin-bottom: 4.5mm;
     padding-top: 4.5mm; border-top: 3pt solid var(--ink); }
h2.minor { font-size: 27pt; border-top-width: 1.5pt; }
h2 .no { font-family: "IBM Plex Mono", monospace; font-weight: 500; font-size: 25pt;
         color: var(--ours); margin-right: 4mm; vertical-align: 3pt; }
p + p { margin-top: 3mm; }
.fig { margin: 5mm 0 1mm; }
.fig svg { width: 100%; height: auto; display: block; }
.cap { font-size: 18.5pt; line-height: 1.3; color: var(--ink2); margin-top: 2.5mm; }
.cap b { color: var(--ink); font-weight: 600; }
em { font-style: normal; font-weight: 600; }
.k-ours { color: var(--ours); font-weight: 600; } .k-foil { color: var(--foil); font-weight: 600; }
.k-nogo { color: var(--nogo); font-weight: 600; }

.eq { margin: 5mm 0 2mm; padding: 4.5mm 6mm; background: #f1efe7; border-left: 3pt solid var(--ink);
      font-size: 22pt; line-height: 1.45; }
.eq .v { font-style: italic; }
.eq .note { color: var(--ink2); font-size: 17.5pt; line-height: 1.3; display: block; margin-top: 1.5mm; }

table { width: 100%; border-collapse: collapse; font-size: 20pt; margin-top: 4mm;
        font-variant-numeric: tabular-nums; }
th { text-align: right; font-weight: 500; color: var(--ink2); font-size: 17pt;
     border-bottom: 1.5pt solid var(--ink); padding: 0 0 2mm 4mm; line-height: 1.2; vertical-align: bottom; }
th:first-child, td:first-child { text-align: left; padding-left: 0; }
td { text-align: right; padding: 2mm 0 2mm 4mm; border-bottom: 0.8pt solid var(--rule); }
tr.us td { font-weight: 600; }
tr.us td:first-child { color: var(--ours); }
tr.foil td:first-child { color: var(--foil); }
table.data { margin-top: 1mm; font-size: 19pt; }
table.data td { text-align: left; vertical-align: top; line-height: 1.3; padding: 2.2mm 0 2.2mm 5mm; }
table.data td:first-child { font-weight: 600; width: 34mm; padding-left: 0; }

footer { position: absolute; left: 30mm; right: 30mm; bottom: 46mm; display: grid;
         grid-template-columns: 1.55fr 1fr; gap: 18mm; font-size: 16pt; line-height: 1.38;
         color: var(--ink2); border-top: 1.2pt solid var(--rule); padding-top: 5mm; }
footer b { color: var(--ink); font-weight: 600; }
footer ol { padding-left: 7mm; font-size: 14.5pt; line-height: 1.35; }
.band { position: absolute; left: 0; right: 0; bottom: 0; height: 36mm; background: var(--band);
        color: #d5d9cf; display: grid; grid-template-columns: 1fr auto 1fr; align-items: center;
        padding: 0 30mm; font-size: 20pt; }
.band b { color: #f3f1ea; font-weight: 600; }
.band .mid { text-align: center; letter-spacing: 0.03em; }
.band .right { display: flex; justify-content: flex-end; align-items: center; gap: 6mm; }
.qr { width: 27mm; height: 27mm; border: 1.5pt dashed #6f7a6a; border-radius: 1.5mm; font-size: 12pt;
      color: #8f9a89; display: flex; align-items: center; justify-content: center; text-align: center;
      line-height: 1.15; }
"""


def pipeline_svg():
    """Six steps left to right; the three that run on the vehicle drawn in blue."""
    W, H = 2950, 300
    steps = [
        ("Weather log", "rain + 2-layer soil moisture", "one 9 km cell, daily"),
        ("Edge forecaster", "GRU 128×2 + ridge", "RCI change, 1–7 days"),
        ("Safety margin", "one per firmness band", "tuned on the other year"),
        ("Soil map", "RCI spread by terrain", "wetness index, 90 m"),
        ("Fuzzy AND", "min(slope, cover, soil)", "Liu et al. tables [1]"),
        ("Route planner", "fastest route, no NO-GO", "convoy shares its tracks"),
    ]
    bw, gap, y0, bh = 400, 110, 12, 186
    out = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
           f'font-family="IBM Plex Sans" fill="none">',
           '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
           'markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#16191d"/></marker></defs>']
    for i, (t, a, b) in enumerate(steps):
        x = i * (bw + gap)
        on_vehicle = i in (1, 2, 3, 4, 5)
        stroke = "#2a78d6" if i in (1, 2) else "#16191d"
        out.append(f'<rect x="{x}" y="{y0}" width="{bw}" height="{bh}" rx="3" '
                   f'stroke="{stroke}" stroke-width="{4 if i in (1, 2) else 2.5}" fill="#fcfbf7"/>')
        out.append(f'<text x="{x + 22}" y="{y0 + 52}" font-family="IBM Plex Mono" font-size="25" '
                   f'fill="{stroke}">{i + 1}</text>')
        out.append(f'<text x="{x + 56}" y="{y0 + 54}" font-size="37" font-weight="600" '
                   f'fill="#16191d">{t}</text>')
        out.append(f'<text x="{x + 22}" y="{y0 + 110}" font-size="27" fill="#4b4f55">{a}</text>')
        out.append(f'<text x="{x + 22}" y="{y0 + 148}" font-size="27" fill="#4b4f55">{b}</text>')
        if i < len(steps) - 1:
            out.append(f'<line x1="{x + bw + 8}" y1="{y0 + bh / 2}" x2="{x + bw + gap - 10}" '
                       f'y2="{y0 + bh / 2}" stroke="#16191d" stroke-width="3" marker-end="url(#ah)"/>')
    xa, xb = 1 * (bw + gap), 5 * (bw + gap) + bw
    yb = y0 + bh + 34
    out.append(f'<path d="M{xa},{yb - 12} V{yb} H{xb} V{yb - 12}" stroke="#2a78d6" stroke-width="3"/>')
    out.append(f'<text x="{(xa + xb) / 2}" y="{yb + 46}" font-size="29" fill="#2a78d6" '
               f'text-anchor="middle">all on the vehicle: no weather service, no network link, '
               f'one CPU core</text>')
    out.append("</svg>")
    return "\n".join(out)


def page(n, svg):
    bt, to = n["bt"], n["to"]
    rows = []
    for p, label, cls in [("Static (terrain only)", "terrain only", ""),
                          ("Today (Liu et al.)", "today's map [1]", "foil"),
                          ("Today + margin", "today + margin", ""),
                          ("Forecast + margin (ours)", "forecast + margin", "us")]:
        r = bt.loc[(p, 3)]
        rows.append(f'<tr class="{cls}"><td>{label}</td><td>{r.stuck_pct:.1f}%</td>'
                    f'<td>{r.held_pct:.1f}%</td><td>+{r.extra_pct:.2f}%</td></tr>')
    mrows = []
    for p, label, cls in [("Forecast + margin (ours)", "minimum (fuzzy AND)", "us"),
                          ("Forecast + margin [liu_weighted]", "weighted mean, with veto [1]", ""),
                          ("Forecast + margin [weighted_no_veto]", "weighted mean, no veto", "")]:
        cells = "".join(f"<td>{bt.loc[(p, h)].stuck_pct:.1f}%</td>" for h in (1, 3, 7))
        mrows.append(f'<tr class="{cls}"><td>{label}</td>{cells}</tr>')
    nogo = n["band_nogo"]

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Plan for the ground you will drive on</title>
<style>{CSS}</style></head>
<body>
<header>
  <div class="venue"><b>ICDCN 2027</b><br>Poster track<br>NITK Surathkal, 4–7 Jan 2027</div>
  <div class="logo">[institution<br>logo]</div>
  <div>
  <h1>Plan for the ground you will drive on, not the ground you see today</h1>
  <div class="sub">Forecast-aware off-road route planning for military vehicles, computed on the
  vehicle with no weather service and no network link</div>
  <div class="authors">{AUTHORS}</div>
  <div class="affil">{AFFIL}</div>
  </div>
</header>

<div class="claim">
  <p>Routes planned on a forecast made <span class="ours">on the vehicle</span> sent
  <b>{n['stuck_cut3']}% fewer</b> trips into mud than routes planned on
  <span class="foil">today's map</span>, three days out, and held back fewer that could have
  gone. At the same risk, they hold back <b>{n['held_cut7']}–{n['held_cut3']}% fewer</b> trips
  than a safety margin on today's map.</p>
  <div class="tiles">
    <div class="tile"><div class="v">{n['cpu_ms']}<small>ms</small></div><div class="k">per 7-day forecast,<br>one CPU core</div></div>
    <div class="tile"><div class="v">{n['msg_bytes']}<small>B</small></div><div class="k">to send it<br>to a convoy</div></div>
    <div class="tile"><div class="v">{n['size_mb']}<small>MB</small></div><div class="k">model size,<br>{n['params_k']} parameters</div></div>
    <div class="tile"><div class="v">{n['replan_ms']}<small>ms</small></div><div class="k">to re-plan a route<br>on a {n['map_kb']}k-cell map</div></div>
  </div>
</div>

<div class="pipe">{pipeline_svg()}</div>

<main>
<div class="col">
  <section>
    <h2><span class="no">01</span>The ground changes faster than the map</h2>
    <p>Off the road, the soil decides who moves. On the Pachmarhi plateau a monsoon week takes
    ground firmness from over 200 psi to under 20, so a route that is GO when planned can be
    <span class="k-nogo">NO-GO</span> when the column arrives.</p>
    <p>Maps describe the ground <em>today</em> [1]. We plan for the day of travel.</p>
    <div class="fig">{svg('f1_season')}</div>
    <div class="cap"><b>Fig. 1</b> Rain and ground firmness (rating cone index, RCI), 2024–25.
    Below its red line, a vehicle band is stuck.</div>
  </section>
  <section>
    <h2><span class="no">02</span>Firmness from soil moisture, limits from the vehicle</h2>
    <div class="eq">
      <span class="v">SWI</span> = 0.35 <span class="v">s</span><sub>0–7 cm</sub> + 0.65 <span class="v">s</span><sub>7–28 cm</sub><br>
      <span class="v">RCI</span> = 300 · e<sup>−3.0 <span class="v">SWI</span></sup> · (1 − 0.15 min(<span class="v">R</span><sub>3d</sub> / 50 mm, 1))<br>
      NO-GO if <span class="v">RCI</span> &lt; VCI₁ &nbsp;·&nbsp; SLOW-GO if &lt; VCI₅₀ &nbsp;·&nbsp; GO otherwise
      <span class="note"><span class="v">s</span>: ERA5-Land soil moisture, scaled 0 (driest) to 1 (wettest)
      in 11 years. The wet end, 15 psi, follows field readings where a 4×4 bogged down [2].</span>
    </div>
    <p>Each of {n['n_vehicles']:,} ground vehicles gets its VCI₁ and VCI₅₀ from the NRMM Mobility
    Index [2, 3]. Our tracked version reproduces the published M1 example.</p>
    <div class="fig">{svg('f2_fleet')}</div>
    <div class="cap"><b>Fig. 2</b> The fleet in five firmness bands: light tracked (A) stuck on
    {nogo['A']:.0f}% of days, wheeled (E) on {nogo['E']:.0f}%.</div>
  </section>
  <section>
    <h2 class="minor">What went in</h2>
    <table class="data">
      <tr><td>Weather</td><td>ERA5-Land soil moisture; IMD rain (ERA5 in 2025); daily 2015–25</td></tr>
      <tr><td>Terrain</td><td>Earth Engine: DEM, WorldCover, Sentinel-2; 90 m grid, {n['map_kb']}k cells</td></tr>
      <tr><td>Roads</td><td>OpenStreetMap, {n['n_roads']} usable segments</td></tr>
      <tr><td>Vehicles</td><td>{n['n_tracked']:,} tracked, {n['n_wheeled']} wheeled, {n['n_half']} half-tracks</td></tr>
    </table>
  </section>
</div>

<div class="col">
  <section>
    <h2><span class="no">03</span>A forecast that fits on the vehicle</h2>
    <p>A GRU and a ridge regression predict the <em>change</em> in RCI 1–7 days ahead from the
    local weather log alone: {n['skill7']}% lower error than "tomorrow equals today" at 7 days.</p>
    <p>A safety margin per band, tuned on one test year and scored on the other, turns it into
    a NO-GO call.</p>
    <div class="fig">{svg('f3_decisions')}</div>
    <div class="cap"><b>Fig. 3</b> Same 90% of NO-GO days caught, {n['fa_red_lo']}–{n['fa_red_hi']}%
    fewer false alarms at 7 days: fewer days a unit stays parked for nothing.</div>
  </section>
  <section>
    <h2><span class="no">04</span>What it cannot see</h2>
    <p>Without a weather forecast, nothing predicts the day the monsoon breaks. The gain comes
    in the days around it.</p>
    <div class="fig">{svg('f7_onset')}</div>
    <div class="cap"><b>Fig. 4</b> Monsoon onset, 2024: firmness three days ahead.</div>
  </section>
  <section>
    <h2><span class="no">05</span>Same trip, two plans</h2>
    <div class="fig">{svg('f4_map')}</div>
    <div class="cap"><b>Fig. 5</b> {n['map_caption']}</div>
  </section>
</div>

<div class="col">
  <section>
    <h2><span class="no">06</span>Two monsoons, {n['bt_trips']:,} trips</h2>
    <p>Every day of 2024–25, 30 trips of 8–25 km per band, leaving 1, 3 or 7 days after planning,
    checked against the real ground on the day of travel. A planner fails two ways: it sends a
    vehicle into mud, or holds back a trip that was possible.</p>
    <div class="fig">{svg('f5_tradeoff')}</div>
    <div class="cap"><b>Fig. 6</b> Both errors as each margin goes from none to 3× the tuned value
    (ringed). The forecast is closer to the corner at every setting.</div>
    <table>
      <tr><th>leaving in 3 days</th><th>stuck</th><th>held back</th><th>extra time</th></tr>
      {''.join(rows)}
    </table>
  </section>
  <section>
    <h2><span class="no">07</span>An average hides the deal-breaker</h2>
    <p>A weighted mean [1] can call a cell passable when one factor says no. The minimum
    cannot.</p>
    <table>
      <tr><th>combining slope, cover, soil</th><th>stuck, 1 d</th><th>3 d</th><th>7 d</th></tr>
      {''.join(mrows)}
    </table>
  </section>
  <section>
    <h2><span class="no">08</span>Convoys: share the track, not the map</h2>
    <p>Each pass churns soft ground. Of ten heavy tracked vehicles, {n['cv_d_follow']} get through
    when they <span class="k-foil">follow the leader</span> and {n['cv_d_wear']} when each re-plans on
    a <span class="k-ours">shared pass-count map</span>, for {n['cv_bytes']} kB of track updates.</p>
    <div class="fig">{svg('f6_convoy')}</div>
  </section>
  <section>
    <h2><span class="no">09</span>Send the forecast, not the map</h2>
    <table>
      <tr><th>what crosses the radio link</th><th>size</th></tr>
      <tr class="us"><td>7-day firmness forecast</td><td>{n['msg_bytes']} B</td></tr>
      <tr><td>a whole convoy's tracks</td><td>{n['cv_bytes']} kB</td></tr>
      <tr><td>one passability map, 1 B per cell</td><td>{n['map_kb']} kB</td></tr>
      <tr><td>maps for 5 bands × 7 days</td><td>{n['maps_mb']} MB</td></tr>
    </table>
    <div class="cap">Vehicles rebuild maps locally: the link carries {n['map_ratio']:,}× less than
    one map.</div>
  </section>
</div>
</main>

<footer>
  <div>
    <b>Limits.</b> Ground firmness is computed from modelled soil moisture (ERA5-Land, one 9 km
    cell), not measured. Spreading it over the map by wetness index is a heuristic with no field
    check. Vehicle masses come from a game-file dataset, with wheeled ones corrected from open
    sources, so we report bands, not named vehicles. Tuned margins differ between the two test
    years by up to {n['margin_gap_max']} psi. <b>Next:</b> cone-penetrometer readings at
    Pachmarhi, Sentinel-1 radar as a soil-moisture input, and ruts that last between days.
  </div>
  <ol>
    <li>Liu Q. et al. Dynamic trafficability map for unmanned vehicles. Sci. Rep. 15:9957, 2025.</li>
    <li>Roșca P. et al. Vehicle cone index for fine-grained soil. J. Mil. Technol. 1(2), 2018.</li>
    <li>Ciobotaru T. Semi-empiric algorithm for vehicle mobility. LEJPT 8(15):19–30, 2009.</li>
    <li>Muñoz-Sabater J. et al. ERA5-Land. Earth Syst. Sci. Data 13:4349–4383, 2021.</li>
    <li>Zanaga D. et al. ESA WorldCover 10 m v200, 2022. Roads © OpenStreetMap contributors.</li>
  </ol>
</footer>
<div class="band">
  <div><b>Contact</b> &nbsp;{CONTACT}</div>
  <div class="mid">ICDCN 2027 &nbsp;·&nbsp; Poster track &nbsp;·&nbsp; NITK Surathkal, Mangaluru</div>
  <div class="right"><span>[project URL]</span><div class="qr">[QR<br>code]</div></div>
</div>
</body></html>
"""
