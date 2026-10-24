"""
The 2-page ICDCN 2027 poster abstract (ACM sigconf, acmart v2.20): fills
paper/icdcn_2027_abstract/abstract_template.tex with numbers from outputs/ and compiles it with Tectonic.

Every number in the abstract comes from the same reader as the poster
(traffic.poster_build.numbers), so the two cannot disagree.

Usage:  python -m traffic.abstract_build
"""

from __future__ import annotations

import re
import subprocess

from traffic import paths
from traffic import poster_build as B

PAPER = paths.ROOT / "paper" / "icdcn_2027_abstract"
TECTONIC = paths.ROOT / "tools" / "tectonic" / "tectonic.exe"


def fill(template: str, n: dict) -> str:
    bt, to = n["bt"], n["to"]
    v = {k: val for k, val in n.items() if not hasattr(val, "loc")}
    k = lambda p, h, c: bt.loc[(p, h), c]
    for h in (1, 3, 7):
        for p, tag in [("Static (terrain only)", "st"), ("Today (Liu et al.)", "td"),
                       ("Today + margin", "tm"), ("Forecast + margin (ours)", "fc"),
                       ("Forecast + margin [liu_weighted]", "lw"),
                       ("Forecast + margin [weighted_no_veto]", "nv")]:
            v[f"{tag}_s{h}"] = f"{k(p, h, 'stuck_pct'):.1f}"
            v[f"{tag}_h{h}"] = f"{k(p, h, 'held_pct'):.1f}"
    v["n_vehicles"] = f"{n['n_vehicles']:,}"
    v["n_days"] = f"{n['n_days']:,}"
    v["n_tracked"] = f"{n['n_tracked']:,}"
    v["bt_trips"] = f"{n['bt_trips']:,}"
    v["map_ratio"] = f"{n['map_ratio']:,}"
    v["nogo_lo"] = f"{min(n['band_nogo'].values()):.0f}"
    v["nogo_hi"] = f"{max(n['band_nogo'].values()):.0f}"
    return re.sub(r"<<(\w+)>>", lambda m: str(v[m.group(1)]), template)


def main():
    import shutil
    (PAPER / "figures").mkdir(exist_ok=True)
    for f in ("f5_tradeoff.pdf", "f4_map.pdf"):          # self-contained for Overleaf
        shutil.copy(B.FIG / f, PAPER / "figures" / f)
    tpl = (PAPER / "abstract_template.tex").read_text(encoding="utf-8")
    tex = fill(tpl, B.numbers())
    (PAPER / "abstract.tex").write_text(tex, encoding="utf-8")
    r = subprocess.run([str(TECTONIC), "-X", "compile", "abstract.tex"], cwd=PAPER,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    errs = [l for l in (r.stdout + r.stderr).splitlines() if l.startswith("error") or "! " in l]
    if r.returncode:
        print("\n".join(errs) or r.stderr[-2000:])
        raise SystemExit(r.returncode)
    from pypdf import PdfReader
    pages = len(PdfReader(PAPER / "abstract.pdf").pages)
    print(f"-> paper/icdcn_2027_abstract/abstract.pdf  {pages} page(s)")


if __name__ == "__main__":
    main()
