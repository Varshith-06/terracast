"""
Check that the TFT's quantile band means what it claims: does the P10-P90 interval
actually contain the truth ~80% of the time, and does the P10 sit below the truth
~90% of the time?

A band that is not calibrated cannot be used to read risk, and the P10 safety rule
rests on the lower quantile being honest.

Usage:  python -m traffic.calibration
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from traffic import paths

OUT = str(paths.FORECAST_OUT)


def main():
    qpath = f"{OUT}/tft_quantiles.npy"
    if not os.path.exists(qpath):
        print("no tft_quantiles.npy — run traffic.run first")
        return
    q = np.load(qpath)                                   # (N, H, 3) = P10, P50, P90
    npz = np.load(f"{OUT}/test_predictions.npz", allow_pickle=True)
    Y = npz["Y"]
    dates = pd.to_datetime(npz["dates"])
    if q.shape[0] != Y.shape[0]:
        print(f"shape mismatch {q.shape} vs {Y.shape}")
        return

    rows = []
    for h in range(1, q.shape[1] + 1):
        lo, mid, hi = q[:, h - 1, 0], q[:, h - 1, 1], q[:, h - 1, 2]
        y = Y[:, h - 1]
        rows.append(dict(
            horizon=h,
            cover_80=100 * np.mean((y >= lo) & (y <= hi)),   # nominal 80
            below_p10=100 * np.mean(y < lo),                 # nominal 10
            above_p90=100 * np.mean(y > hi),                 # nominal 10
            mean_width=float(np.mean(hi - lo)),
            pinball_p10=float(np.mean(np.maximum(0.1 * (y - lo), -0.9 * (y - lo)))),
        ))
    df = pd.DataFrame(rows).round(2)
    print("TFT quantile calibration on the held-out test set")
    print("(nominal: 80% coverage, 10% below P10, 10% above P90)\n")
    print(df.to_string(index=False))

    # monsoon vs dry season: a band that only works in the flat season is not useful
    mons = dates.month.isin([6, 7, 8, 9, 10])
    print("\ncoverage of the P10-P90 band by season")
    for tag, m in [("monsoon Jun-Oct", mons), ("dry Nov-May", ~mons)]:
        lo, hi = q[m, 6, 0], q[m, 6, 2]
        y = Y[m, 6]
        print(f"  {tag:16s} h=7  coverage {100*np.mean((y>=lo)&(y<=hi)):5.1f}%  "
              f"mean width {np.mean(hi-lo):6.1f} kPa  n={m.sum()}")

    df.to_csv(f"{OUT}/tft_calibration.csv", index=False)
    print(f"\n-> {OUT}/tft_calibration.csv")


if __name__ == "__main__":
    main()
