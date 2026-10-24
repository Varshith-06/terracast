"""
The deployed forecaster: GRU (128 x 2) + Ridge on the RCI change, averaged, plus a
per-band, per-horizon safety margin that turns the forecast into a NO-GO call.

Model choice comes from the September comparison (outputs/forecasting): width beat
architecture, attention and bidirectionality hurt, a third network added nothing.
Margins are tuned on one held-out year and scored on the other, in both directions,
so no reported margin was tuned on the days it is scored on.

Usage:  python -m traffic.forecast
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch

from traffic import data as D
from traffic import dl, paths
from traffic import run as R
from traffic import vehicles as V

SEEDS = [0, 1, 2]
RECALL_TARGET = 0.90          # margins are the smallest that catch 9 in 10 NO-GO days
MARGIN_GRID = np.arange(0.0, 40.01, 0.25)
OUT = paths.FORECASTER_OUT


# ----------------------------------------------------------------------- training
def fit_ridge(ds):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    o_tr, o_va, o_te = ds["o_tr"], ds["o_va"], ds["o_te"]
    X = ds["Xtab"]
    Dtr, Dva = R.deltas(ds, o_tr), R.deltas(ds, o_va)
    sc = StandardScaler().fit(X[o_tr])
    full = np.concatenate([o_tr, o_va])
    sc2 = StandardScaler().fit(X[full])
    Dfull = np.vstack([Dtr, Dva])
    pv, pt, models = np.zeros((len(o_va), R.H)), np.zeros((len(o_te), R.H)), []
    for h in range(R.H):
        errs = {a: np.mean((Ridge(alpha=a).fit(sc.transform(X[o_tr]), Dtr[:, h])
                            .predict(sc.transform(X[o_va])) - Dva[:, h]) ** 2)
                for a in [0.1, 1, 10, 100, 1000, 10000]}
        a = min(errs, key=errs.get)
        pv[:, h] = Ridge(alpha=a).fit(sc.transform(X[o_tr]), Dtr[:, h]).predict(sc.transform(X[o_va]))
        m = Ridge(alpha=a).fit(sc2.transform(X[full]), Dfull[:, h])
        pt[:, h] = m.predict(sc2.transform(X[o_te]))
        models.append(m)
    return pv, pt, (sc2, models)


def fit_gru(ds):
    Xseq, Xfut = ds["Xseq"], ds["Xfut"]
    o_tr, o_va, o_te = ds["o_tr"], ds["o_va"], ds["o_te"]
    mu, sd = Xseq[o_tr].mean(0), Xseq[o_tr].std(0)
    sd[sd < 1e-8] = 1.0
    Xs = (Xseq - mu) / sd
    Dtr = R.deltas(ds, o_tr)
    dmu, dsd = Dtr.mean(0), Dtr.std(0)
    packs = {}
    for tag, ors in [("train", o_tr), ("val", o_va), ("test", o_te)]:
        Xp, Xf = dl.make_windows(Xs, Xfut, R.L, R.H, ors)
        Yz = (R.deltas(ds, ors) - dmu) / dsd
        packs[tag] = tuple(torch.tensor(a, dtype=torch.float32, device=dl.DEVICE)
                           for a in (Xp, Xf, Yz))
    pv, pt, nets = [], [], []
    for s in SEEDS:
        torch.manual_seed(s); np.random.seed(s)
        net, _, _ = dl.train_model(dl.LSTMForecaster(Xs.shape[1], R.H, 128, 2, 0.2, "gru"),
                                   packs, epochs=300, lr=1e-3, batch=128, patience=30, seed=s)
        pv.append(dl.predict(net, *packs["val"][:2]))
        pt.append(dl.predict(net, *packs["test"][:2]))
        nets.append(net.cpu())
    return (np.mean(pv, 0) * dsd + dmu, np.mean(pt, 0) * dsd + dmu,
            dict(nets=nets, mu=mu, sd=sd, dmu=dmu, dsd=dsd))


# ------------------------------------------------------------------- decisions
def nogo_counts(y, p, vci1, margin):
    """Pooled confusion counts for NO-GO calls over a set of vehicles.
    y, p: (n,) actual and forecast RCI; vci1: (k,) one-pass VCI of each vehicle."""
    actual = y[None, :] < vci1[:, None]
    called = (p[None, :] - margin) < vci1[:, None]
    return (int((actual & called).sum()), int((~actual & called).sum()),
            int((actual & ~called).sum()))


def _scores(tp, fp, fn, n_vehicle_days):
    rec = tp / (tp + fn) if tp + fn else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    return rec, prec, fp / n_vehicle_days * 365.0


def tune_margin(y, p, vci1):
    for m in MARGIN_GRID:
        tp, fp, fn = nogo_counts(y, p, vci1, m)
        if tp + fn == 0 or tp / (tp + fn) >= RECALL_TARGET:
            return float(m)
    return float(MARGIN_GRID[-1])


def decision_table(y_te, preds, years, fleet):
    """Cross-year margin tuning per (rule, band, horizon). y_te: (n, H) actual levels."""
    rows, margins = [], {}
    for rule, P in preds.items():
        for band, g in fleet.groupby("band"):
            v1 = g["vci1"].values
            for h in range(R.H):
                tp = fp = fn = 0
                ms = {}
                for tune_yr, test_yr in [(2024, 2025), (2025, 2024)]:
                    a, b = years == tune_yr, years == test_yr
                    m = tune_margin(y_te[a, h], P[a, h], v1)
                    ms[tune_yr] = m
                    c = nogo_counts(y_te[b, h], P[b, h], v1, m)
                    tp, fp, fn = tp + c[0], fp + c[1], fn + c[2]
                rec, prec, fa = _scores(tp, fp, fn, len(v1) * len(y_te))
                rows.append(dict(rule=rule, band=band, h=h + 1, vehicles=len(v1),
                                 recall=rec, precision=prec, false_alarms_per_year=fa,
                                 margin_tuned_2024=ms[2024], margin_tuned_2025=ms[2025]))
                margins[(rule, band, h + 1)] = ms
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- edge numbers
def edge_numbers(ds, gru_pack, ridge_pack):
    nets = gru_pack["nets"]
    n_gru = sum(p.numel() for p in nets[0].parameters())
    sc, ridges = ridge_pack
    n_ridge = sum(m.coef_.size + 1 for m in ridges) + 2 * sc.mean_.size
    params = len(nets) * n_gru + n_ridge
    # one forecast for one origin, on CPU, single thread: what a vehicle computer does
    torch.set_num_threads(1)
    t = ds["o_te"][-1]
    xs = (ds["Xseq"][t - R.L + 1: t + 1] - gru_pack["mu"]) / gru_pack["sd"]
    xp = torch.tensor(xs[None], dtype=torch.float32)
    xt = sc.transform(ds["Xtab"][[t]])
    for _ in range(5):                                   # warm-up
        [n(xp) for n in nets]
    reps = 200
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(reps):
            [n.eval()(xp) for n in nets]
            [m.predict(xt) for m in ridges]
    ms = (time.perf_counter() - t0) / reps * 1000
    t0 = time.perf_counter()
    for _ in range(20):
        D.build_features(ds["raw"].iloc[-400:])
    feat_ms = (time.perf_counter() - t0) / 20 * 1000
    return dict(params=int(params), gru_params_each=int(n_gru), gru_count=len(nets),
                size_mb_float32=round(params * 4 / 2**20, 2),
                cpu_forecast_ms=round(ms, 2), cpu_feature_build_ms=round(feat_ms, 1),
                forecast_message_bytes=R.H * 2 + 4)   # 7 x float16 RCI + 4-byte date


# ------------------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ds = R.build_arrays()
    a_va, a_te = R.anchors(ds, ds["o_va"]), R.anchors(ds, ds["o_te"])
    rv, rt, ridge_pack = fit_ridge(ds)
    gv, gt, gru_pack = fit_gru(ds)
    print(f"trained in {time.time()-t0:.0f}s")

    y_va, y_te = R.levels(ds, ds["o_va"]), R.levels(ds, ds["o_te"])
    preds_te = {
        "Persistence": np.repeat(a_te[:, None], R.H, 1),
        "Ridge": rt + a_te[:, None],
        "GRU": gt + a_te[:, None],
        "GRU+Ridge": (gt + rt) / 2 + a_te[:, None],
    }
    preds_va = {"GRU": gv + a_va[:, None], "Ridge": rv + a_va[:, None],
                "GRU+Ridge": (gv + rv) / 2 + a_va[:, None]}

    reg = []
    for name, P in preds_te.items():
        for h in range(R.H):
            e = P[:, h] - y_te[:, h]
            reg.append(dict(model=name, h=h + 1, rmse=float(np.sqrt(np.mean(e ** 2))),
                            mae=float(np.mean(np.abs(e))), bias=float(np.mean(e))))
    reg = pd.DataFrame(reg)
    pers = reg[reg.model == "Persistence"].set_index("h")["rmse"]
    reg["skill_vs_persistence"] = 1 - reg["rmse"] / reg["h"].map(pers)
    reg.to_csv(OUT / "regression_test.csv", index=False)

    fleet = V.add_vci(V.load_vehicles())
    years = ds["dates"].iloc[ds["o_te"]].dt.year.values
    dec = decision_table(y_te, {k: preds_te[k] for k in ["Persistence", "GRU+Ridge"]},
                         years, fleet)
    dec.to_csv(OUT / "decisions_by_band.csv", index=False)

    edge = edge_numbers(ds, gru_pack, ridge_pack)
    json.dump(edge, open(OUT / "edge.json", "w"), indent=2)
    torch.save({"nets": [n.state_dict() for n in gru_pack["nets"]],
                **{k: gru_pack[k] for k in ("mu", "sd", "dmu", "dsd")}}, OUT / "gru.pt")
    np.savez_compressed(
        OUT / "test_forecasts.npz", origins=ds["o_te"],
        dates=ds["dates"].iloc[ds["o_te"]].values.astype("datetime64[D]").astype(str),
        actual=y_te, **{k.replace("+", "_"): v for k, v in preds_te.items()})

    print(reg[reg.h.isin([1, 3, 7])].pivot(index="model", columns="h", values="rmse").round(2))
    print(dec[dec.h.isin([1, 3, 7])].round(3).to_string(index=False))
    print(edge)
    print(f"-> {OUT.relative_to(paths.ROOT)}   total {time.time()-t0:.0f}s")


def decisions_only():
    """Re-tune margins and rescore decisions from the saved test forecasts, for a change
    to the vehicle table that leaves the forecaster itself untouched."""
    fc = np.load(OUT / "test_forecasts.npz")
    years = pd.to_datetime(fc["dates"]).year.values
    fleet = V.add_vci(V.load_vehicles())
    dec = decision_table(fc["actual"], {k: fc[k.replace("+", "_")] for k in ["Persistence", "GRU+Ridge"]},
                         years, fleet)
    dec.to_csv(OUT / "decisions_by_band.csv", index=False)
    print(dec[dec.h.isin([1, 3, 7])].round(3).to_string(index=False))


if __name__ == "__main__":
    import sys
    decisions_only() if "--decisions-only" in sys.argv else main()
