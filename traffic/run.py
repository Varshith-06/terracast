"""
Train and compare time-series models for off-road trafficability (RCI) forecasting.

Setting: operational, no numerical weather prediction available. Every model sees
only observations up to the forecast origin t and predicts RCI at t+1 .. t+7.
Splits are strictly chronological: train 2015-2021, val 2022-2023, test 2024-2025.

Parameterisation: RCI has a lag-1 autocorrelation of 0.99, so the learned models
predict the CHANGE from the origin value, y_{t+h} - y_t, and the anchor y_t is
added back. Persistence is then exactly the zero-change forecast, which makes the
skill score a direct read of how much signal each model adds.

Usage:  python -m traffic.run [--quick] [--skip-dl] [--skip-arima]
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

from traffic import data as D
from traffic import paths

warnings.filterwarnings("ignore")

H = 7
L = 60                       # sequence window for the deep models
EVAL_H = [1, 3, 7]
SEEDS = [0, 1, 2]
OUT = str(paths.FORECAST_OUT)
BASELINES = ("Persistence", "Climatology", "SeasonalNaive365")


# ------------------------------------------------------------------------- metrics
def regression_metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = p - y
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    smape = float(np.mean(2 * np.abs(err) / np.maximum(np.abs(y) + np.abs(p), 1e-9)) * 100)
    return dict(rmse=rmse, mae=mae, bias=bias, r2=r2, smape=smape)


def class_metrics(y, p, vehicle):
    yc = D.rci_to_class(y, vehicle)
    pc = D.rci_to_class(p, vehicle)
    acc = float((yc == pc).mean())
    f1s, nogo_recall = [], np.nan
    for c in range(3):
        tp = float(((pc == c) & (yc == c)).sum())
        fp = float(((pc == c) & (yc != c)).sum())
        fn = float(((pc != c) & (yc == c)).sum())
        if tp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn)
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        if c == 0:
            nogo_recall = rec
    return dict(acc=acc, macro_f1=float(np.mean(f1s)), nogo_recall=float(nogo_recall))


# --------------------------------------------------------------------- data assembly
def build_arrays():
    raw, feats = D.get_dataset()
    y = raw["rci"].values.astype(np.float64)
    dates = feats["Date"]

    tab_cols = D.tabular_columns(feats)
    Xtab = feats[tab_cols].values.astype(np.float64)
    seq_cols = D.sequence_columns(feats)
    Xseq = feats[seq_cols].values.astype(np.float64)
    Xfut = feats[D.KNOWN_FUTURE_COLS].values.astype(np.float64)

    n = len(feats)
    first_valid = int(np.where(~np.isnan(Xtab).any(axis=1))[0][0])
    lo = max(first_valid, L - 1)
    origins = np.arange(lo, n - H)

    tr_m, va_m, te_m = D.split_masks(dates)
    o_tr = origins[tr_m[origins]]
    o_va = origins[va_m[origins]]
    o_te = origins[te_m[origins]]

    Y = np.stack([y[origins + h] for h in range(1, H + 1)], axis=1)   # levels
    anchor = y[origins]
    Yd = Y - anchor[:, None]                                          # changes
    pos = {t: i for i, t in enumerate(origins)}

    return dict(raw=raw, feats=feats, y=y, dates=dates, tab_cols=tab_cols,
                seq_cols=seq_cols, Xtab=Xtab, Xseq=Xseq, Xfut=Xfut,
                origins=origins, o_tr=o_tr, o_va=o_va, o_te=o_te,
                Y=Y, Yd=Yd, anchor=anchor, pos=pos)


def _idx(ds, ors):
    return np.array([ds["pos"][t] for t in ors])


def levels(ds, ors):
    return ds["Y"][_idx(ds, ors)]


def deltas(ds, ors):
    return ds["Yd"][_idx(ds, ors)]


def anchors(ds, ors):
    return ds["anchor"][_idx(ds, ors)]


# ------------------------------------------------------------------ baseline models
def run_baselines(ds, P, V):
    y, o_te, o_va, o_tr = ds["y"], ds["o_te"], ds["o_va"], ds["o_tr"]

    for tag, ors, store in (("te", o_te, P), ("va", o_va, V)):
        store["Persistence"] = np.repeat(y[ors][:, None], H, axis=1)

    dj = ds["dates"].dt.dayofyear.values
    clim = pd.Series(y[o_tr]).groupby(dj[o_tr]).mean()
    clim = clim.reindex(range(1, 367)).interpolate().bfill().ffill()
    for ors, store in ((o_te, P), (o_va, V)):
        store["Climatology"] = np.stack(
            [clim.loc[np.clip(dj[ors + h], 1, 366)].values for h in range(1, H + 1)], axis=1)
        store["SeasonalNaive365"] = np.stack(
            [y[np.maximum(ors + h - 365, 0)] for h in range(1, H + 1)], axis=1)


# --------------------------------------------------------------------- tabular models
def run_tabular(ds, P, V, quick=False):
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb

    Xtab = ds["Xtab"]
    o_tr, o_va, o_te = ds["o_tr"], ds["o_va"], ds["o_te"]
    Xtr, Xva, Xte = Xtab[o_tr], Xtab[o_va], Xtab[o_te]
    Dtr, Dva = deltas(ds, o_tr), deltas(ds, o_va)          # learning targets
    a_va, a_te = anchors(ds, o_va), anchors(ds, o_te)
    Xfull = np.vstack([Xtr, Xva])
    Dfull = np.vstack([Dtr, Dva])

    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s, Xte_s = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    sc2 = StandardScaler().fit(Xfull)
    Xfull_s, Xte_s2 = sc2.transform(Xfull), sc2.transform(Xte)

    # --- Ridge -------------------------------------------------------------
    pv, pt = np.zeros((len(Xva), H)), np.zeros((len(Xte), H))
    alphas = {}
    for h in range(H):
        best_a, best_e = None, np.inf
        for a in [0.1, 1, 10, 100, 1000, 10000]:
            m = Ridge(alpha=a).fit(Xtr_s, Dtr[:, h])
            e = np.mean((m.predict(Xva_s) - Dva[:, h]) ** 2)
            if e < best_e:
                best_a, best_e = a, e
        alphas[h + 1] = best_a
        pv[:, h] = Ridge(alpha=best_a).fit(Xtr_s, Dtr[:, h]).predict(Xva_s)
        pt[:, h] = Ridge(alpha=best_a).fit(Xfull_s, Dfull[:, h]).predict(Xte_s2)
    V["Ridge"], P["Ridge"] = pv + a_va[:, None], pt + a_te[:, None]

    # --- LightGBM ----------------------------------------------------------
    # Two variants. Constant-leaf boosting is piecewise-constant, which is a poor
    # fit for a smooth drying curve; linear_tree puts a linear model in each leaf
    # and is the fair comparison against Ridge.
    base = dict(objective="regression", metric="rmse", learning_rate=0.03,
                num_leaves=31, min_data_in_leaf=20, feature_fraction=0.7,
                bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                verbosity=-1, seed=0, num_threads=os.cpu_count())
    variants = {
        "LightGBM": base,
        "LightGBM(linear)": {**base, "linear_tree": True, "linear_lambda": 1.0,
                             "num_leaves": 15, "min_data_in_leaf": 40},
    }
    sel = {"ridge_alphas": alphas}
    for vname, params in variants.items():
        pv, pt = np.zeros((len(Xva), H)), np.zeros((len(Xte), H))
        gains, iters = {}, {}
        for h in range(H):
            dtr = lgb.Dataset(Xtr, Dtr[:, h], feature_name=ds["tab_cols"])
            dva = lgb.Dataset(Xva, Dva[:, h], reference=dtr)
            b = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            iters[h + 1] = b.best_iteration
            pv[:, h] = b.predict(Xva, num_iteration=b.best_iteration)
            dfull = lgb.Dataset(Xfull, Dfull[:, h], feature_name=ds["tab_cols"])
            final = lgb.train(params, dfull, num_boost_round=max(b.best_iteration, 50))
            pt[:, h] = final.predict(Xte)
            gains[h + 1] = dict(zip(ds["tab_cols"],
                                    final.feature_importance("gain").tolist()))
        V[vname], P[vname] = pv + a_va[:, None], pt + a_te[:, None]
        sel[f"{vname}_best_iterations"] = iters
        if vname == "LightGBM":
            json.dump(gains, open(f"{OUT}/lgbm_feature_gain.json", "w"))
    json.dump(sel, open(f"{OUT}/tabular_selection.json", "w"), indent=2)

    # --- bagged trees ------------------------------------------------------
    n_est = 200 if quick else 500
    for name, Model in [("RandomForest", RandomForestRegressor),
                        ("ExtraTrees", ExtraTreesRegressor)]:
        m = Model(n_estimators=n_est, min_samples_leaf=2, max_features=0.4,
                  n_jobs=-1, random_state=0).fit(Xtr, Dtr)
        V[name] = m.predict(Xva) + a_va[:, None]
        m2 = Model(n_estimators=n_est, min_samples_leaf=2, max_features=0.4,
                   n_jobs=-1, random_state=0).fit(Xfull, Dfull)
        P[name] = m2.predict(Xte) + a_te[:, None]


# ------------------------------------------------------------------------- ARIMA
def _fourier(dates, K=3, period=365.25):
    doy = dates.dt.dayofyear.values.astype(float)
    cols = []
    for k in range(1, K + 1):
        cols += [np.sin(2 * np.pi * k * doy / period), np.cos(2 * np.pi * k * doy / period)]
    return np.column_stack(cols)


def _select_order(endog, exog, max_p=5, max_q=5):
    """AIC grid search; differencing order set by ADF + KPSS."""
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    adf_p = adfuller(endog, autolag="AIC")[1]
    kpss_p = kpss(endog, regression="c", nlags="auto")[1]
    d = 0 if (adf_p < 0.05 and kpss_p > 0.05) else 1
    results = []
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue
            try:
                r = SARIMAX(endog, exog=exog, order=(p, d, q),
                            trend="c" if d == 0 else "n",
                            enforce_stationarity=False, enforce_invertibility=False
                            ).fit(disp=False, maxiter=200)
                if np.isfinite(r.aic):
                    results.append(((p, d, q), float(r.aic), float(r.bic)))
            except Exception:
                continue
    results.sort(key=lambda x: x[1])
    return results[0][0], results, dict(adf_p=float(adf_p), kpss_p=float(kpss_p), d=d)


def _multistep_from_filter(res, n_obs, origins, H):
    """
    Exact h-step-ahead forecasts from every origin using the filtered state:
        y_{t+j|t} = Z (T^j a_{t|t} + sum_{i<j} T^i c) + d_{t+j}
    One Kalman pass over the whole record, then a cheap recursion per origin.
    """
    ssm = res.filter_results
    Z = np.asarray(ssm.design)
    Z = (Z[:, :, 0] if Z.ndim == 3 else Z).ravel()      # single observed series
    T = np.asarray(ssm.transition)
    T = T[:, :, 0] if T.ndim == 3 else T
    c = np.asarray(ssm.state_intercept)
    c = c[:, 0] if c.ndim == 2 else c
    a = np.asarray(ssm.filtered_state)

    oi = np.asarray(ssm.obs_intercept).ravel()
    if oi.size == 1:
        oi = np.repeat(oi, n_obs)

    out = np.zeros((len(origins), H))
    for i, t in enumerate(origins):
        s = a[:, t].copy()
        for j in range(1, H + 1):
            s = T @ s + c
            out[i, j - 1] = Z @ s + oi[t + j]
    return out


def run_arima(ds, P, V):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y, dates = ds["y"], ds["dates"]
    n = len(y)
    tr_end = int(np.where(dates <= pd.Timestamp(D.TRAIN_END))[0][-1]) + 1
    va_end = int(np.where(dates <= pd.Timestamp(D.VAL_END))[0][-1]) + 1
    four = _fourier(dates, K=3)
    info = {}

    for name, exog in [("ARIMA", None), ("ARIMAX-Fourier (DHR)", four)]:
        t0 = time.time()
        # order + params chosen on train only, so val predictions stay honest
        order, grid, stat = _select_order(y[:tr_end], None if exog is None else exog[:tr_end])
        trend = "c" if order[1] == 0 else "n"
        r_tr = SARIMAX(y[:tr_end], exog=None if exog is None else exog[:tr_end],
                       order=order, trend=trend, enforce_stationarity=False,
                       enforce_invertibility=False).fit(disp=False, maxiter=400)
        full = SARIMAX(y, exog=exog, order=order, trend=trend,
                       enforce_stationarity=False, enforce_invertibility=False)
        V[name] = _multistep_from_filter(full.filter(r_tr.params), n, ds["o_va"], H)

        # refit on train+val for the test forecasts
        r_va = SARIMAX(y[:va_end], exog=None if exog is None else exog[:va_end],
                       order=order, trend=trend, enforce_stationarity=False,
                       enforce_invertibility=False).fit(disp=False, maxiter=400,
                                                        start_params=r_tr.params)
        P[name] = _multistep_from_filter(full.filter(r_va.params), n, ds["o_te"], H)

        info[name] = dict(order=list(order), aic=float(r_va.aic), bic=float(r_va.bic),
                          stationarity=stat, seconds=round(time.time() - t0, 1),
                          top5=[(list(o), round(a, 1)) for o, a, _ in grid[:5]])
        print(f"  {name}: order={order}  AIC={r_va.aic:.1f}  ({time.time()-t0:.0f}s)")

    json.dump(info, open(f"{OUT}/arima_selection.json", "w"), indent=2)


# ---------------------------------------------------------------------- deep models
def run_deep(ds, P, V, quick=False):
    import torch
    from traffic import dl

    print(f"  device: {dl.DEVICE}  "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")

    Xseq, Xfut = ds["Xseq"], ds["Xfut"]
    o_tr, o_va, o_te = ds["o_tr"], ds["o_va"], ds["o_te"]

    mu, sd = Xseq[o_tr].mean(0), Xseq[o_tr].std(0)
    sd[sd < 1e-8] = 1.0
    Xs = (Xseq - mu) / sd

    # per-horizon standardisation of the change, so h=1 is not drowned out by h=7
    Dtr = deltas(ds, o_tr)
    dmu, dsd = Dtr.mean(0), Dtr.std(0)
    dsd[dsd < 1e-8] = 1.0

    packs, anch = {}, {}
    for tag, ors in [("train", o_tr), ("val", o_va), ("test", o_te)]:
        Xp, Xf = dl.make_windows(Xs, Xfut, L, H, ors)
        Yz = (deltas(ds, ors) - dmu) / dsd
        packs[tag] = tuple(torch.tensor(a, dtype=torch.float32, device=dl.DEVICE)
                           for a in (Xp, Xf, Yz))
        anch[tag] = anchors(ds, ors)

    n_past, n_fut = Xs.shape[1], Xfut.shape[1]
    target_idx = ds["seq_cols"].index("raw_rci")
    epochs = 60 if quick else 300
    seeds = SEEDS[:1] if quick else SEEDS

    quantiles = [0.1, 0.5, 0.9]
    specs = {
        "LSTM":         lambda: dl.LSTMForecaster(n_past, H, 128, 2, 0.2, "lstm"),
        "GRU":          lambda: dl.LSTMForecaster(n_past, H, 128, 2, 0.2, "gru"),
        "BiGRU-Attn":   lambda: dl.BiGRUAttn(n_past, n_fut, H, 96, 2, 0.2, 4),
        "TCN":          lambda: dl.TCNForecaster(n_past, H, 64, 5, 3, 0.15),
        "Transformer":  lambda: dl.TransformerForecaster(n_past, H, 128, 8, 3, 0.15),
        "N-BEATS":      lambda: dl.NBeatsForecaster(n_past, H, L, target_idx, 256, 3, 4),
        "TFT":          lambda: dl.TFT(n_past, n_fut, H, d=64, heads=4, dropout=0.15),
        "BiGRU-Attn(q)": lambda: dl.BiGRUAttn(n_past, n_fut, H, 96, 2, 0.2, 4,
                                              quantiles=quantiles),
    }
    # models trained on pinball loss -> emit a quantile cube, median is the point forecast
    QUANT = {"TFT", "BiGRU-Attn(q)"}
    SLUG = {"TFT": "tft", "BiGRU-Attn(q)": "bigru"}
    log = {}

    for name, ctor in specs.items():
        t0 = time.time()
        q = quantiles if name in QUANT else None
        pv, pt, vls, qt, att = [], [], [], [], []
        for s in seeds:
            torch.manual_seed(s); np.random.seed(s)
            model, vloss, n_ep = dl.train_model(
                ctor(), packs, epochs=epochs, lr=1e-3, batch=128, patience=30,
                quantiles=q, seed=s)
            pv.append(dl.predict(model, *packs["val"][:2], quantiles=q))
            pt.append(dl.predict(model, *packs["test"][:2], quantiles=q))
            vls.append(vloss)
            if name in QUANT:
                with torch.no_grad():
                    rq = torch.cat([model(packs["test"][0][i:i + 512],
                                          packs["test"][1][i:i + 512])
                                    for i in range(0, len(packs["test"][0]), 512)])
                qt.append(rq.float().cpu().numpy())
            if isinstance(model, dl.BiGRUAttn):     # mean attention over lag, (H, L)
                model.eval()
                with torch.no_grad():
                    acc = []
                    for i in range(0, len(packs["test"][0]), 512):
                        model(packs["test"][0][i:i + 512], packs["test"][1][i:i + 512])
                        acc.append(model.last_attn.float().cpu().numpy())
                att.append(np.concatenate(acc).mean(0))
            del model
            torch.cuda.empty_cache()
        if qt:      # seed-averaged quantile forecasts, back in RCI units
            np.save(f"{OUT}/{SLUG[name]}_quantiles.npy",
                    np.mean(qt, 0) * dsd[None, :, None] + dmu[None, :, None]
                    + anch["test"][:, None, None])
        if att:
            np.save(f"{OUT}/{name.replace('(q)', '_q')}_attention.npy", np.mean(att, 0))
        V[name] = np.mean(pv, 0) * dsd + dmu + anch["val"][:, None]
        P[name] = np.mean(pt, 0) * dsd + dmu + anch["test"][:, None]
        n_par = sum(p.numel() for p in ctor().parameters())
        log[name] = dict(val_loss_mean=float(np.mean(vls)), val_loss_std=float(np.std(vls)),
                         seeds=len(seeds), params=int(n_par), epochs_cap=epochs,
                         seconds=round(time.time() - t0, 1))
        print(f"  {name:12s} val={np.mean(vls):.5f}  params={n_par:,}  "
              f"{time.time()-t0:.0f}s")

    json.dump(log, open(f"{OUT}/dl_training_log.json", "w"), indent=2)


# ------------------------------------------------------------------------ evaluation
def evaluate(ds, P, ors):
    Yte = levels(ds, ors)
    rows = []
    pers = P["Persistence"]
    for name, pred in P.items():
        for h in range(1, H + 1):
            m = regression_metrics(Yte[:, h - 1], pred[:, h - 1])
            rmse_p = np.sqrt(np.mean((pers[:, h - 1] - Yte[:, h - 1]) ** 2))
            m["skill_vs_persistence"] = float(1 - m["rmse"] / rmse_p)
            row = dict(model=name, horizon=h, **m)
            for veh in D.VEHICLES:
                for k, v in class_metrics(Yte[:, h - 1], pred[:, h - 1], veh).items():
                    row[f"{veh}_{k}"] = v
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-dl", action="store_true")
    ap.add_argument("--skip-arima", action="store_true")
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(f"{OUT}/figures", exist_ok=True)

    t0 = time.time()
    ds = build_arrays()
    print(f"features: {len(ds['tab_cols'])} tabular / {len(ds['seq_cols'])} sequence")
    print(f"origins: train {len(ds['o_tr'])}  val {len(ds['o_va'])}  test {len(ds['o_te'])}")
    print(f"test window: {ds['dates'].iloc[ds['o_te'][0]].date()} .. "
          f"{ds['dates'].iloc[ds['o_te'][-1]].date()}")

    P, V = {}, {}
    print("\n[1/4] baselines"); run_baselines(ds, P, V)
    print("[2/4] tabular"); run_tabular(ds, P, V, a.quick)
    if not a.skip_arima:
        print("[3/4] arima"); run_arima(ds, P, V)
    if not a.skip_dl:
        print("[4/4] deep (GPU)"); run_deep(ds, P, V, a.quick)

    # ensemble: top 3 learned models by mean validation RMSE across horizons
    val_res = evaluate(ds, V, ds["o_va"])
    val_rank = (val_res[~val_res.model.isin(BASELINES)]
                .groupby("model")["rmse"].mean().sort_values())
    members = val_rank.index[:3].tolist()
    P["Ensemble(top3)"] = np.mean([P[m] for m in members], axis=0)
    V["Ensemble(top3)"] = np.mean([V[m] for m in members], axis=0)
    print(f"\nensemble members (chosen on validation): {members}")

    # Pessimistic decision rule: read the TFT's P10 instead of its median. Deliberately
    # biased low, so its RMSE is worse by construction -- the point is the GO/NO-GO
    # call, where under-predicting soil strength is the safe direction to err.
    for mdl, slug in [("TFT", "tft"), ("BiGRU-Attn(q)", "bigru")]:
        qpath = f"{OUT}/{slug}_quantiles.npy"
        if os.path.exists(qpath) and mdl in P:
            q = np.load(qpath)
            if q.shape[0] == len(ds["o_te"]):
                P[f"{mdl} P10 (safe rule)"] = q[:, :, 0]

    res = evaluate(ds, P, ds["o_te"])
    res.to_csv(f"{OUT}/results_full.csv", index=False)
    val_res.to_csv(f"{OUT}/results_validation.csv", index=False)
    np.savez_compressed(f"{OUT}/test_predictions.npz",
                        dates=ds["dates"].iloc[ds["o_te"]].values,
                        origins=ds["o_te"], Y=levels(ds, ds["o_te"]), **P)
    json.dump({"ensemble_members": members,
               "val_rmse_mean": val_rank.round(3).to_dict()},
              open(f"{OUT}/ensemble.json", "w"), indent=2)

    piv = res[res.horizon.isin(EVAL_H)].pivot_table(index="model", columns="horizon",
                                                    values="rmse").round(2)
    piv["skill@7%"] = (100 * res[res.horizon == 7]
                       .set_index("model")["skill_vs_persistence"]).round(1)
    print("\n=== test RMSE (kPa RCI) by horizon ===")
    print(piv.sort_values(7).to_string())
    print(f"\ntotal {time.time()-t0:.0f}s -> {OUT}/results_full.csv")


if __name__ == "__main__":
    main()
