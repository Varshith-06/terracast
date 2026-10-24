"""
Retrain the vehicle trafficability forecaster from a daily weather log and write a new
vehicle_trafficability_forecaster.pkl that trafficability_edge.py runs unchanged.

    python train_trafficability.py --log weather.csv
    python train_trafficability.py --log weather.csv --train-end 2023-12-31 --val-end 2024-12-31

What it does (the procedure behind the published model):
  1. weather log -> soil wetness -> rating cone index (RCI, psi) -> causal features
     (trafficability_edge.py, so training and inference use the same code)
  2. split by date: train <= --train-end < validation <= --val-end < test
  3. ridge regression per lead day 1-7 on the RCI change: alpha picked on validation,
     then refit on train + validation
  4. GRU (128 x 2) on the last 60 days, one per seed (default 3): AdamW, one-cycle LR,
     MSE, early stopping on validation (patience 30, up to 300 epochs)
  5. forecast = today's RCI + mean of the GRU and ridge changes
  6. safety margin per vehicle band and lead day: the smallest that catches 90% of
     NO-GO days, tuned on the test period (or validation if there is no test period)

The vehicle table, bands, RCI constants and planner-facing fields come from --template
(the current forecaster pickle), so the planner pickle stays valid. Needs numpy, pandas,
scipy and torch (a CUDA GPU is used if present; CPU works, slower).

Log format: one row per day, no gaps, columns as in the Pachmarhi file: Date (dd-mm-yyyy),
Rainfall, Tmax, Tmin, Tmean, Humidity, Humidity Min, Wind Speed, Wind Speed Max,
Wind Direction, Pressure, Soil Moisture, Soil Moisture Deep, Cloud Cover.
"""

from __future__ import annotations

import argparse
import math
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

import trafficability_edge as E

HERE = Path(__file__).resolve().parent
ALPHAS = [0.1, 1, 10, 100, 1000, 10000]
RECALL_TARGET = 0.90
MARGIN_GRID = np.arange(0.0, 40.01, 0.25)


# --------------------------------------------------------------------------- data
def arrays(log, p, train_end, val_end):
    raw = E.soil_strength(E.read_log(log), p["rci"])
    f = E.build_features(raw, p["lat_deg"])
    L, H = p["L"], p["H"]
    Xtab = f[p["tab_cols"]].values.astype(np.float64)
    Xseq = f[p["seq_cols"]].values.astype(np.float64)
    y = raw["rci"].values.astype(np.float64)
    first = int(np.where(~np.isnan(Xtab).any(axis=1))[0][0])
    origins = np.arange(max(first, L - 1), len(f) - H)
    d = f["Date"].values[origins]
    split = dict(train=origins[d <= np.datetime64(train_end)],
                 val=origins[(d > np.datetime64(train_end)) & (d <= np.datetime64(val_end))],
                 test=origins[d > np.datetime64(val_end)])
    Y = np.stack([y[origins + h] for h in range(1, H + 1)], 1)
    pos = {t: i for i, t in enumerate(origins)}
    levels = lambda o: Y[[pos[t] for t in o]]
    return dict(raw=raw, dates=f["Date"].values, Xtab=Xtab, Xseq=Xseq, y=y, split=split,
                levels=levels, deltas=lambda o: levels(o) - y[o][:, None])


# -------------------------------------------------------------------------- ridge
def _scaler(X):
    mean, sd = X.mean(0), X.std(0)
    return mean, np.where(sd == 0, 1.0, sd)


def _ridge(X, y, alpha):
    """sklearn Ridge(alpha) with intercept: centre, then solve the normal equations."""
    xm, ym = X.mean(0), y.mean()
    Xc = X - xm
    w = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ (y - ym))
    return w, ym - xm @ w


def fit_ridge(A, H):
    tr, va = A["split"]["train"], A["split"]["val"]
    X, Dtr, Dva = A["Xtab"], A["deltas"](tr), A["deltas"](va)
    m1, s1 = _scaler(X[tr])
    full = np.concatenate([tr, va])
    m2, s2 = _scaler(X[full])
    Dfull = np.vstack([Dtr, Dva])
    coefs, ints, alphas = [], [], []
    for h in range(H):
        Ztr, Zva = (X[tr] - m1) / s1, (X[va] - m1) / s1
        err = {}
        for a in ALPHAS:
            w, b = _ridge(Ztr, Dtr[:, h], a)
            err[a] = np.mean((Zva @ w + b - Dva[:, h]) ** 2)
        a = min(err, key=err.get)
        w, b = _ridge((X[full] - m2) / s2, Dfull[:, h], a)
        coefs.append(w); ints.append(b); alphas.append(a)
    return dict(ridge_mean=m2, ridge_scale=s2, ridge_coef=np.stack(coefs),
                ridge_intercept=np.array(ints)), alphas


# ---------------------------------------------------------------------------- GRU
def fit_gru(A, p, seeds, epochs, patience, device):
    import torch
    import torch.nn.functional as F
    L, c = p["L"], p["gru_cfg"]
    tr, va = A["split"]["train"], A["split"]["val"]
    mu, sd = A["Xseq"][tr].mean(0), A["Xseq"][tr].std(0)
    sd[sd < 1e-8] = 1.0
    Xs = ((A["Xseq"] - mu) / sd).astype(np.float32)
    Dtr = A["deltas"](tr)
    dmu, dsd = Dtr.mean(0), Dtr.std(0)
    dev = torch.device(device)

    def pack(o):
        Xp = np.stack([Xs[t - L + 1: t + 1] for t in o])
        Yz = (A["deltas"](o) - dmu) / dsd
        return torch.tensor(Xp, device=dev), torch.tensor(Yz, dtype=torch.float32, device=dev)

    (Xp, Y), (vXp, vY) = pack(tr), pack(va)
    states = []
    for s in seeds:
        torch.manual_seed(s); np.random.seed(s)

        class Net(torch.nn.Module):
            def __init__(m):
                super().__init__()
                m.rnn = torch.nn.GRU(c["n_in"], c["hidden"], c["layers"], batch_first=True,
                                     dropout=c["dropout"])
                m.head = torch.nn.Sequential(torch.nn.Dropout(c["dropout"]),
                                             torch.nn.Linear(c["hidden"], c["hidden"]),
                                             torch.nn.ReLU(), torch.nn.Linear(c["hidden"], c["H"]))

            def forward(m, x):
                out, _ = m.rnn(x)
                return m.head(out[:, -1])

        net = Net()
        torch.manual_seed(s)
        net = net.to(dev)
        batch, lr = 128, 1e-3
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, total_steps=epochs * max(1, math.ceil(len(Xp) / batch)), pct_start=0.2)
        g = torch.Generator(device="cpu"); g.manual_seed(s)
        best, best_state, bad, t0 = float("inf"), None, 0, time.time()
        for ep in range(epochs):
            net.train()
            perm = torch.randperm(len(Xp), generator=g).to(dev)
            for i in range(0, len(Xp), batch):
                idx = perm[i: i + batch]
                opt.zero_grad(set_to_none=True)
                loss = F.mse_loss(net(Xp[idx]), Y[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step(); sched.step()
            net.eval()
            with torch.no_grad():
                v = F.mse_loss(net(vXp), vY).item()
            if v < best - 1e-6:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        net.load_state_dict(best_state)
        states.append({k: t.cpu().numpy().astype(np.float32) for k, t in net.state_dict().items()})
        print(f"  GRU seed {s}: {ep + 1} epochs, best validation MSE {best:.4f}, {time.time() - t0:.0f}s")
    return dict(gru_states=states, seq_mu=mu, seq_sd=sd, dmu=dmu, dsd=dsd)


# ------------------------------------------------------------------------ margins
def tune_margin(y, pred, vci1):
    actual = y[None, :] < vci1[:, None]
    for m in MARGIN_GRID:
        called = (pred[None, :] - m) < vci1[:, None]
        tp, fn = int((actual & called).sum()), int((actual & ~called).sum())
        if tp + fn == 0 or tp / (tp + fn) >= RECALL_TARGET:
            return float(m)
    return float(MARGIN_GRID[-1])


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True, help="daily weather log CSV (11 years were used)")
    ap.add_argument("--template", default=str(HERE / "vehicle_trafficability_forecaster.pkl"))
    ap.add_argument("--out", default=str(HERE / "vehicle_trafficability_forecaster_retrained.pkl"))
    ap.add_argument("--train-end", default="2021-12-31")
    ap.add_argument("--val-end", default="2023-12-31", help="test period is everything after")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--device", default="auto", help="auto, cpu or cuda")
    a = ap.parse_args()
    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device

    p = E.load_pickle(a.template)
    for k in ("ridge_mean", "ridge_scale", "ridge_coef", "ridge_intercept", "gru_states",
              "seq_mu", "seq_sd", "dmu", "dsd", "margins", "margins_cross_year"):
        p.pop(k, None)
    t0 = time.time()
    A = arrays(a.log, p, a.train_end, a.val_end)
    n = {k: len(v) for k, v in A["split"].items()}
    print(f"origins: train {n['train']}, validation {n['val']}, test {n['test']}  ({device})")
    if n["train"] == 0 or n["val"] == 0:
        raise SystemExit("need days in both the training and validation periods")

    ridge, alphas = fit_ridge(A, p["H"])
    print(f"  ridge alphas per lead day: {alphas}")
    gru = fit_gru(A, p, a.seeds, a.epochs, a.patience, device)
    p.update(ridge); p.update(gru)

    # out-of-sample forecasts, through the runtime itself
    with open(a.out, "wb") as f:
        pickle.dump(E.encode(p), f, protocol=4)
    model = E.Forecaster(a.out, backend="numpy")
    allf = model.forecast_all(a.log).set_index("date")
    tag = "test" if n["test"] else "val"
    o = A["split"][tag]
    pred = allf.loc[A["dates"][o], [f"rci_h{h}" for h in range(1, p["H"] + 1)]].values
    y = A["levels"](o)
    years = pd.to_datetime(A["dates"][o]).year.values
    veh = pd.DataFrame(p["vehicles"])
    margins, cross = {}, {}
    for b, grp in veh.groupby("band"):
        v1 = grp.vci1.values
        margins[b] = [tune_margin(y[:, h], pred[:, h], v1) for h in range(p["H"])]
        cross[b] = {h + 1: {f"tuned_{yr}": tune_margin(y[years == yr, h], pred[years == yr, h], v1)
                            for yr in sorted(set(years))} for h in range(p["H"])}
    rmse = {f"h{h + 1}": round(float(np.sqrt(np.mean((pred[:, h] - y[:, h]) ** 2))), 2) for h in range(p["H"])}
    pers = {f"h{h + 1}": round(float(np.sqrt(np.mean((A["y"][o] - y[:, h]) ** 2))), 2) for h in range(p["H"])}

    n_gru = sum(v.size for v in gru["gru_states"][0].values())
    n_ridge = int(ridge["ridge_coef"].size + p["H"])
    raw_log = E.read_log(a.log)
    log_cols = pd.read_csv(a.log, nrows=0).columns
    sample = pd.read_csv(a.log).iloc[-p["window_days"]:]
    p.update(
        margins=margins, margins_cross_year=cross,
        params=dict(gru_each=int(n_gru), gru_count=len(a.seeds), ridge=n_ridge,
                    total=int(len(a.seeds) * n_gru + n_ridge)),
        sample_log={c: sample[c].tolist() for c in log_cols},
        info=dict(p.get("info", {}), trained=pd.Timestamp.now().strftime("%Y-%m-%d"),
                  log=f"{raw_log['Date'].iloc[0].date()} to {raw_log['Date'].iloc[-1].date()}",
                  train=f"to {a.train_end} (ridge refit to {a.val_end})",
                  validation=f"to {a.val_end}", test=f"after {a.val_end}" if n["test"] else "none",
                  margins=f"deployed: tuned on the {tag} period pooled at 90% NO-GO recall; "
                          "margins_cross_year: tuned on each year alone",
                  test_rmse_psi=rmse, test_rmse_persistence_psi=pers),
    )
    with open(a.out, "wb") as f:
        pickle.dump(E.encode(p), f, protocol=4)

    print(f"\n{tag} RMSE psi (model / persistence): "
          + ", ".join(f"{k} {rmse[k]} / {pers[k]}" for k in ("h1", "h3", "h7")))
    print("margins (psi, days 1-7):")
    for b, m in margins.items():
        print(f"  {b}: {m}")
    print(f"-> {a.out}  ({Path(a.out).stat().st_size / 2**20:.2f} MB, {time.time() - t0:.0f}s)")
    print("Rename it to vehicle_trafficability_forecaster.pkl to use it with trafficability_edge.py.")


if __name__ == "__main__":
    main()
