"""
Vehicle trafficability on the edge: the runtime for the two pickles.

    vehicle_trafficability_forecaster.pkl   daily weather log -> ground firmness (rating cone
                                            index, RCI, psi) 1-7 days ahead -> GO / SLOW-GO /
                                            NO-GO per vehicle band. GRU x3 (128x2) + ridge,
                                            531,228 parameters.
    vehicle_trafficability_planner.pkl      firmness -> passability map of the Pachmarhi
                                            plateau (90 m grid, 76,218 cells) -> fastest
                                            route that avoids NO-GO ground (Dijkstra),
                                            with travel time in hours from the speed model

SLOW-GO in numbers (same code as traffic/mobility.py in the repo):
    passes  vehicles one track carries before it fails, 50 ** ((RCI - VCI1) / (VCI50 - VCI1))
    speed   km/h from the NRMM Edition I fine-grained soil relations (drawbar pull, motion
            resistance, soil-limited traction, slip) against slope, with a constant-power rim
            pull; see traffic/mobility.py for sources and assumptions

The pickles hold only plain Python types (arrays as dtype, shape and raw bytes), so they
unpickle with any Python 3.8+ and any numpy. Each carries a sample input: the forecaster a
400-day weather log, the planner 30 trips. Needs numpy, pandas, scipy; torch optional (the
GRU also runs in plain numpy, same forecasts to 1e-4 psi).

    python trafficability_edge.py [--backend numpy|torch]    # run once on the samples

    from trafficability_edge import Forecaster, Planner, end_to_end
    fc = Forecaster("vehicle_trafficability_forecaster.pkl")
    pl = Planner("vehicle_trafficability_planner.pkl")
    log = fc.sample_log()                      # or a CSV / DataFrame with the same columns
    f = fc.forecast(log)                       # dict: date, rci_today, rci[0..6] = days 1..7
    fc.assess(f, band="D")                     # GO / SLOW-GO / NO-GO for days 1..7
    fc.outlook(f, band="D")                    # + passes per track and level-ground km/h
    f, rci, routes = end_to_end(fc, pl, log, "D", pl.sample_trips(), h=3)   # travel in 3 days
    pl.plan(rci, "D", pl.sample_trips(), by="time")   # routes by travel time, hours

Bands group vehicles by VCI1 (firmness needed for one pass): A < 18, B 18-21, C 21-24,
D 24-30, E >= 30 psi. The log needs 400 consecutive days (Date as dd-mm-yyyy).
"""

from __future__ import annotations

import pickle
import struct

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

CLASSES = ("NO-GO", "SLOW-GO", "GO")


# ===================================================== passes and speed (traffic.mobility)
G_ACC, HP_W = 9.81, 745.7
ETA = 0.80            # engine-to-sprocket/wheel efficiency (assumed)
ROAD_PSI = 75.0       # tyre pressure on paved road (assumed)
ROAD_TRACTION = 0.8   # traction coefficient on dry pavement (assumed)
# NRMM Edition I fine-grained soil (TARADCOM TR 12503, subroutines IV3, IV5), keyed by
# (running gear, contact pressure factor >= 4 psi)
_PULL = {("tracked", False): (.544, .0463, .0702, .076), ("tracked", True): (.455, .0392, .0526, .056),
         ("wheeled", False): (.3885, .0265, .0358, .046), ("wheeled", True): (.379, .0219, .0257, .033)}
_RES = {("tracked", False): (.045, 2.3075, 6.5, .4, .072), ("tracked", True): (.045, 2.3075, 6.5, .4, .052),
        ("wheeled", False): (.035, .861, 3.249, .3, .043), ("wheeled", True): (.045, 2.3075, 6.5, .4, .029)}
_TFOR = {("tracked", False): (.758, .82), ("tracked", True): (.671, .71),
         ("wheeled", False): (.674, .76), ("wheeled", True): (.585, .655)}
_SLIP = {("tracked", False): (.0257, -.0161, .01519, .8353), ("tracked", True): (.0733, -.0063, .00734, .7177),
         ("wheeled", False): (.0621, -.021, .01888, .7794), ("wheeled", True): (.084, -.016, .01414, .6697)}


def passes(rci, vci1, vci50):
    """Vehicles one track carries before it fails: 0 below VCI1, 50 at VCI50, more above."""
    rci = np.asarray(rci, float)
    return np.where(rci < vci1, 0.0, 50.0 ** ((rci - vci1) / (vci50 - vci1)))


def speed(rci, sin_t, veh, paved=False, need=None):
    """Steady speed (km/h) on soil of strength rci (psi) along a slope of sine sin_t
    (+ uphill). veh: dict(kind, vci1, cpf, hp_per_t, vmax_kmh). 0 where it cannot move."""
    rci, sin_t, paved = np.broadcast_arrays(np.asarray(rci, float), np.asarray(sin_t, float),
                                            np.asarray(paved, bool))
    need = veh["vci1"] if need is None else need
    key = ("wheeled" if veh["kind"] == "wheeled" else "tracked", bool(veh["cpf"] >= 4.0))
    cos_t = np.sqrt(1.0 - sin_t ** 2)
    x = rci - need
    xp = np.maximum(x, 0.0)
    a, b, c, s = _PULL[key]
    t = a + b * xp
    d = np.where(x < 0, s * x, t - np.sqrt(t * t - c * xp))
    p, q, r0, u, s = _RES[key]
    r = np.where(x < 0, u - s * x, p + q / (xp + r0))
    k1, k2 = _TFOR[key]
    cf = d - k1 + r
    tfor = np.where(paved, ROAD_TRACTION, cf + k2)
    r = np.where(paved, .007 + .0939 / ROAD_PSI if veh["kind"] == "wheeled" else .045, r)
    load = sin_t + r * cos_t
    a, b, c, dd = _SLIP[key]
    y = load / cos_t - cf
    yy = np.minimum(y, dd - 1e-6)
    sl = np.where(y >= dd, 1.0, np.clip(a * yy + b + c / (dd - yy), 0.0, 1.0))
    sl = np.where(paved, 0.0, sl)
    v = ETA * veh["hp_per_t"] * HP_W / (1000.0 * G_ACC * np.maximum(load, 1e-9)) * 3.6 * (1 - sl)
    v = np.where(load <= 0, veh["vmax_kmh"], np.minimum(v, veh["vmax_kmh"]))
    return np.where((tfor * cos_t < load) | (~paved & (x < 0)), 0.0, v)


def load_pickle(path):
    """Arrays are stored as (dtype, shape, raw bytes), so the pickles load with any numpy
    version (numpy 1.x pickles and numpy 2.x pickles are not interchangeable)."""
    with open(path, "rb") as f:
        return _decode(pickle.load(f))


def _decode(o):
    if isinstance(o, dict):
        if o.get("__ndarray__"):
            return np.frombuffer(o["data"], dtype=np.dtype(o["dtype"])).reshape(o["shape"]).copy()
        return {k: _decode(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_decode(v) for v in o]
    if isinstance(o, tuple):
        return tuple(_decode(v) for v in o)
    return o


def encode(o):
    """Inverse of _decode: numpy arrays and scalars -> plain Python objects."""
    if isinstance(o, np.ndarray):
        a = np.ascontiguousarray(o)
        return {"__ndarray__": True, "dtype": a.dtype.str, "shape": list(a.shape), "data": a.tobytes()}
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, dict):
        return {str(k) if isinstance(k, np.generic) else k: encode(v) for k, v in o.items()}
    if isinstance(o, list):
        return [encode(v) for v in o]
    if isinstance(o, tuple):
        return tuple(encode(v) for v in o)
    return o


# ============================================================ weather log -> features
_RENAME = {"Rainfall": "rain", "Tmax": "tmax", "Tmin": "tmin", "Tmean": "tmean",
           "Humidity": "rh", "Humidity Min": "rh_min", "Wind Speed": "wind",
           "Wind Speed Max": "wind_max", "Wind Direction": "wdir", "Pressure": "pres",
           "Soil Moisture": "sm_surf", "Soil Moisture Deep": "sm_deep",
           "Cloud Cover": "cloud", "Rainfall Source": "rain_source"}


def read_log(log, last: int | None = None) -> pd.DataFrame:
    """Daily weather log: a CSV path or DataFrame with the columns of the Pachmarhi file
    (Date as dd-mm-yyyy, Rainfall, Tmax, ..., Soil Moisture, Soil Moisture Deep)."""
    df = log.copy() if isinstance(log, pd.DataFrame) else pd.read_csv(log)
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    df = df.sort_values("Date").reset_index(drop=True).rename(columns=_RENAME)
    if last is not None:
        df = df.iloc[-last:].reset_index(drop=True)
    if not df["Date"].diff().dt.days.dropna().eq(1).all():
        raise ValueError("weather log has missing days")
    return df


def _hargreaves(tmax, tmin, tmean, doy, lat_deg):
    phi = np.deg2rad(lat_deg)
    dr = 1.0 + 0.033 * np.cos(2 * np.pi * doy / 365.0)
    dec = 0.409 * np.sin(2 * np.pi * doy / 365.0 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1.0, 1.0))
    ra = (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))
    return 0.0023 * (tmean + 17.8) * np.sqrt(np.maximum(tmax - tmin, 0.0)) * ra * 0.408


def soil_strength(df, c):
    """Soil moisture + rain -> wetness index (SWI) -> rating cone index (RCI, psi)."""
    df = df.copy()
    span = c["theta_sat"] - c["theta_dry"]
    df["swi"] = (c["w_surf"] * ((df["sm_surf"] - c["theta_dry"]) / span).clip(0, 1)
                 + c["w_deep"] * ((df["sm_deep"] - c["theta_dry"]) / span).clip(0, 1))
    api3 = df["rain"].rolling(3, min_periods=1).sum()
    rain = 1.0 - c["rain_max"] * np.minimum(api3 / c["rain_scale"], 1.0)
    df["rci"] = c["rci_dry"] * np.exp(-c["k"] * df["swi"]) * rain
    ti = 100.0 * (np.log(df["rci"]) - np.log(c["rci_sat"])) / (np.log(c["rci_dry"]) - np.log(c["rci_sat"]))
    df["ti"] = ti.clip(0, 100)
    return df


def _lags(out, s, name, lags):
    for l in lags:
        out[f"{name}_lag{l}"] = s.shift(l)


def _rolls(out, s, name, wins, stats=("mean",)):
    for w in wins:
        r = s.shift(1).rolling(w, min_periods=max(2, w // 2))
        for st in ("mean", "std", "min", "max"):
            if st in stats:
                out[f"{name}_r{st}{w}"] = getattr(r, st)()
        if "sum" in stats:
            out[f"{name}_rsum{w}"] = s.shift(1).rolling(w, min_periods=1).sum()


def build_features(d, lat_deg):
    """Strictly causal features: row t uses observations up to day t only."""
    d = d.copy()
    doy = d["Date"].dt.dayofyear.values
    out = {"Date": d["Date"]}
    for k in (1, 2, 3):
        out[f"doy_sin{k}"] = np.sin(2 * k * np.pi * doy / 365.25)
        out[f"doy_cos{k}"] = np.cos(2 * k * np.pi * doy / 365.25)
    out["monsoon"] = d["Date"].dt.month.isin([6, 7, 8, 9]).astype(float)
    d["et0"] = _hargreaves(d["tmax"].values, d["tmin"].values, d["tmean"].values, doy, lat_deg)
    d["wbal"] = d["rain"] - d["et0"]

    _lags(out, d["rci"], "rci", [1, 2, 3, 4, 5, 7, 10, 14, 21, 28])
    _rolls(out, d["rci"], "rci", [3, 7, 14, 30], stats=("mean", "std", "min", "max"))
    out["rci_d1"] = d["rci"].shift(1) - d["rci"].shift(2)
    out["rci_d3"] = d["rci"].shift(1) - d["rci"].shift(4)
    out["rci_d7"] = d["rci"].shift(1) - d["rci"].shift(8)
    for col in ["sm_surf", "sm_deep"]:
        _lags(out, d[col], col, [1, 2, 3, 5, 7, 14])
        _rolls(out, d[col], col, [7, 30])
    out["sm_grad"] = d["sm_surf"].shift(1) - d["sm_deep"].shift(1)

    _lags(out, d["rain"], "rain", [1, 2, 3, 4, 5, 6, 7])
    _rolls(out, d["rain"], "rain", [3, 7, 14, 30, 60], stats=("sum",))
    out["rain_max7"] = d["rain"].shift(1).rolling(7, min_periods=1).max()
    r = d["rain"].values
    for k in (0.85, 0.95):
        api = np.zeros(len(d))
        for i in range(1, len(d)):
            api[i] = k * api[i - 1] + r[i - 1]
        out[f"api_{int(k * 100)}"] = api
    wet = (d["rain"] > 1.0).astype(int)
    dsr, w = np.zeros(len(d)), wet.values
    for i in range(1, len(d)):
        dsr[i] = 0 if w[i - 1] else dsr[i - 1] + 1
    out["days_since_rain"] = dsr
    out["wet_days_30"] = wet.shift(1).rolling(30, min_periods=1).sum()

    for col in ["et0", "tmax", "tmin", "tmean", "rh", "rh_min", "wind", "wind_max", "pres", "cloud"]:
        _lags(out, d[col], col, [1, 3, 7])
        _rolls(out, d[col], col, [7, 30])
    _rolls(out, d["et0"], "et0", [7, 14, 30], stats=("sum",))
    _rolls(out, d["wbal"], "wbal", [7, 14, 30], stats=("sum",))
    out["wdir_sin"] = np.sin(np.deg2rad(d["wdir"].shift(1)))
    out["wdir_cos"] = np.cos(np.deg2rad(d["wdir"].shift(1)))
    for c in ["rci", "ti", "swi", "sm_surf", "sm_deep", "rain", "et0", "wbal", "tmax",
              "tmin", "tmean", "rh", "rh_min", "wind", "wind_max", "pres", "cloud"]:
        out[f"raw_{c}"] = d[c]
    return pd.DataFrame(out, index=d.index)


# ======================================================================= forecaster
class _NumpyGRU:
    """GRU (2 layers) + MLP head in plain numpy, same equations as torch.nn.GRU."""

    def __init__(self, sd, layers):
        self.sd, self.layers = sd, layers

    def __call__(self, x):                       # x: (B, L, n_in) float32
        h_seq = x
        for k in range(self.layers):
            Wi, Wh = self.sd[f"rnn.weight_ih_l{k}"], self.sd[f"rnn.weight_hh_l{k}"]
            bi, bh = self.sd[f"rnn.bias_ih_l{k}"], self.sd[f"rnn.bias_hh_l{k}"]
            H = Wh.shape[1]
            gi = h_seq @ Wi.T + bi               # input part for all steps at once
            h = np.zeros((x.shape[0], H), np.float32)
            outs = []
            for t in range(x.shape[1]):
                gh = h @ Wh.T + bh
                r = 1 / (1 + np.exp(-(gi[:, t, :H] + gh[:, :H])))
                z = 1 / (1 + np.exp(-(gi[:, t, H:2 * H] + gh[:, H:2 * H])))
                n = np.tanh(gi[:, t, 2 * H:] + r * gh[:, 2 * H:])
                h = (1 - z) * n + z * h
                outs.append(h)
            h_seq = np.stack(outs, 1)
        y = np.maximum(h_seq[:, -1] @ self.sd["head.1.weight"].T + self.sd["head.1.bias"], 0)
        return y @ self.sd["head.3.weight"].T + self.sd["head.3.bias"]


class Forecaster:
    def __init__(self, path, backend="auto"):
        self.p = load_pickle(path)
        p = self.p
        self.L, self.H, self.window = p["L"], p["H"], p["window_days"]
        self.backend = backend
        if backend in ("auto", "torch"):
            try:
                import torch
                torch.set_num_threads(1)
                self._torch = torch
                self.nets = [self._torch_net(sd) for sd in p["gru_states"]]
                self.backend = "torch"
            except ImportError:
                if backend == "torch":
                    raise
                self.backend = "numpy"
        if self.backend == "numpy":
            self.nets = [_NumpyGRU(sd, p["gru_cfg"]["layers"]) for sd in p["gru_states"]]
        self.vehicles = pd.DataFrame(p["vehicles"])

    def sample_log(self):
        """The 400-day weather log stored in the pickle (ends 31 Dec 2025)."""
        return pd.DataFrame(self.p["sample_log"])

    def _torch_net(self, sd):
        torch, c = self._torch, self.p["gru_cfg"]
        nn = torch.nn

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.rnn = nn.GRU(c["n_in"], c["hidden"], c["layers"], batch_first=True,
                               dropout=c["dropout"])
                s.head = nn.Sequential(nn.Dropout(c["dropout"]), nn.Linear(c["hidden"], c["hidden"]),
                                       nn.ReLU(), nn.Linear(c["hidden"], c["H"]))

            def forward(s, x):
                out, _ = s.rnn(x)
                return s.head(out[:, -1])

        net = Net()
        net.load_state_dict({k: torch.from_numpy(v) for k, v in sd.items()})
        return net.eval()

    # ------------------------------------------------------------------ pieces
    def features(self, log, all_days=False):
        """-> dates, today's RCI, tabular rows, normalised GRU windows (one per origin)."""
        p = self.p
        raw = soil_strength(read_log(log, None if all_days else self.window), p["rci"])
        f = build_features(raw, p["lat_deg"])
        Xtab = f[p["tab_cols"]].values.astype(np.float64)
        Xseq = ((f[p["seq_cols"]].values - p["seq_mu"]) / p["seq_sd"]).astype(np.float32)
        ok = ~np.isnan(Xtab).any(axis=1)
        ok[: self.L - 1] = False
        origins = np.where(ok)[0] if all_days else np.array([len(f) - 1])
        if not ok[origins].all():
            raise ValueError(f"need {self.window} days of log (at least {self.L} + 60)")
        win = np.stack([Xseq[t - self.L + 1: t + 1] for t in origins])
        return raw["Date"].values[origins], raw["rci"].values[origins], Xtab[origins], win

    def ridge(self, Xtab):
        p = self.p
        Z = (Xtab - p["ridge_mean"]) / p["ridge_scale"]
        return Z @ p["ridge_coef"].T + p["ridge_intercept"]

    def gru(self, win):
        p = self.p
        if self.backend == "torch":
            with self._torch.no_grad():
                x = self._torch.from_numpy(win)
                z = np.mean([np.concatenate([n(x[i:i + 512]).numpy() for i in range(0, len(x), 512)])
                             for n in self.nets], axis=0)
        else:
            z = np.mean([np.concatenate([n(win[i:i + 512]) for i in range(0, len(win), 512)])
                         for n in self.nets], axis=0)
        return z * p["dsd"] + p["dmu"]

    # ------------------------------------------------------------------ public
    def forecast(self, log):
        """Forecast from the last day of the log. -> dict(date, rci_today, rci[H])."""
        dates, today, Xtab, win = self.features(log)
        rci = today[:, None] + (self.gru(win) + self.ridge(Xtab)) / 2
        return dict(date=pd.Timestamp(dates[0]), rci_today=float(today[0]), rci=rci[0])

    def forecast_all(self, log):
        """Forecast from every day of a long log (batched). -> DataFrame."""
        dates, today, Xtab, win = self.features(log, all_days=True)
        rci = today[:, None] + (self.gru(win) + self.ridge(Xtab)) / 2
        out = pd.DataFrame({"date": dates, "rci_today": today})
        for h in range(self.H):
            out[f"rci_h{h + 1}"] = rci[:, h]
        return out

    def band_of(self, vci1):
        for name, lo, hi in self.p["bands"]:
            if lo <= vci1 < hi:
                return name
        raise ValueError(vci1)

    def band_vehicle(self, band):
        """The band's planning vehicle: majority type, median VCI1 / VCI50."""
        return self.p["band_vehicles"][band]

    def margin(self, band, h):
        return self.p["margins"][band][h - 1]

    def assess(self, fc, band=None, vehicle=None, vci1=None, vci50=None):
        """GO / SLOW-GO / NO-GO for each of the next H days, after the band's margin."""
        if vehicle is not None:
            r = self.vehicles.set_index("name").loc[vehicle]
            vci1, vci50 = float(r.vci1), float(r.vci50)
        elif band is not None:
            v = self.band_vehicle(band)
            vci1, vci50 = v["vci1"], v["vci50"]
        b = self.band_of(vci1)
        safe = fc["rci"] - np.array(self.p["margins"][b])
        cls = np.where(safe >= vci50, 2, np.where(safe >= vci1, 1, 0))
        return [CLASSES[c] for c in cls]

    def outlook(self, fc, band=None, vehicle=None):
        """Per day 1..H: status, firmness after the band's margin, passes one track carries
        (50+ is GO) and level-ground speed off road (km/h, before land cover)."""
        if vehicle is not None:
            r = self.vehicles.set_index("name").loc[vehicle]
            v = dict(kind=r.type, vci1=float(r.vci1), vci50=float(r.vci50), cpf=float(r.cpf),
                     hp_per_t=float(r.hp / r.mass_t), vmax_kmh=float(r.vmax_kmh))
        else:
            v = self.band_vehicle(band)
        safe = fc["rci"] - np.array(self.p["margins"][self.band_of(v["vci1"])])
        status = self.assess(fc, vci1=v["vci1"], vci50=v["vci50"])
        n, sp = passes(safe, v["vci1"], v["vci50"]), speed(safe, 0.0, v)
        return [dict(day=h + 1, status=status[h], rci=round(float(safe[h]), 1),
                     passes=None if status[h] == "NO-GO" else ("50+" if n[h] >= 50 else int(n[h])),
                     speed_kmh=round(float(sp[h]), 1)) for h in range(self.H)]

    @staticmethod
    def message(fc):
        """The forecast as broadcast to a convoy: 4-byte day number + H x float16."""
        day = int((fc["date"] - pd.Timestamp("1970-01-01")).days)
        return struct.pack("<i", day) + np.asarray(fc["rci"], "<f2").tobytes()

    @staticmethod
    def from_message(b):
        day, = struct.unpack("<i", b[:4])
        return dict(date=pd.Timestamp("1970-01-01") + pd.Timedelta(days=day),
                    rci=np.frombuffer(b[4:], "<f2").astype(np.float64))


# ========================================================================== planner
_NBRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class Planner:
    def __init__(self, path):
        self.p = load_pickle(path)
        p = self.p
        self.shape = tuple(p["shape"])
        self.n = self.shape[0] * self.shape[1]
        self.paved, self.decay = p["paved"], p["decay"]
        r, c = np.indices(self.shape)
        self._edges = []
        for dr, dc in _NBRS:
            ok = (r + dr >= 0) & (r + dr < self.shape[0]) & (c + dc >= 0) & (c + dc < self.shape[1])
            u = (r * self.shape[1] + c)[ok]
            v = ((r + dr) * self.shape[1] + (c + dc))[ok]
            self._edges.append((u, v, p["cell_m"] / 1000 * np.hypot(dr, dc)))
        self._z = p["elevation"].ravel()
        self._sin = [(self._z[v] - self._z[u]) / np.hypot(self._z[v] - self._z[u], step * 1000)
                     for u, v, step in self._edges]

    def sample_trips(self):
        """30 trips of 8-25 km between road cells, as (source cell, destination cell)."""
        return [tuple(t) for t in self.p["sample_trips"]]

    def soil(self, rci, vci1, vci50, passes=None):
        """Soil membership per cell: 0 below VCI1 (NO-GO), SLOW_FLOOR..1 up to VCI50.
        passes: earlier vehicles over each cell (convoy wear) raise the threshold."""
        rc = rci * self.decay
        need = np.full(self.shape, vci1)
        if passes is not None:
            need = vci1 + (vci50 - vci1) * np.log(passes + 1.0) / np.log(50.0)
        f = self.p["slow_floor"]
        mu = f + (1 - f) * np.clip((rc - need) / np.maximum(vci50 - need, 1e-6), 0, 1)
        mu = np.where(rc < need, 0.0, mu)
        return np.where(self.paved, 1.0, mu)

    def passability(self, kind, mu_soil):
        """Fuzzy AND (minimum) of slope, land cover and soil."""
        s, c = self.p["static"]["wheeled" if kind == "halftrack" else kind]
        return np.minimum(np.minimum(s, c), mu_soil)

    def graph(self, mu):
        f = mu.ravel()
        us, vs, ws = [], [], []
        for u, v, step in self._edges:
            ok = (f[u] > 0) & (f[v] > 0)
            us.append(u[ok]); vs.append(v[ok])
            ws.append(step * 0.5 * (1 / f[u[ok]] + 1 / f[v[ok]]))
        return csr_matrix((np.concatenate(ws), (np.concatenate(us), np.concatenate(vs))),
                          shape=(self.n, self.n))

    def routes(self, mu, trips, G=None):
        """Fastest route for each (src, dst) cell pair; None if no route exists.
        One Dijkstra per distinct source."""
        G = self.graph(mu) if G is None else G
        srcs = sorted({s for s, _ in trips})
        dist, pred = dijkstra(G, directed=True, indices=srcs, return_predecessors=True)
        row = {s: i for i, s in enumerate(srcs)}
        out = []
        for s, d in trips:
            i = row[s]
            if not np.isfinite(dist[i, d]):
                out.append(None)
                continue
            path, k = [d], d
            while k != s:
                k = pred[i, k]
                path.append(k)
            out.append(np.array(path[::-1]))
        return out

    def path_time(self, path, mu):
        """Travel time in hours-equivalent units (km / speed fraction)."""
        f = mu.ravel()[path]
        if (f <= 0).any():
            return np.inf
        r, c = np.divmod(path, self.shape[1])
        step = self.p["cell_m"] / 1000 * np.hypot(np.diff(r), np.diff(c))
        return float(np.sum(step * 0.5 * (1 / f[:-1] + 1 / f[1:])))

    # ------------------------------------------------- travel time (speed model)
    def _cell_speed(self, cells, sin_t, rci, v, passes_=None):
        """km/h per cell: soil/slope speed (paved roads ignore the soil) times the
        land-cover score off road; 0 where slope or cover rule the cell out."""
        s, c = (a.ravel()[cells] for a in self.p["static"]["wheeled" if v["kind"] == "halftrack" else v["kind"]])
        need = None
        if passes_ is not None:
            need = v["vci1"] + (v["vci50"] - v["vci1"]) * np.log(passes_.ravel()[cells] + 1.0) / np.log(50.0)
        sp = speed(rci * self.decay.ravel()[cells], sin_t, v, self.paved.ravel()[cells], need)
        return np.where(s > 0, sp * c, 0.0)

    def hours_graph(self, rci, v, passes_=None):
        us, vs, ws = [], [], []
        for (u, w, step), sin_t in zip(self._edges, self._sin):
            a = self._cell_speed(u, sin_t, rci, v, passes_)
            b = self._cell_speed(w, sin_t, rci, v, passes_)
            ok = (a > 0) & (b > 0)
            us.append(u[ok]); vs.append(w[ok])
            ws.append(step * 0.5 * (1 / a[ok] + 1 / b[ok]))
        return csr_matrix((np.concatenate(ws), (np.concatenate(us), np.concatenate(vs))),
                          shape=(self.n, self.n))

    def path_hours(self, path, rci, band, passes_=None):
        """Travel time (hours) of a path for the band's vehicle; inf if a stretch is
        impassable (NO-GO soil, or a climb the soil cannot carry it up)."""
        v = self.p["band_vehicles"][band]
        r, c = np.divmod(path, self.shape[1])
        step = self.p["cell_m"] / 1000 * np.hypot(np.diff(r), np.diff(c))
        z = self._z[path]
        sin_t = np.diff(z) / np.hypot(np.diff(z), step * 1000)
        a = self._cell_speed(path[:-1], sin_t, rci, v, passes_)
        b = self._cell_speed(path[1:], sin_t, rci, v, passes_)
        if (a <= 0).any() or (b <= 0).any():
            return np.inf
        return float(np.sum(step * 0.5 * (1 / a + 1 / b)))

    def plan(self, rci, band, trips, by="passability"):
        """Routes for a band's planning vehicle on ground of firmness `rci` (psi, already
        net of the safety margin). by="passability": the published planner, time in
        km / speed-fraction units; by="time": routes minimising travel time, in hours.
        -> list of (path or None, time)."""
        v = self.p["band_vehicles"][band]
        if by == "time":
            return [(p, None if p is None else self.path_hours(p, rci, band))
                    for p in self.routes(None, trips, self.hours_graph(rci, v))]
        mu = self.passability(v["kind"], self.soil(rci, v["vci1"], v["vci50"]))
        return [(p, None if p is None else self.path_time(p, mu)) for p in self.routes(mu, trips)]

    def convoy(self, rci, band, src, dst, n=10, wear_aware=True):
        """n vehicles from src to dst. wear_aware: each re-plans on the shared pass-count
        map (4 B per cell of track broadcast); else all follow the leader's route.
        -> vehicles through, bytes broadcast."""
        v = self.p["band_vehicles"][band]
        passes = np.zeros(self.shape)
        flat = passes.ravel()
        stage = np.zeros(self.shape, bool)
        for p in (src, dst):
            r, c = divmod(int(p), self.shape[1])
            stage[max(r - 1, 0): r + 2, max(c - 1, 0): c + 2] = True
        stage = stage.ravel()
        through, shared, leader = 0, 0, None
        for _ in range(n):
            mu = self.passability(v["kind"], self.soil(rci, v["vci1"], v["vci50"], passes))
            if not wear_aware and leader is not None:
                path = leader
            else:
                path = self.routes(mu, [(src, dst)])[0]
                if path is None:
                    break
                leader = path if leader is None else leader
                shared += 4 * len(path) if wear_aware else 0
            if not np.isfinite(self.path_time(path, mu)):
                if not wear_aware:
                    break
                continue
            through += 1
            np.add.at(flat, path[~stage[path]], 1)
        return through, shared


def end_to_end(fc_model: Forecaster, planner: Planner, log, band, trips, h, by="passability"):
    """Weather log -> forecast -> margin -> map -> routes for travel in h days."""
    fc = fc_model.forecast(log)
    rci = float(fc["rci"][h - 1]) - fc_model.margin(band, h)
    return fc, rci, planner.plan(rci, band, trips, by=by)


# ============================================================================ demo
if __name__ == "__main__":
    import argparse
    import time
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Run both pickles once on their sample inputs.")
    ap.add_argument("--backend", choices=["auto", "torch", "numpy"], default="auto")
    ap.add_argument("--band", default="D")
    ap.add_argument("--h", type=int, default=3, help="days until travel (1-7)")
    a = ap.parse_args()
    here = Path(__file__).resolve().parent

    t = time.perf_counter()
    fc_model = Forecaster(here / "vehicle_trafficability_forecaster.pkl", backend=a.backend)
    planner = Planner(here / "vehicle_trafficability_planner.pkl")
    load_ms = (time.perf_counter() - t) * 1000
    log, trips = fc_model.sample_log(), planner.sample_trips()

    t = time.perf_counter()
    f = fc_model.forecast(log)
    fc_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    f, rci, res = end_to_end(fc_model, planner, log, a.band, trips, a.h)
    e2e_ms = (time.perf_counter() - t) * 1000

    print(f"backend {fc_model.backend}; loaded both pickles in {load_ms:.0f} ms")
    print(f"forecast from {f['date'].date()}: RCI today {f['rci_today']:.1f} psi, days 1-7: "
          + " ".join(f"{x:.1f}" for x in f["rci"]) + f"   ({fc_ms:.0f} ms)")
    for b in "ABCDE":
        print(f"  band {b}: " + " ".join(s_[:2] for s_ in fc_model.assess(f, band=b)) + "   (GO / SL / NO, days 1-7)")
    print(f"band {a.band} outlook (passes per track, level-ground km/h):")
    for o in fc_model.outlook(f, band=a.band):
        print(f"  day {o['day']}: {o['status']:7s} RCI {o['rci']:5.1f}  passes {o['passes']}  {o['speed_kmh']} km/h")
    ok = [p_ for p_, _ in res if p_ is not None]
    print(f"band {a.band}, travel in {a.h} days: RCI after margin {rci:.1f} psi; "
          f"{len(ok)}/{len(trips)} trips have a route; mean {np.mean([len(p_) for p_ in ok]) if ok else 0:.0f} cells")
    tm = [t_ for p_, t_ in planner.plan(rci, a.band, trips, by="time") if p_ is not None]
    if tm:
        print(f"  routed by travel time: mean {60 * np.mean(tm):.0f} min per trip")
    print(f"end to end (log -> forecast -> map -> {len(trips)} routes): {e2e_ms:.0f} ms")
    print(f"forecast message: {len(Forecaster.message(f))} bytes")
