"""
Data loading, soil-strength (RCI) target construction, and feature engineering
for off-road military vehicle trafficability forecasting at Pachmarhi.

Source record: data/weather/pachmarhi_daily_2015_2025.csv (4018 daily rows, 2015-01-01..2025-12-31).
See data/weather/metadata_sheet.csv for provenance of each column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from traffic import paths

# ----------------------------------------------------------------------------- site
LAT_DEG = 22.4675          # Pachmarhi, Satpura plateau
ELEV_M = 1067

# --------------------------------------------------------------- soil-strength model
# Volumetric water-content envelope taken from the ERA5-Land record itself:
# observed range is 20.70..48.85 vol% across both layers over 11 years.
# THETA_DRY is treated as the residual / near-wilting end, THETA_SAT as porosity.
THETA_DRY = 20.0
THETA_SAT = 49.0

# Rating Cone Index of this fine-grained plateau soil when air-dry and the exponential
# decay constant that drives it down to RCI_SAT at full saturation. Units are the
# psi-convention cone-index units the NRMM vehicle cone index uses (not kPa).
# RCI_SAT was 25 until 2026-10-11. That floor sat above the one-pass VCI of most light
# and medium tracked vehicles, so they could never be NO-GO. 15 follows the wettest
# field evidence available: Rosca et al. (2018) measured RCI 14-25 on natural
# fine-grained lanes where a 4x4 immobilised.
RCI_DRY = 300.0
RCI_SAT = 15.0
K_DECAY = float(np.log(RCI_DRY / RCI_SAT))     # ~2.996

# Critical layer for vehicle sinkage is roughly 15-30 cm, so the 7-28 cm layer
# carries most of the weight; the 0-7 cm skin still governs initial shear.
W_SURF, W_DEEP = 0.35, 0.65

# Free-water / mud-film penalty from recent rainfall, over and above what the
# daily-mean soil moisture already reflects. Caps at a 15% strength reduction.
RAIN_PENALTY_MAX = 0.15
RAIN_PENALTY_SCALE = 50.0      # mm of 3-day antecedent rainfall for full penalty

# --------------------------------------------------------------- vehicle definitions
# VCI1 / VCI50 = vehicle cone index for 1 and 50 passes (kPa).
# GO:      RCI >= VCI50   (sustained traffic, 50 passes)
# SLOW-GO: VCI1 <= RCI < VCI50   (a few passes; ruts, reduced speed)
# NO-GO:   RCI < VCI1     (immobilisation on the first pass)
VEHICLES = {
    "tracked_light": dict(label="Tracked light (ICV, ~14 t)", vci1=20.0, vci50=30.0),
    "tracked_heavy": dict(label="Tracked heavy (MBT, ~46 t)", vci1=30.0, vci50=45.0),
    "wheeled_6x6": dict(label="Wheeled 6x6 logistics truck", vci1=40.0, vci50=60.0),
}

TARGET = "rci"
CLASS_NAMES = ["NO-GO", "SLOW-GO", "GO"]

# ----------------------------------------------------------------------- date splits
TRAIN_END = "2021-12-31"
VAL_END = "2023-12-31"      # test is everything after


def hargreaves_et0(tmax, tmin, tmean, doy, lat_deg=LAT_DEG):
    """Reference evapotranspiration, mm/day (FAO-56 Hargreaves)."""
    phi = np.deg2rad(lat_deg)
    dr = 1.0 + 0.033 * np.cos(2 * np.pi * doy / 365.0)
    dec = 0.409 * np.sin(2 * np.pi * doy / 365.0 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1.0, 1.0))
    ra = (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws)
    )
    return 0.0023 * (tmean + 17.8) * np.sqrt(np.maximum(tmax - tmin, 0.0)) * ra * 0.408


def load_raw(path=paths.WEATHER_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.rename(columns={
        "Rainfall": "rain", "Tmax": "tmax", "Tmin": "tmin", "Tmean": "tmean",
        "Humidity": "rh", "Humidity Min": "rh_min", "Wind Speed": "wind",
        "Wind Speed Max": "wind_max", "Wind Direction": "wdir",
        "Pressure": "pres", "Soil Moisture": "sm_surf",
        "Soil Moisture Deep": "sm_deep", "Cloud Cover": "cloud",
        "Rainfall Source": "rain_source",
    })
    assert df["Date"].diff().dt.days.dropna().eq(1).all(), "date gaps in record"
    return df


def add_soil_strength(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the trafficability target chain: SWI -> RCI -> class per vehicle."""
    df = df.copy()
    span = THETA_SAT - THETA_DRY
    df["swi_surf"] = ((df["sm_surf"] - THETA_DRY) / span).clip(0, 1)
    df["swi_deep"] = ((df["sm_deep"] - THETA_DRY) / span).clip(0, 1)
    df["swi"] = W_SURF * df["swi_surf"] + W_DEEP * df["swi_deep"]

    api3 = df["rain"].rolling(3, min_periods=1).sum()
    rain_factor = 1.0 - RAIN_PENALTY_MAX * np.minimum(api3 / RAIN_PENALTY_SCALE, 1.0)

    df["rci"] = RCI_DRY * np.exp(-K_DECAY * df["swi"]) * rain_factor
    # 0..100 index, monotone in RCI, handy for reporting
    df["ti"] = 100.0 * (np.log(df["rci"]) - np.log(RCI_SAT)) / (np.log(RCI_DRY) - np.log(RCI_SAT))
    df["ti"] = df["ti"].clip(0, 100)

    for key, spec in VEHICLES.items():
        cls = np.where(df["rci"] >= spec["vci50"], 2,
                       np.where(df["rci"] >= spec["vci1"], 1, 0))
        df[f"class_{key}"] = cls.astype(np.int8)
    return df


def rci_to_class(rci, vehicle: str) -> np.ndarray:
    spec = VEHICLES[vehicle]
    rci = np.asarray(rci, dtype=float)
    return np.where(rci >= spec["vci50"], 2, np.where(rci >= spec["vci1"], 1, 0)).astype(np.int8)


# ------------------------------------------------------------------ feature building
def _lags(out, s, name, lags):
    for l in lags:
        out[f"{name}_lag{l}"] = s.shift(l)


def _rolls(out, s, name, wins, stats=("mean",)):
    for w in wins:
        r = s.shift(1).rolling(w, min_periods=max(2, w // 2))
        if "mean" in stats:
            out[f"{name}_rmean{w}"] = r.mean()
        if "std" in stats:
            out[f"{name}_rstd{w}"] = r.std()
        if "min" in stats:
            out[f"{name}_rmin{w}"] = r.min()
        if "max" in stats:
            out[f"{name}_rmax{w}"] = r.max()
        if "sum" in stats:
            out[f"{name}_rsum{w}"] = s.shift(1).rolling(w, min_periods=1).sum()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strictly causal features. Every column at row t uses only observations at
    time <= t, matching an operational setting with no numerical weather forecast
    available. Calendar/Fourier terms are the only genuinely known-future inputs.
    """
    d = df.copy()
    doy = d["Date"].dt.dayofyear.values
    # accumulate in a dict and build the frame once -- 128 incremental inserts into a
    # DataFrame fragments its block manager and is markedly slower
    out: dict = {}
    out["Date"] = d["Date"]

    # --- calendar / annual cycle (known in the future)
    out["doy_sin1"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos1"] = np.cos(2 * np.pi * doy / 365.25)
    out["doy_sin2"] = np.sin(4 * np.pi * doy / 365.25)
    out["doy_cos2"] = np.cos(4 * np.pi * doy / 365.25)
    out["doy_sin3"] = np.sin(6 * np.pi * doy / 365.25)
    out["doy_cos3"] = np.cos(6 * np.pi * doy / 365.25)
    out["monsoon"] = d["Date"].dt.month.isin([6, 7, 8, 9]).astype(float)

    # --- evaporative demand and water balance
    d["et0"] = hargreaves_et0(d["tmax"].values, d["tmin"].values, d["tmean"].values, doy)
    d["wbal"] = d["rain"] - d["et0"]

    # --- target history (dominant signal: lag-1 autocorrelation is 0.99)
    _lags(out, d["rci"], "rci", [1, 2, 3, 4, 5, 7, 10, 14, 21, 28])
    _rolls(out, d["rci"], "rci", [3, 7, 14, 30], stats=("mean", "std", "min", "max"))
    out["rci_d1"] = d["rci"].shift(1) - d["rci"].shift(2)
    out["rci_d3"] = d["rci"].shift(1) - d["rci"].shift(4)
    out["rci_d7"] = d["rci"].shift(1) - d["rci"].shift(8)

    for col in ["sm_surf", "sm_deep"]:
        _lags(out, d[col], col, [1, 2, 3, 5, 7, 14])
        _rolls(out, d[col], col, [7, 30], stats=("mean",))
    out["sm_grad"] = d["sm_surf"].shift(1) - d["sm_deep"].shift(1)

    # --- rainfall forcing
    _lags(out, d["rain"], "rain", [1, 2, 3, 4, 5, 6, 7])
    _rolls(out, d["rain"], "rain", [3, 7, 14, 30, 60], stats=("sum",))
    out["rain_max7"] = d["rain"].shift(1).rolling(7, min_periods=1).max()
    # antecedent precipitation index, exponential memory
    for k in (0.85, 0.95):
        api = np.zeros(len(d))
        r = d["rain"].values
        for i in range(1, len(d)):
            api[i] = k * api[i - 1] + r[i - 1]
        out[f"api_{int(k*100)}"] = api
    wet = (d["rain"] > 1.0).astype(int)
    dsr = np.zeros(len(d))
    for i in range(1, len(d)):
        dsr[i] = 0 if wet.iloc[i - 1] else dsr[i - 1] + 1
    out["days_since_rain"] = dsr
    out["wet_days_30"] = wet.shift(1).rolling(30, min_periods=1).sum()

    # --- drying forcing
    for col in ["et0", "tmax", "tmin", "tmean", "rh", "rh_min", "wind", "wind_max",
                "pres", "cloud"]:
        _lags(out, d[col], col, [1, 3, 7])
        _rolls(out, d[col], col, [7, 30], stats=("mean",))
    _rolls(out, d["et0"], "et0", [7, 14, 30], stats=("sum",))
    _rolls(out, d["wbal"], "wbal", [7, 14, 30], stats=("sum",))
    out["wdir_sin"] = np.sin(np.deg2rad(d["wdir"].shift(1)))
    out["wdir_cos"] = np.cos(np.deg2rad(d["wdir"].shift(1)))

    # --- keep contemporaneous raw channels for the sequence models
    keep = ["rci", "ti", "swi", "sm_surf", "sm_deep", "rain", "et0", "wbal", "tmax",
            "tmin", "tmean", "rh", "rh_min", "wind", "wind_max", "pres", "cloud"]
    for c in keep:
        out[f"raw_{c}"] = d[c]

    return pd.DataFrame(out, index=d.index)


FEATURE_PREFIX_EXCLUDE = ("Date", "raw_")


def tabular_columns(feats: pd.DataFrame) -> list[str]:
    """Columns usable by tabular models: everything already lagged/rolled."""
    return [c for c in feats.columns if not c.startswith(FEATURE_PREFIX_EXCLUDE)]


def sequence_columns(feats: pd.DataFrame) -> list[str]:
    """Channels fed to the sequence models as an L-day window ending at t."""
    return [c for c in feats.columns if c.startswith("raw_")] + [
        "doy_sin1", "doy_cos1", "doy_sin2", "doy_cos2", "doy_sin3", "doy_cos3", "monsoon",
        "api_85", "api_95", "days_since_rain",
    ]


KNOWN_FUTURE_COLS = ["doy_sin1", "doy_cos1", "doy_sin2", "doy_cos2",
                     "doy_sin3", "doy_cos3", "monsoon"]


def split_masks(dates: pd.Series):
    tr = dates <= pd.Timestamp(TRAIN_END)
    va = (dates > pd.Timestamp(TRAIN_END)) & (dates <= pd.Timestamp(VAL_END))
    te = dates > pd.Timestamp(VAL_END)
    return tr.values, va.values, te.values


def get_dataset(path=paths.WEATHER_CSV):
    raw = load_raw(path)
    raw = add_soil_strength(raw)
    feats = build_features(raw)
    return raw, feats
