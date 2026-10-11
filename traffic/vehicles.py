"""
Vehicle table: cleaning, wheeled/tracked classification, and soft-soil vehicle cone
index (VCI1, VCI50) from the NRMM / TOP 02-2-619A Mobility Index.

Source record: data/vehicles/army_vehicles_attributes.csv, a War Thunder game-file datamine. Only
mass and engine power are used. Masses are per-template values in the game files
(about 90 distinct values across ~1,250 rows), so a named vehicle's mass can be well
off its real figure; results should be reported by firmness band, not by name.

Formulas
  Wheeled  Rosca et al. (2018), J. Military Technology 1(2), doi:10.32754/JMT.2018.2.07
  Tracked  Ciobotaru (2009), Leonardo El. J. Pract. Technol. 8(15):19-30.
           Checked here against its M1 example: MI 98.59, VCI1 26.34, VCI50 60.48.
Cone-index values are in psi-convention units, the same units as data.RCI.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from traffic import paths

KG_TO_LB = 2.20462

# ------------------------------------------------------------------- cleaning rules
# game-only rows: scenery, wrecks, tutorials, event/cup skins, joke entries
DROP_PATTERNS = [
    r"railway", r"^space_", r"^santa", r"^aaa_dummy", r"destroyed", r"burn", r"damaged",
    r"tutorial", r"_race$", r"arena", r"yt_cup", r"_event$", r"kit_3rank", r"_promo$",
    r"animal_version", r"without_crew", r"snowball", r"bulldozer",
]
MASS_PLACEHOLDER_KG = 5500.0      # the game's default for (mostly wheeled) templates

# --------------------------------------------------------------- classification
# Anything not matched below is tracked: the file is dominated by tanks, SPGs, IFVs.
HALFTRACK = [
    r"sdkfz_251", r"sdkfz_6", r"sdkfz_9", r"halftrack", r"tcm_20", r"citroen_kegresse",
    r"zis_4[23]", r"panzerwerfer_42",
]
WHEELED = [
    r"btr_(?!zd)", r"brdm", r"boxer", r"vilkas", r"lav_ad", r"m1128", r"m1296", r"patria",
    r"centauro", r"freccia", r"vbci", r"vab_", r"vcac", r"vextra", r"aml_90", r"amx_10rc",
    r"jaguar_ebrc", r"panhard_ebr", r"amd_35", r"rooikat", r"ratel", r"eland", r"badger",
    r"fv721_fox", r"daimler", r"armored_car", r"marmon", r"sdkfz_22[12]", r"sdkfz_234",
    r"luchs", r"radpanzer", r"th_[48]00", r"_ab_4[13]", r"as_42", r"aubl", r"fiat_66",
    r"oto_r3", r"vbc_pt2", r"type_16", r"type_87_rcv", r"icv", r"m8_greyhound", r"t17e2",
    r"t18_e2", r"t55e1", r"ba_11", r"gaz_", r"zis_12", r"truck", r"studebaker", r"cckw",
    r"dodge", r"lancia", r"breda_52", r"fiat_cm52", r"tpk_641", r"breda_501", r"antelope",
    r"cm_34", r"ptl_02", r"zsl_92", r"ztl_11", r"wma_301", r"pgz_625", r"hj_9", r"eitan",
    r"_g6_", r"ystervark", r"bosvark", r"sarc", r"concept3", r"landsverk", r"pbil",
    r"tgdgb", r"csaba", r"m53_59", r"9a33", r"flarakrad", r"garford", r"pantsyr_s1$",
    r"bm_13n", r"bm_31", r"9k79", r"9a52", r"samp_t", r"nasams", r"sl_amraam", r"iris_slm",
    r"sky_sabre", r"spyder", r"tansam", r"chusam", r"jp_type_93", r"crotale", r"hummer",
    r"launcher_vertical",
]

# Placeholder-mass rows: approximate public combat masses (t) per base model, and axle
# count for wheeled ones. Rounded figures from open sources, not certified data; a row
# on the placeholder with no entry here is dropped.
MASS_FIX = [
    # pattern               mass_t  axles
    (r"btr_80a",             14.6, 4), (r"btr_82at",       15.4, 4), (r"btr_3e1",      16.4, 4),
    (r"boxer_crv",           38.0, 4), (r"boxer|vilkas",   34.0, 4), (r"lav_ad",       13.0, 4),
    (r"m1128",               18.8, 4), (r"m1296",          20.0, 4), (r"patria",       26.0, 4),
    (r"centauro_2",          30.0, 4), (r"centauro",       25.0, 4), (r"freccia",      28.0, 4),
    (r"vbci2",               32.0, 4), (r"vbci",           29.0, 4), (r"vab_|vcac",    14.0, 2),
    (r"vextra",              25.0, 4), (r"aml_90",          5.5, 2), (r"amx_10rc",     15.9, 3),
    (r"jaguar_ebrc",         25.0, 3), (r"amd_35",          8.2, 2), (r"sdkfz_234",    11.7, 4),
    (r"luchs",               19.5, 4), (r"radpanzer_90",   19.5, 4), (r"garford",       8.6, 2),
    (r"_ab_4[13]",            7.5, 2), (r"as_42",           6.0, 2), (r"aubl|fiat_6616", 8.0, 2),
    (r"fiat_6614",            8.5, 2), (r"oto_r3",          3.5, 2), (r"vbc_pt2",      29.0, 4),
    (r"csaba",                6.0, 2), (r"jp_type_16",     26.0, 4), (r"type_87_rcv",  15.0, 4),
    (r"m8_greyhound",         7.8, 3), (r"t17e2",          12.0, 2), (r"t18_e2",       26.8, 4),
    (r"eland",                6.0, 2), (r"fv721_fox",       6.1, 2), (r"daimler",       7.6, 2),
    (r"armored_car_aec",     12.7, 2), (r"ratel",          18.5, 3), (r"rooikat",      28.0, 4),
    (r"badger",              28.0, 4), (r"_g6_",           47.0, 3), (r"eitan",        35.0, 4),
    (r"ba_11",                8.1, 3), (r"gaz_|truck_gaz",  3.5, 2), (r"m53_59",       10.3, 3),
    (r"pantsyr_s1$",         30.0, 4), (r"9a33",           17.5, 3), (r"bm_13n|bm_31",  7.0, 3),
    (r"studebaker",           7.0, 3), (r"zsl_92",         15.0, 3), (r"ptl_02",       19.0, 3),
    (r"ztl_11",              22.0, 4), (r"cm_34",          26.0, 4), (r"pbil_m40",      7.8, 2),
    (r"jp_type_93",           3.5, 2), (r"icv",            20.0, 4),
    # tracked rows that also carry the placeholder
    (r"flakpanzer_i_",        5.5, 0), (r"panzerjager_1",   6.4, 0), (r"lorraine_37l",  6.1, 0),
    (r"olifant_mk_2",        58.0, 0), (r"chaparral",      12.6, 0),
]

# 8x8 / 6x6 names for wheeled rows whose mass came from the file (no axle count there)
AXLES_BY_NAME = [(r"panhard_ebr", 4), (r"btr_152|cckw|zis_12|lancia|breda_52|fiat_cm52|"
                                       r"tpk_641|9k79", 3), (r"9a52", 4)]

# ------------------------------------------------- assumed running gear (inches)
# Tracked: interpolated on combat mass through open-source reference vehicles
# (M22 Locust, M113, M2 Bradley, T-72, M1, Tiger II-class). Shoe pitch ~6.5 in.
_TR_MASS = np.array([5.0, 12.0, 27.0, 41.0, 54.0, 70.0])
_TR_WIDTH = np.array([10.0, 15.0, 21.0, 23.0, 25.0, 30.0])
_TR_LENGTH = np.array([80.0, 105.0, 150.0, 168.0, 180.0, 165.0])
_TR_WHEELS = np.array([8, 10, 12, 12, 14, 18], dtype=float)
_TR_CLEAR = np.array([13.0, 16.0, 18.0, 18.5, 19.0, 19.0])
SHOE_PITCH_IN = 6.5

# Wheeled: default axle count by gross mass, then tyre outside diameter / section width
# / clearance by axle load, since tyre size follows the load each axle carries
_AXLE_T = [2.5, 4.5, 7.0, 10.0]                       # axle-load class bounds, t
_TYRES = [(37.0, 12.5, 16.0),      # 37x12.5R16.5   HMMWV class
          (42.0, 13.0, 16.0),      # 13.00-18       BRDM / light armoured car
          (48.0, 14.0, 18.0),      # 14.00-20       BTR / medium truck
          (54.0, 16.0, 18.0),      # 16.00R20       Boxer / heavy 8x8
          (63.0, 21.0, 18.0)]      # 21.00R25       G6 / heavy launcher


def _default_axles(mass_t):
    return 2 if mass_t < 8 else (3 if mass_t < 16 else 4)


def _tyre(axle_t):
    d, b, hc = _TYRES[int(np.searchsorted(_AXLE_T, axle_t))]
    return dict(d_in=d, b_in=b, clear_in=hc)


# Half-track: no standard Mobility Index relation. The track unit is rated with the
# tracked formula; the steered front axle is rated as a wheeled axle carrying this
# share of the weight; the worse of the two governs.
HALFTRACK_FRONT_SHARE = 0.25


# -------------------------------------------------------------- Mobility Index
def _wf_wheeled(axle_lb):
    """Weight factor on average axle load (lb), NRMM piecewise fit."""
    k = axle_lb / 1000.0
    if axle_lb < 2000:
        return 0.553 * k
    if axle_lb < 13500:
        return 0.033 * k + 1.050          # the range Rosca et al. (2018) state
    if axle_lb < 20000:
        return 0.142 * k - 0.420
    return 0.278 * k - 3.115


def _wf_tracked(gross_lb):
    if gross_lb < 50000:
        return 1.0
    if gross_lb < 70000:
        return 1.2
    if gross_lb < 100000:
        return 1.4
    return 1.8


def _ef(mass_t, hp):
    return 1.05 if hp / mass_t < 10.0 else 1.0


def mi_tracked(mass_t, hp, width_in, length_in, wheels, clear_in,
               shoe_area_in2=None, grouser_ge_1p5=False, manual=False):
    w = mass_t * 1000 * KG_TO_LB
    shoe = shoe_area_in2 if shoe_area_in2 is not None else width_in * SHOE_PITCH_IN
    cpf = w / (2 * length_in * width_in)              # two tracks
    tef = width_in / 100.0
    gf = 1.1 if grouser_ge_1p5 else 1.0
    bf = w / (10.0 * wheels * shoe)                   # bogie (wheel-load) factor
    cf = clear_in / 10.0
    tf = 1.05 if manual else 1.0
    return (cpf * _wf_tracked(w) / (tef * gf) + bf - cf) * _ef(mass_t, hp) * tf


def mi_wheeled(mass_t, hp, axles, d_in, b_in, clear_in,
               tyres_per_axle=2, chains=False, manual=False, w_axle=None):
    if w_axle is None:
        w_axle = mass_t * 1000 * KG_TO_LB / axles
    cpf = w_axle / (0.5 * tyres_per_axle * d_in * b_in)
    tef = (10.0 + b_in) / 100.0
    gf = 1.05 if chains else 1.0
    wlf = w_axle / 2000.0
    cf = clear_in / 10.0
    tf = 1.05 if manual else 1.0
    return (cpf * _wf_wheeled(w_axle) / (tef * gf) + wlf - cf) * _ef(mass_t, hp) * tf


def vci_tracked(mi):
    return (7.0 + 0.2 * mi - 39.2 / (mi + 5.6),
            19.27 + 0.43 * mi - 125.79 / (mi + 7.08))


def vci_wheeled(mi, dcf=1.0):
    """DCF = (0.15 / (deflection/section height))**0.25 = 1 at 15% tyre deflection."""
    if mi <= 115:
        return ((11.48 + 0.2 * mi - 39.2 / (mi + 3.74)) * dcf,
                (28.23 + 0.43 * mi - 92.67 / (mi + 3.74)) * dcf)
    return 4.1 * mi ** 0.446 * dcf, 9.0 * mi ** 0.446 * dcf


# ------------------------------------------------------------------ pipeline
def _first_match(name, table):
    for pat, *vals in table:
        if re.search(pat, name):
            return vals
    return None


def classify(name: str) -> str:
    if any(re.search(p, name) for p in HALFTRACK):
        return "halftrack"
    if any(re.search(p, name) for p in WHEELED):
        return "wheeled"
    return "tracked"


def load_vehicles(path=paths.VEHICLES_CSV) -> pd.DataFrame:
    raw = pd.read_csv(path)
    df = pd.DataFrame({
        "name": raw["vehicle_name"].astype(str),
        "mass_t": pd.to_numeric(raw["mass_kg"], errors="coerce") / 1000.0,
        "hp": pd.to_numeric(raw["horsePowers_hp"], errors="coerce"),
    })
    n0 = len(df)
    df = df[~df["name"].str.contains("|".join(DROP_PATTERNS))]
    df = df[df["hp"].between(20, 3000) & df["mass_t"].notna()]
    df["type"] = df["name"].map(classify)
    df["axles"] = np.nan
    df["mass_source"] = "file"

    on_ph = np.isclose(df["mass_t"] * 1000, MASS_PLACEHOLDER_KG)
    for i in df.index[on_ph]:
        fix = _first_match(df.at[i, "name"], MASS_FIX)
        if fix is None:
            df.at[i, "mass_source"] = "drop"
            continue
        df.at[i, "mass_t"], ax = fix
        df.at[i, "mass_source"] = "public_approx"
        if ax:
            df.at[i, "axles"] = ax
    df = df[(df["mass_source"] != "drop") & df["mass_t"].between(2.0, 80.0)]
    df.attrs["n_raw"] = n0
    return df.reset_index(drop=True)


def add_vci(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mi, v1, v50 = [], [], []
    for i, r in out.iterrows():
        m, hp = r["mass_t"], r["hp"]
        tr = dict(width_in=np.interp(m, _TR_MASS, _TR_WIDTH),
                  length_in=np.interp(m, _TR_MASS, _TR_LENGTH),
                  wheels=2 * round(np.interp(m, _TR_MASS, _TR_WHEELS) / 2),
                  clear_in=np.interp(m, _TR_MASS, _TR_CLEAR))
        ax = r["axles"]
        if np.isnan(ax):
            named = _first_match(r["name"], AXLES_BY_NAME)
            ax = named[0] if named else _default_axles(m)
        if r["type"] == "tracked":
            x = mi_tracked(m, hp, **tr)
            a, b = vci_tracked(x)
        elif r["type"] == "wheeled":
            x = mi_wheeled(m, hp, ax, **_tyre(m / ax))
            a, b = vci_wheeled(x)
        else:
            front_t = HALFTRACK_FRONT_SHARE * m
            xt = mi_tracked(m, hp, **tr)
            xw = mi_wheeled(m, hp, 1, w_axle=front_t * 1000 * KG_TO_LB, **_tyre(front_t))
            (at, bt), (aw, bw) = vci_tracked(xt), vci_wheeled(xw)
            x, a, b = (xt, at, bt) if at >= aw else (xw, aw, bw)
            ax = np.nan
        out.at[i, "axles"] = ax
        mi.append(x); v1.append(a); v50.append(b)
    out["mi"], out["vci1"], out["vci50"] = mi, v1, v50
    out["band"] = out["vci1"].map(band_of)
    return out


# Firmness bands on VCI1, for reporting and for tuning one safety margin per band.
# Edges follow the vehicle mix: A is light tracked, B-D medium to heavy tracked with
# light wheeled and half-tracks, E is medium and heavy wheeled.
BANDS = [("A", 0.0, 18.0), ("B", 18.0, 21.0), ("C", 21.0, 24.0), ("D", 24.0, 30.0),
         ("E", 30.0, np.inf)]


def band_of(vci1):
    for name, lo, hi in BANDS:
        if lo <= vci1 < hi:
            return name
    raise ValueError(vci1)


def m1_check():
    """Ciobotaru (2009) M1 example: 120,000 lb, 25x180 in track, 14 wheels, 19 in."""
    mi = mi_tracked(120000 / KG_TO_LB / 1000, 1500, 25, 180, 14, 19, shoe_area_in2=190)
    return mi, *vci_tracked(mi)


if __name__ == "__main__":
    print("M1 check  MI %.2f  VCI1 %.2f  VCI50 %.2f   (paper: 98.59 / 26.34 / 60.48)"
          % m1_check())
    v = add_vci(load_vehicles())
    print(v.groupby("type")[["mass_t", "vci1", "vci50"]].describe().round(1).T)
    paths.VEHICLES_OUT.mkdir(parents=True, exist_ok=True)
    v.to_csv(paths.VEHICLES_VCI_CSV, index=False)
    print("->", paths.VEHICLES_VCI_CSV.relative_to(paths.ROOT))
