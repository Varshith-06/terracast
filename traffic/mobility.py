"""
Passes and speed on soft soil for one vehicle: what SLOW-GO means in numbers.

Passes (ours). VCI1 and VCI50 are the soil strengths a vehicle needs for 1 and for 50
passes in the same track (Rula & Nuttall 1971). The sources give only those two curves;
between them the requirement is taken to rise with log(passes), the same curve the convoy
wear model uses, so ground of strength RCI carries
    n = 50 ** ((RCI - VCI1) / (VCI50 - VCI1))
vehicles in one track: 0 below VCI1 (NO-GO), 1 at VCI1, 50 at VCI50 (GO).

Speed. The NRMM Edition I areal module for fine-grained soil: Haley, Jurkat & Brady
(1979), NATO Reference Mobility Model Edition I Users Guide Vol. I, TARADCOM TR 12503,
subroutines IV3 (FGSPC, FGSPR), IV5 (TFORCF, SLIP), IV8 and IV9, whose relations come from
Rula & Nuttall (1971) as revised by Turnage (1972). With excess strength x = RCI - VCI1:
    D/W(x)    drawbar pull coefficient at 20% slip                    FGSPC
    R/W(x)    motion resistance coefficient of the powered running gear  FGSPR
    TFOR/W    soil-limited tractive effort at 100% slip = D/W + R/W + c  TFORCF
    resistance on a slope of angle t:  W sin t + R/W W cos t           IV8
    the vehicle is immobilised if TFOR cos t is below that resistance;
    otherwise its speed is where the rim pull meets the resistance, less slip  IV9, SLIP
Each relation has two forms, for contact pressure factor below and above 4 psi.
On paved road the resistance is NRMM's hard-surface rolling resistance (sec. III.C.3.b).

Ours, because the vehicle table has no powertrain data:
  - rim pull is the constant-power curve F = ETA * P / v (NRMM builds it gear by gear from
    engine torque and gear ratios), with P the table's engine power;
  - ETA, the engine-to-track or wheel efficiency, is assumed;
  - the top speed is the game template value (mostly 60 or 75 km/h), not a real figure;
  - vegetation override, ride comfort and obstacles, which NRMM also checks, are not
    modelled (the planner scales speed by the land-cover score instead);
  - the NO-GO line stays at VCI1, as in the decisions. NRMM's own level-ground
    immobilisation point lies 0.7-2.1 psi below it, so this is the cautious side;
  - slip is set to 0, not NRMM's 1, when the force coefficient goes negative (downhill).
"""

from __future__ import annotations

import numpy as np

G = 9.81
HP_W = 745.7
ETA = 0.80            # engine-to-sprocket/wheel efficiency (assumed)
ROAD_PSI = 75.0       # tyre pressure on paved road (assumed; 8x8s run 50-105 psi on road)
ROAD_TRACTION = 0.8   # traction coefficient on dry pavement (assumed): limits only steep grades

# NRMM I fine-grained soil, keyed by (running gear, contact pressure factor >= 4 psi)
# pull, FGSPC:       a + b x - sqrt((a + b x)^2 - c x) for x >= 0, s x below
_PULL = {("tracked", False): (.544, .0463, .0702, .076),
         ("tracked", True): (.455, .0392, .0526, .056),
         ("wheeled", False): (.3885, .0265, .0358, .046),
         ("wheeled", True): (.379, .0219, .0257, .033)}
# resistance, FGSPR: p + q / (x + r) for x >= 0, u - s x below
_RES = {("tracked", False): (.045, 2.3075, 6.5, .4, .072),
        ("tracked", True): (.045, 2.3075, 6.5, .4, .052),
        ("wheeled", False): (.035, .861, 3.249, .3, .043),
        ("wheeled", True): (.045, 2.3075, 6.5, .4, .029)}
# TFORCF: CF = D/W - k1 + R/W, TFOR/W = CF + k2
_TFOR = {("tracked", False): (.758, .82), ("tracked", True): (.671, .71),
         ("wheeled", False): (.674, .76), ("wheeled", True): (.585, .655)}
# SLIP: S = a y + b + c / (d - y), y = F / (W cos t) - CF
_SLIP = {("tracked", False): (.0257, -.0161, .01519, .8353),
         ("tracked", True): (.0733, -.0063, .00734, .7177),
         ("wheeled", False): (.0621, -.021, .01888, .7794),
         ("wheeled", True): (.084, -.016, .01414, .6697)}


def passes(rci, vci1, vci50):
    """Vehicles one track carries before it fails: 0 below VCI1, 50 at VCI50, more above."""
    rci = np.asarray(rci, float)
    return np.where(rci < vci1, 0.0, 50.0 ** ((rci - vci1) / (vci50 - vci1)))


def gear(kind, cpf):
    """Half-tracks take the tracked relations: the track unit carries and drives them."""
    return ("wheeled" if kind == "wheeled" else "tracked", bool(cpf >= 4.0))


def pull(x, key):
    a, b, c, s = _PULL[key]
    xp = np.maximum(x, 0.0)
    t = a + b * xp
    return np.where(x < 0, s * x, t - np.sqrt(t * t - c * xp))


def resistance(x, key):
    p, q, r, u, s = _RES[key]
    return np.where(x < 0, u - s * x, p + q / (np.maximum(x, 0.0) + r))


def slip(y, key):
    a, b, c, d = _SLIP[key]
    yy = np.minimum(y, d - 1e-6)
    s = np.clip(a * yy + b + c / (d - yy), 0.0, 1.0)
    return np.where(y >= d, 1.0, s)


def speed(rci, sin_t, veh, paved=False, need=None):
    """Steady speed (km/h) of a vehicle on soil of strength `rci` (psi) along a slope with
    sine `sin_t` (+ uphill). veh: dict(kind, vci1, cpf, hp_per_t, vmax_kmh). paved: road
    cells, which ignore the soil. need: the strength required in place of VCI1 (convoy
    wear). 0 where the vehicle cannot move."""
    rci, sin_t, paved = np.broadcast_arrays(np.asarray(rci, float), np.asarray(sin_t, float),
                                            np.asarray(paved, bool))
    need = veh["vci1"] if need is None else need
    key = gear(veh["kind"], veh["cpf"])
    cos_t = np.sqrt(1.0 - sin_t ** 2)
    x = rci - need
    d, r = pull(x, key), resistance(x, key)
    k1, k2 = _TFOR[key]
    cf = d - k1 + r
    tfor = cf + k2
    if veh["kind"] == "wheeled":
        r_road = .007 + .0939 / ROAD_PSI
    else:
        r_road = .045
    r = np.where(paved, r_road, r)
    tfor = np.where(paved, ROAD_TRACTION, tfor)
    load = sin_t + r * cos_t                                   # resistance / W
    s = np.where(paved, 0.0, slip(load / cos_t - cf, key))
    v = ETA * veh["hp_per_t"] * HP_W / (1000.0 * G * np.maximum(load, 1e-9)) * 3.6 * (1 - s)
    v = np.where(load <= 0, veh["vmax_kmh"], np.minimum(v, veh["vmax_kmh"]))
    stuck = (tfor * cos_t < load) | (~paved & (x < 0))
    return np.where(stuck, 0.0, v)


if __name__ == "__main__":
    # level-ground speed of each band's planning vehicle as the soil firms up
    from traffic import planner as PL
    from traffic import vehicles as V
    reps = PL.representatives(V.add_vci(V.load_vehicles()))
    print("band  kind     VCI1  hp/t   CPF   km/h at VCI1 +0, +2, +5, +10, +20, +40 psi   passes at +5, +10")
    for b, v in reps.items():
        xs = np.array([0, 2, 5, 10, 20, 40.0])
        sp = speed(v["vci1"] + xs, 0.0, v)
        n = passes(v["vci1"] + np.array([5.0, 10.0]), v["vci1"], v["vci50"])
        print(f"  {b}   {v['kind']:8s} {v['vci1']:5.1f} {v['hp_per_t']:5.1f} {v['cpf']:5.1f}   "
              + " ".join(f"{s:5.1f}" for s in sp) + "      " + " ".join(f"{p:5.1f}" for p in n))
