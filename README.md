# Off-road trafficability forecasting — Pachmarhi (2015–2025)

Time-series models that forecast **military vehicle trafficability** on the Satpura
plateau from an 11-year daily meteorological record.

## What is being predicted

The weather record does not contain trafficability, so it is derived. The chain is:

```
rainfall + evaporative demand
        ↓  (already integrated by ERA5-Land)
soil moisture, 0–7 cm and 7–28 cm       [observed columns]
        ↓  soil wetness index, profile-weighted 0.35 / 0.65
        ↓  RCI = 300 · exp(−2.485 · SWI) · rain_factor
Rating Cone Index (kPa)                 [the modelled target]
        ↓  compare against vehicle cone index
GO / SLOW-GO / NO-GO, per vehicle class [the decision]
```

`RCI` is soil bearing strength. A vehicle moves if `RCI ≥ VCI50` (its cone index for
50 passes), struggles between `VCI1` and `VCI50`, and immobilises below `VCI1`.
Three vehicle classes are carried through: a ~14 t tracked ICV, a ~46 t tracked MBT,
and a wheeled 6×6 logistics truck.

The profile weighting means the target depends on **both** soil layers plus recent
rainfall, so it is not a relabelling of a single input column.

## Forecast setting

Operational, **no numerical weather prediction available**. Every feature at forecast
origin `t` uses observations at time `≤ t` only; the sole known-future inputs are
calendar/Fourier terms. Models emit `t+1 … t+7` directly (no recursive rollout).

Because RCI has a lag-1 autocorrelation of 0.99, learned models predict the **change**
`y_{t+h} − y_t` and the anchor is added back. Persistence is then exactly the
zero-change forecast, and the skill score reads as "signal added beyond yesterday".

| Split | Period | Origins |
|---|---|---|
| Train | 2015-01-01 – 2021-12-31 | 2498 |
| Validation | 2022-01-01 – 2023-12-31 | 730 |
| Test (held out) | 2024-01-01 – 2025-12-24 | 724 |

## Models compared

- **Baselines** — persistence, day-of-year climatology, seasonal naïve (t−365)
- **Classical** — ARIMA, and ARIMAX with 3 Fourier harmonics (dynamic harmonic
  regression). Order chosen by AIC grid search with `d` set by ADF + KPSS.
  Multi-step forecasts come from the filtered Kalman state, verified to match
  `statsmodels`' own dynamic prediction to machine precision.
- **Tabular ML** — Ridge, LightGBM, RandomForest, ExtraTrees on 128 causal features
- **Deep (GPU)** — LSTM, GRU, TCN, Transformer, N-BEATS, and a Temporal Fusion
  Transformer with variable-selection networks, gated residual networks,
  interpretable multi-head attention and quantile output
- **Ensemble** — unweighted mean of the top 3 learned models by validation RMSE
- **TFT P10 safety rule** — the GO/NO-GO call read off the TFT's 10th percentile
  instead of its median. Scored in the decision table only; its RMSE is worse by
  construction because it is deliberately biased low.

Deep models train on an RTX 4060 (CUDA), 3 seeds each, averaged, early-stopped on the
validation block.

## Run it

```bash
pip install -r requirements.txt   # torch: see the note at the top of the file
python -m traffic.check           # setup check: packages, inputs, loaders, terrain grids
python -m traffic.vehicles        # vehicle table + VCI1/VCI50 -> outputs/vehicles/
python -m traffic.forecast        # GRU+Ridge forecaster, per-band margins, edge numbers -> outputs/forecaster/
python -m traffic.terrain         # 90 m terrain grid, OSM roads, wetness index -> outputs/terrain/
python -m traffic.planner         # route backtest + convoy (~12 min) -> outputs/planner/
python -m traffic.planner --tradeoff      # margin-scale sweep (~15 min)
python -m traffic.poster_figs     # poster figures -> poster/figures/
python -m traffic.poster_build    # poster/poster.html, poster.pdf (A0), poster_preview.png
python -m traffic.abstract_build  # paper/abstract.pdf: 2-page ACM sigconf abstract (needs tools/tectonic)
python -m traffic.run             # trains everything -> outputs/forecasting/results_full.csv
python -m traffic.report          # figures + outputs/forecasting/RESULTS.md
python -m traffic.run --quick     # 1 seed, 60 epochs, for a fast check
```

Modules run from any working directory; every path lives in `traffic/paths.py`.

## Layout

```
data/
  weather/      pachmarhi_daily_2015_2025.csv, extreme events, metadata sheet
  vehicles/     army_vehicles_attributes.csv (War Thunder datamine, see caveats)
  terrain/      per-season 17-band GeoTIFF stacks (30 m, EPSG:32643) + parquet tables
    quicklooks/ maps.zip, the original PNG renders, superseded by the GeoTIFFs
docs/           project explainer (ICDCN poster plan), August report
  references/   Liu et al. 2025 (inspiration), Rosca et al. 2018 (VCI formulas)
traffic/
  paths.py      every file location
  data.py       loading, RCI target chain, 128 causal features
  vehicles.py   vehicle cleaning, tracked/wheeled split, Mobility Index -> VCI1/VCI50
  dl.py         PyTorch sequence models + training loop
  run.py        experiment orchestration, metrics, splits
  report.py     figures and the written comparison
  artifact.py   shareable HTML report
  forecast.py   deployed GRU+Ridge, cross-year safety margins per firmness band
  terrain.py    90 m terrain grid, fuzzy memberships (Liu et al. 2025 tables)
  planner.py    forecast-aware routing, 2024-25 backtest, trade-off sweep, convoy wear
  poster_*.py   ICDCN 2027 poster: figures, page copy, build to PDF
  check.py      setup check
outputs/
  forecasting/  September model comparison. NOTE: computed with RCI_SAT = 25, before the
                2026-10-11 change to 15; rerun traffic.run before quoting these numbers
  forecaster/   the deployed GRU+Ridge: forecasts, per-band margins, decisions, edge.json
  terrain/      terrain_90m.npz
  planner/      backtest.csv, tradeoff.csv, convoy.csv, timing.json, run logs
  vehicles/     vehicles_vci.csv
  eda/          exploratory plots of the weather record
poster/         poster.pdf (A0 portrait), figures/, fonts/ (IBM Plex, local)
paper/          ICDCN 2027 submission: abstract_template.tex (edit this), refs.bib, acmart.cls v2.20,
                figures/; abstract.tex/.pdf are generated. Folder uploads to Overleaf as-is
tools/          tectonic LaTeX engine (from conda-forge), not in version control
```

## Caveats

- Soil moisture is an **unassimilated ERA5-Land model state**, not an observation;
  the RCI target inherits that. Field cone-penetrometer readings would be needed to
  calibrate `RCI_DRY`, `K_DECAY` and the soil-texture constants for this site.
- The record is a single ~9 km grid cell. Trafficability in the field also depends on
  slope, vegetation, drainage and soil texture, none of which are in this dataset.
- The rainfall day definition changes at 2025-01-01 (IMD 08:30–08:30 IST → ERA5
  00:00–24:00 IST), inside the test window.
- Thresholds are doctrinal reference values, not measured for specific vehicles.

## Headline result

> **Superseded (September follow-up, see `docs/Project_Explainer_ICDCN_Poster.pdf` §5.3).**
> The TFT P10 rule below was rejected: it reaches 91.4% recall by calling NO-GO on
> 27.6% of days against a true rate of 19.2% (precision 63.5%), and its margin is
> unstable across years. Do not quote 91.4%. The chosen model is GRU (128×2) + Ridge
> with a per-vehicle safety margin. The fixed thresholds it was tuned on (MBT 30/45,
> truck 40/60) are being replaced by Mobility-Index VCIs from `traffic/vehicles.py`, so
> margins are re-tuned per VCI band. Cone-index values are psi-convention units, not
> kPa, despite the labels below.

Test window 2024-01-01 – 2025-12-24, 724 rolling origins, 7-day lead:

| | RMSE (kPa) | Skill vs persistence | MBT NO-GO recall |
|---|---:|---:|---:|
| Ensemble (TCN+Transformer+GRU) | 19.71 | +24.8% | 59.0% |
| TCN | 19.91 | +24.0% | 52.5% |
| ARIMAX-Fourier (DHR) | 22.32 | +14.9% | 59.0% |
| Persistence | 26.21 | 0.0% | 61.9% |
| **TFT P10 decision rule** | — | — | **91.4%** |

Two things worth carrying forward:

1. **Model the change, not the level.** RCI's lag-1 autocorrelation is 0.99. Predicting
   `y(t+h)` directly means learning to copy yesterday; predicting `y(t+h) − y(t)` and
   adding the anchor back is what makes every learned model beat persistence.
2. **RMSE and decision quality disagree.** The best regressor catches fewer NO-GO days
   than plain persistence, because squared error rewards staying near the middle of the
   distribution and NO-GO days are the tail. Reading the TFT's P10 instead of its median
   raises NO-GO recall to 91.4% while *also* giving the best overall accuracy (85.2%) —
   the median is genuinely biased high near the transitions. The band is well calibrated
   (84% empirical coverage against a nominal 80%; see `outputs/tft_calibration.csv`).
