# Trafficability forecasting - model comparison

Test period **2024-01-01 - 2025-12-24** (724 rolling forecast origins), never seen during training or model selection.


## Headline: test RMSE (kPa RCI)

| model                         | h=1   | h=3   | h=7   | skill@7 (%) |
|-------------------------------|------:|------:|------:|------------:|
| Ensemble(top3)                |  5.76 | 12.82 | 19.95 |       23.90 |
| TCN                           |  5.76 | 13.13 | 20.60 |       21.40 |
| Transformer                   |  5.86 | 13.00 | 20.63 |       21.30 |
| GRU                           |  6.14 | 13.51 | 20.81 |       20.60 |
| N-BEATS                       |  6.20 | 13.59 | 20.83 |       20.50 |
| BiGRU-Attn                    |  6.78 | 14.35 | 21.08 |       19.60 |
| TFT                           |  6.15 | 13.06 | 21.11 |       19.50 |
| Ridge                         |  6.37 | 13.98 | 21.96 |       16.20 |
| ARIMAX-Fourier (DHR)          |  5.76 | 14.05 | 22.32 |       14.90 |
| LightGBM(linear)              |  6.27 | 14.81 | 23.04 |       12.10 |
| LSTM                          |  7.84 | 17.81 | 23.20 |       11.50 |
| RandomForest                  |  6.57 | 14.89 | 23.58 |       10.00 |
| ExtraTrees                    |  6.93 | 15.64 | 23.86 |        9.00 |
| ARIMA                         |  5.85 | 14.72 | 25.11 |        4.20 |
| LightGBM                      |  6.26 | 14.76 | 25.15 |        4.00 |
| BiGRU-Attn(q)                 |  7.07 | 15.79 | 25.43 |        3.00 |
| Persistence                   |  6.79 | 15.33 | 26.21 |        0.00 |
| TFT P10 (safe rule)           |  8.47 | 19.20 | 32.76 |      -25.00 |
| Climatology                   | 34.65 | 34.54 | 33.83 |      -29.10 |
| BiGRU-Attn(q) P10 (safe rule) | 12.56 | 28.32 | 43.44 |      -65.80 |
| SeasonalNaive365              | 53.08 | 52.95 | 52.50 |     -100.30 |


## Trafficability decision quality (tracked heavy / MBT)

| model                         | GO/SLOW-GO/NO-GO accuracy % | macro F1 % | NO-GO recall % |
|-------------------------------|----------------------------:|-----------:|---------------:|
| TFT P10 (safe rule)           |                        85.2 |       69.0 |           91.4 |
| GRU                           |                        83.6 |       67.9 |           64.0 |
| BiGRU-Attn                    |                        83.4 |       67.6 |           61.9 |
| Ridge                         |                        83.3 |       67.3 |           58.3 |
| Transformer                   |                        82.7 |       66.8 |           54.0 |
| BiGRU-Attn(q) P10 (safe rule) |                        82.9 |       65.9 |           79.1 |
| ARIMAX-Fourier (DHR)          |                        82.9 |       65.8 |           59.0 |
| Ensemble(top3)                |                        82.3 |       65.6 |           54.0 |
| N-BEATS                       |                        82.2 |       64.9 |           56.1 |
| LightGBM(linear)              |                        81.4 |       64.1 |           44.6 |
| BiGRU-Attn(q)                 |                        82.0 |       63.7 |           57.6 |
| RandomForest                  |                        81.1 |       63.7 |           41.0 |
| TCN                           |                        80.7 |       63.4 |           39.6 |
| SeasonalNaive365              |                        80.1 |       63.1 |           56.8 |
| LSTM                          |                        80.5 |       62.7 |           37.4 |
| Persistence                   |                        79.6 |       62.0 |           61.9 |
| LightGBM                      |                        80.1 |       61.9 |           42.4 |
| ExtraTrees                    |                        80.2 |       61.8 |           33.8 |
| TFT                           |                        79.7 |       60.1 |           42.4 |
| Climatology                   |                        78.0 |       55.3 |           21.6 |
| ARIMA                         |                        75.1 |       49.2 |            5.8 |


## Best model per horizon

- **h=1**: ARIMAX-Fourier (DHR) - RMSE 5.76 kPa, MAE 2.74, R2 0.9942, skill vs persistence +15.2%
- **h=3**: Ensemble(top3) - RMSE 12.82 kPa, MAE 6.94, R2 0.9714, skill vs persistence +16.4%
- **h=7**: Ensemble(top3) - RMSE 19.95 kPa, MAE 12.44, R2 0.9308, skill vs persistence +23.9%
