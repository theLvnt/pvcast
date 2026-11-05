# Results — 15-minute-ahead PV forecasting (SKIPP'D, Stanford 30 kW array)

Held-out test: 20 days, 13371 samples (1-min issue times); cloudy subset: 9468 samples. Training: 64 days (+16 validation days).

**All test samples**

| Model | RMSE (kW) | MAE (kW) | nRMSE (%) | Skill vs smart persistence (%) |
|---|---|---|---|---|
| persistence | 3.747 | 2.148 | 12.45 | -3.2 |
| smart_persistence | 3.632 | 1.792 | 12.07 | 0.0 |
| gbm_history | 3.373 | 1.834 | 11.21 | 7.1 |
| gbm_weather | 3.382 | 1.874 | 11.24 | 6.9 |
| cnn_image | 2.892 | 1.567 | 9.61 | 20.4 |
| fusion | 2.914 | 1.581 | 9.68 | 19.8 |
| fusion (camera offline) | 3.147 | 1.817 | 10.45 | 13.4 |

**Cloudy / variable conditions (k* < 0.8 or 15-min k* std > 0.05)**

| Model | RMSE (kW) | MAE (kW) | nRMSE (%) | Skill vs smart persistence (%) |
|---|---|---|---|---|
| persistence | 4.219 | 2.591 | 14.02 | -2.5 |
| smart_persistence | 4.117 | 2.284 | 13.68 | 0.0 |
| gbm_history | 3.774 | 2.189 | 12.54 | 8.3 |
| gbm_weather | 3.783 | 2.221 | 12.57 | 8.1 |
| cnn_image | 3.258 | 1.895 | 10.82 | 20.9 |
| fusion | 3.279 | 1.885 | 10.90 | 20.3 |
| fusion (camera offline) | 3.534 | 2.162 | 11.74 | 14.2 |

Inference (CPU, ms/sample): gbm_history 0.02, gbm_weather 0.02, cnn_image 2.26, fusion 2.16