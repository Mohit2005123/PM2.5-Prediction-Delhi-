# PM2.5 Forecasting

End-to-end AIML pipeline for hourly PM2.5 prediction across Indian cities using meteorology, feature engineering, and a model zoo (Random Forest, CNN+BiLSTM, Stacked LSTM, BiGRU, Transformer).

---

##  TL;DR

- **Goal**: Predict next-hour PM2.5
- **Data**: Open-Meteo weather + derived & calendar features.
- **Models**: RF (per-city), CNN+BiLSTM, Stacked LSTM, BiGRU, Transformer.
- **Best (this run)**: CNN+BiLSTM → MAE≈6.56, RMSE≈8.24, MAPE≈11.65% (test).
- **Outputs**: Per-model plots, metric tables, and CSV predictions (e.g., `outputs/rf_test_predictions_<city>.csv`).

---

## Project Overview

This repository contains a self-contained notebook that:

1. Ingests & cleans weather data
2. Performs EDA (trends, correlation, outliers)
3. Engineers features (calendar, lags, scaling, PCA)
4. Explores clusters with K-Means (optional insights)
5. Trains/evaluates per-city ML & DL models
6. Saves predictions/plots + prints a summary leaderboard

---

##  Architecture (High-Level)

```mermaid
flowchart TD
    A[Raw Weather Data\n(Open-Meteo)] --> B[Data Cleaning\n& Alignment]
    B --> C[EDA & Diagnostics\n(trends, outliers, correlation)]
    C --> D[Feature Engineering\n(lags, calendar, scaling, PCA)]
    D --> E{Split by City?}
    E -- Yes --> F[Per-City Tabular ML\nRandom Forest]
    E -- Yes --> G[Per-City DL Windows\nSequences (48->1)]
    G --> H[DL Models:\nCNN+BiLSTM / Stacked LSTM / BiGRU / Transformer]
    F --> I[Evaluation\n(RMSE, MAE, R², sMAPE)]
    H --> I
    I --> J[Plots & CSVs\nper city/model]
```

---

##  Models Implemented

1. **Random Forest (baseline, per-city)**: Tabular baseline with MI feature analysis + median imputation.
2. **CNN + BiLSTM (hybrid)**: CNN for short-term motifs → BiLSTM for long-range context. Best performer in this run.
3. **LSTM (stacked)**: Multi-layer LSTM for hierarchical temporal representation.
4. **BiGRU (single)**: Lightweight gated RNN with bidirectional context.
5. **Transformer (d=1, h=2)**: Minimal self-attention baseline for sequences.

---

## Results (snapshot)


*Exact numbers may vary with data slice, seeds, and cities.*

| Model | MAE | RMSE | MAPE % | Notes |
|-------|-----|------|--------|-------|
| **CNN + BiLSTM** | **6.56** | **8.24** | **11.65** |  Best RMSE on test split |
| Stacked LSTM | 7.34 | 9.06 | 12.41 | Solid |
| BiGRU (single) | 7.80 | 9.66 | 12.76 | Fast / light |
| Transformer (d=1, h=2) | 10.39 | 12.16 | 18.59 | Under-parameterized |
| Random Forest (Delhi ex) | 8.95 | 12.76 | 13.04 | R²≈0.703 |

### Result Images (add yours)




### Minimal requirements (edit to match your notebook):

```
pandas
numpy
scikit-learn
matplotlib
seaborn
scipy
tensorflow>=2.14
keras
```

---

## How to Run

###  Run the notebook

1. Open `notebooks/Pollution_Prediction_OpenMeteo_v2_8-4.ipynb`.
2. Run cells top-to-bottom.
3. DL section creates windows (LOOKBACK=48 → HORIZON=1), trains models, plots predictions, and prints a leaderboard.
4. RF trains per city and writes predictions to `outputs/`.

---

##  Key Configuration (in-notebook)

- **LOOKBACK**: 48
- **HORIZON**: 1
- **Split**: time-ordered 80% train / 20% test
- **DL training**: epochs=30, batch_size=32, ReduceLROnPlateau
- **RF**: median imputation, Mutual Information features, per-city models

*Expose these as constants at the top of the notebook for easy tuning.*

---

##  Evaluation Metrics

- **RMSE** – overall error magnitude
- **MAE** – average absolute deviation
- **MAPE / sMAPE** – percentage error
- **R²** – variance explained (tabular RF)

```mermaid
flowchart LR
    A[Predictions] --> B[Compute MAE]
    A --> C[Compute RMSE]
    A --> D[Compute MAPE/sMAPE]
    A --> E[Compute R² (tabular)]
    B & C & D & E --> F[Model Leaderboard]
```

---

## Additional Work Done (non-experiments)

- **EDA**: Trend lines, distributions, outliers, seasonality
- **Correlation & MI**: Feature relevance for PM2.5
- **PCA**: Dimensionality reduction for stability/visualization
- **K-Means**: Unsupervised grouping for pollution regimes
- **Visualization**: City-wise plots, prediction vs actual, error histograms

*(These support data readiness and model quality and are summarized in your practical file under "Additional Work Done".)*

---



## Known Limitations / Future Work

- Transformer is minimal (d=1, h=2) → try larger d, deeper encoders, richer positional encodings
- Explore multi-step forecasts (6/24 hours)
- Add exogenous signals (emissions, traffic, events)
- Run hyperparameter search (Optuna/KerasTuner)
- Try global (cross-city) sequence models

---



##  Acknowledgements

- **Open-Meteo** for weather data
- **Libraries**: pandas, scikit-learn, TensorFlow/Keras, matplotlib
- Teammates & mentors for reviews