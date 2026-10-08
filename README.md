<div align="center">

# ₿ Crypto MLOps

### Real-Time Multi-Coin Crypto Prediction & Monitoring Platform

**Binance → Feature Engineering → CatBoost + GRU → Ensemble → Monitoring → FastAPI → Streamlit → Docker**

<br>

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-ML-FFCC00?style=for-the-badge)](https://catboost.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=for-the-badge)](https://mlflow.org/)

<br>

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://crypto-mlops.streamlit.app/)

<br>

**5 cryptocurrencies • 1-minute streaming • 61 features • CatBoost + GRU • live monitoring**

</div>

---

## ⚡ What is this?

**Crypto MLOps** is an end-to-end machine-learning system that continuously ingests live cryptocurrency market data, generates features, runs a production ML ensemble, stores predictions, monitors system health, and exposes everything through an API and dashboard.

This is not just a model inside a notebook.

It is a complete local MLOps pipeline covering:

- real-time data ingestion
- feature engineering
- model experimentation
- sequence modeling
- ensemble inference
- model monitoring
- API serving
- dashboarding
- containerized deployment

The system is designed to answer one practical question:

> **Can a noisy financial ML experiment be turned into a continuously running, observable, reproducible application?**

---

## 🎯 Prediction Task

The production model predicts the **direction of the next 5-minute return**.

```python
future_return_5m > 0  -> UP
future_return_5m < 0  -> DOWN
future_return_5m == 0 -> excluded
```

Low-confidence predictions are converted into:

```text
UP / DOWN / HOLD
```

Production confidence threshold:

```text
0.55
```

---

## 🏗️ Architecture

```text
                         ┌─────────────────────┐
                         │   Binance Markets   │
                         │   Live WebSocket     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   SQLite Database   │
                         │  1-Minute OHLCV     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Feature Engineering │
                         │     V2 Pipeline     │
                         │      61 Features    │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┴────────────────┐
                    │                                │
                    ▼                                ▼
             ┌─────────────┐                  ┌─────────────┐
             │   CatBoost  │                  │     GRU     │
             │ Classifier  │                  │  Sequence   │
             │    55%      │                  │    45%      │
             └──────┬──────┘                  └──────┬──────┘
                    │                                │
                    └───────────────┬────────────────┘
                                    │
                                    ▼
                          ┌────────────────────┐
                          │ Ensemble Inference │
                          │   UP / DOWN / HOLD │
                          └──────────┬─────────┘
                                     │
                                     ▼
                         ┌─────────────────────┐
                         │ Live Predictions    │
                         │ + Monitoring        │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
             ┌─────────────┐                 ┌──────────────┐
             │   FastAPI   │                 │  Monitoring  │
             │     API     │                 │   Metrics    │
             └──────┬──────┘                 └──────┬───────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                          ┌──────────────────┐
                          │    Streamlit     │
                          │  Control Center  │
                          └──────────────────┘
```

---

# 📈 Dashboard

The dashboard is designed as a **production-style monitoring control center**.

### Market overview

- BTCUSDT
- ETHUSDT
- SOLUSDT
- BNBUSDT
- XRPUSDT
- current price
- UP probability
- confidence
- current signal

### Model analytics

- ensemble market probability
- UP probability history
- confidence history
- 50% neutral threshold
- 55% production action threshold

### Individual asset view

- candlestick chart
- EMA 20
- Bollinger Bands
- volume
- CatBoost probability
- GRU probability
- ensemble probability
- prediction history
- inference latency
- model version

### MLOps telemetry

- data freshness
- sequence continuity
- prediction completeness
- inference latency
- feature drift
- confidence
- UP / DOWN / HOLD distribution

---

# 🧠 Model Stack

## Classical ML

Evaluated:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

## Deep Learning

Evaluated:

- GRU
- LSTM
- CNN-LSTM

### Final production combination

```text
┌───────────────┐
│   CatBoost    │
│   55% weight  │
└───────┬───────┘
        │
        ▼
┌─────────────────────┐
│      ENSEMBLE       │
│                     │
│ UP / DOWN / HOLD    │
└─────────▲───────────┘
          │
          │
┌─────────┴───────────┐
│        GRU          │
│      45% weight     │
└─────────────────────┘
```

---

# 📊 Model Results

## Binary Direction Benchmark

| Model | Macro-F1 | Accuracy |
|---|---:|---:|
| Logistic Regression | 0.520525 | 0.520596 |
| Random Forest | 0.521783 | 0.522099 |
| XGBoost | 0.522451 | 0.522469 |
| LightGBM | 0.522509 | 0.522555 |
| **CatBoost** | **0.523403** | **0.523431** |
| Inverse 5m Baseline | 0.517319 | 0.517366 |

CatBoost produced the strongest validation result among the classical ML models.

---

## Sequence Model

Sequence configuration:

```text
Sequence length:       60 minutes
Prediction horizon:     5 minutes
Features per timestep:  61
```

Binary GRU:

```text
Validation Macro-F1: 0.522358
Validation Accuracy: 0.522440
```

---

## Ensemble Validation

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| CatBoost | 0.523581 | 0.523550 |
| GRU | 0.523384 | 0.523290 |
| **55/45 Ensemble** | **0.525364** | **0.525358** |

---

## Frozen Test Evaluation

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| CatBoost | 0.515686 | 0.515670 |
| GRU | 0.515812 | 0.515641 |
| **55/45 Ensemble** | **0.516428** | **0.516428** |
| Inverse Baseline | 0.509532 | 0.509496 |

> **Evaluation note:** The test set was inspected multiple times during development and should not be treated as a pristine untouched benchmark.

---

# ⚙️ Production Inference

The original live pipeline launched multiple Python subprocesses for every new candle.

That produced approximately:

```text
8.5–9.7 seconds
```

per inference cycle.

A resident inference service was then implemented.

Models and preprocessing components are loaded once:

```text
CatBoost
GRU
Feature scaler
Feature configuration
```

Observed production timings:

```text
GRU inference:       ~6 ms
CatBoost inference: ~23 ms
Resident cycle:     ~510 ms
```

This reduced live inference latency by approximately **96%** compared with the earlier subprocess-based pipeline.

---

# 🔎 Production Monitoring

The system continuously checks whether the pipeline is healthy.

### Data health

```text
Freshness
Sequence continuity
Symbol completeness
Missing intervals
```

### Prediction health

```text
Prediction count
Mean inference latency
Maximum inference latency
Mean confidence
Maximum confidence
UP / DOWN / HOLD
```

### Feature drift

```text
Mean |z|
Max |z|
% of features where |z| > 3
```

Example healthy production state:

```text
Data freshness:           PASS
Sequence continuity:      PASS
Prediction completeness:  PASS
Symbols:                  5/5
Predictions:              5/5
Inference latency:        ~510 ms
Feature |z| > 3:          0.00%
```

---

# 🧪 What the Experiments Revealed

One of the biggest findings was:

> **Better ML metrics do not automatically mean a profitable trading strategy.**

The experiments showed that:

- very short-horizon crypto returns contain weak predictive signal
- binary classification was modestly better than naive baselines
- CatBoost and GRU provide complementary representations
- the ensemble gives a small measurable improvement
- confidence gating improves selectivity but reduces coverage
- transaction costs can eliminate the apparent high-frequency edge

The project therefore focuses on:

**MLOps engineering + reproducibility + monitoring + deployment**

rather than making unsupported profitability claims.

---

# 🧰 Tech Stack

| Area | Technology |
|---|---|
| Language | Python 3.11 |
| Market Data | Binance WebSocket / REST |
| Data Processing | Pandas, NumPy, PyArrow |
| Machine Learning | Scikit-learn |
| Gradient Boosting | CatBoost, XGBoost, LightGBM |
| Deep Learning | PyTorch |
| Sequence Model | GRU |
| Experiment Tracking | MLflow |
| Database | SQLite |
| API | FastAPI |
| Dashboard | Streamlit |
| Visualization | Plotly |
| Containers | Docker |
| Orchestration | Docker Compose |
| Version Control | Git / GitHub |

---

# 🚀 Run the Project

## Requirements

- Docker Desktop
- Docker Compose
- Git
- Internet connection

No paid cloud infrastructure is required.

Designed for:

```text
₹0 / $0 cloud cost
```

---

## Clone

```bash
git clone https://github.com/rohithap0819/crypto-mlops.git
cd crypto-mlops
```

## Start Everything

```bash
docker compose up -d --build
```

## Check Services

```bash
docker compose ps
```

Expected services:

```text
crypto-mlops-ingestion
crypto-mlops-inference
crypto-mlops-api
crypto-mlops-dashboard
```

---

# 🌐 URLs

### Streamlit Dashboard

```text
http://localhost:8501
```

### FastAPI

```text
http://localhost:8000
```

### FastAPI Docs

```text
http://localhost:8000/docs
```

### Health Check

```text
http://localhost:8000/health
```

---

# 🐳 Useful Docker Commands

### Start

```bash
docker compose up -d
```

### Rebuild everything

```bash
docker compose up -d --build
```

### Rebuild dashboard only

```bash
docker compose up -d --build dashboard
```

### Stop

```bash
docker compose down
```

### Status

```bash
docker compose ps
```

### Inference logs

```bash
docker compose logs -f inference
```

### Ingestion logs

```bash
docker compose logs -f ingestion
```

### API logs

```bash
docker compose logs -f api
```

### Dashboard logs

```bash
docker compose logs -f dashboard
```

---

# 🩹 Recover a Live Data Gap

If the WebSocket has been interrupted and the inference service reports a missing 1-minute sequence:

```bash
docker compose exec inference python src/ingestion/backfill_recent_binance.py
```

The backfill utility retrieves recent closed Binance candles and restores the required market-data continuity.

---

# 📁 Project Structure

```text
crypto-mlops/
│
├── src/
│   ├── api/
│   │   └── main.py
│   │
│   ├── dashboard/
│   │   └── app.py
│   │
│   ├── features/
│   │   ├── build_features_v2.py
│   │   └── live_feature_builder.py
│   │
│   ├── ingestion/
│   │   ├── binance_ws_ingestor.py
│   │   ├── bootstrap_sqlite_from_historical.py
│   │   └── backfill_recent_binance.py
│   │
│   ├── inference/
│   │   ├── live_inference_runner.py
│   │   └── resident_inference_service.py
│   │
│   ├── models/
│   │   ├── binary_sequence_models.py
│   │   ├── prepare_sequence_data.py
│   │   ├── train_binary_baselines.py
│   │   ├── compare_target_variants.py
│   │   ├── diagnose_binary_direction.py
│   │   ├── build_live_gru_sequence.py
│   │   ├── predict_live_gru.py
│   │   ├── predict_live_catboost.py
│   │   ├── predict_live_ensemble.py
│   │   ├── register_production_ensemble_mlflow.py
│   │   ├── backtest_ensemble.py
│   │   ├── backtest_ensemble_v2.py
│   │   ├── backtest_ensemble_v3.py
│   │   ├── compare_forecast_horizons.py
│   │   ├── compare_magnitude_targets.py
│   │   └── evaluate_ensemble_per_coin.py
│   │
│   └── monitoring/
│       └── monitor_live.py
│
├── data/
├── models/
├── reports/
│
├── Dockerfile
├── docker-compose.yml
├── requirements-docker.txt
├── .dockerignore
├── .gitignore
└── README.md
```

---

# 🧩 MLOps Design

### Experiment Tracking

MLflow experiments include:

```text
CryptoML_Regression
CryptoML_Classification
CryptoML_Optuna
CryptoML_Binary_Sequence
CryptoML_Final_Binary
CryptoML_Production_Ensemble
```

### Production Model Version

```text
catboost_v1_gru_v1_ensemble_v1
```

### Production Feature Set

```text
V2
61 ML features
```

### Production Sequence

```text
60 timesteps × 61 features
```

### Ensemble

```text
CatBoost 55%
GRU      45%
```

### Action Threshold

```text
0.55
```

---

# 💡 Engineering Decisions

| Decision | Reason |
|---|---|
| Binary direction target | Better aligned with the observed signal than the original 3-class formulation |
| CatBoost + GRU | Combines engineered tabular and temporal modeling |
| Resident inference | Reduces repeated model-loading/process overhead |
| Confidence threshold | Avoids forcing low-confidence decisions |
| HOLD state | Keeps uncertain predictions non-actionable |
| SQLite | Simple local zero-cost persistence |
| Docker Compose | Reproducible multi-service deployment |
| FastAPI | Separates API/data access from UI |
| Streamlit | Interactive production-style monitoring interface |
| Feature z-score monitoring | Simple live distribution-shift detection |
| REST backfill | Recovers gaps after WebSocket interruptions |
| Execution-aware backtesting | Avoids unrealistic trading assumptions |

---

# 🔬 Model Governance

The production ensemble was frozen after model and target comparisons.

The system does not retrain itself automatically from live data.

This makes the deployed model reproducible and separates:

```text
model version
data state
feature version
inference state
monitoring state
```

---

# 🔮 Future Improvements

Potential next steps include:

- Automated model retraining
- CI/CD
- Model promotion gates
- Automated rollback
- Prometheus + Grafana
- Kafka
- PostgreSQL / TimescaleDB
- Redis
- Cloud deployment
- Kubernetes
- Feature store
- SHAP explainability
- Delayed-label production monitoring
- Automated drift-triggered retraining
- Canary deployment
- Multi-horizon prediction
- Online model evaluation

---

# 📌 Project Goal

The objective is **not** to claim guaranteed cryptocurrency trading profitability.

The objective is to demonstrate a complete ML lifecycle:

```text
Experiment
    ↓
Evaluate
    ↓
Select
    ↓
Register
    ↓
Deploy
    ↓
Infer
    ↓
Monitor
    ↓
Visualize
```

In other words:

> **From ML experiment → to a continuously running MLOps system.**

---

# 🌐 Repository

https://github.com/rohithap0819/crypto-mlops

---

# 👤 Author

**Rohith AP**

---

<div align="center">

### Built as an end-to-end MLOps engineering project

**Real-Time Data • Machine Learning • Deep Learning • Monitoring • APIs • Docker**

</div>

---

## Disclaimer

This project is intended for educational, engineering, and portfolio purposes.

Cryptocurrency prediction is inherently uncertain. Model outputs should not be interpreted as guaranteed trading signals or financial advice.
