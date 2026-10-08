# Crypto MLOps — Real-Time Multi-Coin Cryptocurrency Prediction Platform

A production-style, end-to-end MLOps system for real-time cryptocurrency market prediction, model inference, monitoring, and visualization.

The project combines **Binance live market data, feature engineering, CatBoost, GRU, ensemble inference, SQLite, FastAPI, Streamlit, Docker Compose, and MLOps monitoring** into a single local deployment.

The system currently supports:

- BTCUSDT
- ETHUSDT
- SOLUSDT
- BNBUSDT
- XRPUSDT

The primary prediction task is **5-minute cryptocurrency direction prediction**:

> UP when the future 5-minute return is positive  
> DOWN when the future 5-minute return is negative

The final production system uses a **CatBoost + GRU ensemble** with a confidence threshold for actionable predictions.

---

## Project Overview

The project was designed around a realistic MLOps pipeline rather than a standalone machine-learning notebook.

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
             │             │                  │    Model    │
             └──────┬──────┘                  └──────┬──────┘
                    │                                │
                    └───────────────┬────────────────┘
                                    │
                                    ▼
                          ┌────────────────────┐
                          │ Ensemble Inference │
                          │ 55% CatBoost       │
                          │ 45% GRU            │
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
             └──────┬──────┘                 └──────────────┘
                    │
                    ▼
             ┌─────────────────┐
             │ Streamlit       │
             │ MLOps Dashboard │
             └─────────────────┘
```

---

# Key Features

## Real-Time Market Ingestion

The platform connects to the Binance public WebSocket and continuously receives closed 1-minute candles for five cryptocurrencies.

Supported streams:

```text
btcusdt@kline_1m
ethusdt@kline_1m
solusdt@kline_1m
bnbusdt@kline_1m
xrpusdt@kline_1m
```

Only completed candles are persisted.

Market data is stored locally in SQLite.

---

## Historical Data

Historical cryptocurrency data is used for model development and evaluation.

The project uses approximately one year of 1-minute OHLCV data and builds a processed machine-learning dataset.

Large datasets are intentionally excluded from Git because they are generated artifacts and can exceed GitHub file-size limits.

---

# Feature Engineering

The V2 feature pipeline produces **61 model features**.

Feature groups include:

### Returns

- 1-minute return
- 5-minute return
- 15-minute return
- 30-minute return

### Lag Features

- 1-minute lag
- 2-minute lag
- 3-minute lag
- 5-minute lag
- 10-minute lag
- 15-minute lag
- 30-minute lag
- 60-minute lag

Additional 5-minute lag features are included.

### Rolling Statistics

Rolling means and standard deviations over multiple windows:

- 5 minutes
- 15 minutes
- 30 minutes
- 60 minutes

### Technical Indicators

- Volatility
- RSI
- MACD
- ATR
- SMA / EMA relationships
- Bollinger Bands

### Candle Structure

- Candle body
- Upper wick
- Lower wick
- Relative candle structure

### Time Features

- Time-of-day sine/cosine
- Day-of-week sine/cosine

### Cross-Asset Features

Cross-market returns and market-level volatility features capture relationships among the five cryptocurrencies.

---

# Prediction Target

The production classification target is a binary 5-minute direction target.

```python
future_return_5m > 0  -> UP
future_return_5m < 0  -> DOWN
future_return_5m == 0 -> removed
```

This binary target was selected after comparing it against the original three-class formulation.

---

# Model Development

Several model families were evaluated.

## Regression

Evaluated models:

- Linear Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

Regression targets included:

- Next 1-minute return
- Future 5-minute return

The experiments showed that very short-horizon return prediction has extremely weak raw predictive signal, with performance close to zero-return baselines.

---

## Classification

Evaluated models:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

The binary direction benchmark produced the following validation results:

| Model | Macro-F1 | Accuracy |
|---|---:|---:|
| Logistic Regression | 0.520525 | 0.520596 |
| Random Forest | 0.521783 | 0.522099 |
| XGBoost | 0.522451 | 0.522469 |
| LightGBM | 0.522509 | 0.522555 |
| **CatBoost** | **0.523403** | **0.523431** |
| Inverse 5m Baseline | 0.517319 | 0.517366 |

CatBoost produced the best validation result among the classical machine-learning models.

---

# Sequence Modeling

A sequence-based deep-learning stage was added to model temporal dependencies directly.

Sequence configuration:

```text
Sequence length: 60 minutes
Prediction horizon: 5 minutes
Features per timestep: 61
```

Three architectures were evaluated:

- GRU
- LSTM
- CNN-LSTM

The binary GRU achieved:

```text
Best validation Macro-F1: 0.522358
Best validation Accuracy: 0.522440
```

The production system uses the trained GRU as the temporal component of the ensemble.

---

# Final Production Ensemble

The final production inference system combines the two models:

```text
CatBoost weight = 55%
GRU weight      = 45%
```

The ensemble probability is:

```text
ensemble_probability_up =
    0.55 × CatBoost probability
    +
    0.45 × GRU probability
```

A prediction becomes actionable only when model confidence reaches the production threshold:

```text
Confidence threshold = 0.55
```

Therefore:

```text
High-confidence UP   -> UP
High-confidence DOWN -> DOWN
Low confidence       -> HOLD
```

This prevents the system from forcing a directional decision when the models are uncertain.

---

# Ensemble Evaluation

The frozen CatBoost + GRU ensemble produced the following results.

## Validation

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| CatBoost | 0.523581 | 0.523550 |
| GRU | 0.523384 | 0.523290 |
| **55/45 Ensemble** | **0.525364** | **0.525358** |

The ensemble provided a modest improvement over either individual model.

## Frozen Test Evaluation

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| CatBoost | 0.515686 | 0.515670 |
| GRU | 0.515812 | 0.515641 |
| **55/45 Ensemble** | **0.516428** | **0.516428** |
| Inverse Baseline | 0.509532 | 0.509496 |

The improvement is modest, which is expected for a noisy high-frequency financial prediction problem.

> **Evaluation note:** The test set was inspected multiple times during development and should therefore not be treated as a pristine untouched benchmark.

---

# Confidence Gating

The project also evaluates prediction confidence instead of relying only on raw classification accuracy.

With a frozen confidence threshold of `0.55`:

```text
Coverage: approximately 11%
Accuracy: approximately 53.9%
```

This demonstrates the difference between:

- overall classification performance
- high-confidence prediction performance

The production service therefore uses:

```text
Confidence < 0.55  -> HOLD
Confidence >= 0.55 -> directional signal
```

---

# Backtesting

Execution-aware backtesting was performed to evaluate whether model predictions translate into economically useful trading signals.

The backtesting assumptions included:

- Signal generated at candle close
- Entry at next 1-minute open
- Exit after 5 minutes
- No overlapping positions for the same symbol
- Separate capital allocation per asset

A critical finding was that small predictive improvements can disappear once realistic transaction costs are introduced.

At approximately:

```text
5 bps transaction cost
```

the strategy performance collapsed substantially.

This led to an important project conclusion:

> A model can achieve slightly better-than-baseline classification metrics without producing a profitable high-frequency trading strategy after transaction costs.

The project therefore focuses on **MLOps engineering, real-time inference, monitoring, and deployment rather than unsupported profitability claims**.

---

# Live Inference Architecture

The original live system executed multiple Python subprocesses for every new candle.

That approach produced inference cycles around:

```text
~8.5–9.7 seconds
```

A resident inference service was then implemented.

The resident service loads the following once:

```text
CatBoost model
GRU model
Feature scaler
Feature configuration
```

It then processes each completed candle inside the same process.

Observed production inference performance:

```text
GRU inference:       ~6 ms
CatBoost inference: ~23 ms
Resident cycle:     ~510 ms
```

This reduced live inference latency by approximately 96% compared with the earlier subprocess-based pipeline.

---

# Live Monitoring

The monitoring system continuously evaluates production health.

## Data Health

- Data freshness
- Symbol completeness
- Candle continuity
- Missing intervals

## Prediction Health

- Prediction count
- Mean inference latency
- Maximum inference latency
- Mean confidence
- Maximum confidence
- HOLD / UP / DOWN counts

## Feature Monitoring

Feature z-score statistics are compared against the training distribution.

Monitoring includes:

```text
Mean |z|
Max |z|
Percentage of features with |z| > 3
```

## Production Status Example

```text
Data freshness:           PASS
Sequence continuity:      PASS
Prediction completeness:  PASS
Symbols:                  5/5
Predictions:              5/5
Mean inference latency:   ~510 ms
Feature |z| > 3:           0.00%
```

---

# SQLite Storage

The live system uses:

```text
data/crypto_live.db
```

Current tables:

```text
market_klines_1m
live_predictions
monitoring_metrics
```

## `market_klines_1m`

Stores:

- Open time
- Close time
- Open price
- High price
- Low price
- Close price
- Volume
- Quote volume
- Trade count
- Taker-buy volume
- Event time
- Receive time

## `live_predictions`

Stores:

- Symbol
- Candle timestamp
- Close price
- CatBoost probability
- GRU probability
- Ensemble probability
- Ensemble confidence
- Raw prediction
- Final signal
- Actionable flag
- Inference latency
- Model version
- Ensemble weights
- Confidence threshold

## `monitoring_metrics`

Stores:

- Data freshness
- Sequence continuity
- Prediction completeness
- Inference latency
- Confidence statistics
- Feature drift metrics
- Model version

---

# FastAPI

FastAPI exposes the production data layer to the dashboard.

Main endpoints:

```text
GET /
GET /health

GET /market/latest
GET /market/history

GET /predictions/latest
GET /predictions/history

GET /monitoring/latest
GET /monitoring/history
```

FastAPI reads the live SQLite database and provides the latest market, inference, and monitoring state to the dashboard.

---

# Streamlit Dashboard

The Streamlit dashboard provides a production-style control center.

## Main Dashboard

The dashboard includes:

- Live pipeline status
- Production model version
- Five-coin market overview
- Current price
- UP probability
- Prediction confidence
- Ensemble market chart
- Historical UP probability
- Historical confidence
- MLOps health

## Individual Coin View

Each asset has a detailed view containing:

- Current price
- Ensemble signal
- CatBoost probability
- GRU probability
- Ensemble probability
- Inference latency
- Candlestick chart
- EMA 20
- Bollinger Bands
- Volume
- Prediction history
- Confidence history
- MLOps health

---

# Docker Architecture

The project runs locally using Docker Compose.

Services:

```text
ingestion
inference
api
dashboard
```

Architecture:

```text
┌───────────────────────────────────────────────┐
│                Docker Compose                 │
│                                               │
│  ┌─────────────┐       ┌─────────────────┐   │
│  │  ingestion  │──────▶│   SQLite DB     │   │
│  └─────────────┘       └────────┬────────┘   │
│                                 │            │
│                                 ▼            │
│                        ┌─────────────────┐   │
│                        │    inference    │   │
│                        │ CatBoost + GRU  │   │
│                        └────────┬────────┘   │
│                                 │            │
│                                 ▼            │
│                        ┌─────────────────┐   │
│                        │      FastAPI    │   │
│                        └────────┬────────┘   │
│                                 │            │
│                                 ▼            │
│                        ┌─────────────────┐   │
│                        │    Streamlit    │   │
│                        └─────────────────┘   │
│                                               │
└───────────────────────────────────────────────┘
```

---

# Running the Project

## Requirements

- Windows / Linux / macOS
- Docker Desktop
- Docker Compose
- Git
- Internet connection for Binance live market data

No paid cloud infrastructure is required.

The project was specifically designed around:

```text
₹0 / $0 cloud cost
```

using local Docker resources and public Binance market data.

---

## Clone the Repository

```bash
git clone https://github.com/rohithap0819/crypto-mlops.git
cd crypto-mlops
```

---

# Start the Full System

Build and start all services:

```bash
docker compose up -d --build
```

Check container status:

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

# Dashboard

Open:

```text
http://localhost:8501
```

---

# FastAPI

Open:

```text
http://localhost:8000
```

Health endpoint:

```text
http://localhost:8000/health
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

---

# View Logs

## Ingestion

```bash
docker compose logs -f ingestion
```

## Inference

```bash
docker compose logs -f inference
```

## API

```bash
docker compose logs -f api
```

## Dashboard

```bash
docker compose logs -f dashboard
```

---

# Backfill Recent Market Data

If the live stream develops a recent data gap, use the Binance REST backfill utility:

```bash
docker compose exec inference python src/ingestion/backfill_recent_binance.py
```

The backfill process retrieves recent closed candles and stores them in SQLite.

This is useful after:

- Restarting the system
- Temporary network interruptions
- Binance WebSocket disconnections
- Missing recent candles

The inference pipeline validates candle continuity before generating predictions.

---

# Stop the System

```bash
docker compose down
```

Restart the dashboard:

```bash
docker compose restart dashboard
```

Rebuild the dashboard after source changes:

```bash
docker compose up -d --build dashboard
```

---

# Project Structure

```text
crypto-mlops/
│
├── src/
│   │
│   ├── ingestion/
│   │   ├── binance_ws_ingestor.py
│   │   ├── bootstrap_sqlite_from_historical.py
│   │   └── backfill_recent_binance.py
│   │
│   ├── features/
│   │   ├── build_features_v2.py
│   │   └── live_feature_builder.py
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
│   ├── inference/
│   │   ├── live_inference_runner.py
│   │   └── resident_inference_service.py
│   │
│   ├── monitoring/
│   │   └── monitor_live.py
│   │
│   ├── api/
│   │   └── main.py
│   │
│   └── dashboard/
│       └── app.py
│
├── data/
│   ├── processed/
│   └── crypto_live.db
│
├── models/
│   ├── final/
│   └── binary_sequence_v1/
│
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

# MLOps Components

The project implements the following MLOps practices.

## Experiment Tracking

MLflow is used during model development and experiment tracking.

Experiments include:

```text
CryptoML_Regression
CryptoML_Classification
CryptoML_Optuna
CryptoML_Binary_Sequence
CryptoML_Final_Binary
CryptoML_Production_Ensemble
```

Tracked information includes:

- Model configuration
- Hyperparameters
- Evaluation metrics
- Feature versions
- Sequence configuration
- Ensemble weights
- Confidence threshold
- Model metadata

---

## Model Versioning

Production model version:

```text
catboost_v1_gru_v1_ensemble_v1
```

The production ensemble is registered with its:

- Feature version
- Feature list
- Ensemble weights
- Confidence threshold
- Model metadata
- Git commit information
- Checksums

---

## Feature Consistency

The live feature pipeline reuses the V2 feature-generation logic used during model development.

This reduces the risk of training-serving feature mismatch.

---

## Model Scaler

The GRU sequence model uses the scaler fitted on training data rather than fitting a new scaler on live data.

This ensures consistent feature scaling between training and production inference.

---

## Monitoring

The production monitoring layer checks:

- Data freshness
- Sequence continuity
- Prediction completeness
- Inference latency
- Prediction confidence
- Feature drift

---

## Containerization

The entire application is deployed using Docker Compose.

This allows the ingestion, inference, API, and dashboard services to run as separate components.

---

# MLOps Design Decisions

Several engineering decisions were made based on measured experiments rather than assumptions.

## Binary Direction Instead of 3-Class Direction

The original target used:

```text
DOWN
NEUTRAL
UP
```

The binary direction target performed better and provided a more appropriate signal formulation for the production experiment.

---

## Ensemble Instead of a Single Model

The production ensemble combines two modeling approaches:

```text
CatBoost -> engineered tabular features
GRU      -> temporal sequence information
```

This allows the system to combine different representations of the market state.

---

## Resident Inference Service

The initial subprocess-based live inference pipeline was replaced by a resident service that loads models once.

This substantially reduced inference overhead.

---

## Confidence Threshold

Low-confidence predictions are converted to:

```text
HOLD
```

instead of forcing an UP or DOWN decision.

---

## Execution-Aware Evaluation

Backtesting explicitly considered:

- Signal timing
- Entry timing
- Holding period
- Position overlap
- Transaction costs

This prevents unrealistic backtesting assumptions.

---

# Important Findings

One of the most important conclusions of the project is that **better classification metrics do not automatically imply a profitable trading strategy**.

The experiments showed:

- Very short-horizon return prediction has extremely weak signal.
- Binary direction classification performs only modestly better than naive baselines.
- CatBoost and GRU provide complementary modeling approaches.
- Their ensemble provides a small measurable improvement.
- Confidence filtering increases selectivity while reducing coverage.
- Transaction costs can eliminate apparent high-frequency strategy performance.

The project therefore treats the prediction system as an **experimental predictive component**, while focusing on reliable data ingestion, inference, monitoring, reproducibility, and deployment.

---

# Limitations

## Predictive Signal

Cryptocurrency markets are highly noisy and non-stationary.

The observed predictive improvements are modest.

## Transaction Costs

Backtesting shows that realistic transaction costs can overwhelm the small predictive edge.

## Local Deployment

The system currently runs locally rather than on cloud infrastructure.

## Data Dependency

The system depends on Binance market availability and internet connectivity.

## SQLite

SQLite is appropriate for this local portfolio project, but larger production systems would typically use a more scalable database.

## Monitoring Scope

The current monitoring system focuses mainly on:

- data quality
- prediction health
- latency
- feature drift

Advanced model-performance monitoring requires delayed ground-truth labels and longer evaluation windows.

---

# Future Improvements

Potential future extensions include:

- Kafka-based streaming
- Redis caching
- PostgreSQL / TimescaleDB
- Cloud deployment
- Kubernetes
- Feature-store integration
- Automated retraining
- Automated model validation gates
- Model registry promotion workflows
- Delayed-label production monitoring
- Automated data-quality alerts
- Grafana / Prometheus integration
- SHAP-based explainability
- Multi-horizon prediction
- Online model evaluation
- CI/CD deployment pipelines
- Canary model deployment
- Automated rollback

---

# Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy, PyArrow |
| Machine Learning | Scikit-learn |
| Gradient Boosting | CatBoost, XGBoost, LightGBM |
| Deep Learning | PyTorch |
| Sequence Model | GRU |
| Experiment Tracking | MLflow |
| Database | SQLite |
| Market Data | Binance WebSocket / REST |
| API | FastAPI |
| Dashboard | Streamlit |
| Visualization | Plotly |
| Containerization | Docker |
| Orchestration | Docker Compose |
| Version Control | Git / GitHub |

---

# End-to-End Lifecycle

The project covers the full machine-learning lifecycle:

```text
Live Market Data
       ↓
Reliable Ingestion
       ↓
SQLite Storage
       ↓
Feature Engineering
       ↓
Experimentation
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Model Selection
       ↓
Model Registration
       ↓
Live Inference
       ↓
Prediction Persistence
       ↓
Monitoring
       ↓
FastAPI
       ↓
Streamlit Dashboard
       ↓
Docker Deployment
```

---

# Portfolio Highlights

This project demonstrates practical experience with:

- End-to-end MLOps architecture
- Real-time streaming data
- Data engineering
- Feature engineering
- Classical machine learning
- Deep-learning sequence models
- Ensemble modeling
- MLflow experiment tracking
- Model governance
- Live model inference
- Data-quality monitoring
- Feature drift monitoring
- API development
- Dashboard development
- Docker containerization
- Git/GitHub workflow
- Execution-aware backtesting
- Production-style debugging
- Service-oriented architecture

---

# Repository

GitHub:

https://github.com/rohithap0819/crypto-mlops

---

# Author

**Rohith AP**

---

# Disclaimer

This project is intended for **educational, engineering, and portfolio purposes**.

Cryptocurrency prediction is inherently uncertain. Model outputs should not be interpreted as guaranteed trading signals or financial advice.