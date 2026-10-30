# PRD: Real-Time Credit Card Fraud Detection System

| | |
|---|---|
| **Owner** | Kapish |
| **Type** | End-to-end ML project (data science + ML engineering) |
| **Build budget** | 6 hours, full scope (see Milestones) |
| **Status** | Ready to build with Claude Code |

---

## 1. Overview

Build an end-to-end fraud detection system on highly imbalanced credit card transaction data. It covers the full path from raw data to a served model: rigorous EDA, leakage-free time-based evaluation, multiple modelling approaches (supervised and anomaly-based), cost-sensitive threshold selection, explainability, a real-time scoring API, and an analyst dashboard.

The goal is a portfolio project that shows ML judgement, not only model fitting: correct metrics for imbalance, no leakage, business-cost framing, and production-minded serving.

## 2. Problem Statement

Fraud is rare (about 0.17% of transactions in the chosen dataset). Accuracy is meaningless: a model that predicts "legit" for everything scores 99.8%. A useful system must:

1. Catch as much fraud as possible (recall).
2. Avoid flooding analysts with false alarms (precision).
3. Pick an operating point based on business cost, not a default 0.5 threshold.
4. Explain why a transaction was flagged.
5. Score a transaction in real time.

## 3. Goals

| # | Goal |
|---|------|
| G1 | Reproducible pipeline from raw CSV to trained, saved model |
| G2 | Honest evaluation with a chronological split and imbalance-aware metrics |
| G3 | Compare at least 4 approaches against a simple baseline |
| G4 | Cost-based threshold selection with a stated cost model |
| G5 | Per-transaction explanations (SHAP) |
| G6 | REST API for real-time scoring |
| G7 | Analyst dashboard for exploring flagged transactions and results |
| G8 | Containerised, tested, documented repo |

## 4. Non-Goals

- Real payment-network integration or real customer data.
- Streaming infrastructure (Kafka etc.). Real-time is demonstrated through the API and a transaction replay simulator.
- Deep learning beyond a small autoencoder used as an anomaly-detection comparison.

## 5. Dataset

**Primary: ULB Credit Card Fraud Detection (Kaggle, `mlg-ulb/creditcardfraud`)**

- 284,807 transactions over 2 days (European cardholders, Sept 2013)
- 492 frauds (0.172%)
- Features: `Time` (seconds from first transaction), `V1`-`V28` (PCA-anonymised), `Amount`, `Class` (target)

Notes and constraints:
- Features are already PCA-transformed, so feature engineering is limited to `Time`/`Amount`-derived features and interactions; domain-feature claims must not be invented.
- The CSV is not committed to git. `scripts/download_data.py` fetches it via the Kaggle API (needs `kaggle.json`) and prints manual download instructions if credentials are missing.
- Duplicates exist in this dataset (about 1,000 rows). They must be handled explicitly and documented, with dedup applied before splitting.

## 6. Users and Use Cases

| User | Need |
|------|------|
| **Fraud analyst** | See which transactions need review, ranked by risk, with reasons |
| **Risk manager** | Understand the cost trade-off and choose the operating threshold |
| **Integrating system** | Call an API with a transaction and get a score and decision |
| **Recruiter / reviewer** | Quickly see rigour: README, results table, live demo |

## 7. Functional Requirements

### 7.1 Data and EDA
- FR1: Load, validate schema, report nulls, duplicates, dtypes.
- FR2: EDA notebook covering class imbalance, `Amount` and `Time` distributions by class, feature separation (KDE or boxplots of top features by class), correlation structure, fraud rate by hour of day (derived from `Time`).
- FR3: Save key plots to `reports/figures/`.

### 7.2 Splitting (critical)
- FR4: Chronological split by `Time`: first 70% train, next 15% validation, last 15% test.
- FR5: Report fraud count in each split. Raise a clear error if any split has fewer than 20 frauds.
- FR6: Any scaler, imputer, resampler, or encoder is fit on **train only**.
- FR7: Resampling (SMOTE or similar) is applied only inside the training fold, never to validation or test.
- FR8: Hyperparameter tuning uses time-aware CV on the train set (`TimeSeriesSplit`) plus the validation set. The test set is touched once, at the end.

### 7.3 Modelling
Compare at least these, all reported on the same test set:

| ID | Model | Imbalance handling |
|----|-------|--------------------|
| M0 | Dummy baseline (predict by prior) | n/a |
| M1 | Logistic Regression | `class_weight="balanced"` |
| M2 | Random Forest | `class_weight="balanced_subsample"` |
| M3 | XGBoost or LightGBM | `scale_pos_weight` |
| M4 | XGBoost or LightGBM + SMOTE (train only) | resampling |
| M5 | Isolation Forest | unsupervised, fit on train |
| M6 | Autoencoder (small, PyTorch or Keras) | trained on legit-only train rows, anomaly score = reconstruction error |

- FR9: Tune M3 with Optuna (at least 30 trials) optimising PR-AUC on time-aware CV.
- FR10: Calibrate probabilities for the best supervised model (isotonic or Platt) and show a reliability curve.
- FR11: Optional stretch inside scope: a stacked or averaged ensemble of M3 and M6 scores, evaluated like the others.
- FR12: Log all runs (params, metrics, artifacts) with MLflow (local file store).

### 7.4 Evaluation
Primary metric: **PR-AUC (average precision)**. Also report:
- ROC-AUC (secondary, with a note on why it is optimistic here)
- Precision, recall, F1, F2 at the chosen threshold
- Recall at fixed precision (90%) and precision at fixed recall (80%)
- Confusion matrix at chosen threshold
- PR curve and ROC curve overlay for all models
- Bootstrap 95% confidence intervals on test PR-AUC for the top 2 models (at least 500 resamples), since the test set has few frauds

### 7.5 Cost-Sensitive Threshold
- FR13: Define the cost model in `config.yaml`:
  - False negative cost = transaction `Amount` (money lost)
  - False positive cost = fixed review cost (default **5.0**, configurable; this is an assumption, not a measured number)
  - True positive cost = review cost (frauds caught still get reviewed)
- FR14: Choose the threshold on the **validation** set to minimise total cost. Report the same threshold's cost on test.
- FR15: Report savings versus two alternatives: "flag nothing" and the default 0.5 threshold.
- FR16: Produce a cost-versus-threshold curve and a sensitivity table over review cost in {1, 5, 10, 25}.

### 7.6 Explainability
- FR17: Global SHAP summary and bar plot for the best tree model.
- FR18: Local SHAP explanation (waterfall or top-5 contributing features) for individual flagged transactions.
- FR19: The API's explain endpoint returns the top-5 features with signed contributions.
- Note in the README that `V1`-`V28` are anonymised, so explanations are in terms of component names, not business meaning.

### 7.7 Serving API (FastAPI)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Liveness and model version |
| `/predict` | POST | Single transaction in, `{fraud_probability, is_fraud, threshold, model_version, latency_ms}` out |
| `/predict/batch` | POST | List of transactions in, list of results out (max 1,000) |
| `/explain` | POST | Prediction plus top-5 SHAP contributions |
| `/metrics` | GET | Request count, flag rate, p50/p95 latency (simple in-memory counters or Prometheus format) |

Requirements:
- Pydantic request and response models with validation (all 30 features required, `Amount` >= 0).
- Model and threshold loaded once at startup from `models/`.
- p95 latency under 50 ms for a single `/predict` call locally (excluding `/explain`).
- Structured logging of each request (no raw card data exists in this dataset, but log only ids and scores).

### 7.8 Dashboard (Streamlit)
Pages:
1. **Overview**: dataset stats, class balance, fraud rate over time.
2. **Model comparison**: results table, PR and ROC overlays, bootstrap CIs.
3. **Threshold and cost**: interactive slider for threshold and review cost, live recomputation of precision, recall, and total cost on the test set.
4. **Transaction explorer**: ranked table of test transactions by fraud score, click a row to see its SHAP explanation.
5. **Live replay simulator**: replays test transactions in time order through the API and shows a running feed of flags, running cost saved, and alert rate.

### 7.9 Engineering
- FR20: `Makefile` targets: `setup`, `data`, `train`, `evaluate`, `serve`, `dashboard`, `test`, `lint`, `docker`.
- FR21: Unit tests for: split correctness (no time overlap, ordering), no-leakage checks, cost function, API schemas, API happy path and validation errors.
- FR22: `Dockerfile` for the API and `docker-compose.yml` running API and dashboard together.
- FR23: GitHub Actions workflow running lint and tests.
- FR24: README with architecture diagram, results table, how to run, design decisions, limitations.

## 8. Success Criteria

These are targets to aim for, not guarantees; the README must report actual numbers whatever they are.

| Metric (test set, time-split) | Target |
|---|---|
| PR-AUC of best model | At least 0.75 and clearly above Logistic Regression baseline |
| Recall at 90% precision | Reported with 95% CI |
| Cost savings vs flag-nothing | Positive and reported in currency and percent |
| API p95 latency | Under 50 ms |
| Test coverage on core modules | At least 70% |
| Reproducibility | `make setup data train evaluate` works on a clean clone with a fixed seed (42) |

If the time-split score lands well below published random-split numbers, that is expected and should be explained, not hidden.

## 9. Architecture

```
            Kaggle CSV
                |
        scripts/download_data.py
                |
   src/data  (load, validate, dedup, time split)
                |
   src/features (Time/Amount features, scaler fit on train)
                |
   src/models  (M0-M6 train, Optuna, calibration) --> MLflow
                |
   src/evaluation (metrics, bootstrap CI, cost model, plots)
                |
        models/  (best_model.joblib, threshold.json, explainer)
           |                           |
   src/api (FastAPI)           app/dashboard.py (Streamlit)
           ^                           |
           +---- replay simulator -----+
```

## 10. Repository Structure

```
fraud-detection/
├── CLAUDE.md
├── PRD.md
├── TASKS.md
├── README.md
├── Makefile
├── config.yaml
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── .gitignore
├── .github/workflows/ci.yml
├── data/
│   ├── raw/                  # gitignored
│   └── processed/            # gitignored
├── notebooks/
│   └── 01_eda.ipynb
├── scripts/
│   ├── download_data.py
│   └── replay.py
├── src/
│   └── fraud/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── models.py
│       ├── train.py
│       ├── evaluate.py
│       ├── cost.py
│       ├── explain.py
│       └── api/
│           ├── main.py
│           └── schemas.py
├── app/
│   └── dashboard.py
├── models/                   # gitignored except metadata
├── reports/
│   ├── figures/
│   └── results.md
└── tests/
    ├── test_data.py
    ├── test_cost.py
    ├── test_features.py
    └── test_api.py
```

## 11. Tech Stack

Python 3.11, pandas, numpy, scikit-learn, imbalanced-learn, XGBoost or LightGBM, Optuna, PyTorch (autoencoder), SHAP, MLflow, FastAPI, Uvicorn, Pydantic v2, Streamlit, Plotly, pytest, ruff, Docker, GitHub Actions.

## 12. Milestones (6-hour build)

| Phase | Scope | Time |
|---|---|---|
| 1 | Scaffold repo, config, data download, validation, dedup, time split, split tests | 0:40 |
| 2 | EDA notebook and figures | 0:35 |
| 3 | Baseline, LR, RF, XGB/LGBM, SMOTE variant, MLflow logging | 1:05 |
| 4 | Isolation Forest, autoencoder, Optuna tuning, calibration | 0:55 |
| 5 | Evaluation suite: metrics, bootstrap CIs, cost model, threshold, plots | 0:40 |
| 6 | SHAP explanations | 0:20 |
| 7 | FastAPI service, schemas, tests | 0:40 |
| 8 | Streamlit dashboard and replay simulator | 0:35 |
| 9 | Docker, CI, README, results writeup | 0:30 |
| | **Total** | **6:00** |

Scope is fixed. If a phase overruns, compress the polish of that phase rather than dropping a deliverable. The order above puts the riskiest ML work first and presentation last.

## 13. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Data leakage inflating scores | Chronological split, train-only fitting, dedicated tests |
| Very few frauds in val/test | Bootstrap CIs, report counts, avoid over-tuning on val |
| SMOTE overfitting or leakage | Apply inside train only; compare against class-weighting |
| Autoencoder eats time | Keep it tiny (e.g. 30-16-8-16-30), CPU, under 20 epochs |
| Kaggle credentials missing | Download script prints manual steps; documented in README |
| Over-claiming from anonymised features | README limitations section, no invented business meaning |

## 14. Deliverables

1. Working repo meeting section 10
2. `reports/results.md` with model comparison, CIs, cost analysis, figures
3. Trained model artifacts and threshold file
4. Running API and dashboard (via `docker compose up`)
5. README with architecture, results, limitations, and a 2-3 minute demo flow

## 15. Acceptance Checklist

- [x] Chronological split verified by test
- [x] No fit on validation or test data anywhere
- [x] 6+ models compared on identical test set, with baseline
- [x] PR-AUC is the headline metric, with bootstrap CI
- [x] Threshold chosen on validation by cost, reported on test
- [x] SHAP global and local plots exist
- [x] `/predict`, `/predict/batch`, `/explain`, `/health`, `/metrics` work and are tested
- [x] Dashboard has all 5 pages
- [x] `docker compose up` brings up both services
- [x] CI green
- [x] README states assumptions (review cost) and limitations
