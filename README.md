# FraudLens: credit card fraud detection, end to end

Fraud is about 0.17% of transactions, so accuracy is meaningless: predicting "legit" for everything scores 99.8%. This project goes from raw data to a served model, and tries to be honest about each step: a **chronological** train/val/test split, no fitting on validation or test data, **PR-AUC** as the headline metric, a threshold chosen by **business cost** rather than 0.5, SHAP explanations, a scoring API and an analyst dashboard.

Dataset: [ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions over 2 days, 492 frauds). `V1`-`V28` are anonymised PCA components, so this project never attaches business meaning to them.

## Results at a glance

Test set = the last 15% of transactions in time, **52 frauds**. Every number below is from `reports/results.md`, produced by `make evaluate`; none were tuned to hit a target.

| Model | Test PR-AUC | Recall @ 90% precision |
|---|---|---|
| Random Forest | 0.778 | 0.750 |
| XGBoost + SMOTE | 0.769 | 0.731 |
| XGBoost, tuned with Optuna | 0.766 | 0.750 |
| XGBoost (`scale_pos_weight`) | 0.765 | 0.750 |
| **Deployed: XGBoost, calibrated** | **0.749** | **0.712** |
| Logistic Regression | 0.699 | 0.673 |
| Rank-average (tuned XGB + autoencoder) | 0.594 | 0.000 |
| Autoencoder (legit-only) | 0.132 | 0.000 |
| Isolation Forest | 0.048 | 0.000 |
| Dummy (prior) | 0.001 | 0.000 |

Bootstrap 95% confidence intervals (500 resamples):

| Model | Test PR-AUC | Recall @ 90% precision |
|---|---|---|
| Random Forest | 0.778 [0.665, 0.872] | 0.750 [0.616, 0.855] |
| XGBoost | 0.765 [0.645, 0.872] | 0.750 [0.600, 0.854] |
| Deployed (calibrated XGBoost) | 0.749 [0.626, 0.858] | 0.712 [0.559, 0.852] |

How to read this:

- The intervals overlap almost completely. With 52 test frauds, the top four supervised models cannot be told apart.
- The PRD target of PR-AUC >= 0.75 is narrowly missed by the deployed model (0.749) and met by the Random Forest (0.778), with intervals that span the target either way. The deployed model was chosen on validation before test was scored, and was not switched afterwards.
- Test PR-AUC is below validation (0.847 for the deployed model). That is expected: fraud patterns drift over time, and a chronological split never lets the model see the future. Published numbers on this dataset mostly use random splits, which leak time and look better.
- Isolation Forest and the autoencoder are weak on their own, and mixing the autoencoder into a rank-average ensemble made things worse. They are reported anyway.
- ROC-AUC (0.979 for the deployed model) is shown in `reports/results.md` but is optimistic here, because the huge legit class keeps the false positive rate tiny.

### Cost analysis

A missed fraud costs its `Amount`; every flagged transaction costs a review (`review_cost`, default 5.0); a true negative costs nothing. **`review_cost` is an assumption, not a measured value**, and every cost figure depends on it. The threshold (0.2812) was chosen on validation, then applied to test:

| Strategy | Test cost | Saving vs flag-nothing |
|---|---|---|
| Flag nothing | 6,169 | |
| Deployed model, validation-chosen threshold | 2,632 | 3,536 (57.3%) |
| Plain 0.5 threshold | 2,592 | 3,576 (58.0%) |

At that threshold: precision 0.750, recall 0.750 (39 frauds caught, 13 false alarms, 13 missed). The cost-optimal threshold is **not** better than a plain 0.5 on test (by a few transactions), so the cost model shows no reliable advantage over 0.5 for this calibrated model. Savings versus flag-nothing are 60.7%, 57.3%, 53.1% and 43.3% for review costs 1, 5, 10 and 25. Amounts are in the dataset's currency units (the dataset does not say which currency).

### API latency

500 sequential `POST /predict` calls on localhost, one worker: **p95 36 ms** round trip (27 ms server-side), under the 50 ms target. Three idle runs gave 29, 37 and 36 ms, so expect some spread on a busy laptop.

## Architecture

```
            Kaggle CSV
                |
        scripts/download_data.py
                |
   fraud.data       load, validate, dedupe, chronological split
                |
   fraud.features   log1p(Amount), hour-of-day; scaler fit on train only
                |
   fraud.models / anomaly / tune / calibrate   M0-M7, Optuna, isotonic  --> MLflow
                |
   fraud.evaluate / cost / report   metrics, bootstrap CIs, cost threshold, results.md
                |
        models/  (joblib files, metadata.json, threshold.json)
           |                              |
   fraud.api (FastAPI)            app/dashboard.py (Streamlit)
           ^                              |
           +------ fraud.replay ----------+
```

All logic lives in `src/fraud/` and is imported by the scripts, API, dashboard and notebook.

## Quick start

Requires Python 3.11. The commands below use `make`; on Windows without `make`, `mingw32-make` works, or run the underlying commands from the Makefile.

```bash
make setup      # venv + pinned dependencies
make data       # download the dataset (needs ~/.kaggle/kaggle.json), or follow the printed manual steps
make train      # train M0-M7, tune, calibrate, save to models/   (about 10 minutes on a laptop CPU)
make evaluate   # score the test set once, CIs, cost analysis, SHAP, reports/results.md
make serve      # API on http://localhost:8000  (docs at /docs)
make dashboard  # dashboard on http://localhost:8501
make test       # pytest with coverage
make lint       # ruff check + format check
make docker     # API + dashboard in containers
```

The trained model files are committed (about 10 MB), so `make serve` and `make dashboard` work on a fresh clone without retraining; the dashboard also needs the dataset (`make data`).

If the Kaggle download fails, `make data` stops and prints manual steps. It never generates fake data.

### Docker

```bash
make data            # the dashboard reads data/raw/creditcard.csv (mounted, not baked in)
docker compose up --build
```

API on `:8000`, dashboard on `:8501`. One image serves both (see `Dockerfile`, `docker-compose.yml`); the runtime image uses `requirements-serve.txt`, which has no torch, mlflow or jupyter.

### Using the API

```bash
curl -X POST localhost:8000/predict -H "content-type: application/json" \
  -d '{"Time": 0, "V1": 0, ..., "V28": 0, "Amount": 25.0}'
```

All 30 features are required and `Amount` must be >= 0; anything else returns a 422. Responses include `fraud_probability`, `is_fraud`, `threshold`, `model_version` and `latency_ms`.

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness, model version, threshold |
| `POST /predict` | one transaction |
| `POST /predict/batch` | up to 1000 transactions |
| `POST /explain` | prediction plus top-5 signed SHAP contributions |
| `GET /metrics` | request counts, flag rate, p50/p95 latency |

Request logs are one JSON line each (id, score, latency); feature values are never logged.

## Demo flow (2-3 minutes)

1. `make serve` and `make dashboard`.
2. **Overview**: 473 frauds in 283,726 rows; fraud rate over time with the split boundaries marked.
3. **Model comparison**: the test table, PR and ROC overlays and the bootstrap intervals. Point out how much the intervals overlap.
4. **Threshold and cost**: set review cost to 25 and press "Use validation-optimal threshold" (the slider jumps to ~0.43); watch precision, recall and total cost change.
5. **Transaction explorer**: click a high-scoring row to see its SHAP contributions.
6. **Live replay**: choose "Around a fraud", press Start, and watch transactions stream through the API with a running cost-saved total.

## Design decisions

The full list, with the reasons, is in [`reports/decisions.md`](reports/decisions.md). The ones that matter most:

- **Chronological split, dedup first.** 1,081 exact duplicates are removed before splitting, then 70/15/15 by `Time`. Splits have 366 / 55 / 52 frauds.
- **No leakage.** The scaler is fit on train only (tested); SMOTE sits inside the training pipeline so it never touches validation or test; the calibrator is fit on **out-of-fold train** predictions, not validation, so validation stays clean for threshold selection. This deviates from TASKS.md ("fit on validation") to follow the project's no-leakage rule.
- **Test set scored once**, after every choice was frozen. Optuna tuned on time-ordered CV over train only.
- **The tuned model did not beat the untuned one on validation** (0.837 vs 0.858), so the untuned XGBoost is the one deployed. No re-tuning to chase it.
- **Deployment limited to the XGBoost family**, for TreeExplainer and the latency target.
- **SHAP explains the raw XGBoost score** (log-odds, before the isotonic step); it explains the ranking, not calibrated probabilities.
- **Bug found by a test:** FastAPI's default 422 response echoes the bad input, and a `NaN` there turned into a 500. A custom handler returns the location and reason only.

## Limitations

- **Small test set.** 52 frauds in test and 55 in validation, so all results are noisy. Differences between the top models are within the bootstrap intervals.
- **Anonymised features.** `V1`-`V28` are PCA components; explanations name them but cannot say what they mean in the real world.
- **Two days of data, one dataset, one set of European cardholders from 2013.** Nothing here shows how the model would behave on other periods, regions or fraud patterns, or under concept drift.
- **The cost model is an assumption.** `review_cost` is not measured, and treating `Amount` as the cost of a miss ignores chargeback fees, customer friction and repeated fraud on the same card.
- **Validation is reused.** It is used to pick the deployed model, to compare calibration and to choose the threshold, so the validation numbers are optimistic (validation cost at the chosen threshold is far below test cost). The test set is the number to trust.
- **Calibration is weak evidence.** At 0.1% fraud, most reliability-curve bins hold almost no frauds, and isotonic calibration creates tied scores that lower PR-AUC slightly (0.765 raw vs 0.749 calibrated).
- **Latency** was measured on one laptop with one worker and sequential calls; it says nothing about throughput under concurrent load.
- **Not production-grade.** No authentication, rate limiting, drift monitoring or retraining pipeline; metrics are in memory and reset on restart.

## Reproducibility

Dependencies are pinned. Seed 42 is set for numpy, sklearn, XGBoost, PyTorch and Optuna. The models are pickles, so they must load with the same scikit-learn and XGBoost versions they were trained with (hence the pins). See `reports/decisions.md` for the clean-clone check.

## Repository layout

```
src/fraud/    data, features, models, anomaly, tune, calibrate, train, evaluate, cost, report, explain, replay, api/
app/          dashboard.py (Streamlit)
scripts/      download_data.py, replay.py
notebooks/    01_eda.ipynb
tests/        pytest suite
models/       trained artifacts, metadata.json, threshold.json
reports/      results.md, decisions.md, shap_examples.md, figures/
config.yaml   every tunable (seed, split, cost model, training, API)
```
