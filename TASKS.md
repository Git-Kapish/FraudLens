# TASKS.md: Build Plan (6 hours)

Work in order. Tick items when done and verified. Budget in brackets.

## Phase 1: Scaffold, data, split [0:40]
- [x] Create repo structure from PRD section 10; add `pyproject.toml`, `requirements.txt`, `.gitignore`, `Makefile`, `config.yaml`
- [x] `src/fraud/config.py` loads `config.yaml`
- [x] `scripts/download_data.py`: Kaggle API download of `mlg-ulb/creditcardfraud` into `data/raw/`; print manual steps if credentials missing
- [x] `src/fraud/data.py`: load, schema validation, null and dtype report, duplicate removal (log count)
- [x] Chronological split 70/15/15 by `Time`; print fraud count per split; error if any split has fewer than 20 frauds
- [x] `tests/test_data.py`: ordering, no overlap, split sizes, dedup before split
- [x] Verify: `make data && make test`

## Phase 2: EDA [0:35]
- [x] `notebooks/01_eda.ipynb` using `src/fraud` imports
- [x] Class imbalance, Amount and Time distributions by class, top feature separation, correlation heatmap, fraud rate by hour
- [x] Save figures to `reports/figures/`
- [x] Write 5-6 bullet takeaways at the end of the notebook

## Phase 3: Baselines and core models [1:05]
- [x] `src/fraud/features.py`: Amount (log1p, scaled), hour-of-day from Time, scaler fit on train only
- [x] `src/fraud/models.py`: factory for M0 Dummy, M1 LogReg (balanced), M2 RandomForest (balanced_subsample), M3 XGBoost with `scale_pos_weight`, M4 XGBoost + SMOTE (train only)
- [x] MLflow logging (params, metrics, model artifact) for every run
- [x] `tests/test_features.py`: scaler fit on train only
- [x] Verify: validation PR-AUC table printed for M0-M4

## Phase 4: Anomaly models, tuning, calibration [0:55]
- [x] M5 Isolation Forest (fit on train, score = negative decision function)
- [x] M6 small autoencoder (30-16-8-16-30), trained on legit-only train rows, under 20 epochs, score = reconstruction error
- [x] Optuna tuning of XGBoost, at least 30 trials, PR-AUC on `TimeSeriesSplit` over train
- [x] Calibrate best supervised model (isotonic or Platt, fit on validation) and plot reliability curve — done with isotonic on out-of-fold train scores instead, see decisions.md #7
- [x] Optional ensemble of tuned XGBoost and autoencoder scores (rank-average), evaluated like the rest
- [x] Save best model, scaler, and metadata to `models/`

## Phase 5: Evaluation and cost [0:40]
- [x] `src/fraud/evaluate.py`: PR-AUC, ROC-AUC, precision, recall, F1, F2, recall at 90% precision, precision at 80% recall, confusion matrix
- [x] Bootstrap 95% CI (at least 500 resamples) on test PR-AUC for top 2 models
- [x] `src/fraud/cost.py`: cost function, best threshold on validation, savings versus flag-nothing and threshold 0.5
- [x] Cost-versus-threshold curve and review-cost sensitivity table {1, 5, 10, 25}
- [x] PR and ROC overlay plots for all models
- [x] Test set evaluated once; `reports/results.md` generated with tables and figures
- [x] `tests/test_cost.py` with hand-computed cases
- [x] Save `models/threshold.json`

## Phase 6: Explainability [0:20]
- [x] `src/fraud/explain.py`: SHAP TreeExplainer on the best tree model
- [x] Global summary and bar plots to `reports/figures/`
- [x] Function returning top-5 signed feature contributions for one transaction
- [x] Waterfall plots for 3 example flagged test transactions

## Phase 7: FastAPI service [0:40]
- [x] `src/fraud/api/schemas.py` Pydantic v2 models (30 features, `Amount >= 0`)
- [x] `src/fraud/api/main.py` with lifespan loading of model, threshold, explainer
- [x] Endpoints: `/health`, `/predict`, `/predict/batch`, `/explain`, `/metrics`
- [x] Structured request logging (id, score, latency only)
- [x] `tests/test_api.py` using `TestClient`: happy path, missing field, negative Amount, batch limit
- [x] Measure p95 latency over 500 calls with `scripts/replay.py` and record it in `reports/results.md`

## Phase 8: Dashboard and replay [0:35]
- [x] `app/dashboard.py` with 5 pages: Overview, Model comparison, Threshold and cost (live sliders), Transaction explorer (SHAP per row), Live replay
- [x] `scripts/replay.py` streams test transactions in time order to the API with adjustable speed
- [x] Live replay page shows running flags, alert rate, and cumulative cost saved
- [x] Verify all pages load with no errors on trained artifacts

## Phase 9: Ship [0:30]
- [ ] `Dockerfile` (API) and `docker-compose.yml` (API + dashboard)
- [ ] `.github/workflows/ci.yml`: ruff + pytest
- [ ] README: overview, architecture diagram, results table with CIs, cost analysis, how to run, design decisions, assumptions (review cost), limitations (anonymised features, 2-day window, 492 frauds, single dataset)
- [ ] `reports/decisions.md` with any judgment calls made
- [ ] Final check against PRD section 15 acceptance checklist
- [ ] Clean-clone test: `make setup data train evaluate` works

## Definition of done
Every item in PRD section 15 is ticked, `make test` and `make lint` pass, and `README.md` reports actual measured numbers.
