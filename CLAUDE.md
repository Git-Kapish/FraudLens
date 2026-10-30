# CLAUDE.md: Fraud Detection System

Read `PRD.md` first. It is the source of truth for scope. Follow `TASKS.md` phase by phase. Do not drop any deliverable; if a phase runs long, simplify its polish, not its scope.

## Project in one line
End-to-end credit card fraud detection on the ULB Kaggle dataset: time-split evaluation, 6+ models, cost-based threshold, SHAP, FastAPI scoring service, Streamlit dashboard, Docker, CI.

## Stack
Python 3.11, pandas, numpy, scikit-learn, imbalanced-learn, XGBoost (or LightGBM), Optuna, PyTorch (small autoencoder), SHAP, MLflow, FastAPI, Uvicorn, Pydantic v2, Streamlit, Plotly, pytest, ruff, Docker, GitHub Actions.

## Layout
```
src/fraud/        # all library code (data, features, models, train, evaluate, cost, explain, api/)
app/dashboard.py  # Streamlit
scripts/          # download_data.py, replay.py
notebooks/        # 01_eda.ipynb
tests/            # pytest
models/           # saved artifacts (gitignored)
reports/          # figures/ and results.md
config.yaml       # all tunables (seed, split ratios, cost model, paths)
```
Library code lives in `src/fraud/` and is imported by the scripts, API, dashboard, and notebook. No logic duplicated in notebooks.

## Non-negotiable ML rules
1. **Chronological split by `Time`**: 70% train, 15% val, 15% test. Never random split.
2. **No leakage**: scalers, resamplers, calibrators and any fitted transform are fit on train only. Validation and test are transform-only.
3. SMOTE (or any resampling) is applied only to training data, inside CV folds when tuning.
4. Tune with `TimeSeriesSplit` on train plus the validation set. **The test set is evaluated once**, after all choices are frozen.
5. **Headline metric is PR-AUC (average precision).** Never report accuracy as a result. ROC-AUC is secondary.
6. Choose the decision threshold on **validation** by minimising the cost model in `config.yaml`, then report that same threshold on test.
7. Deduplicate before splitting and document how many rows were removed.
8. Fixed seed (42) everywhere: numpy, sklearn, xgboost, torch, optuna.
9. Report real numbers, whatever they are. Do not tune to hit the PRD targets, and do not hide a low time-split score. Explain it.
10. `V1`-`V28` are anonymised PCA components. Never invent business meaning for them.

## Cost model (assumption, keep configurable)
- False negative cost = transaction `Amount`
- False positive cost = `review_cost` (default 5.0)
- True positive cost = `review_cost`
- True negative cost = 0
Baselines to compare against: flag-nothing, and threshold 0.5.
State in the README that `review_cost` is an assumption, not a measured value.

## API rules
- Pydantic v2 models in `src/fraud/api/schemas.py`; all 30 features required; `Amount >= 0`.
- Load model, threshold, and SHAP explainer once at startup (lifespan), not per request.
- Endpoints: `/health`, `/predict`, `/predict/batch` (max 1000), `/explain`, `/metrics`.
- Return `model_version` and `latency_ms` in prediction responses.
- Return 422 on invalid input via Pydantic; never crash on bad payloads.
- Target p95 under 50 ms for `/predict` locally.

## Code style
- Type hints on public functions, short docstrings, small pure functions.
- `ruff` for lint and format; keep `make lint` clean.
- Config via `config.yaml` loaded in `src/fraud/config.py`; no magic numbers in code.
- Log with the `logging` module, not `print`, inside `src/`.
- Prefer clear code over clever code.

## Testing (pytest)
Must cover: split ordering and no time overlap, no fitting on val/test (assert scaler stats equal train stats), cost function on hand-computed cases, API schema validation, API happy path and error path. Run `make test` before finishing each phase.

## Commands
```
make setup       # venv + install
make data        # download and validate dataset
make train       # train all models, log to MLflow, save best + threshold
make evaluate    # test-set metrics, CIs, cost analysis, figures, reports/results.md
make serve       # uvicorn src.fraud.api.main:app --reload
make dashboard   # streamlit run app/dashboard.py
make test        # pytest with coverage
make lint        # ruff check + format check
make docker      # docker compose up --build
```

## Working agreement
- Work through `TASKS.md` in order; tick each item when done and verified.
- After each phase: run `make test` and `make lint`, then give a 2-3 line summary of what was built and any numbers worth noting.
- If the dataset cannot be downloaded (no Kaggle credentials), stop and print the manual download steps instead of generating fake data. For unit tests only, a small synthetic fixture in `tests/` is fine.
- Never commit `data/`, `mlruns/`, or `kaggle.json`. Trained `models/*.joblib` files ARE committed (small, user decision).
- When a decision is ambiguous, pick the simpler option, note it in `reports/decisions.md`, and continue.
