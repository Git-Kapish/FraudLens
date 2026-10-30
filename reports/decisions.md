# Decisions

Judgment calls made while building, in order.

1. **Python 3.11 via uv.** The machine had 3.12-3.14 only. The venv is built with `uv venv --python 3.11 --seed .venv` so torch and shap wheels match the spec. `make setup` still uses `$(PY) -m venv` (default `python3.11`).
2. **Makefile on Windows.** The venv dir is `Scripts` on Windows and `bin` elsewhere, picked with `$(OS)`. Run with `mingw32-make` locally.
3. **Split by row position after sorting by Time.** `Time` has 1-second resolution, so a value shared across a boundary can land in two splits. Tests assert `train.max <= val.min`, not strict `<`.
4. **Dedup removes exact duplicate rows**, applied before the split (rule 7). The count is logged and recorded here once the real data is loaded.

## Measured on the real data
- Raw: 284,807 rows, 492 frauds. Dedup removed **1,081** exact duplicate rows, leaving 283,726.
- Split (train/val/test): 198,608 / 42,559 / 42,559 rows with **366 / 55 / 52** frauds. Val and test are small, so PR-AUC will be noisy there; bootstrap CIs are essential.
5. **MLflow file store opt-in.** MLflow 3.17 raises on `file:./mlruns` unless `MLFLOW_ALLOW_FILE_STORE=true`. The PRD asks for a local file store, so `train.init_mlflow()` sets that variable instead of switching to sqlite. `mlruns/` is gitignored.
6. **Raw `Time` is not a model feature.** It is replaced by an hour-of-day feature (relative to the first transaction, since the real clock offset is unknown). `log1p(Amount)` and hour are standardised with train statistics; V1-V28 pass through unscaled.

## Phase 4
7. **Calibrator is fit on out-of-fold TRAIN predictions, not on validation.** TASKS.md says "fit on validation", but CLAUDE.md rule 2 says calibrators are fit on train only. A calibrator fit on in-sample train scores would be useless, so I use `TimeSeriesSplit` out-of-fold scores on train. This follows the non-negotiable rule and keeps validation clean for threshold selection. The earliest ~1/6 of train rows never appear as a test fold and so are not used for calibration. Isotonic is monotone, so ranking metrics are unchanged except for ties.
8. **Optuna objective is CV PR-AUC on `TimeSeriesSplit(5)` over train only.** PRD FR8 says "train plus validation"; validation is instead used for model selection, the calibration check and the cost threshold, so it stays out of the tuning loop. `scale_pos_weight` is recomputed from each fold's train labels.
9. **Deployed model is limited to the XGBoost family** (`training.deployable_models`: M3, M3 tuned, M4), chosen by validation PR-AUC. Reasons: TreeExplainer for `/explain` and the 50 ms p95 target for `/predict`. Logistic regression and random forest are still compared in the results table.
10. **Ensemble uses train-score percentiles, not batch ranks**, so a single API request can be scored on its own. Percentiles come from the stored sorted train scores of each component.
11. **Autoencoder standardises inputs with the mean/std of legit-only train rows** (30-16-8-16-30, 15 epochs, Adam 1e-3, batch 512, CPU).

### Phase 4 results (validation set, 55 frauds; test untouched)
| Model | Val PR-AUC |
|---|---|
| Dummy (prior) | 0.0013 |
| Logistic Regression | 0.8267 |
| Random Forest | 0.8625 |
| XGBoost (scale_pos_weight) | 0.8578 |
| XGBoost + SMOTE | 0.8176 |
| Isolation Forest | 0.0402 |
| Autoencoder | 0.0602 |
| XGBoost tuned (Optuna, 30 trials) | 0.8368 |
| Rank-average (tuned XGB + AE) | 0.7510 |

- **Tuning did not help on validation.** Best CV PR-AUC on train was 0.7799, but the tuned model scores 0.8368 on val vs 0.8578 for untuned M3. With 55 val frauds the gap is within noise; I report it as is and deploy by the rule (best val PR-AUC among deployable models), which picks **M3**. No re-tuning to chase val.
- **Unsupervised models are weak** (0.04 and 0.06). Expected: fraud here is not simply "far from normal" in these PCA components. They are reported, not hidden.
- **Ensemble is worse than its best member** (0.751), because the weak autoencoder dilutes the XGBoost ranking. Reported as a negative result.
- **Calibration:** val Brier is 0.00031 both raw and calibrated, so calibration changes little here. The reliability curve is only informative in the top bin: at 0.13% fraud, quantile bins below that hold almost no frauds.
- **Val PR-AUC of 0.82-0.86 is for validation only.** Do not expect the same on test; the time split and 52 test frauds will make that number noisier.

## Phase 5
12. **CI models chosen on validation.** "Top 2 models" for bootstrap CIs are the top 2 by validation PR-AUC (Random Forest, XGBoost) plus the deployed model, so test results never pick which models get reported.
13. **Each model gets its own val-chosen cost threshold** for the operating-point table, including non-probability scorers (isolation forest, autoencoder). The 0.5 baseline is only reported for models with `predict_proba`.
14. **Flag-nothing is a candidate threshold** (just above the top score), so a high review cost can legitimately choose it. Ties between equal-cost thresholds go to the higher one (fewer flags).
15. **Deployed model is not switched after seeing test.** Random Forest scores higher on test (0.778 vs 0.749 deployed) but within overlapping CIs, and the selection rule was frozen on validation. Reported in results.md caveats.
16. **Honest result on the cost model:** on test, the val-chosen threshold (0.2812, cost 2,632) is no better than a plain 0.5 (cost 2,592) for the calibrated model. Both save about 57-58% vs flag-nothing. Not tuned further.
17. **Test-set results (deployed, calibrated XGBoost):** PR-AUC 0.749 [0.626, 0.858], recall at 90% precision 0.712 [0.559, 0.852], at the val-chosen threshold precision 0.750, recall 0.750 (TP 39, FP 13, FN 13), cost saving 3,536 (57.3%) vs flag-nothing at review cost 5. `review_cost` is an assumption.

## Phase 6
18. **SHAP explains the raw XGBoost margin, not the calibrated probability.** Contributions are log-odds of the `scale_pos_weight` model before the isotonic step (base value 3.71 because that weight inflates raw scores). They explain the ranking; isotonic is a monotone map afterwards. Additivity is tested (sum of contributions + base = margin).
19. **Global SHAP plots use a validation sample** (all 55 frauds + random legit rows, 5,000 total, seed 42), keeping the test set for the metrics and the three waterfall examples.
20. **Waterfall examples are picked by score only**: the highest, middle and lowest-scoring flagged true frauds on test. Not chosen for how tidy the explanation looks. Top features by mean |SHAP| on val: V4, V14, V12, V10, V11. These are anonymised PCA components, so no business meaning is claimed.
21. **`make evaluate` now also runs `fraud.explain`**, so SHAP figures and `reports/shap_examples.md` regenerate with the rest of the reports.
