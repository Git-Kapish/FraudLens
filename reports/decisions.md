# Decisions

Judgment calls made while building, in order.

1. **Python 3.11 via uv.** The machine had 3.12-3.14 only. The venv is built with `uv venv --python 3.11 --seed .venv` so torch and shap wheels match the spec. `make setup` still uses `$(PY) -m venv` (default `python3.11`).
2. **Makefile on Windows.** The venv dir is `Scripts` on Windows and `bin` elsewhere, picked with `$(OS)`. Run with `mingw32-make` locally.
3. **Split by row position after sorting by Time.** `Time` has 1-second resolution, so a value shared across a boundary can land in two splits. Tests assert `train.max <= val.min`, not strict `<`.
4. **Dedup removes exact duplicate rows**, applied before the split (rule 7). The count is logged and recorded here once the real data is loaded.

## Measured on the real data
- Raw: 284,807 rows, 492 frauds. Dedup removed **1,081** exact duplicate rows, leaving 283,726.
- Split (train/val/test): 198,608 / 42,559 / 42,559 rows with **366 / 55 / 52** frauds. Val and test are small, so PR-AUC will be noisy there; bootstrap CIs are essential.
