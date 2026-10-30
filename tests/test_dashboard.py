"""Smoke test: every dashboard page renders on real artifacts without an exception.

Skipped where the dataset or trained models are absent (e.g. CI).
"""

import json
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from fraud.config import ROOT, resolve_path

APP = str(Path(ROOT) / "app" / "dashboard.py")
PAGES = [
    "Overview",
    "Model comparison",
    "Threshold and cost",
    "Transaction explorer",
    "Live replay",
]
needs_artifacts = pytest.mark.skipif(
    not (
        resolve_path("raw_data").exists()
        and (resolve_path("models_dir") / "threshold.json").exists()
        and (resolve_path("results_md").parent / "test_results.json").exists()
    ),
    reason="dataset or trained artifacts missing",
)


def _api_up() -> bool:
    try:
        return httpx.get("http://127.0.0.1:8000/health", timeout=1.0).status_code == 200
    except httpx.HTTPError:
        return False


@needs_artifacts
@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


@needs_artifacts
def test_threshold_button_moves_slider_to_validation_optimum():
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.sidebar.radio[0].set_value("Threshold and cost").run()
    path = resolve_path("results_md").parent / "test_results.json"
    expected = next(
        r for r in json.loads(path.read_text())["sensitivity"] if r["review_cost"] == 25
    )
    # review cost 25 has a different val-optimal threshold than the default 5
    at.slider[0].set_value(25.0).run()
    assert at.session_state["thr"] != pytest.approx(expected["threshold"], abs=1e-3)
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["thr"] == pytest.approx(expected["threshold"], abs=1e-6)


@needs_artifacts
@pytest.mark.skipif(not _api_up(), reason="API not running on :8000")
def test_live_replay_runs_against_the_api():
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.sidebar.radio[0].set_value("Live replay").run()
    at.slider[0].set_value(50).run()
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.success and "Done: 50 transactions" in at.success[0].value
