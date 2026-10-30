import numpy as np
import pandas as pd
import pytest

from fraud import report

LATENCY = {
    "n_calls": 500,
    "target_p95_ms": 50,
    "meets_target": True,
    "server_p50_ms": 10.0,
    "server_p95_ms": 20.0,
    "round_trip_p50_ms": 15.0,
    "round_trip_p95_ms": 30.0,
}


def test_md_table_renders_header_separator_and_rows():
    table = report.md_table(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))
    assert table.splitlines() == ["| a | b |", "|---|---|", "| 1 | x |", "| 2 | y |"]


def test_latency_block_reports_target_verdict():
    assert "not measured" in report.latency_block(None).lower()
    assert "(met)" in report.latency_block(LATENCY)
    assert "NOT met" in report.latency_block({**LATENCY, "meets_target": False})


def test_update_latency_replaces_only_the_marked_section(tmp_path, monkeypatch):
    path = tmp_path / "results.md"
    path.write_text(
        f"# Results\nkeep me\n{report.LATENCY_START}\nold\n{report.LATENCY_END}\ntail\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(report, "resolve_path", lambda key: path)
    report.update_latency(LATENCY)
    text = path.read_text(encoding="utf-8")
    assert "old" not in text and "30.00" in text
    assert text.startswith("# Results\nkeep me\n") and text.endswith("tail\n")


def test_update_latency_needs_the_markers(tmp_path, monkeypatch):
    path = tmp_path / "results.md"
    path.write_text("# no markers here", encoding="utf-8")
    monkeypatch.setattr(report, "resolve_path", lambda key: path)
    with pytest.raises(ValueError, match="latency section"):
        report.update_latency(LATENCY)


def test_thin_keeps_both_end_points_and_caps_length():
    xs, ys = report._thin(np.arange(1000.0), np.arange(1000.0) * 2, n=50)
    assert len(xs) == len(ys) == 50
    assert xs[0] == 0 and xs[-1] == 999 and ys[-1] == 1998
    short, _ = report._thin(np.arange(5.0), np.arange(5.0), n=50)
    assert len(short) == 5
