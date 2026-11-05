"""Tests for the strict Phase 4A report helpers."""

import json
from pathlib import Path

import pytest

from evaluation.common import (
    EvaluationError,
    GroundTruth,
    error_summary,
    latency_summary,
    load_ground_truth,
    read_csv,
    read_result,
    write_csv,
    write_json,
)


def test_read_csv_rejects_unknown_columns(tmp_path):
    path = tmp_path / "input.csv"
    path.write_text("stem,width_mm,typo\nA01,10,1\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="unknown column"):
        read_csv(path, required=("stem", "width_mm"), what="input")


def test_read_csv_rejects_missing_columns(tmp_path):
    path = tmp_path / "input.csv"
    path.write_text("stem\nA01\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="missing column"):
        read_csv(path, required=("stem", "width_mm"), what="input")


def test_ground_truth_requires_a_named_instrument(tmp_path):
    path = tmp_path / "truth.csv"
    path.write_text(
        "stem,width_mm,height_mm,instrument\nA01,100,60,\n", encoding="utf-8"
    )
    with pytest.raises(EvaluationError, match="instrument is empty"):
        load_ground_truth(path)


def test_ground_truth_loads_and_keeps_provenance(tmp_path):
    path = tmp_path / "truth.csv"
    path.write_text(
        "stem,width_mm,height_mm,instrument,measured_utc,notes\n"
        "A01,100.2,60.1,digital calliper,2026-11-05,three readings\n",
        encoding="utf-8",
    )
    truth = load_ground_truth(path)
    assert truth["A01"] == GroundTruth(
        stem="A01",
        width_mm=100.2,
        height_mm=60.1,
        instrument="digital calliper",
        measured_utc="2026-11-05",
        notes="three readings",
    )


def test_ground_truth_rejects_non_positive_dimensions(tmp_path):
    path = tmp_path / "truth.csv"
    path.write_text(
        "stem,width_mm,height_mm,instrument\nA01,0,60,calliper\n", encoding="utf-8"
    )
    with pytest.raises(EvaluationError, match="must be positive"):
        load_ground_truth(path)


def test_error_summary_matches_hand_calculation():
    metrics = error_summary([1.0, -3.0], [100.0, 50.0])
    assert metrics["n"] == 2
    assert metrics["mae_mm"] == pytest.approx(2.0)
    assert metrics["rmse_mm"] == pytest.approx((5.0) ** 0.5)
    assert metrics["max_abs_error_mm"] == pytest.approx(3.0)
    assert metrics["mean_percentage_error_pct"] == pytest.approx(3.5)
    assert metrics["mean_signed_error_mm"] == pytest.approx(-1.0)


def test_error_summary_with_no_rows_is_explicitly_null():
    metrics = error_summary([], [])
    assert metrics["n"] == 0
    assert metrics["mae_mm"] is None
    assert metrics["rmse_mm"] is None


def test_latency_summary_reports_median_and_p95():
    metrics = latency_summary([1.0, 2.0, 3.0, 4.0, 5.0])
    assert metrics["samples"] == 5
    assert metrics["median_s"] == pytest.approx(3.0)
    assert metrics["p95_s"] == pytest.approx(4.8)
    assert metrics["min_s"] == pytest.approx(1.0)
    assert metrics["max_s"] == pytest.approx(5.0)


def test_write_json_is_atomic_and_writable(tmp_path):
    path = write_json(tmp_path / "nested" / "report.json", {"ok": True})
    assert json.loads(path.read_text(encoding="utf-8")) == {"ok": True}
    assert not (tmp_path / "nested" / "report.json.tmp").exists()


def test_write_csv_writes_header_and_rows(tmp_path):
    path = write_csv(tmp_path / "out.csv", ["a", "b"], [[1, 2], [3, 4]])
    assert path.read_text(encoding="utf-8").splitlines() == ["a,b", "1,2", "3,4"]


def _write_result(tmp_path: Path, payload: dict) -> None:
    (tmp_path / "A01_result.json").write_text(json.dumps(payload), encoding="utf-8")


def test_read_result_marks_missing_as_failure(tmp_path):
    view = read_result(tmp_path, "A01")
    assert not view.ok
    assert view.code == "missing_result"
    assert view.result_path is None


def test_read_result_extracts_dimensions_only_for_ok(tmp_path):
    _write_result(
        tmp_path,
        {
            "status": "ok",
            "code": None,
            "measurement": {"width_mm": 100.0, "height_mm": 60.0, "reliability": "high"},
        },
    )
    view = read_result(tmp_path, "A01")
    assert view.ok
    assert view.width_mm == pytest.approx(100.0)
    assert view.height_mm == pytest.approx(60.0)
    assert view.reliability == "high"


def test_read_result_derives_low_reliability(tmp_path):
    _write_result(
        tmp_path,
        {
            "status": "ok",
            "code": None,
            "measurement": {"width_mm": 100.0, "height_mm": 60.0, "reliability": "low"},
        },
    )
    view = read_result(tmp_path, "A01")
    assert view.ok
    assert view.code == "low_reliability"


def test_read_result_treats_rejection_as_no_dimensions(tmp_path):
    _write_result(
        tmp_path,
        {
            "status": "rejected",
            "code": "missing_reference",
            "measurement": {"width_mm": 100.0, "height_mm": 60.0},
        },
    )
    view = read_result(tmp_path, "A01")
    assert not view.ok
    assert view.code == "missing_reference"
    assert view.width_mm is None
