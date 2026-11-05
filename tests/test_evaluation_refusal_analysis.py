"""Synthetic tests for refusal analysis and category coverage reporting."""

import json
from pathlib import Path

import pytest

from evaluation.common import EvaluationError
from evaluation.refusal_analysis import (
    build_refusal_analysis,
    category_matches,
    load_cases,
    main,
)


def _write_cases(path: Path, rows: list[str]) -> Path:
    path.write_text(
        "case_id,stem,expected_code,expected_category,synthetic,notes\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_result(
    results: Path,
    stem: str,
    *,
    status: str,
    code: str | None,
    message: str = "",
    reliability: str | None = None,
) -> None:
    results.mkdir(parents=True, exist_ok=True)
    measurement = None
    if status == "ok":
        measurement = {
            "width_mm": 100.0,
            "height_mm": 60.0,
            "reliability": reliability or "high",
        }
    payload = {
        "status": status,
        "code": code,
        "message": message,
        "measurement": measurement,
    }
    (results / f"{stem}_result.json").write_text(json.dumps(payload), encoding="utf-8")


def test_all_required_categories_can_be_covered_by_labelled_synthetic_cases(tmp_path):
    cases = _write_cases(
        tmp_path / "cases.csv",
        [
            "C1,R1,missing_reference,missing_reference,true,no marker",
            "C2,R2,invalid_geometry,invalid_marker_geometry,true,clipped marker",
            "C3,R3,partial_occlusion,occlusion,true,split mask",
            "C4,R4,missing_object,unreliable_segmentation,true,empty mask",
            "C5,R5,missing_calibration,invalid_calibration,true,missing camera",
            "C6,R6,excessive_tilt,excessive_tilt,true,future tilt code",
        ],
    )
    results = tmp_path / "results"
    _write_result(results, "R1", status="rejected", code="missing_reference")
    _write_result(
        results,
        "R2",
        status="rejected",
        code="invalid_geometry",
        message="reference marker is present but unusable: clipped",
    )
    _write_result(results, "R3", status="rejected", code="partial_occlusion")
    _write_result(results, "R4", status="rejected", code="missing_object")
    _write_result(results, "R5", status="unavailable", code="missing_calibration")
    _write_result(results, "R6", status="rejected", code="excessive_tilt")

    payload = build_refusal_analysis(
        cases_path=cases,
        results_dir=results,
        out_dir=tmp_path / "out",
    )

    assert payload["status"] == "ok"
    assert payload["summary"]["n_cases"] == 6
    assert payload["summary"]["n_passed"] == 6
    assert payload["summary"]["n_failed"] == 0
    assert payload["summary"]["coverage_gaps"] == []
    assert payload["summary"]["synthetic_counts"] == {
        "synthetic": 6,
        "real": 0,
        "unknown": 0,
    }
    assert payload["summary"]["excessive_tilt_implemented"] is False
    markdown = (tmp_path / "out" / "refusal_analysis.md").read_text(encoding="utf-8")
    assert "No case is labelled real" in markdown
    assert "## Required category coverage" in markdown


def test_missing_excessive_tilt_case_is_reported_as_a_coverage_gap(tmp_path):
    cases = _write_cases(
        tmp_path / "cases.csv",
        ["C1,R1,missing_reference,missing_reference,true,"],
    )
    results = tmp_path / "results"
    _write_result(results, "R1", status="rejected", code="missing_reference")
    payload = build_refusal_analysis(
        cases_path=cases, results_dir=results, out_dir=tmp_path / "out"
    )
    assert payload["status"] == "ok"
    assert "excessive_tilt" in payload["summary"]["coverage_gaps"]
    excessive = next(
        item for item in payload["categories"] if item["category"] == "excessive_tilt"
    )
    assert excessive["covered"] is False
    assert "no excessive-tilt threshold" in excessive["note"]


def test_case_mismatch_is_reported_and_cli_returns_1(tmp_path):
    cases = _write_cases(
        tmp_path / "cases.csv",
        ["C1,R1,missing_reference,missing_reference,true,"],
    )
    results = tmp_path / "results"
    _write_result(results, "R1", status="rejected", code="partial_occlusion")

    payload = build_refusal_analysis(
        cases_path=cases, results_dir=results, out_dir=tmp_path / "out"
    )
    assert payload["status"] == "mismatch"
    assert payload["summary"]["n_passed"] == 0
    assert payload["summary"]["n_failed"] == 1

    code = main(
        [
            "--cases",
            str(cases),
            "--results",
            str(results),
            "--out",
            str(tmp_path / "out2"),
        ]
    )
    assert code == 1


def test_missing_results_are_unavailable_and_cli_returns_3(tmp_path):
    cases = _write_cases(
        tmp_path / "cases.csv",
        ["C1,R1,missing_reference,missing_reference,true,"],
    )
    payload = build_refusal_analysis(
        cases_path=cases,
        results_dir=tmp_path / "missing_results",
        out_dir=tmp_path / "out",
    )
    assert payload["status"] == "unavailable"
    assert payload["code"] == "missing_results"
    assert payload["cases"][0]["observed_code"] == "missing_result"

    code = main(
        [
            "--cases",
            str(cases),
            "--results",
            str(tmp_path / "missing_results"),
            "--out",
            str(tmp_path / "out2"),
        ]
    )
    assert code == 3


def test_invalid_marker_geometry_requires_a_reference_marker_message():
    assert category_matches(
        "invalid_marker_geometry",
        "invalid_geometry",
        "reference marker is present but unusable",
    )
    assert not category_matches(
        "invalid_marker_geometry",
        "invalid_geometry",
        "object contour is degenerate",
    )
    assert category_matches(
        "invalid_object_geometry",
        "invalid_geometry",
        "object contour is degenerate",
    )


def test_low_reliability_is_derived_and_can_satisfy_unreliable_segmentation(tmp_path):
    cases = _write_cases(
        tmp_path / "cases.csv",
        ["C1,R1,low_reliability,unreliable_segmentation,true,"],
    )
    results = tmp_path / "results"
    _write_result(results, "R1", status="ok", code=None, reliability="low")
    payload = build_refusal_analysis(
        cases_path=cases, results_dir=results, out_dir=tmp_path / "out"
    )
    assert payload["status"] == "ok"
    assert payload["cases"][0]["observed_code"] == "low_reliability"
    assert payload["cases"][0]["observed_category"] == "unreliable_segmentation"


def test_unlabelled_cases_are_counted_as_unknown(tmp_path):
    cases = _write_cases(
        tmp_path / "cases.csv",
        ["C1,R1,missing_reference,missing_reference,,"],
    )
    results = tmp_path / "results"
    _write_result(results, "R1", status="rejected", code="missing_reference")
    payload = build_refusal_analysis(
        cases_path=cases, results_dir=results, out_dir=tmp_path / "out"
    )
    assert payload["summary"]["synthetic_counts"] == {
        "synthetic": 0,
        "real": 0,
        "unknown": 1,
    }
    markdown = (tmp_path / "out" / "refusal_analysis.md").read_text(encoding="utf-8")
    assert "Some cases are unlabelled" in markdown


def test_cases_reject_duplicate_stems(tmp_path):
    path = _write_cases(
        tmp_path / "cases.csv",
        [
            "C1,R1,missing_reference,missing_reference,true,",
            "C2,R1,partial_occlusion,occlusion,true,",
        ],
    )
    with pytest.raises(EvaluationError, match="repeats stem"):
        load_cases(path)


def test_cases_reject_unknown_category(tmp_path):
    path = _write_cases(
        tmp_path / "cases.csv",
        ["C1,R1,missing_reference,not_a_category,true,"],
    )
    with pytest.raises(EvaluationError, match="unknown expected_category"):
        load_cases(path)


def test_cases_reject_unknown_code(tmp_path):
    path = _write_cases(
        tmp_path / "cases.csv",
        ["C1,R1,not_a_code,missing_reference,true,"],
    )
    with pytest.raises(EvaluationError, match="unknown expected_code"):
        load_cases(path)
