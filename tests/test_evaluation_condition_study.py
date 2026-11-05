"""Synthetic tests for the tilt/condition study framework."""

import json
from pathlib import Path

import pytest

from evaluation.common import EvaluationError
from evaluation.condition_study import (
    build_condition_study,
    load_angle_manifest,
    main,
)


def _write_manifest(path: Path, rows: list[str], header: str | None = None) -> Path:
    path.write_text(
        (header or "stem,measured_angle_deg,target_angle_deg,condition,notes")
        + "\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_truth(path: Path) -> Path:
    path.write_text(
        "stem,width_mm,height_mm,instrument,measured_utc,notes\n"
        "T00_01,99.0,60.0,digital calliper,2026-11-05,\n"
        "T10_01,100.0,60.0,digital calliper,2026-11-05,\n"
        "T20_01,100.0,60.0,digital calliper,2026-11-05,\n",
        encoding="utf-8",
    )
    return path


def _write_result(results: Path, stem: str, *, width: float | None, height: float | None, reliability: str = "high") -> None:
    results.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "ok",
        "code": None,
        "measurement": {
            "width_mm": width,
            "height_mm": height,
            "reliability": reliability,
        },
    }
    (results / f"{stem}_result.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_rejection(results: Path, stem: str, code: str) -> None:
    results.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "rejected",
        "code": code,
        "message": "synthetic rejection",
        "measurement": None,
    }
    (results / f"{stem}_result.json").write_text(json.dumps(payload), encoding="utf-8")


def test_manifest_requires_a_measured_angle_not_a_target(tmp_path):
    path = tmp_path / "tilt.csv"
    path.write_text(
        "stem,target_angle_deg\nT00_01,0\n", encoding="utf-8"
    )
    with pytest.raises(EvaluationError, match="target is not a measurement"):
        load_angle_manifest(path)


def test_manifest_accepts_angle_alias_and_keeps_target_informational(tmp_path):
    path = _write_manifest(
        tmp_path / "tilt.csv",
        ["T10_01,10,40,fixed,measured with digital level"],
    )
    rows = load_angle_manifest(path)
    assert rows[0]["angle_deg"] == pytest.approx(10.0)
    assert rows[0]["target_angle_deg"] == pytest.approx(40.0)


def test_manifest_rejects_disagreeing_angle_columns(tmp_path):
    path = tmp_path / "tilt.csv"
    path.write_text(
        "stem,measured_angle_deg,angle_deg\nT00_01,10,11\n", encoding="utf-8"
    )
    with pytest.raises(EvaluationError, match="disagree"):
        load_angle_manifest(path)


def test_manifest_rejects_duplicate_stems(tmp_path):
    path = _write_manifest(
        tmp_path / "tilt.csv",
        ["T00_01,0,,,", "T00_01,10,,,"],
    )
    with pytest.raises(EvaluationError, match="repeats stem"):
        load_angle_manifest(path)


def test_manifest_rejects_out_of_range_angle(tmp_path):
    path = _write_manifest(tmp_path / "tilt.csv", ["T00_01,91,,,"])
    with pytest.raises(EvaluationError, match="must be <="):
        load_angle_manifest(path)


def test_condition_study_groups_by_angle_and_computes_errors(tmp_path):
    manifest = _write_manifest(
        tmp_path / "tilt.csv",
        [
            "T00_01,0,,fixed,",
            "T10_01,10,,fixed,",
            "T20_01,20,,fixed,",
        ],
    )
    truth = _write_truth(tmp_path / "truth.csv")
    results = tmp_path / "results"
    _write_result(results, "T00_01", width=100.0, height=60.0)
    _write_result(results, "T10_01", width=102.0, height=61.0)
    _write_rejection(results, "T20_01", "missing_reference")

    payload = build_condition_study(
        manifest_path=manifest,
        results_dir=results,
        ground_truth_path=truth,
        out_dir=tmp_path / "out",
    )

    assert payload["status"] == "ok"
    assert [group["angle_deg"] for group in payload["groups"]] == [0.0, 10.0, 20.0]
    assert payload["overall"]["n_instances"] == 3
    assert payload["overall"]["n_ok"] == 2
    assert payload["overall"]["n_failed"] == 1
    assert payload["overall"]["failure_rate_pct"] == pytest.approx(100.0 / 3.0)
    metrics = payload["overall"]["error_metrics"]
    assert metrics["available"] is True
    assert metrics["width"]["mae_mm"] == pytest.approx(1.5)
    assert metrics["height"]["mae_mm"] == pytest.approx(0.5)
    assert metrics["combined"]["max_abs_error_mm"] == pytest.approx(2.0)

    angle_20 = payload["groups"][2]
    assert angle_20["n_failed"] == 1
    assert angle_20["codes"] == {"missing_reference": 1}
    assert angle_20["error_metrics"]["available"] is False


def test_condition_study_without_ground_truth_reports_failure_rate_only(tmp_path):
    manifest = _write_manifest(tmp_path / "tilt.csv", ["T00_01,0,,,"])
    results = tmp_path / "results"
    _write_result(results, "T00_01", width=100.0, height=60.0)

    payload = build_condition_study(
        manifest_path=manifest,
        results_dir=results,
        out_dir=tmp_path / "out",
    )
    assert payload["status"] == "partial"
    assert payload["overall"]["failure_rate_pct"] == pytest.approx(0.0)
    assert payload["overall"]["error_metrics"]["available"] is False
    assert "no ground-truth" in payload["overall"]["error_metrics"]["reason"]


def test_condition_study_missing_result_is_a_failure(tmp_path):
    manifest = _write_manifest(tmp_path / "tilt.csv", ["T00_01,0,,,"])
    payload = build_condition_study(
        manifest_path=manifest,
        results_dir=tmp_path / "missing_results",
        out_dir=tmp_path / "out",
    )
    assert payload["status"] == "unavailable"
    assert payload["code"] == "missing_results"
    assert payload["overall"]["n_failed"] == 1
    assert payload["overall"]["codes"] == {"missing_result": 1}


def test_condition_study_missing_ground_truth_file_is_reported(tmp_path):
    manifest = _write_manifest(tmp_path / "tilt.csv", ["T00_01,0,,,"])
    results = tmp_path / "results"
    _write_result(results, "T00_01", width=100.0, height=60.0)
    payload = build_condition_study(
        manifest_path=manifest,
        results_dir=results,
        ground_truth_path=tmp_path / "missing_truth.csv",
        out_dir=tmp_path / "out",
    )
    assert payload["status"] == "partial"
    assert payload["code"] == "ground_truth_unavailable"
    assert "not found" in payload["ground_truth_error"]


def test_condition_study_reports_missing_truth_stems(tmp_path):
    manifest = _write_manifest(tmp_path / "tilt.csv", ["T00_01,0,,,"])
    truth = tmp_path / "truth.csv"
    truth.write_text(
        "stem,width_mm,height_mm,instrument\nOTHER,100,60,calliper\n",
        encoding="utf-8",
    )
    results = tmp_path / "results"
    _write_result(results, "T00_01", width=100.0, height=60.0)
    payload = build_condition_study(
        manifest_path=manifest,
        results_dir=results,
        ground_truth_path=truth,
        out_dir=tmp_path / "out",
    )
    assert payload["status"] == "partial"
    assert payload["missing_ground_truth_stems"] == ["T00_01"]


def test_condition_study_writes_json_csv_and_markdown(tmp_path):
    manifest = _write_manifest(tmp_path / "tilt.csv", ["T00_01,0,,,"])
    results = tmp_path / "results"
    _write_result(results, "T00_01", width=100.0, height=60.0)
    out = tmp_path / "out"
    build_condition_study(
        manifest_path=manifest, results_dir=results, out_dir=out
    )
    assert (out / "condition_study.json").is_file()
    assert (out / "condition_study.csv").is_file()
    assert (out / "condition_study.md").is_file()
    markdown = (out / "condition_study.md").read_text(encoding="utf-8")
    assert "By measured angle" in markdown
    assert "Angle provenance" in markdown


def test_cli_returns_3_when_results_are_missing(tmp_path):
    manifest = _write_manifest(tmp_path / "tilt.csv", ["T00_01,0,,,"])
    code = main(
        [
            "--manifest",
            str(manifest),
            "--results",
            str(tmp_path / "missing"),
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert code == 3


def test_cli_returns_2_for_target_only_manifest(tmp_path):
    manifest = tmp_path / "tilt.csv"
    manifest.write_text("stem,target_angle_deg\nT00_01,0\n", encoding="utf-8")
    code = main(["--manifest", str(manifest), "--results", str(tmp_path)])
    assert code == 2
