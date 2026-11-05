"""Synthetic tests for accuracy reporting.

Ground truth here is a hand-written test fixture with a fictional instrument
name. It verifies the arithmetic and reporting, not a real measurement study.
"""

import json
from pathlib import Path

import pytest

from evaluation.accuracy_study import build_accuracy_study, main


def _write_truth(path: Path, rows: list[str]) -> Path:
    path.write_text(
        "stem,width_mm,height_mm,instrument,measured_utc,notes\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_result(
    results: Path,
    stem: str,
    *,
    width: float | None,
    height: float | None,
    reliability: str = "high",
) -> None:
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


def test_accuracy_metrics_match_hand_calculation(tmp_path):
    truth = _write_truth(
        tmp_path / "truth.csv",
        [
            "A01,100,60,digital calliper,2026-11-05,",
            "A02,100,60,digital calliper,2026-11-05,",
        ],
    )
    results = tmp_path / "results"
    _write_result(results, "A01", width=101.0, height=60.0)
    _write_result(results, "A02", width=98.0, height=62.0)

    payload = build_accuracy_study(
        ground_truth_path=truth, results_dir=results, out_dir=tmp_path / "out"
    )

    assert payload["status"] == "ok"
    metrics = payload["metrics"]
    assert metrics["n_total"] == 2
    assert metrics["n_ok"] == 2
    assert metrics["n_failed"] == 0
    assert metrics["failure_rate_pct"] == pytest.approx(0.0)
    assert metrics["width"]["mae_mm"] == pytest.approx(1.5)
    assert metrics["width"]["rmse_mm"] == pytest.approx((2.5) ** 0.5)
    assert metrics["width"]["mean_percentage_error_pct"] == pytest.approx(1.5)
    assert metrics["width"]["max_abs_error_mm"] == pytest.approx(2.0)
    assert metrics["height"]["mae_mm"] == pytest.approx(1.0)
    assert metrics["height"]["rmse_mm"] == pytest.approx((2.0) ** 0.5)
    assert metrics["combined"]["mae_mm"] == pytest.approx(1.25)
    assert metrics["combined"]["rmse_mm"] == pytest.approx(1.5)
    assert metrics["combined"]["max_abs_error_mm"] == pytest.approx(2.0)
    assert payload["all_instances_measured"] is True


def test_accuracy_reports_failures_separately_from_error(tmp_path):
    truth = _write_truth(
        tmp_path / "truth.csv",
        [
            "A01,100,60,digital calliper,,",
            "A02,100,60,digital calliper,,",
            "A03,100,60,digital calliper,,",
        ],
    )
    results = tmp_path / "results"
    _write_result(results, "A01", width=100.0, height=60.0)
    _write_rejection(results, "A03", "missing_reference")

    payload = build_accuracy_study(
        ground_truth_path=truth, results_dir=results, out_dir=tmp_path / "out"
    )

    assert payload["status"] == "ok"
    metrics = payload["metrics"]
    assert metrics["n_total"] == 3
    assert metrics["n_ok"] == 1
    assert metrics["n_failed"] == 2
    assert metrics["failure_rate_pct"] == pytest.approx(200.0 / 3.0)
    assert metrics["failure_codes"] == {"missing_reference": 1, "missing_result": 1}
    assert payload["all_instances_measured"] is False
    # A02 has no result, so only A01 contributes to the error.
    assert metrics["width"]["mae_mm"] == pytest.approx(0.0)
    assert metrics["height"]["mae_mm"] == pytest.approx(0.0)


def test_accuracy_reports_missing_ground_truth_as_unavailable(tmp_path):
    payload = build_accuracy_study(
        ground_truth_path=tmp_path / "missing_truth.csv",
        results_dir=tmp_path / "results",
        out_dir=tmp_path / "out",
    )
    assert payload["status"] == "unavailable"
    assert payload["code"] == "missing_ground_truth"
    assert payload["metrics"]["available"] is False
    assert "not found" in payload["metrics"]["reason"]
    assert (tmp_path / "out" / "accuracy.json").is_file()
    assert (tmp_path / "out" / "accuracy.md").is_file()


def test_accuracy_reports_missing_results_as_unavailable(tmp_path):
    truth = _write_truth(
        tmp_path / "truth.csv", ["A01,100,60,digital calliper,,"]
    )
    payload = build_accuracy_study(
        ground_truth_path=truth,
        results_dir=tmp_path / "missing_results",
        out_dir=tmp_path / "out",
    )
    assert payload["status"] == "unavailable"
    assert payload["code"] == "missing_results"
    assert payload["metrics"]["failure_rate_pct"] == pytest.approx(100.0)
    assert payload["metrics"]["failure_codes"] == {"missing_result": 1}


def test_accuracy_reports_all_refusals_as_unavailable_metrics(tmp_path):
    truth = _write_truth(
        tmp_path / "truth.csv",
        [
            "A01,100,60,digital calliper,,",
            "A02,100,60,digital calliper,,",
        ],
    )
    results = tmp_path / "results"
    _write_rejection(results, "A01", "missing_reference")
    _write_rejection(results, "A02", "partial_occlusion")
    payload = build_accuracy_study(
        ground_truth_path=truth, results_dir=results, out_dir=tmp_path / "out"
    )
    assert payload["status"] == "unavailable"
    assert payload["code"] == "no_valid_measurements"
    assert payload["metrics"]["available"] is False
    assert payload["metrics"]["failure_rate_pct"] == pytest.approx(100.0)
    assert payload["metrics"]["failure_codes"] == {
        "missing_reference": 1,
        "partial_occlusion": 1,
    }


def test_accuracy_writes_per_instance_csv_and_markdown(tmp_path):
    truth = _write_truth(
        tmp_path / "truth.csv", ["A01,100,60,digital calliper,2026-11-05,"]
    )
    results = tmp_path / "results"
    _write_result(results, "A01", width=101.0, height=59.0, reliability="low")
    out = tmp_path / "out"
    build_accuracy_study(ground_truth_path=truth, results_dir=results, out_dir=out)
    csv_text = (out / "accuracy_per_instance.csv").read_text(encoding="utf-8")
    assert "A01" in csv_text
    assert "101.000000" in csv_text
    markdown = (out / "accuracy.md").read_text(encoding="utf-8")
    assert "MAE (mm)" in markdown
    assert "Failure / refusal rate" in markdown
    assert "digital calliper" in markdown


def test_accuracy_cli_returns_3_when_ground_truth_is_missing(tmp_path):
    code = main(
        [
            "--ground-truth",
            str(tmp_path / "missing_truth.csv"),
            "--results",
            str(tmp_path / "results"),
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert code == 3


def test_accuracy_cli_returns_2_for_malformed_ground_truth(tmp_path):
    truth = tmp_path / "truth.csv"
    truth.write_text(
        "stem,width_mm,height_mm,instrument\nA01,100,60,\n", encoding="utf-8"
    )
    code = main(
        [
            "--ground-truth",
            str(truth),
            "--results",
            str(tmp_path / "results"),
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert code == 2
