"""Accuracy study: compare saved system measurements with manual ground truth.

The ground-truth CSV must contain values measured with a recorded instrument.
This module never derives, estimates or fabricates a ground-truth dimension. It
computes MAE, RMSE, mean percentage error, maximum absolute error, failure /
refusal rate and a per-instance comparison.

Metrics are computed over successful measurements only. The failure rate is
reported separately so a high refusal rate cannot be hidden behind a good MAE.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from evaluation.common import (
    EvaluationError,
    GroundTruth,
    ResultView,
    error_summary,
    fmt,
    load_ground_truth,
    md_table,
    read_result,
    utc_now,
    write_csv,
    write_json,
    write_markdown,
)

STUDY_SCHEMA = "measureit.accuracy_study/1"

DEFINITIONS = {
    "mae_mm": "mean of |system - ground truth| in millimetres",
    "rmse_mm": "sqrt(mean((system - ground truth)^2)) in millimetres",
    "mean_percentage_error_pct": "mean of 100 * |system - ground truth| / ground truth",
    "max_abs_error_mm": "largest |system - ground truth| in millimetres",
    "failure_rate_pct": "100 * (instances without a valid measurement) / all ground-truth instances",
    "combined": "width and height errors pooled together, one row per dimension",
}


def _instance(
    truth: GroundTruth, view: ResultView
) -> dict[str, Any]:
    width_error = height_error = None
    width_abs = height_abs = None
    width_pct = height_pct = None
    if view.ok:
        width_error = float(view.width_mm) - truth.width_mm
        height_error = float(view.height_mm) - truth.height_mm
        width_abs = abs(width_error)
        height_abs = abs(height_error)
        width_pct = 100.0 * width_abs / truth.width_mm
        height_pct = 100.0 * height_abs / truth.height_mm

    return {
        "stem": truth.stem,
        "result_path": None if view.result_path is None else str(view.result_path),
        "status": view.status,
        "code": view.code,
        "message": view.message,
        "reliability": view.reliability,
        "width_mm": view.width_mm,
        "height_mm": view.height_mm,
        "truth_width_mm": truth.width_mm,
        "truth_height_mm": truth.height_mm,
        "instrument": truth.instrument,
        "measured_utc": truth.measured_utc,
        "truth_notes": truth.notes,
        "width_error_mm": width_error,
        "height_error_mm": height_error,
        "width_abs_error_mm": width_abs,
        "height_abs_error_mm": height_abs,
        "width_error_pct": width_pct,
        "height_error_pct": height_pct,
    }


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def _unavailable_metrics(reason: str, *, n_total: int, n_ok: int, codes: dict[str, int]) -> dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "definitions": DEFINITIONS,
        "n_total": n_total,
        "n_ok": n_ok,
        "n_failed": n_total - n_ok,
        "failure_rate_pct": None if n_total == 0 else 100.0 * (n_total - n_ok) / n_total,
        "failure_codes": codes,
        "width": error_summary([], []),
        "height": error_summary([], []),
        "combined": error_summary([], []),
    }


def build_accuracy_study(
    *,
    ground_truth_path: str | Path,
    results_dir: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    truth_path = Path(ground_truth_path)
    results_path = Path(results_dir)
    out_path = Path(out_dir)

    truth: dict[str, GroundTruth] = {}
    truth_error = None
    if not truth_path.is_file():
        truth_error = f"ground-truth CSV not found: {truth_path}"
    else:
        try:
            truth = load_ground_truth(truth_path)
        except EvaluationError as exc:
            if "has no instances" in str(exc):
                truth = {}
                truth_error = str(exc)
            else:
                raise

    instances = [
        _instance(truth_row, read_result(results_path, stem))
        for stem, truth_row in sorted(truth.items())
    ]
    successful = [instance for instance in instances if instance["width_error_mm"] is not None]
    results_available = any(instance["result_path"] is not None for instance in instances)
    failure_codes = Counter(
        instance["code"] for instance in instances if instance["width_error_mm"] is None
    )

    if truth_error is not None:
        status = "unavailable"
        code = "missing_ground_truth"
        message = truth_error
    elif not results_available:
        status = "unavailable"
        code = "missing_results"
        message = (
            f"no saved inference results were found for the ground-truth stems "
            f"in {results_path}"
        )
    elif not successful:
        status = "unavailable"
        code = "no_valid_measurements"
        message = (
            "none of the ground-truth instances has a valid saved measurement; "
            "accuracy cannot be computed"
        )
    else:
        status = "ok"
        code = None
        message = "accuracy computed from saved measurements and manual ground truth"

    if successful:
        width_errors = [float(instance["width_error_mm"]) for instance in successful]
        height_errors = [float(instance["height_error_mm"]) for instance in successful]
        width_truth = [float(instance["truth_width_mm"]) for instance in successful]
        height_truth = [float(instance["truth_height_mm"]) for instance in successful]
        metrics = {
            "available": True,
            "reason": None,
            "definitions": DEFINITIONS,
            "n_total": len(instances),
            "n_ok": len(successful),
            "n_failed": len(instances) - len(successful),
            "failure_rate_pct": 100.0 * (len(instances) - len(successful)) / len(instances),
            "failure_codes": dict(sorted(failure_codes.items())),
            "width": error_summary(width_errors, width_truth),
            "height": error_summary(height_errors, height_truth),
            "combined": error_summary(
                width_errors + height_errors, width_truth + height_truth
            ),
        }
    else:
        metrics = _unavailable_metrics(
            message,
            n_total=len(instances),
            n_ok=0,
            codes=dict(sorted(failure_codes.items())),
        )

    instruments = sorted(
        {truth_row.instrument for truth_row in truth.values() if truth_row.instrument}
    )
    payload = {
        "schema": STUDY_SCHEMA,
        "created_utc": utc_now(),
        "status": status,
        "code": code,
        "message": message,
        "ground_truth": {
            "path": str(truth_path),
            "instances": len(truth),
            "instruments": instruments,
        },
        "results_dir": str(results_path),
        "results_available": results_available,
        "all_instances_measured": bool(instances) and len(successful) == len(instances),
        "metrics": metrics,
        "instances": instances,
        "notes": [
            "ground truth is manual instrument data; this tool never generates it",
            "error metrics use successful measurements only; the failure rate is "
            "reported separately",
            "percentage error is relative to the ground-truth dimension",
            "synthetic fixtures used in tests are software checks, not real-world "
            "accuracy evidence",
        ],
    }

    out_path.mkdir(parents=True, exist_ok=True)
    payload["report_paths"] = {
        "json": str(out_path / "accuracy.json"),
        "csv": str(out_path / "accuracy_per_instance.csv"),
        "markdown": str(out_path / "accuracy.md"),
    }
    write_json(out_path / "accuracy.json", payload)

    header = [
        "stem",
        "status",
        "code",
        "reliability",
        "width_mm",
        "height_mm",
        "truth_width_mm",
        "truth_height_mm",
        "width_error_mm",
        "height_error_mm",
        "width_abs_error_mm",
        "height_abs_error_mm",
        "width_error_pct",
        "height_error_pct",
        "instrument",
        "measured_utc",
        "result_path",
        "message",
    ]
    write_csv(
        out_path / "accuracy_per_instance.csv",
        header,
        [
            [_csv_value(instance[name]) for name in header]
            for instance in instances
        ],
    )

    lines = [
        "# Measurement accuracy study",
        "",
        f"- Generated: {payload['created_utc']}",
        f"- Ground truth: `{truth_path}` ({len(truth)} instance(s))",
        f"- Results: `{results_path}`",
        f"- Status: **{status}**" + ("" if code is None else f" (`{code}`)"),
        f"- Instruments: {', '.join(instruments) if instruments else 'not recorded'}",
        "",
        "## Definitions",
        "",
        "| metric | definition |",
        "|---|---|",
    ]
    for name, definition in DEFINITIONS.items():
        lines.append(f"| `{name}` | {definition} |")
    lines.append("")

    if successful:
        lines.extend(["## Summary", ""])
        rows = []
        for name, label in (
            ("width", "width"),
            ("height", "height"),
            ("combined", "combined (width + height)"),
        ):
            summary = metrics[name]
            rows.append(
                [
                    label,
                    summary["n"],
                    fmt(summary["mae_mm"], 3),
                    fmt(summary["rmse_mm"], 3),
                    fmt(summary["mean_percentage_error_pct"], 3),
                    fmt(summary["max_abs_error_mm"], 3),
                    fmt(summary["median_abs_error_mm"], 3),
                ]
            )
        lines.extend(
            md_table(
                [
                    "dimension",
                    "n",
                    "MAE (mm)",
                    "RMSE (mm)",
                    "mean % error",
                    "max abs (mm)",
                    "median abs (mm)",
                ],
                rows,
            )
        )
        lines.extend(
            [
                "",
                f"- Valid measurements: **{metrics['n_ok']} / {metrics['n_total']}**",
                f"- Failure / refusal rate: **{fmt(metrics['failure_rate_pct'], 2)} %**",
                f"- Failure codes: {metrics['failure_codes'] or 'none'}",
                "",
                "Metrics above are computed only on the successful instances. A "
                "high failure rate means this is not the whole population.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Summary",
                "",
                f"Accuracy metrics are unavailable: {message}.",
                "",
            ]
        )

    lines.extend(["## Per-instance comparison", ""])
    rows = []
    for instance in instances:
        system = (
            "—"
            if instance["width_mm"] is None
            else f"{instance['width_mm']:.2f} x {instance['height_mm']:.2f}"
        )
        truth_text = (
            f"{instance['truth_width_mm']:.2f} x {instance['truth_height_mm']:.2f}"
        )
        errors = (
            "—"
            if instance["width_abs_error_mm"] is None
            else f"{instance['width_abs_error_mm']:.2f} / "
            f"{instance['height_abs_error_mm']:.2f}"
        )
        rows.append(
            [
                instance["stem"],
                instance["status"],
                instance["code"],
                system,
                truth_text,
                errors,
                instance["reliability"] or "—",
                instance["instrument"],
            ]
        )
    lines.extend(
        md_table(
            [
                "stem",
                "status",
                "code",
                "system (mm)",
                "truth (mm)",
                "abs error w/h (mm)",
                "reliability",
                "instrument",
            ],
            rows,
        )
    )

    lines.extend(
        [
            "",
            "## What this does not prove",
            "",
            "- The numbers are only as good as the manual measurements in the "
            "ground-truth CSV; this tool cannot detect a mis-measured object.",
            "- Successful-case error does not describe refused frames; read the "
            "failure rate and codes together with the error table.",
            "- Synthetic fixtures in the test suite are not real-world evidence.",
            "",
        ]
    )
    write_markdown(out_path / "accuracy.md", lines)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.accuracy_study",
        description="Compare saved measurements with manually measured ground "
        "truth and report MAE, RMSE, percentage error and refusal rate.",
    )
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/accuracy"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = build_accuracy_study(
            ground_truth_path=args.ground_truth,
            results_dir=args.results,
            out_dir=args.out,
        )
    except EvaluationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    metrics = payload["metrics"]
    print(f"{payload['status'].upper()}: {payload['message']}")
    if metrics["available"]:
        for name in ("width", "height", "combined"):
            summary = metrics[name]
            print(
                f"{name}: n={summary['n']} MAE={summary['mae_mm']:.3f} mm "
                f"RMSE={summary['rmse_mm']:.3f} mm "
                f"mean%={summary['mean_percentage_error_pct']:.3f} "
                f"max|e|={summary['max_abs_error_mm']:.3f} mm"
            )
    failure_rate = metrics["failure_rate_pct"]
    print(
        "failure/refusal rate: "
        + ("unavailable" if failure_rate is None else f"{failure_rate:.2f}%")
    )
    print(f"json: {args.out / 'accuracy.json'}")
    print(f"csv: {args.out / 'accuracy_per_instance.csv'}")
    print(f"markdown: {args.out / 'accuracy.md'}")
    return 0 if metrics["available"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
