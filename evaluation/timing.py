"""Reproducible CPU inference benchmark.

Stages are timed separately: model loading, undistortion, segmentation,
reference-marker detection, geometry/measurement, and total inference. The
report states the CPU environment, image dimensions, warm-up runs and
repetitions, and gives median/p95 latency. It is a **software benchmark**, not
an accuracy claim.

A missing calibration, measured reference size or checkpoint produces an
explicit unavailable report. No timing is invented.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch

from evaluation.common import (
    EvaluationError,
    latency_summary,
    md_table,
    utc_now,
    write_csv,
    write_json,
    write_markdown,
)
from measureit.camera import CameraModel, CameraModelError
from measureit.io import ImageLoadError, load_image
from measurement.engine import MeasurementResult, measure
from measurement.infer import EXIT_UNAVAILABLE, EXIT_USAGE, segment_with_model
from measurement.reference import Marker, ReferenceError, ReferenceSpec, detect_marker
from segmentation.model import IMAGENET_MEAN, IMAGENET_STD
from segmentation.train import load_checkpoint

BENCHMARK_SCHEMA = "measureit.benchmark/1"


@dataclass
class BenchmarkOutput:
    payload: dict[str, Any]
    exit_code: int
    json_path: Path | None
    markdown_path: Path | None
    csv_path: Path | None


def collect_environment() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "torch": str(torch.__version__),
        "torch_threads": int(torch.get_num_threads()),
        "opencv": str(cv2.__version__),
        "numpy": str(np.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def _base_payload(
    image_path: Path,
    width: int,
    height: int,
    *,
    warmup: int,
    repetitions: int,
    refine: bool,
    threads: int,
) -> dict[str, Any]:
    return {
        "schema": BENCHMARK_SCHEMA,
        "created_utc": utc_now(),
        "status": None,
        "code": None,
        "message": "",
        "scope": "software_timing_only_not_accuracy",
        "image": {
            "path": str(image_path),
            "width": int(width),
            "height": int(height),
        },
        "environment": None,
        "camera": None,
        "reference": None,
        "checkpoint": None,
        "settings": {
            "warmup_runs": int(warmup),
            "repetitions": int(repetitions),
            "refine": bool(refine),
            "requested_threads": int(threads),
        },
        "model": None,
        "timings": {},
        "geometry_measurement_available": False,
        "geometry_measurement_reason": None,
        "measurement": None,
        "repetitions": [],
        "notes": [
            "timings are software latency measurements, not accuracy evidence",
            "total_inference includes undistort, segmentation, reference "
            "detection and geometry/measurement; it excludes image loading and "
            "model loading",
            "p95 uses linear interpolation over the recorded repetitions",
            "no real-world performance or accuracy claim follows from a "
            "synthetic test fixture",
        ],
    }


def _milliseconds(value: float | None) -> str:
    return "—" if value is None else f"{value * 1000.0:.2f}"


def _write_report(payload: dict[str, Any], out_dir: Path) -> tuple[Path, Path, Path | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = write_json(out_dir / "benchmark.json", payload)

    csv_path = None
    repetitions = payload.get("repetitions") or []
    if repetitions:
        csv_path = write_csv(
            out_dir / "benchmark_repetitions.csv",
            [
                "repetition",
                "undistort_s",
                "segmentation_s",
                "reference_detection_s",
                "geometry_measurement_s",
                "total_inference_s",
                "measurement_status",
                "measurement_code",
            ],
            [
                [
                    row["repetition"],
                    "" if row["undistort_s"] is None else f"{row['undistort_s']:.9f}",
                    "" if row["segmentation_s"] is None else f"{row['segmentation_s']:.9f}",
                    "" if row["reference_detection_s"] is None else f"{row['reference_detection_s']:.9f}",
                    "" if row["geometry_measurement_s"] is None else f"{row['geometry_measurement_s']:.9f}",
                    "" if row["total_inference_s"] is None else f"{row['total_inference_s']:.9f}",
                    row["measurement_status"],
                    row["measurement_code"] or "",
                ]
                for row in repetitions
            ],
        )

    lines = [
        "# CPU inference benchmark",
        "",
        f"- Generated: {payload['created_utc']}",
        f"- Status: **{payload['status']}**"
        + ("" if payload.get("code") is None else f" (`{payload['code']}`)"),
        f"- Scope: **{payload['scope']}**",
        "",
    ]
    if payload["status"] == "unavailable":
        lines.extend(
            [
                "No benchmark was run because a required physical artifact is "
                "missing.",
                "",
                f"Reason: {payload['message']}",
                "",
            ]
        )
    else:
        environment = payload["environment"]
        settings = payload["settings"]
        model = payload["model"]
        image = payload["image"]
        lines.extend(
            [
                "This report measures software latency only. It says nothing "
                "about measurement accuracy, which requires instrument ground "
                "truth.",
                "",
                "## Environment",
                "",
            ]
        )
        lines.extend(
            md_table(
                ["item", "value"],
                [
                    ["platform", environment["platform"]],
                    ["processor", environment["processor"]],
                    ["CPU count", environment["cpu_count"]],
                    ["Python", environment["python"]],
                    ["torch", environment["torch"]],
                    ["torch threads", environment["torch_threads"]],
                    ["OpenCV", environment["opencv"]],
                    ["NumPy", environment["numpy"]],
                    ["CUDA available", environment["cuda_available"]],
                ],
            )
        )
        lines.extend(["", "## Input and settings", ""])
        lines.extend(
            md_table(
                ["item", "value"],
                [
                    ["image", image["path"]],
                    ["image size", f"{image['width']}x{image['height']}"],
                    ["model architecture", model["architecture"]],
                    ["model parameters", model["parameters"]],
                    ["model input size", model["input_size"]],
                    ["warm-up runs", settings["warmup_runs"]],
                    ["repetitions", settings["repetitions"]],
                    ["crop refinement", settings["refine"]],
                    ["torch threads", environment["torch_threads"]],
                ],
            )
        )
        lines.extend(["", "## Latency", ""])
        timing_rows = []
        for name, label in (
            ("model_load", "model load"),
            ("undistort", "undistort"),
            ("segmentation", "segmentation"),
            ("reference_detection", "reference marker detection"),
            ("geometry_measurement", "geometry + measurement"),
            ("total_inference", "total inference"),
        ):
            summary = payload["timings"].get(name, latency_summary([]))
            timing_rows.append(
                [
                    label,
                    summary["samples"],
                    _milliseconds(summary["median_s"]),
                    _milliseconds(summary["p95_s"]),
                    _milliseconds(summary["min_s"]),
                    _milliseconds(summary["max_s"]),
                ]
            )
        lines.extend(
            md_table(
                ["stage", "samples", "median (ms)", "p95 (ms)", "min (ms)", "max (ms)"],
                timing_rows,
            )
        )
        measurement = payload["measurement"]
        lines.extend(["", "## Measurement status", ""])
        lines.append(
            f"- Status: `{measurement['status']}`"
            + ("" if measurement["code"] is None else f" (`{measurement['code']}`)")
        )
        if payload["geometry_measurement_available"]:
            lines.append("- Geometry/measurement timing was collected on a valid measurement.")
        else:
            lines.append(
                "- Geometry/measurement timing is unavailable: "
                f"{payload['geometry_measurement_reason']}"
            )
        lines.append("")
        if payload.get("code") == "measurement_not_ok":
            lines.append(
                "The supplied frame did not produce a valid measurement, so this "
                "is a partial benchmark. Use a frame that measures successfully "
                "for a complete stage breakdown."
            )
            lines.append("")

    lines.extend(
        [
            "## Scope and honesty",
            "",
            "- A timing here is a property of this machine and software build, "
            "not of measurement accuracy.",
            "- Synthetic fixtures used in tests are software checks; they do not "
            "establish real-world performance.",
            "- No timing is reported unless the corresponding stage ran.",
            "",
        ]
    )
    markdown_path = write_markdown(out_dir / "benchmark.md", lines)
    return json_path, markdown_path, csv_path


def run_benchmark(
    *,
    image_path: str | Path,
    camera_path: str | Path | None,
    reference_path: str | Path | None,
    checkpoint_path: str | Path | None,
    out_dir: str | Path,
    warmup: int = 2,
    repetitions: int = 10,
    threads: int = 4,
    refine: bool = True,
    write: bool = True,
) -> BenchmarkOutput:
    if warmup < 0:
        raise EvaluationError("warmup must be >= 0")
    if repetitions < 1:
        raise EvaluationError("repetitions must be >= 1")
    if threads < 1:
        raise EvaluationError("threads must be >= 1")

    image_path = Path(image_path)
    out_dir = Path(out_dir)
    try:
        image = load_image(image_path)
    except (ImageLoadError, OSError) as exc:
        raise EvaluationError(f"cannot read image {image_path}: {exc}") from exc

    height, width = image.shape[:2]
    payload = _base_payload(
        image_path,
        width,
        height,
        warmup=warmup,
        repetitions=repetitions,
        refine=refine,
        threads=threads,
    )

    def finish(status: str, code: str | None, message: str, exit_code: int) -> BenchmarkOutput:
        payload["status"] = status
        payload["code"] = code
        payload["message"] = message
        json_path = markdown_path = csv_path = None
        if write:
            json_path, markdown_path, csv_path = _write_report(payload, out_dir)
        return BenchmarkOutput(
            payload=payload,
            exit_code=exit_code,
            json_path=json_path,
            markdown_path=markdown_path,
            csv_path=csv_path,
        )

    camera_file = None if camera_path is None else Path(camera_path)
    if camera_file is None or not camera_file.is_file():
        return finish(
            "unavailable",
            "missing_calibration",
            f"camera calibration not found: {camera_file}; run the calibration stage first",
            EXIT_UNAVAILABLE,
        )
    try:
        camera = CameraModel.load(camera_file)
    except CameraModelError as exc:
        return finish("unavailable", "invalid_calibration", str(exc), EXIT_UNAVAILABLE)
    payload["camera"] = {
        "path": str(camera_file),
        "image_size": [int(camera.image_size[0]), int(camera.image_size[1])],
        "rms_reprojection_px": float(camera.rms_reprojection_px),
        "n_views_used": int(camera.n_views_used),
    }
    if (width, height) != camera.image_size:
        return finish(
            "unavailable",
            "calibration_image_size_mismatch",
            f"image is {width}x{height} but the camera was calibrated at "
            f"{camera.image_size[0]}x{camera.image_size[1]}",
            EXIT_UNAVAILABLE,
        )

    reference_file = None if reference_path is None else Path(reference_path)
    if reference_file is None or not reference_file.is_file():
        return finish(
            "unavailable",
            "missing_reference_config",
            f"reference configuration not found: {reference_file}",
            EXIT_UNAVAILABLE,
        )
    try:
        reference = ReferenceSpec.load(reference_file)
    except ReferenceError as exc:
        return finish(
            "unavailable", "invalid_reference_config", str(exc), EXIT_UNAVAILABLE
        )
    payload["reference"] = {
        "path": str(reference_file),
        "dictionary": reference.dictionary,
        "marker_id": int(reference.marker_id),
        "marker_size_mm": reference.marker_size_mm,
        "usable": bool(reference.usable),
    }
    if not reference.usable:
        return finish(
            "unavailable",
            "reference_size_unknown",
            "marker_size_mm is null: print the marker, measure its outer black "
            "square, and set the value in the reference file",
            EXIT_UNAVAILABLE,
        )

    checkpoint_file = None if checkpoint_path is None else Path(checkpoint_path)
    if checkpoint_file is None or not checkpoint_file.is_file():
        return finish(
            "unavailable",
            "missing_checkpoint",
            f"checkpoint not found: {checkpoint_file}; train the model first",
            EXIT_UNAVAILABLE,
        )

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(int(threads))
    payload["environment"] = collect_environment()
    try:
        try:
            started = time.perf_counter()
            model, meta = load_checkpoint(checkpoint_file)
            model_load_s = time.perf_counter() - started
        except Exception as exc:  # noqa: BLE001 - artifacts fail in loader-specific ways
            return finish(
                "unavailable", "invalid_checkpoint", str(exc), EXIT_UNAVAILABLE
            )

        image_size = int(meta.get("image_size", 512))
        normalization = meta.get("normalization", {})
        mean = normalization.get("mean", IMAGENET_MEAN)
        std = normalization.get("std", IMAGENET_STD)
        payload["checkpoint"] = {
            "path": str(checkpoint_file),
            "architecture": meta.get("architecture", {}).get("name"),
            "image_size": image_size,
            "epoch": meta.get("epoch"),
        }
        payload["model"] = {
            "architecture": meta.get("architecture", {}).get(
                "name", "unet_mobilenetv3_small"
            ),
            "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
            "input_size": image_size,
        }

        def segment(image_bgr: np.ndarray) -> np.ndarray:
            return segment_with_model(
                model,
                image_bgr,
                image_size=image_size,
                mean=mean,
                std=std,
                refine=refine,
            )

        def run_once() -> dict[str, Any]:
            started_total = time.perf_counter()
            started = time.perf_counter()
            undistorted = camera.undistort(image)
            undistort_s = time.perf_counter() - started

            started = time.perf_counter()
            mask = segment(undistorted)
            segmentation_s = time.perf_counter() - started

            started = time.perf_counter()
            detection_error = None
            try:
                marker: Marker | None = detect_marker(undistorted, reference)
            except ReferenceError as exc:
                marker = None
                detection_error = str(exc)
            reference_detection_s = time.perf_counter() - started

            geometry_s = None
            if marker is None:
                result = measure(undistorted, mask, reference)
                geometry_reason = detection_error or "reference marker was not detected"
            else:
                started = time.perf_counter()
                result = measure(undistorted, mask, reference, marker=marker)
                elapsed = time.perf_counter() - started
                if result.ok:
                    geometry_s = elapsed
                    geometry_reason = None
                else:
                    geometry_reason = (
                        f"measurement was {result.status}"
                        f"{'' if result.code is None else '/' + result.code}"
                    )

            total_s = time.perf_counter() - started_total
            return {
                "undistort_s": undistort_s,
                "segmentation_s": segmentation_s,
                "reference_detection_s": reference_detection_s,
                "geometry_measurement_s": geometry_s,
                "total_inference_s": total_s,
                "result": result,
                "geometry_reason": geometry_reason,
            }

        for _ in range(warmup):
            run_once()

        reps: list[dict[str, Any]] = []
        geometry_reason = None
        for index in range(1, repetitions + 1):
            measurement = run_once()
            result: MeasurementResult = measurement["result"]
            if measurement["geometry_measurement_s"] is None:
                geometry_reason = measurement["geometry_reason"]
            reps.append(
                {
                    "repetition": index,
                    "undistort_s": measurement["undistort_s"],
                    "segmentation_s": measurement["segmentation_s"],
                    "reference_detection_s": measurement["reference_detection_s"],
                    "geometry_measurement_s": measurement["geometry_measurement_s"],
                    "total_inference_s": measurement["total_inference_s"],
                    "measurement_status": result.status,
                    "measurement_code": result.code,
                }
            )

        payload["timings"] = {
            "model_load": latency_summary([model_load_s]),
            "undistort": latency_summary([row["undistort_s"] for row in reps]),
            "segmentation": latency_summary([row["segmentation_s"] for row in reps]),
            "reference_detection": latency_summary(
                [row["reference_detection_s"] for row in reps]
            ),
            "geometry_measurement": latency_summary(
                [
                    row["geometry_measurement_s"]
                    for row in reps
                    if row["geometry_measurement_s"] is not None
                ]
            ),
            "total_inference": latency_summary(
                [row["total_inference_s"] for row in reps]
            ),
        }
        payload["repetitions"] = reps
        payload["geometry_measurement_available"] = any(
            row["geometry_measurement_s"] is not None for row in reps
        )
        payload["geometry_measurement_reason"] = (
            None if payload["geometry_measurement_available"] else geometry_reason
        )

        all_ok = all(row["measurement_status"] == "ok" for row in reps)
        first_code = next(
            (
                row["measurement_code"]
                for row in reps
                if row["measurement_code"] is not None
            ),
            None,
        )
        payload["measurement"] = {
            "status": "ok" if all_ok else "not_ok",
            "code": None if all_ok else first_code or "measurement_not_ok",
            "ok": all_ok,
            "repetition_statuses": [row["measurement_status"] for row in reps],
        }
        if all(row["measurement_status"] == "ok" for row in reps):
            return finish(
                "ok",
                None,
                "benchmark completed on a valid measurement",
                0,
            )
        return finish(
            "partial",
            "measurement_not_ok",
            "timings were collected, but at least one repetition did not produce "
            "a valid measurement; geometry timing may be unavailable",
            0,
        )
    finally:
        torch.set_num_threads(previous_threads)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.timing",
        description="Benchmark CPU inference stages with warm-ups and repetitions.",
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument(
        "--camera", type=Path, default=Path("calibration/params/camera.json")
    )
    parser.add_argument(
        "--reference", type=Path, default=Path("measurement/reference.json")
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("runs/seg/best.pt")
    )
    parser.add_argument("--out", type=Path, default=Path("results/benchmark"))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="skip the full-resolution crop refinement pass",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = run_benchmark(
            image_path=args.image,
            camera_path=args.camera,
            reference_path=args.reference,
            checkpoint_path=args.checkpoint,
            out_dir=args.out,
            warmup=args.warmup,
            repetitions=args.repetitions,
            threads=args.threads,
            refine=not args.no_refine,
        )
    except (EvaluationError, OSError, ValueError, ImageLoadError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_USAGE

    payload = output.payload
    if output.json_path is not None:
        print(f"json: {output.json_path}")
    if output.markdown_path is not None:
        print(f"markdown: {output.markdown_path}")
    if output.csv_path is not None:
        print(f"repetitions: {output.csv_path}")

    if payload["status"] == "unavailable":
        print(f"UNAVAILABLE: {payload['message']}")
        return output.exit_code

    total = payload["timings"]["total_inference"]
    print(
        f"{payload['status'].upper()}: total inference median "
        f"{_milliseconds(total['median_s'])} ms, p95 "
        f"{_milliseconds(total['p95_s'])} ms over {total['samples']} repetition(s)"
    )
    return output.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
