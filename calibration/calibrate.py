"""Calibrate the camera from real ChArUco photographs.

This command never invents a camera model. If the printed board has not been
measured, it stops. If the reprojection error misses the gate, it writes the
report but refuses to save ``camera.json``.
"""

from __future__ import annotations

import argparse
import csv
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from calibration.board import BoardSpec, MissingMeasurementError, build_charuco, load_spec
from calibration.pipeline import (
    CalibrationError,
    CalibrationResult,
    NoImagesError,
    NoUsableViewsError,
    View,
    calibrate_views,
    detect_view,
    list_images,
)
from measureit.camera import CameraModel

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_NO_IMAGES = 2
EXIT_NO_USABLE_VIEWS = 3
EXIT_NOT_MEASURED = 4
EXIT_GATE_FAILED = 5


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_size(text: str) -> tuple[int, int] | None:
    if text.lower() == "auto":
        return None
    cleaned = text.lower().replace(" ", "").replace("*", "x")
    try:
        width, height = (int(v) for v in cleaned.split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("size must look like 2048x1536 or 'auto'") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("size must be positive")
    return (width, height)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m calibration.calibrate",
        description="Detect ChArUco corners in real photographs and calibrate the camera.",
    )
    parser.add_argument("--images", default="calibration/images", type=Path)
    parser.add_argument("--board", default="calibration/board/board_spec.json", type=Path)
    parser.add_argument("--out", default="calibration/params/camera.json", type=Path)
    parser.add_argument("--report-dir", default="calibration/report", type=Path)
    parser.add_argument(
        "--working-size",
        default="2048x1536",
        help="resolution (WxH) every image is normalised to, or 'auto' to use the source size",
    )
    parser.add_argument("--min-corners", type=int, default=12)
    parser.add_argument("--min-views", type=int, default=18)
    parser.add_argument("--max-rms", type=float, default=1.0)
    parser.add_argument("--max-view-rms", type=float, default=1.5)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="undistortion crop: 0 keeps only valid pixels, 1 keeps the full source",
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="report corner coverage without calibrating (works before the board is measured)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except MissingMeasurementError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return EXIT_NOT_MEASURED
    except NoImagesError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return EXIT_NO_IMAGES
    except (NoUsableViewsError, CalibrationError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return EXIT_NO_USABLE_VIEWS
    except cv2.error as exc:
        print(f"OpenCV error: {exc}", file=sys.stderr)
        return EXIT_USAGE


def _run(args: argparse.Namespace) -> int:
    spec = load_spec(args.board)
    working = _parse_size(args.working_size)

    if args.detect_only:
        try:
            board = build_charuco(spec, measured=True)
        except MissingMeasurementError:
            print("note: board not measured yet, detecting against the nominal geometry")
            board = build_charuco(spec, measured=False)
    else:
        board = build_charuco(spec, measured=True)  # raises when unmeasured

    paths = list_images(args.images)
    detector = cv2.aruco.CharucoDetector(board)
    views = [detect_view(path, detector, args.min_corners, working) for path in paths]

    image_size = working or next(
        (v.image_size for v in views if v.image_size is not None), None
    )
    if image_size is None:
        raise NoUsableViewsError("none of the images could be read")

    for view in views:
        if view.detected and view.image_size != tuple(image_size):
            view.reason = (
                f"image is {view.image_size[0]}x{view.image_size[1]}, expected "
                f"{image_size[0]}x{image_size[1]}"
            )
            view.corners = None
            view.ids = None

    if args.detect_only:
        return _detect_only(args, spec, views, image_size)

    result = calibrate_views(
        views,
        board,
        image_size,
        min_views=args.min_views,
        max_view_rms=args.max_view_rms,
    )

    new_matrix, roi = cv2.getOptimalNewCameraMatrix(
        result.camera_matrix, result.dist_coeffs, image_size, args.alpha, image_size, centerPrincipalPoint=False
    )

    reasons: list[str] = []
    if result.rms_reprojection_px > args.max_rms:
        reasons.append(
            f"rms {result.rms_reprojection_px:.2f} px exceeds the {args.max_rms:.2f} px gate"
        )
    if len(result.used) < args.min_views:
        reasons.append(
            f"only {len(result.used)} views used, {args.min_views} required"
        )
    status = "PASS" if not reasons else "FAIL"

    model = CameraModel(
        image_size=tuple(image_size),
        camera_matrix=result.camera_matrix,
        dist_coeffs=result.dist_coeffs,
        new_camera_matrix=new_matrix,
        rms_reprojection_px=result.rms_reprojection_px,
        n_views_used=len(result.used),
        created_utc=_utc_now(),
        board={
            "squares_x": spec.squares_x,
            "squares_y": spec.squares_y,
            "dictionary": spec.dictionary,
            "square_mm_nominal": spec.square_mm_nominal,
            "square_mm_measured_x": spec.measured_square_x_mm,
            "square_mm_measured_y": spec.measured_square_y_mm,
            "square_mm_measured_mean": spec.measured_square_mm,
            "measurement_instrument": spec.measurement_instrument,
        },
        source_images=tuple(str(v.path) for v in result.used),
    )

    report_path = write_report(
        args=args,
        spec=spec,
        views=views,
        result=result,
        model=model,
        image_size=image_size,
        status=status,
        reasons=reasons,
        roi=roi,
    )

    if status == "PASS":
        model.save(args.out)
        print(f"calibration PASS: rms {model.rms_reprojection_px:.3f} px from {model.n_views_used} views")
        print(f"camera model: {args.out}")
        print(f"report: {report_path}")
        return EXIT_OK

    print(f"calibration FAIL: {'; '.join(reasons)}", file=sys.stderr)
    print(f"report written to {report_path}; camera.json was NOT saved", file=sys.stderr)
    return EXIT_GATE_FAILED


def _detect_only(
    args: argparse.Namespace, spec: BoardSpec, views: list[View], image_size: tuple[int, int]
) -> int:
    args.report_dir.mkdir(parents=True, exist_ok=True)
    summary = args.report_dir / "detection_summary.csv"
    usable = 0
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "corners", "markers", "reason"])
        for view in views:
            if view.detected:
                usable += 1
            writer.writerow([view.name, view.n_corners, view.n_markers, view.reason])

    print(f"images: {len(views)}  usable: {usable}  size: {image_size[0]}x{image_size[1]}")
    for view in views:
        flag = "ok" if view.detected else "REJECT"
        print(f"  {flag:6s} {view.name:32s} corners={view.n_corners:3d} {view.reason}")
    print(f"summary: {summary}")
    return EXIT_OK if usable >= args.min_views else EXIT_NO_USABLE_VIEWS


def _draw_corners(image: np.ndarray, view: View, color: tuple[int, int, int]) -> None:
    if view.corners is None:
        return
    for x, y in view.corners:
        cv2.circle(image, (int(round(x)), int(round(y))), 3, color, -1, lineType=cv2.LINE_AA)


def _write_overlays(report_dir: Path, views: list[View]) -> None:
    out = report_dir / "views"
    out.mkdir(parents=True, exist_ok=True)
    for view in views:
        image = cv2.imread(str(view.path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        color = (0, 200, 0) if view.used else (0, 0, 220)
        _draw_corners(image, view, color)
        label = f"{view.name} corners={view.n_corners} rms={view.rms_px or 0:.2f}px"
        cv2.putText(
            image, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA
        )
        half = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / f"{view.path.stem}.png"), half)


def _write_undistort_check(
    report_dir: Path, model: CameraModel, view: View, image_size: tuple[int, int]
) -> Path:
    raw = cv2.imread(str(view.path), cv2.IMREAD_COLOR)
    if raw is None:
        raise CalibrationError(f"could not re-read {view.path} for the undistortion check")
    undistorted = model.undistort(raw)

    width, height = image_size
    for canvas in (raw, undistorted):
        for x in range(0, width, 100):
            cv2.line(canvas, (x, 0), (x, height - 1), (0, 255, 0), 1, cv2.LINE_AA)
        for y in range(0, height, 100):
            cv2.line(canvas, (0, y), (width - 1, y), (0, 255, 0), 1, cv2.LINE_AA)

    cv2.putText(raw, "raw (distorted)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(
        undistorted, "undistorted", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
    )

    combined = cv2.hconcat([raw, undistorted])
    combined = cv2.resize(combined, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    target = report_dir / "undistort_check.png"
    cv2.imwrite(str(target), combined)
    return target


def write_report(
    *,
    args: argparse.Namespace,
    spec: BoardSpec,
    views: list[View],
    result: CalibrationResult,
    model: CameraModel,
    image_size: tuple[int, int],
    status: str,
    reasons: list[str],
    roi: tuple[int, int, int, int] | None,
) -> Path:
    report_dir: Path = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    _write_overlays(report_dir, views)
    used_errors = sorted(result.used, key=lambda v: v.rms_px or 0.0)
    middle = used_errors[len(used_errors) // 2]
    check_path = _write_undistort_check(report_dir, model, middle, image_size)

    csv_path = report_dir / "per_view_errors.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "corners", "markers", "rms_px", "used", "reason"])
        for view in views:
            writer.writerow(
                [
                    view.name,
                    view.n_corners,
                    view.n_markers,
                    "" if view.rms_px is None else f"{view.rms_px:.4f}",
                    int(view.used),
                    view.reason,
                ]
            )

    worst = max(result.used, key=lambda v: v.rms_px or 0.0)
    displacement = model.max_undistort_displacement_px()
    rejected = [v for v in views if not v.used]

    lines: list[str] = []
    lines.append("# Camera calibration report\n")
    lines.append(f"- **Status: {status}**")
    for reason in reasons:
        lines.append(f"- Reason: {reason}")
    lines.append(f"- Generated: {_utc_now()}")
    lines.append(f"- Command: `{shlex.join(sys.argv)}`")
    lines.append("")
    lines.append("## Counts and input")
    lines.append("")
    lines.append(f"- Images found in `{args.images}`: **{len(views)}**")
    lines.append(f"- Views used: **{len(result.used)}**")
    lines.append(f"- Views rejected: **{len(rejected)}**")
    lines.append(f"- Resolution: **{image_size[0]} x {image_size[1]}** px (working frame)")
    lines.append(
        "- Source JPEGs: EXIF orientation applied, then resized with INTER_AREA when the "
        "source differs from the working frame"
    )
    lines.append(
        f"- Board: {spec.squares_x} x {spec.squares_y} squares, {spec.dictionary}, "
        f"{spec.inner_corners} inner corners"
    )
    lines.append(
        f"- Printed square (measured): **{spec.measured_square_mm:.3f} mm** "
        f"(x {spec.measured_square_x_mm:.3f}, y {spec.measured_square_y_mm:.3f}); "
        f"instrument: {spec.measurement_instrument}"
    )
    lines.append(f"- Undistortion alpha: {args.alpha}")
    if roi is not None:
        lines.append(f"- Valid region after undistortion: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
    lines.append("")
    lines.append("## Accuracy")
    lines.append("")
    lines.append(f"- **RMS reprojection error: {model.rms_reprojection_px:.3f} px**")
    lines.append(
        f"- Per-view error: median {np.median(result.view_rms):.3f} px, "
        f"worst {worst.rms_px:.3f} px ({worst.name})"
    )
    lines.append(
        f"- Maximum displacement from undistortion at the frame border: "
        f"**{displacement:.2f} px**"
    )
    lines.append("")
    lines.append("## Intrinsics")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append(f"| fx | {model.fx:.3f} px |")
    lines.append(f"| fy | {model.fy:.3f} px |")
    lines.append(f"| cx | {model.cx:.3f} px |")
    lines.append(f"| cy | {model.cy:.3f} px |")
    lines.append(
        "| distortion (k1, k2, p1, p2, k3) | "
        + ", ".join(f"{v:.6g}" for v in model.dist_coeffs)
        + " |"
    )
    lines.append("")
    lines.append("## Per-view results")
    lines.append("")
    lines.append("| image | corners | markers | rms px | used | reason |")
    lines.append("|---|---:|---:|---:|:--:|---|")
    for view in views:
        rms_text = "" if view.rms_px is None else f"{view.rms_px:.3f}"
        lines.append(
            f"| {view.name} | {view.n_corners} | {view.n_markers} | {rms_text} | "
            f"{'yes' if view.used else 'no'} | {view.reason} |"
        )
    lines.append("")
    lines.append("## Evidence")
    lines.append("")
    lines.append(f"- Corner overlays: `{report_dir / 'views'}`")
    lines.append(f"- Per-view CSV: `{csv_path}`")
    lines.append(f"- Raw vs undistorted: `{check_path}`")
    lines.append("")
    lines.append(
        "This report was produced from real captured images only. Synthetic images are used "
        "by the test suite and never written here."
    )

    report_path = report_dir / "calibration_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


if __name__ == "__main__":
    raise SystemExit(main())
