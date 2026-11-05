"""Command line for the ChArUco board: generate it, then record its real size."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from calibration.board import BoardError, BoardSpec, generate, load_spec, record_measurement


def _parse_squares(text: str) -> tuple[int, int]:
    try:
        x, y = (int(v) for v in text.lower().replace("*", "x").split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("squares must look like 9x7") from exc
    if x < 3 or y < 3:
        raise argparse.ArgumentTypeError("need at least 3x3 squares")
    return x, y


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m calibration.make_board",
        description="Generate a print-ready ChArUco board and record the measured print size.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="write the PDF, PNG and board spec")
    gen.add_argument("--out", default=Path("calibration/board"), type=Path)
    gen.add_argument("--page", default="A4", choices=["A4", "LETTER"])
    gen.add_argument("--squares", default="9x7", type=_parse_squares)
    gen.add_argument("--square-mm", type=float, default=25.0)
    gen.add_argument("--marker-mm", type=float, default=18.0)
    gen.add_argument("--dictionary", default="DICT_4X4_50")
    gen.add_argument(
        "--keep-measurement",
        action="store_true",
        help="carry an existing measured square size into the regenerated spec",
    )

    measure = sub.add_parser("measure", help="record the printed square size")
    measure.add_argument("--spec", default=Path("calibration/board/board_spec.json"), type=Path)
    measure.add_argument("--square-x-mm", type=float, required=True)
    measure.add_argument("--square-y-mm", type=float, required=True)
    measure.add_argument(
        "--instrument",
        required=True,
        help='instrument and its stated accuracy, e.g. "Mitutoyo 500-196-30, 0.01 mm, +/-0.02 mm"',
    )
    measure.add_argument("--date", default=None)

    show = sub.add_parser("show", help="print the current board status")
    show.add_argument("--spec", default=Path("calibration/board/board_spec.json"), type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "generate":
            return _generate(args)
        if args.command == "measure":
            return _measure(args)
        return _show(args)
    except BoardError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def _generate(args: argparse.Namespace) -> int:
    squares_x, squares_y = args.squares
    spec = BoardSpec(
        squares_x=squares_x,
        squares_y=squares_y,
        square_mm_nominal=args.square_mm,
        marker_mm_nominal=args.marker_mm,
        dictionary=args.dictionary,
        page=args.page,
    )

    spec_path = args.out / "board_spec.json"
    if args.keep_measurement and spec_path.is_file():
        previous = load_spec(spec_path)
        spec = BoardSpec(
            **{
                **{k: getattr(spec, k) for k in spec.__dataclass_fields__},
                "measured_square_x_mm": previous.measured_square_x_mm,
                "measured_square_y_mm": previous.measured_square_y_mm,
                "measurement_instrument": previous.measurement_instrument,
                "measured_on": previous.measured_on,
            }
        )

    paths = generate(spec, args.out)
    board_w, board_h = spec.board_size_mm
    page_w, page_h = spec.page_size_mm

    print(f"board      : {spec.squares_x}x{spec.squares_y} squares, {spec.inner_corners} inner corners")
    print(f"board size : {board_w:.1f} x {board_h:.1f} mm on {args.page} ({page_w:.1f} x {page_h:.1f} mm)")
    print(f"printed at : 100% scale, matte cardstock, glued to a rigid flat backing")
    print(f"pdf        : {paths['pdf']}")
    print(f"png        : {paths['png']}")
    print(f"spec       : {paths['spec']}")
    print("")
    print("Next: print the PDF, measure three squares horizontally and three vertically, then")
    print("      python -m calibration.make_board measure --square-x-mm <mean> --square-y-mm <mean> \\")
    print("             --instrument \"<model, resolution, stated accuracy>\"")
    return 0


def _measure(args: argparse.Namespace) -> int:
    spec, warnings = record_measurement(
        args.spec,
        square_x_mm=args.square_x_mm,
        square_y_mm=args.square_y_mm,
        instrument=args.instrument,
        measured_on=args.date,
    )
    print(f"recorded: square x={spec.measured_square_x_mm:.3f} mm, y={spec.measured_square_y_mm:.3f} mm")
    print(f"mean     : {spec.measured_square_mm:.3f} mm (nominal {spec.square_mm_nominal:.3f} mm)")
    for warning in warnings:
        print(f"WARNING: {warning}")
    print(f"spec     : {args.spec}")
    return 0


def _show(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    board_w, board_h = spec.board_size_mm
    print(f"board   : {spec.squares_x}x{spec.squares_y}, {spec.dictionary}, {spec.inner_corners} corners")
    print(f"size    : {board_w:.1f} x {board_h:.1f} mm, page {spec.page}")
    if spec.measured_square_x_mm is None:
        print("measured: NOT MEASURED - calibration is blocked until the printed square is measured")
        return 0
    print(f"measured: {spec.measured_square_mm:.3f} mm (x {spec.measured_square_x_mm:.3f}, y {spec.measured_square_y_mm:.3f})")
    if spec.axis_difference_pct is not None:
        print(f"axes    : {spec.axis_difference_pct:.3f}% apart")
    print(f"tool    : {spec.measurement_instrument}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
