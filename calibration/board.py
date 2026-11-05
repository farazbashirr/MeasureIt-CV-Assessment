"""Printable ChArUco board and the measured-size guard.

Two different numbers live here and must never be confused:

* ``square_mm_nominal``  what the PDF asks the printer for;
* ``measured_square_*``  what a ruler or calliper actually finds on paper.

Calibration refuses to run until the measured values exist, because
calibrating against a nominal size silently bakes the printer's scale error
into every millimetre the system will ever report.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PAGE_SIZES_MM: dict[str, tuple[float, float]] = {
    "A4": (297.0, 210.0),
    "LETTER": (279.4, 215.9),
}

PIXELS_PER_MM = 12.0  # 304.8 dpi equivalent; exact integer pixels per square
MIN_PAGE_MARGIN_MM = 10.0


class BoardError(RuntimeError):
    """The board description or the print measurement is unusable."""


class MissingMeasurementError(BoardError):
    """The printed board has not been measured yet."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class BoardSpec:
    squares_x: int = 9
    squares_y: int = 7
    square_mm_nominal: float = 25.0
    marker_mm_nominal: float = 18.0
    dictionary: str = "DICT_4X4_50"
    page: str = "A4"
    pixels_per_mm: float = PIXELS_PER_MM
    measured_square_x_mm: float | None = None
    measured_square_y_mm: float | None = None
    measurement_instrument: str | None = None
    measured_on: str | None = None
    generated_utc: str = field(default_factory=_utc_now)
    schema: str = "measureit.charuco/1"

    # -- derived -----------------------------------------------------------

    @property
    def inner_corners(self) -> int:
        return (self.squares_x - 1) * (self.squares_y - 1)

    @property
    def page_size_mm(self) -> tuple[float, float]:
        try:
            return PAGE_SIZES_MM[self.page.upper()]
        except KeyError as exc:
            raise BoardError(
                f"unknown page {self.page!r}; choose one of {sorted(PAGE_SIZES_MM)}"
            ) from exc

    @property
    def board_size_mm(self) -> tuple[float, float]:
        return (
            self.squares_x * self.square_mm_nominal,
            self.squares_y * self.square_mm_nominal,
        )

    @property
    def marker_ratio(self) -> float:
        return self.marker_mm_nominal / self.square_mm_nominal

    @property
    def measured_square_mm(self) -> float:
        if self.measured_square_x_mm is None or self.measured_square_y_mm is None:
            raise MissingMeasurementError(
                "the printed board has not been measured: record measured_square_x_mm and "
                "measured_square_y_mm with `python -m calibration.make_board measure` "
                "before calibrating"
            )
        return 0.5 * (self.measured_square_x_mm + self.measured_square_y_mm)

    @property
    def axis_difference_pct(self) -> float | None:
        if self.measured_square_x_mm is None or self.measured_square_y_mm is None:
            return None
        mean = self.measured_square_mm
        return abs(self.measured_square_x_mm - self.measured_square_y_mm) / mean * 100.0

    # -- validation --------------------------------------------------------

    def validate(self) -> None:
        if self.squares_x < 3 or self.squares_y < 3:
            raise BoardError("a ChArUco board needs at least 3 x 3 squares")
        if self.inner_corners < 12:
            raise BoardError(
                f"only {self.inner_corners} inner corners; at least 12 are needed"
            )
        if self.square_mm_nominal <= 0 or self.marker_mm_nominal <= 0:
            raise BoardError("square and marker sizes must be positive")
        if self.marker_mm_nominal >= self.square_mm_nominal:
            raise BoardError("the marker must be smaller than the square")
        if not 0.4 <= self.marker_ratio <= 0.85:
            raise BoardError(
                f"marker/square ratio {self.marker_ratio:.2f} is outside the printable "
                "range 0.40-0.85"
            )
        board_w, board_h = self.board_size_mm
        page_w, page_h = self.page_size_mm
        if board_w > page_w - 2 * MIN_PAGE_MARGIN_MM or board_h > page_h - 2 * MIN_PAGE_MARGIN_MM:
            raise BoardError(
                f"board {board_w:.1f}x{board_h:.1f} mm does not fit {self.page} "
                f"({page_w:.1f}x{page_h:.1f} mm) with {MIN_PAGE_MARGIN_MM:.0f} mm margins"
            )
        if not hasattr(cv2.aruco, self.dictionary):
            raise BoardError(f"OpenCV has no ArUco dictionary named {self.dictionary!r}")

    # -- io ----------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["inner_corners"] = self.inner_corners
        payload["board_size_mm"] = list(self.board_size_mm)
        payload["page_size_mm"] = list(self.page_size_mm)
        payload["measured_square_mm"] = (
            None if self.measured_square_x_mm is None or self.measured_square_y_mm is None
            else self.measured_square_mm
        )
        payload["notes"] = (
            "Nominal sizes describe the PDF, not the paper. Fill measured_square_*_mm "
            "and measurement_instrument before calibrating."
        )
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BoardSpec":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})


def load_spec(path: str | Path) -> BoardSpec:
    source = Path(path)
    if not source.is_file():
        raise BoardError(f"board spec not found: {source}")
    spec = BoardSpec.from_dict(json.loads(source.read_text(encoding="utf-8")))
    spec.validate()
    return spec


def save_spec(spec: BoardSpec, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(spec.to_dict(), indent=2) + "\n", encoding="utf-8")


# -- board geometry --------------------------------------------------------


def build_charuco(spec: BoardSpec, *, measured: bool = True) -> cv2.aruco.CharucoBoard:
    """Build the OpenCV board.

    ``measured=True`` uses the measured square size, which is the only value
    calibration may use. ``measured=False`` uses the nominal size for
    rendering and for synthetic tests.
    """
    if measured:
        square_mm = spec.measured_square_mm
    else:
        square_mm = spec.square_mm_nominal
    square_m = square_mm / 1000.0
    marker_m = square_m * spec.marker_ratio
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.dictionary))
    return cv2.aruco.CharucoBoard(
        (spec.squares_x, spec.squares_y), square_m, marker_m, dictionary
    )


def render_board_image(spec: BoardSpec) -> np.ndarray:
    """Render the board at exactly ``pixels_per_mm`` with no margin."""
    spec.validate()
    board = build_charuco(spec, measured=False)
    px_per_square = int(round(spec.square_mm_nominal * spec.pixels_per_mm))
    size = (spec.squares_x * px_per_square, spec.squares_y * px_per_square)
    image = board.generateImage(size, marginSize=0, borderBits=1)
    if image is None:
        raise BoardError("OpenCV failed to render the ChArUco board")
    return image


# -- PDF layout ------------------------------------------------------------

_PT_PER_MM = 72.0 / 25.4


def _mm(value: float) -> float:
    return value * _PT_PER_MM


def write_board_pdf(spec: BoardSpec, board_png: Path, pdf_path: Path) -> None:
    """Write a two-page A4 PDF: the board, then the capture protocol."""
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    spec.validate()
    page_w, page_h = spec.page_size_mm
    board_w, board_h = spec.board_size_mm
    x = (page_w - board_w) / 2.0
    y = 12.0

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = canvas.Canvas(str(pdf_path), pagesize=(_mm(page_w), _mm(page_h)))
    doc.setTitle("Measure It - ChArUco calibration board")
    doc.setAuthor("Measure It")

    doc.drawImage(
        ImageReader(str(board_png)),
        _mm(x),
        _mm(y),
        width=_mm(board_w),
        height=_mm(board_h),
        preserveAspectRatio=False,
        mask=None,
    )

    # 100 mm scale bar, drawn as vectors so it is independent of the raster.
    bar_y = y + board_h + 10.0
    doc.setStrokeColorRGB(0, 0, 0)
    doc.setLineWidth(0.6)
    doc.line(_mm(x), _mm(bar_y), _mm(x + 100.0), _mm(bar_y))
    for offset in range(0, 101, 10):
        tick = 2.5 if offset % 50 else 4.0
        doc.line(
            _mm(x + offset), _mm(bar_y), _mm(x + offset), _mm(bar_y + tick)
        )
    doc.setFont("Helvetica", 8)
    doc.drawString(_mm(x), _mm(bar_y + 5.0), "100.00 mm scale bar")
    doc.drawString(
        _mm(x) + 200.0,
        _mm(bar_y + 5.0),
        "Print at 100% (no fit-to-page). Then measure the printed square and record it.",
    )

    doc.setFont("Helvetica", 7)
    doc.drawString(
        _mm(x),
        _mm(5.0),
        f"ChArUco {spec.squares_x}x{spec.squares_y} | nominal square "
        f"{spec.square_mm_nominal:.2f} mm | marker {spec.marker_mm_nominal:.2f} mm | "
        f"{spec.dictionary} | inner corners {spec.inner_corners}",
    )
    doc.showPage()

    _write_protocol_page(doc, spec)
    doc.save()


def _write_protocol_page(doc: Any, spec: BoardSpec) -> None:
    page_w, _ = spec.page_size_mm
    left = 18.0
    width = page_w - 2 * left
    cursor = 196.0

    doc.setFont("Helvetica-Bold", 14)
    doc.drawString(_mm(left), _mm(cursor), "ChArUco capture protocol")
    cursor -= 8.0

    blocks: list[tuple[str, list[str]]] = [
        (
            "1. Print and mount",
            [
                f"Print this page at 100% on matte {spec.page} cardstock. Do not use 'fit to page'.",
                "Glue the sheet flat to a rigid board (foam board, MDF or glass). A taped sheet",
                "still bows, and warp cannot be corrected later: it shows up as reprojection",
                "error that will not come down.",
                "Check the 100 mm bar above: it must measure 100.0 mm +/- 0.3 mm.",
            ],
        ),
        (
            "2. Measure the print, then record it",
            [
                "Measure three squares horizontally and three vertically with the instrument you",
                "will use for the object, and enter the means:",
                "  python -m calibration.make_board measure --spec calibration/board/board_spec.json \\",
                "      --square-x-mm <mean> --square-y-mm <mean> --instrument \"<model, 0.01 mm, +/-0.02 mm>\"",
                "If the two axes differ by more than 0.3% the print is stretched: reprint.",
            ],
        ),
        (
            "3. Lock the camera (Pixel 8 Pro)",
            [
                "Main (wide) lens only. Disable automatic lens switching, ultra-wide and telephoto.",
                "Zoom fixed at 1.0x. Lock focus at the working distance and never change it again.",
                "Lock ISO and white balance. Disable HDR, night mode and portrait mode.",
                "4:3, maximum JPEG quality, no HEIC. Keep one phone orientation for every photo.",
            ],
        ),
        (
            "4. Capture 30 or more views",
            [
                "Move the board, not the camera: tilt +/-30 degrees around both axes, rotate it",
                "in plane, and place it centre, corners and edges of the frame.",
                "Vary distance so the board fills 40-70% of the frame. Reject blurred frames.",
                "Diffuse daylight only: no direct lamp reflection on the board.",
                "Every view needs at least 12 inner corners, 20+ is better. Never crop the images.",
            ],
        ),
        (
            "5. Transfer and calibrate",
            [
                "Copy the original JPEGs over USB/MTP (not a messaging app) into calibration/images/.",
                "The command applies EXIF rotation and resizes to the 2048x1536 working frame.",
                "Run:  python -m calibration.calibrate --images calibration/images",
                "Read calibration/report/calibration_report.md, then look at undistort_check.png.",
            ],
        ),
    ]

    for title, lines in blocks:
        doc.setFont("Helvetica-Bold", 10)
        doc.drawString(_mm(left), _mm(cursor), title)
        cursor -= 5.2
        doc.setFont("Helvetica", 9)
        for line in lines:
            doc.drawString(_mm(left), _mm(cursor), line)
            cursor -= 4.4
        cursor -= 3.0

    doc.setFont("Helvetica-Bold", 9.5)
    doc.drawString(_mm(left), _mm(cursor), "Measurement record (write it down before calibrating)")
    cursor -= 5.0
    doc.setFont("Helvetica", 9)
    for label in (
        "Square size, horizontal mean (mm):  ........................................",
        "Square size, vertical mean (mm):    ........................................",
        "Instrument, resolution, accuracy:   ........................................",
        "Date and initials:                  ........................................",
    ):
        doc.drawString(_mm(left), _mm(cursor), label)
        cursor -= 4.6

    doc.showPage()


# -- generation and measurement recording ----------------------------------


def generate(spec: BoardSpec, outdir: str | Path) -> dict[str, Path]:
    spec.validate()
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    image = render_board_image(spec)
    png_path = out / "charuco_board.png"
    if not cv2.imwrite(str(png_path), image):
        raise BoardError(f"could not write {png_path}")

    pdf_path = out / "charuco_board.pdf"
    write_board_pdf(spec, png_path, pdf_path)

    spec_path = out / "board_spec.json"
    save_spec(spec, spec_path)
    return {"png": png_path, "pdf": pdf_path, "spec": spec_path}


def record_measurement(
    spec_path: str | Path,
    *,
    square_x_mm: float,
    square_y_mm: float,
    instrument: str,
    measured_on: str | None = None,
) -> tuple[BoardSpec, list[str]]:
    """Store the measured print size and return any warnings."""
    if square_x_mm <= 0 or square_y_mm <= 0:
        raise BoardError("measured square sizes must be positive")
    if not instrument.strip():
        raise BoardError("record the instrument that produced the measurement")

    spec = load_spec(spec_path)
    mean = 0.5 * (square_x_mm + square_y_mm)
    warnings: list[str] = []

    axis_pct = abs(square_x_mm - square_y_mm) / mean * 100.0
    if axis_pct > 0.3:
        warnings.append(
            f"horizontal and vertical squares differ by {axis_pct:.2f}% (> 0.3%): "
            "the print is stretched and should be reprinted"
        )

    nominal_error_pct = abs(mean - spec.square_mm_nominal) / spec.square_mm_nominal * 100.0
    if nominal_error_pct > 5.0:
        warnings.append(
            f"measured square is {nominal_error_pct:.1f}% away from nominal "
            f"{spec.square_mm_nominal:.2f} mm; check the print scale and your instrument"
        )

    updated = BoardSpec(
        **{
            **{
                key: getattr(spec, key)
                for key in spec.__dataclass_fields__  # type: ignore[attr-defined]
            },
            "measured_square_x_mm": float(square_x_mm),
            "measured_square_y_mm": float(square_y_mm),
            "measurement_instrument": instrument.strip(),
            "measured_on": measured_on or _utc_now(),
        }
    )
    save_spec(updated, spec_path)
    return updated, warnings
