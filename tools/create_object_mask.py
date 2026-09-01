#!/usr/bin/env python3
"""Create a binary object mask from one RGB image, box, and foreground point.

The mask is intentionally kept in original RGB resolution.  It can therefore
be combined with the registered original-resolution depth image before either
is center-cropped for HUG.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument(
        "--point",
        type=int,
        nargs=2,
        metavar=("U", "V"),
        help="Foreground pixel in original RGB coordinates. Omit to click it in a window.",
    )
    parser.add_argument(
        "--rect",
        type=int,
        nargs=4,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        help="Object bounding box. If omitted, draw one in the OpenCV window.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--refine",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Open the foreground/background correction window. Defaults to on for GUI selection.",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        help="Optional RGB preview showing the selected object and seed point.",
    )
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    return args


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read RGB image: {path}")
    return image


def _validate_rect(rect: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, rect_width, rect_height = rect
    if rect_width <= 1 or rect_height <= 1:
        raise ValueError("Object rectangle must have width and height greater than one pixel")
    if x < 0 or y < 0 or x + rect_width > width or y + rect_height > height:
        raise ValueError(
            f"Rectangle {rect} is outside the {width}x{height} RGB image"
        )
    return rect


def _select_rect(image: np.ndarray, point: tuple[int, int]) -> tuple[int, int, int, int]:
    """Select a valid rectangle with all controls visible in the image view."""
    title = "Object rectangle"
    start: list[tuple[int, int]] = []
    end: list[tuple[int, int]] = []
    dragging = [False]
    status = [""]

    def render() -> np.ndarray:
        display = image.copy()
        cv2.drawMarker(
            display, point, (0, 0, 255), markerType=cv2.MARKER_CROSS,
            markerSize=22, thickness=2,
        )
        if start and end:
            x0, y0 = start[0]
            x1, y1 = end[0]
            cv2.rectangle(display, (x0, y0), (x1, y1), (0, 255, 255), 2)
        cv2.rectangle(display, (0, 0), (min(display.shape[1] - 1, 900), 92), (0, 0, 0), -1)
        cv2.putText(display, "LEFT-DRAG: object box containing red cross", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display, "ENTER / SPACE: confirm    C / ESC: cancel", (12, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2, cv2.LINE_AA)
        if status[0]:
            cv2.putText(display, status[0], (12, 82), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 180, 255), 1, cv2.LINE_AA)
        return display

    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            start[:] = [(x, y)]
            end[:] = [(x, y)]
            dragging[0] = True
            status[0] = ""
        elif event == cv2.EVENT_MOUSEMOVE and dragging[0]:
            end[:] = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP and dragging[0]:
            end[:] = [(x, y)]
            dragging[0] = False
        else:
            return
        cv2.imshow(title, render())

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, on_mouse)
    cv2.imshow(title, render())
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("c"), 27):
            cv2.destroyWindow(title)
            raise RuntimeError("Object-rectangle selection cancelled")
        if key not in (13, 32):
            continue
        if not start or not end:
            status[0] = "Draw a rectangle before confirming."
            cv2.imshow(title, render())
            continue
        x0, y0 = start[0]
        x1, y1 = end[0]
        rect = (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
        try:
            rect = _validate_rect(rect, image.shape[1], image.shape[0])
        except ValueError as exc:
            status[0] = str(exc)
            cv2.imshow(title, render())
            continue
        cv2.destroyWindow(title)
        return rect


def _select_point(image: np.ndarray) -> tuple[int, int]:
    """Select a foreground point with all controls visible in the image view."""
    title = "Object foreground point"
    selection: list[tuple[int, int]] = []

    def render() -> np.ndarray:
        display = image.copy()
        if selection:
            cv2.drawMarker(display, selection[0], (0, 0, 255),
                           markerType=cv2.MARKER_CROSS, markerSize=22, thickness=2)
        cv2.rectangle(display, (0, 0), (min(display.shape[1] - 1, 820), 64), (0, 0, 0), -1)
        cv2.putText(display, "LEFT-CLICK: a pixel on the object to grasp", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display, "ENTER / SPACE: confirm    C / ESC: cancel", (12, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2, cv2.LINE_AA)
        return display

    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        selection[:] = [(x, y)]
        cv2.imshow(title, render())

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, on_mouse)
    cv2.imshow(title, render())
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32) and selection:
            cv2.destroyWindow(title)
            return selection[0]
        if key in (ord("c"), 27):
            cv2.destroyWindow(title)
            raise RuntimeError("Object-point selection cancelled")


def _segment(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    point: tuple[int, int],
    iterations: int,
) -> np.ndarray:
    mask = np.full(image.shape[:2], cv2.GC_BGD, dtype=np.uint8)
    x, y, width, height = rect
    mask[y : y + height, x : x + width] = cv2.GC_PR_FGD
    u, v = point
    mask[v, u] = cv2.GC_FGD
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        image,
        mask,
        None,
        background_model,
        foreground_model,
        iterations,
        cv2.GC_INIT_WITH_MASK,
    )
    return mask


def _binary_mask(grabcut_mask: np.ndarray) -> np.ndarray:
    return np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)


def _refine_mask(
    image: np.ndarray, grabcut_mask: np.ndarray, iterations: int
) -> np.ndarray:
    """Let the user correct a GrabCut result with foreground/background strokes."""
    title = "Refine: left=object, right=background, g=recompute, Enter=save, c=cancel"
    markings: list[tuple[tuple[int, int], bool]] = []

    def render() -> np.ndarray:
        binary = _binary_mask(grabcut_mask)
        preview = image.copy()
        preview[binary == 0] = (preview[binary == 0] * 0.25).astype(np.uint8)
        for point, is_foreground in markings:
            color = (0, 220, 0) if is_foreground else (0, 0, 255)
            cv2.circle(preview, point, 4, color, -1)
        cv2.rectangle(preview, (0, 0), (min(preview.shape[1] - 1, 900), 64), (0, 0, 0), -1)
        cv2.putText(preview, "LEFT: object   RIGHT: background   G: recompute", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(preview, "ENTER / S: save    C / ESC: cancel", (12, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2, cv2.LINE_AA)
        return preview

    def mark(x: int, y: int, is_foreground: bool) -> None:
        label = cv2.GC_FGD if is_foreground else cv2.GC_BGD
        cv2.circle(grabcut_mask, (x, y), 4, label, -1)
        markings.append(((x, y), is_foreground))

    def on_mouse(event: int, x: int, y: int, flags: int, _userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN or (
            event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON
        ):
            mark(x, y, True)
        elif event == cv2.EVENT_RBUTTONDOWN or (
            event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_RBUTTON
        ):
            mark(x, y, False)
        else:
            return
        cv2.imshow(title, render())

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, on_mouse)
    cv2.imshow(title, render())
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, ord("s")):
            break
        if key == ord("g"):
            background_model = np.zeros((1, 65), dtype=np.float64)
            foreground_model = np.zeros((1, 65), dtype=np.float64)
            cv2.grabCut(
                image,
                grabcut_mask,
                None,
                background_model,
                foreground_model,
                iterations,
                cv2.GC_INIT_WITH_MASK,
            )
            cv2.imshow(title, render())
        if key in (ord("c"), 27):
            cv2.destroyWindow(title)
            raise RuntimeError("Object-mask refinement cancelled")
    cv2.destroyWindow(title)
    return grabcut_mask

def _write_preview(path: Path, image: np.ndarray, mask: np.ndarray, point: tuple[int, int]) -> None:
    preview = image.copy()
    preview[mask == 0] = (preview[mask == 0] * 0.25).astype(np.uint8)
    cv2.drawMarker(
        preview,
        point,
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=20,
        thickness=2,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), preview):
        raise IOError(f"Failed to write preview: {path}")


def main() -> None:
    args = _parse_args()
    image = _read_rgb(args.rgb)
    height, width = image.shape[:2]
    point = (
        (int(args.point[0]), int(args.point[1]))
        if args.point is not None
        else _select_point(image)
    )
    if not (0 <= point[0] < width and 0 <= point[1] < height):
        raise ValueError(f"Foreground point {point} is outside the {width}x{height} RGB image")
    if args.rect:
        rect = _validate_rect(tuple(args.rect), width, height)
    else:
        while True:
            rect = _select_rect(image, point)
            try:
                rect = _validate_rect(rect, width, height)
                break
            except ValueError as exc:
                print(f"Invalid object rectangle ({exc}); drag a box with the left mouse button and press Enter to retry.")
    x, y, rect_width, rect_height = rect
    if not (x <= point[0] < x + rect_width and y <= point[1] < y + rect_height):
        raise ValueError(
            f"--point {point} must be inside the object rectangle {rect}; "
            "redraw the rectangle so it contains the red cross"
        )

    grabcut_mask = _segment(image, rect, point, args.iterations)
    refine = args.refine if args.refine is not None else args.rect is None
    if refine:
        grabcut_mask = _refine_mask(image, grabcut_mask, args.iterations)
    mask = _binary_mask(grabcut_mask)
    area = int(np.count_nonzero(mask))
    if area == 0:
        raise RuntimeError("GrabCut produced an empty object mask")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), mask):
        raise IOError(f"Failed to write object mask: {args.output}")
    preview_path = args.preview or args.output.with_name(args.output.stem + "_preview.png")
    _write_preview(preview_path, image, mask, point)
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "rgb": str(args.rgb.resolve()),
                "resolution": [width, height],
                "foreground_point": list(point),
                "rectangle": list(rect),
                "iterations": args.iterations,
                "mask_area_pixels": area,
                "mask_area_fraction": area / float(width * height),
                "preview": str(preview_path.resolve()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("Object mask written")
    print(f"  mask: {args.output}")
    print(f"  preview: {preview_path}")
    print(f"  foreground pixels: {area} ({area / float(width * height):.2%})")
    print(f"  metadata: {metadata_path}")


if __name__ == "__main__":
    main()
