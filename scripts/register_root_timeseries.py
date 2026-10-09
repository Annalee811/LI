#!/usr/bin/env python3
"""Register same-site T2/T3/T4 root images to the T1 coordinate system.

Input layout (filenames must match across periods)::

    data/T1/sample.png
    data/T2/sample.png
    data/T3/sample.png
    data/T4/sample.png

The source images are opened read-only. All generated files are written below
``results/registration/full_frame`` (or ``--output-dir``).

The script never forces a moving frame to fill T1 when the photographs only
partly overlap.  It writes both a T1-sized registered layer (transparent/NA
outside the real overlap) and a full-extent layer that preserves every source
pixel.  This distinction is essential for longitudinal root measurements:
inventing correspondences in an unobserved area would create false changes.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch

from romatch import roma_outdoor


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
PERIODS = ("T2", "T3", "T4")


class RegistrationError(RuntimeError):
    """An error with an explicit pipeline stage and image pair."""

    def __init__(self, stage: str, pair: str, message: str):
        super().__init__(f"[{stage}] {pair}: {message}")
        self.stage = stage
        self.pair = pair


@dataclass
class QualityMetrics:
    filename: str
    moving_period: str
    transform_type: str
    transform_source: str
    registration_status: str
    t1_coverage_fraction: float
    moving_visible_fraction: float
    overlap_class: str
    dense_confident_fraction: float
    fixed_support_grid_fraction: float
    moving_support_grid_fraction: float
    droplet_exclusion_enabled: bool
    t1_droplet_mask_fraction: float
    moving_droplet_mask_fraction: float
    root_soil_analysis_fraction: float
    raw_candidate_matches: int
    candidate_matches: int
    water_excluded_candidate_fraction: float
    selected_matches: int
    inlier_matches: int
    inlier_ratio: float
    mean_match_confidence: float
    root_priority_fraction: float
    reprojection_rmse_px: float | None
    reprojection_median_px: float | None
    reprojection_p95_px: float | None
    root_reprojection_rmse_px: float | None
    valid_overlap_fraction: float
    root_soil_ncc_before: float | None
    root_soil_ncc_after: float | None
    soil_lab_l_trimmed_ncc_after: float | None
    soil_lab_a_trimmed_ncc_after: float | None
    soil_lab_b_trimmed_ncc_after: float | None


@dataclass(frozen=True)
class FrameOverlap:
    """Area accounting for a moving frame projected into T1 coordinates."""

    intersection_area_px2: float
    t1_coverage_fraction: float
    moving_visible_fraction: float
    transformed_moving_area_px2: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register same-named T2/T3/T4 root images to T1 with RoMa."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/registration/full_frame")
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device. 'auto' uses CUDA when available.",
    )
    parser.add_argument(
        "--transform",
        choices=("auto", "similarity", "affine", "homography"),
        default="auto",
        help=(
            "Global model. 'auto' uses a homography for broadly supported full-frame "
            "pairs and a safer similarity transform for partial overlaps."
        ),
    )
    parser.add_argument("--coarse-res", type=int, default=560)
    parser.add_argument("--upsample-res", type=int, default=864)
    parser.add_argument("--max-matches", type=int, default=5000)
    parser.add_argument("--min-confidence", type=float, default=0.05)
    parser.add_argument("--ransac-threshold", type=float, default=3.0)
    parser.add_argument(
        "--ignore-droplets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Detect circular condensation droplets and exclude them from transform "
            "fitting, visual comparison, and quality metrics (default: enabled)."
        ),
    )
    parser.add_argument(
        "--droplet-hough-threshold",
        type=float,
        default=23.0,
        help=(
            "Circular Hough accumulator threshold at a normalized 1200 px width. "
            "Higher values make the conservative droplet mask smaller (default: 23)."
        ),
    )
    parser.add_argument(
        "--min-full-overlap",
        type=float,
        default=0.85,
        help=(
            "Minimum coverage of both frames required to label a pair full-frame. "
            "Lower-coverage pairs are retained and explicitly marked partial overlap."
        ),
    )
    parser.add_argument(
        "--strict-full-overlap",
        action="store_true",
        help=(
            "Fail a pair instead of writing a partial-overlap result when either frame "
            "coverage is below --min-full-overlap."
        ),
    )
    parser.add_argument(
        "--reuse-warp-dir",
        type=Path,
        default=None,
        help=(
            "Optional prior result root containing <T1 filename>/<period>/roma_warp.npz. "
            "This resumes post-processing without rerunning RoMa."
        ),
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue other image pairs after one pair fails.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print a traceback in addition to the stage-specific error.",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RegistrationError("model setup", "all", "CUDA was requested but is unavailable")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def configure_torch_cache() -> Path:
    """Use a writable cache; the cloud home directory can be read-only."""

    configured = os.environ.get("TORCH_HOME")
    cache_dir = Path(configured) if configured else Path("/tmp/torch-cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(cache_dir))
    return cache_dir


def discover_inputs(data_dir: Path) -> list[tuple[str, dict[str, Path]]]:
    period_dirs = {period: data_dir / period for period in ("T1", *PERIODS)}
    # Single-sequence convenience layout: data/T1.png, data/T2.png, ...
    # This is checked only when the directory-based batch layout is absent.
    if not period_dirs["T1"].is_dir():
        flat: dict[str, Path] = {}
        for period in ("T1", *PERIODS):
            candidates = [
                path
                for path in sorted(data_dir.glob(f"{period}.*"))
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ]
            if len(candidates) > 1:
                raise RegistrationError(
                    "input discovery",
                    "all",
                    f"multiple flat-layout candidates found for {period}: "
                    + ", ".join(map(str, candidates)),
                )
            if candidates:
                flat[period] = candidates[0]
        if "T1" not in flat:
            raise RegistrationError(
                "input discovery",
                "all",
                f"neither {period_dirs['T1']} nor a flat-layout T1 image exists",
            )
        available = [period for period in PERIODS if period in flat]
        if not available:
            raise RegistrationError(
                "input discovery", "all", "T1 exists, but no T2/T3/T4 image was found"
            )
        return [(flat["T1"].name, flat)]

    missing_dirs = [str(period_dirs[p]) for p in PERIODS if not period_dirs[p].is_dir()]
    if len(missing_dirs) == len(PERIODS):
        raise RegistrationError(
            "input discovery", "all", f"no moving-period directories exist below {data_dir}"
        )

    by_period: dict[str, dict[str, Path]] = {}
    for period, directory in period_dirs.items():
        if not directory.is_dir():
            by_period[period] = {}
            continue
        files = {
            path.name: path
            for path in sorted(directory.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        by_period[period] = files

    if not by_period["T1"]:
        raise RegistrationError(
            "input discovery", "all", f"no supported images found in {period_dirs['T1']}"
        )

    missing: list[str] = []
    jobs: list[tuple[str, dict[str, Path]]] = []
    for filename, fixed_path in by_period["T1"].items():
        paths = {"T1": fixed_path}
        available = [period for period in PERIODS if filename in by_period[period]]
        if not available:
            missing.append(f"{filename}: no T2/T3/T4 counterpart")
            continue
        paths.update({period: by_period[period][filename] for period in available})
        jobs.append((filename, paths))

    if missing:
        raise RegistrationError(
            "input discovery",
            "all",
            "moving-period counterparts are missing (" + "; ".join(missing) + ")",
        )
    return jobs


def read_color(path: Path, pair: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RegistrationError("image loading", pair, f"cannot read {path}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise RegistrationError("image loading", pair, f"expected a 3-channel image: {path}")
    return image


def robust_normalize(array: np.ndarray) -> np.ndarray:
    low, high = np.percentile(array, (2.0, 98.0))
    if high <= low + 1e-6:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array.astype(np.float32) - low) / (high - low), 0.0, 1.0)


def robust_normalize_uint8(
    image: np.ndarray, mask: np.ndarray | None = None
) -> np.ndarray:
    """Contrast-normalize an image using only measured pixels when masked."""

    values = image.ravel() if mask is None else image[mask > 0]
    if values.size < 32:
        return image.astype(np.uint8, copy=True)
    low, high = np.percentile(values, (2.0, 98.0))
    if high <= low + 1e-6:
        return image.astype(np.uint8, copy=True)
    scaled = np.clip(
        (image.astype(np.float32) - low) * 255.0 / (high - low), 0, 255
    )
    return scaled.astype(np.uint8)


def root_priority_map(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate an illumination-robust root likelihood without altering input data.

    Bright and dark top-hat responses both contribute, allowing pale or dark roots
    against spatially varying soil. The map is only a sampling priority; RoMa
    confidence and RANSAC still decide which correspondences are geometrically valid.
    """

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    size = max(15, int(round(min(gray.shape) * 0.035)))
    size = min(size + (size + 1) % 2, 81)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    white = cv2.morphologyEx(clahe, cv2.MORPH_TOPHAT, kernel)
    black = cv2.morphologyEx(clahe, cv2.MORPH_BLACKHAT, kernel)
    local_contrast = robust_normalize(np.maximum(white, black))

    gx = cv2.Sobel(clahe, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(clahe, cv2.CV_32F, 0, 1, ksize=3)
    edge = robust_normalize(cv2.magnitude(gx, gy))
    likelihood = cv2.GaussianBlur(0.75 * local_contrast + 0.25 * edge, (0, 0), 1.0)
    # A sparse mask is used only for visualization. The continuous likelihood
    # above is retained for correspondence weighting.
    threshold = float(np.quantile(likelihood, 0.90))
    mask = (likelihood >= threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return likelihood.astype(np.float32), mask


def long_line_protection_mask(gray: np.ndarray, length: int = 19) -> np.ndarray:
    """Find persistent bright/dark thin lines that should not be masked as water.

    Condensation circles and fine roots can intersect.  A bank of directional
    openings protects elongated structures before the circular nuisance mask is
    finalized.  This is intentionally a *protection* mask, not a root
    segmentation result.
    """

    smoothed = cv2.GaussianBlur(gray, (0, 0), 0.7).astype(np.float32)
    residual = smoothed - cv2.GaussianBlur(smoothed, (0, 0), 4.0)
    threshold = max(8.0, float(np.percentile(np.abs(residual), 88.0)))
    bright = (residual > threshold).astype(np.uint8)
    dark = (residual < -threshold).astype(np.uint8)
    protected = np.zeros(gray.shape, dtype=np.uint8)
    radius = (length - 1) / 2.0
    for angle_degrees in range(0, 180, 15):
        angle = np.deg2rad(angle_degrees)
        dx = radius * np.cos(angle)
        dy = radius * np.sin(angle)
        kernel = np.zeros((length, length), dtype=np.uint8)
        cv2.line(
            kernel,
            (round(radius - dx), round(radius - dy)),
            (round(radius + dx), round(radius + dy)),
            1,
            1,
        )
        protected |= cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)
        protected |= cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    return cv2.dilate(
        protected, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )


def conservative_root_line_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Return a sparse, high-precision visualization of elongated root-like lines.

    The output is descriptive only: it is never used as a registration objective,
    because optimizing root overlap would incorrectly remove true biological growth.
    """

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, 1200.0 / image_bgr.shape[1])
    normalized = cv2.resize(
        gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )
    root_lines = long_line_protection_mask(normalized, length=19) * 255
    # Suppress straight image-frame boundaries created by acquisition/cropping.
    border = max(2, round(min(root_lines.shape) * 0.004))
    root_lines[:border] = 0
    root_lines[-border:] = 0
    root_lines[:, :border] = 0
    root_lines[:, -border:] = 0
    return cv2.resize(
        root_lines,
        (gray.shape[1], gray.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )


def detect_droplet_mask(
    image_bgr: np.ndarray, hough_threshold: float = 23.0
) -> tuple[np.ndarray, int]:
    """Detect circular condensation droplets while retaining root/soil context.

    Detection is performed at a normalized width so the parameters remain stable
    across the two camera resolutions in this dataset. The mask is deliberately
    conservative: pixels inside detected circles are excluded from estimation and
    comparison, but the source image itself is never modified.
    """

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, 1200.0 / image_bgr.shape[1])
    resized = cv2.resize(
        gray,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )
    blurred = cv2.medianBlur(resized, 5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=4,
        param1=80,
        param2=hough_threshold,
        minRadius=2,
        maxRadius=18,
    )
    normalized_mask = np.zeros(resized.shape, dtype=np.uint8)
    if circles is None:
        return np.zeros(gray.shape, dtype=np.uint8), 0

    # A circular Hough response by itself is much too permissive on granular
    # soil.  Validate each candidate using radial gradient evidence around most
    # of its circumference.  This rejects soil pits and isolated root bends.
    smooth = cv2.GaussianBlur(resized, (0, 0), 0.8)
    gradient_x = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    angles = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
    cosine = np.cos(angles)
    sine = np.sin(angles)
    height, width = resized.shape
    validated: list[tuple[float, float, float]] = []
    for x_coord, y_coord, radius in circles[0]:
        if (
            radius < 4
            or x_coord - radius - 3 < 0
            or y_coord - radius - 3 < 0
            or x_coord + radius + 3 >= width
            or y_coord + radius + 3 >= height
        ):
            continue
        radial_responses: list[np.ndarray] = []
        alignments: list[np.ndarray] = []
        for radial_offset in (-1, 0, 1):
            sample_x = np.rint(
                x_coord + (radius + radial_offset) * cosine
            ).astype(np.int32)
            sample_y = np.rint(
                y_coord + (radius + radial_offset) * sine
            ).astype(np.int32)
            gx = gradient_x[sample_y, sample_x]
            gy = gradient_y[sample_y, sample_x]
            magnitude = np.hypot(gx, gy)
            radial = np.abs(gx * cosine + gy * sine)
            radial_responses.append(radial)
            alignments.append(radial / np.maximum(magnitude, 1.0))
        radial_stack = np.stack(radial_responses)
        aligned_stack = np.stack(alignments)
        response = np.max(
            np.where(aligned_stack > 0.75, radial_stack, 0.0), axis=0
        )
        sector_support = float((response.reshape(24, 3).max(axis=1) >= 35).mean())
        angular_support = float((response >= 35).mean())
        if (
            sector_support >= 0.875
            and angular_support >= 0.70
            and float(np.median(response)) >= 60.0
        ):
            validated.append((x_coord, y_coord, radius))

    # Mask only the droplet core; then restore any long root-like structure that
    # crosses it.  Missing a weak droplet is preferable to deleting a real root.
    for x_coord, y_coord, radius in validated:
        cv2.circle(
            normalized_mask,
            (round(x_coord), round(y_coord)),
            max(2, round(radius * 0.85)),
            255,
            thickness=-1,
        )
    protected_lines = long_line_protection_mask(resized)
    normalized_mask[protected_lines > 0] = 0
    mask = cv2.resize(
        normalized_mask,
        (gray.shape[1], gray.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return mask, len(validated)


def normalized_to_pixels(coords: np.ndarray, height: int, width: int) -> np.ndarray:
    pixels = np.empty_like(coords, dtype=np.float32)
    pixels[:, 0] = width * 0.5 * (coords[:, 0] + 1.0)
    pixels[:, 1] = height * 0.5 * (coords[:, 1] + 1.0)
    return pixels


def sample_root_likelihood(root_map: np.ndarray, points: np.ndarray) -> np.ndarray:
    x = np.clip(np.rint(points[:, 0]).astype(np.int32), 0, root_map.shape[1] - 1)
    y = np.clip(np.rint(points[:, 1]).astype(np.int32), 0, root_map.shape[0] - 1)
    return root_map[y, x]


def spatially_balanced_indices(
    fixed_points: np.ndarray,
    score: np.ndarray,
    image_shape: tuple[int, int],
    maximum: int,
    grid_size: int = 10,
) -> np.ndarray:
    """Take the strongest matches per grid cell so soil texture cannot dominate."""

    height, width = image_shape
    cell_x = np.clip((fixed_points[:, 0] / max(width, 1) * grid_size).astype(int), 0, grid_size - 1)
    cell_y = np.clip((fixed_points[:, 1] / max(height, 1) * grid_size).astype(int), 0, grid_size - 1)
    cell_ids = cell_y * grid_size + cell_x
    per_cell = max(1, int(np.ceil(maximum / (grid_size * grid_size))))
    selected: list[np.ndarray] = []
    for cell_id in range(grid_size * grid_size):
        members = np.flatnonzero(cell_ids == cell_id)
        if members.size == 0:
            continue
        take = min(per_cell, members.size)
        local = np.argpartition(score[members], -take)[-take:]
        selected.append(members[local])
    if not selected:
        return np.empty(0, dtype=np.int64)
    indices = np.concatenate(selected)
    indices = indices[np.argsort(score[indices])[::-1]]
    return indices[:maximum]


def select_matches(
    warp: np.ndarray,
    certainty: np.ndarray,
    fixed_shape: tuple[int, int],
    moving_shape: tuple[int, int],
    root_map: np.ndarray,
    fixed_droplet_mask: np.ndarray,
    moving_droplet_mask: np.ndarray,
    min_confidence: float,
    max_matches: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    flat_warp = warp.reshape(-1, 4)
    flat_certainty = certainty.reshape(-1)
    finite = np.isfinite(flat_warp).all(axis=1) & np.isfinite(flat_certainty)
    in_bounds = (np.abs(flat_warp) <= 1.0).all(axis=1)
    valid = finite & in_bounds & (flat_certainty >= min_confidence)
    raw_candidate_count = int(valid.sum())
    if raw_candidate_count < 12:
        raise ValueError(
            f"only {raw_candidate_count} matches exceed confidence {min_confidence:.3f}"
        )

    matches = flat_warp[valid]
    confidence = flat_certainty[valid].astype(np.float32)
    fixed = normalized_to_pixels(matches[:, :2], *fixed_shape)
    moving = normalized_to_pixels(matches[:, 2:], *moving_shape)
    water = (sample_root_likelihood(fixed_droplet_mask, fixed) > 0) | (
        sample_root_likelihood(moving_droplet_mask, moving) > 0
    )
    fixed = fixed[~water]
    moving = moving[~water]
    confidence = confidence[~water]
    candidate_count = int((~water).sum())
    if candidate_count < 12:
        raise ValueError(
            f"only {candidate_count} non-droplet matches remain after excluding "
            f"{raw_candidate_count - candidate_count} water-droplet matches"
        )
    root_score = sample_root_likelihood(root_map, fixed)
    # Retain non-root context for global stability while strongly prioritizing roots.
    score = confidence * (0.25 + 0.75 * root_score)
    selected = spatially_balanced_indices(fixed, score, fixed_shape, max_matches)
    if selected.size < 12:
        raise ValueError(f"spatial sampling retained only {selected.size} matches")
    return (
        fixed[selected],
        moving[selected],
        confidence[selected],
        root_score[selected],
        candidate_count,
        raw_candidate_count,
    )


def estimate_transform(
    moving: np.ndarray,
    fixed: np.ndarray,
    transform_type: str,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    if transform_type == "homography":
        matrix, inliers = cv2.findHomography(
            moving, fixed, cv2.RANSAC, threshold, maxIters=10000, confidence=0.999
        )
    elif transform_type == "similarity":
        affine, inliers = cv2.estimateAffinePartial2D(
            moving,
            fixed,
            method=cv2.RANSAC,
            ransacReprojThreshold=threshold,
            maxIters=10000,
            confidence=0.999,
            refineIters=20,
        )
        matrix = None if affine is None else np.vstack([affine, [0.0, 0.0, 1.0]])
    else:
        affine, inliers = cv2.estimateAffine2D(
            moving,
            fixed,
            method=cv2.RANSAC,
            ransacReprojThreshold=threshold,
            maxIters=10000,
            confidence=0.999,
            refineIters=20,
        )
        matrix = None if affine is None else np.vstack([affine, [0.0, 0.0, 1.0]])
    if matrix is None or inliers is None:
        raise ValueError(f"{transform_type} RANSAC could not estimate a transform")
    matrix = matrix.astype(np.float64)
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError(f"estimated {transform_type} matrix is singular or non-finite")
    return matrix, inliers.ravel().astype(bool)


def project_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(points[:, None, :].astype(np.float32), matrix)[:, 0]


def measure_frame_overlap(
    matrix: np.ndarray,
    fixed_shape: tuple[int, int],
    moving_shape: tuple[int, int],
) -> FrameOverlap:
    """Measure the intersection relative to both T1 and the projected frame."""

    fixed_h, fixed_w = fixed_shape
    moving_h, moving_w = moving_shape
    moving_corners = np.array(
        [[0, 0], [moving_w, 0], [moving_w, moving_h], [0, moving_h]], np.float32
    )
    transformed = project_points(moving_corners, matrix).astype(np.float32)
    area = abs(float(cv2.contourArea(transformed)))
    if area <= 1e-6 or not np.isfinite(transformed).all():
        return FrameOverlap(0.0, 0.0, 0.0, 0.0)
    fixed_corners = np.array(
        [[0, 0], [fixed_w, 0], [fixed_w, fixed_h], [0, fixed_h]], np.float32
    )
    try:
        intersection_area, _ = cv2.intersectConvexConvex(transformed, fixed_corners)
    except cv2.error:
        return FrameOverlap(0.0, 0.0, 0.0, area)
    fixed_area = float(fixed_h * fixed_w)
    return FrameOverlap(
        intersection_area_px2=float(max(intersection_area, 0.0)),
        t1_coverage_fraction=float(
            np.clip(intersection_area / max(fixed_area, 1.0), 0.0, 1.0)
        ),
        moving_visible_fraction=float(
            np.clip(intersection_area / max(area, 1.0), 0.0, 1.0)
        ),
        transformed_moving_area_px2=area,
    )


def grid_support_fraction(
    points: np.ndarray, image_shape: tuple[int, int], grid_size: int = 10
) -> float:
    """Fraction of grid cells containing at least one geometrically valid match."""

    if points.size == 0:
        return 0.0
    height, width = image_shape
    cell_x = np.clip(
        (points[:, 0] / max(width, 1) * grid_size).astype(int), 0, grid_size - 1
    )
    cell_y = np.clip(
        (points[:, 1] / max(height, 1) * grid_size).astype(int), 0, grid_size - 1
    )
    return float(np.unique(cell_y * grid_size + cell_x).size / (grid_size * grid_size))


def normalized_cross_correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float | None:
    valid = mask.astype(bool)
    if valid.sum() < 32:
        return None
    av = a[valid].astype(np.float64)
    bv = b[valid].astype(np.float64)
    av -= av.mean()
    bv -= bv.mean()
    denominator = np.linalg.norm(av) * np.linalg.norm(bv)
    if denominator <= 1e-12:
        return None
    return float(np.dot(av, bv) / denominator)


def trimmed_normalized_cross_correlation(
    a: np.ndarray, b: np.ndarray, mask: np.ndarray, trim_fraction: float = 0.10
) -> float | None:
    """ZNCC after discarding the largest temporal differences.

    Low-pass Lab soil channels are compared this way so illumination differences,
    residual droplets, and genuine new/disappearing roots do not dominate the
    geometric quality metric.
    """

    valid = mask.astype(bool)
    if valid.sum() < 100:
        return None
    av = a[valid].astype(np.float64)
    bv = b[valid].astype(np.float64)
    difference = np.abs(av - bv)
    cutoff = np.quantile(difference, 1.0 - trim_fraction)
    keep = difference <= cutoff
    if keep.sum() < 100:
        return None
    av = av[keep]
    bv = bv[keep]
    av -= av.mean()
    bv -= bv.mean()
    denominator = np.linalg.norm(av) * np.linalg.norm(bv)
    if denominator <= 1e-12:
        return None
    return float(np.dot(av, bv) / denominator)


def compute_metrics(
    filename: str,
    period: str,
    transform_type: str,
    transform_source: str,
    registration_status: str,
    overlap: FrameOverlap,
    overlap_class: str,
    dense_confident_fraction: float,
    fixed_support_fraction: float,
    moving_support_fraction: float,
    droplet_exclusion_enabled: bool,
    fixed_droplet_mask: np.ndarray,
    moving_droplet_mask: np.ndarray,
    moving_droplet_registered: np.ndarray,
    fixed: np.ndarray,
    moving: np.ndarray,
    confidence: np.ndarray,
    root_score: np.ndarray,
    inliers: np.ndarray,
    matrix: np.ndarray,
    candidate_count: int,
    raw_candidate_count: int,
    fixed_image: np.ndarray,
    moving_image: np.ndarray,
    registered: np.ndarray,
    valid_mask: np.ndarray,
) -> QualityMetrics:
    projected = project_points(moving, matrix)
    errors = np.linalg.norm(projected - fixed, axis=1)
    inlier_errors = errors[inliers]
    root_inliers = inliers & (root_score >= 0.5)
    root_errors = errors[root_inliers]

    fixed_gray = cv2.cvtColor(fixed_image, cv2.COLOR_BGR2GRAY)
    registered_gray = cv2.cvtColor(registered, cv2.COLOR_BGR2GRAY)
    before = cv2.resize(
        cv2.cvtColor(moving_image, cv2.COLOR_BGR2GRAY),
        (fixed_gray.shape[1], fixed_gray.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    before_droplet = cv2.resize(
        moving_droplet_mask,
        (fixed_gray.shape[1], fixed_gray.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    before_mask = (
        (fixed_droplet_mask == 0) & (before_droplet == 0)
    ).astype(np.uint8)
    analysis_mask = (
        (valid_mask > 0)
        & (fixed_droplet_mask == 0)
        & (moving_droplet_registered == 0)
    ).astype(np.uint8)
    fixed_lab = cv2.cvtColor(fixed_image, cv2.COLOR_BGR2LAB).astype(np.float32)
    registered_lab = cv2.cvtColor(registered, cv2.COLOR_BGR2LAB).astype(np.float32)
    fixed_soil = cv2.GaussianBlur(fixed_lab, (0, 0), 5.0)
    registered_soil = cv2.GaussianBlur(registered_lab, (0, 0), 5.0)
    lab_ncc = [
        trimmed_normalized_cross_correlation(
            fixed_soil[..., channel],
            registered_soil[..., channel],
            analysis_mask,
        )
        for channel in range(3)
    ]
    return QualityMetrics(
        filename=filename,
        moving_period=period,
        transform_type=transform_type,
        transform_source=transform_source,
        registration_status=registration_status,
        t1_coverage_fraction=overlap.t1_coverage_fraction,
        moving_visible_fraction=overlap.moving_visible_fraction,
        overlap_class=overlap_class,
        dense_confident_fraction=dense_confident_fraction,
        fixed_support_grid_fraction=fixed_support_fraction,
        moving_support_grid_fraction=moving_support_fraction,
        droplet_exclusion_enabled=droplet_exclusion_enabled,
        t1_droplet_mask_fraction=float((fixed_droplet_mask > 0).mean()),
        moving_droplet_mask_fraction=float((moving_droplet_mask > 0).mean()),
        root_soil_analysis_fraction=float(analysis_mask.mean()),
        raw_candidate_matches=raw_candidate_count,
        candidate_matches=candidate_count,
        water_excluded_candidate_fraction=float(
            1.0 - candidate_count / max(raw_candidate_count, 1)
        ),
        selected_matches=int(len(fixed)),
        inlier_matches=int(inliers.sum()),
        inlier_ratio=float(inliers.mean()),
        mean_match_confidence=float(confidence.mean()),
        root_priority_fraction=float((root_score >= 0.5).mean()),
        reprojection_rmse_px=(
            float(np.sqrt(np.mean(inlier_errors**2))) if inlier_errors.size else None
        ),
        reprojection_median_px=(
            float(np.median(inlier_errors)) if inlier_errors.size else None
        ),
        reprojection_p95_px=(
            float(np.percentile(inlier_errors, 95)) if inlier_errors.size else None
        ),
        root_reprojection_rmse_px=(
            float(np.sqrt(np.mean(root_errors**2))) if root_errors.size else None
        ),
        valid_overlap_fraction=float((valid_mask > 0).mean()),
        root_soil_ncc_before=normalized_cross_correlation(
            fixed_gray, before, before_mask
        ),
        root_soil_ncc_after=normalized_cross_correlation(
            fixed_gray, registered_gray, analysis_mask
        ),
        soil_lab_l_trimmed_ncc_after=lab_ncc[0],
        soil_lab_a_trimmed_ncc_after=lab_ncc[1],
        soil_lab_b_trimmed_ncc_after=lab_ncc[2],
    )


def resize_for_visualization(image: np.ndarray, max_side: int = 1200) -> tuple[np.ndarray, float]:
    scale = min(1.0, max_side / max(image.shape[:2]))
    if scale == 1.0:
        return image.copy(), scale
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return resized, scale


def draw_matches(
    fixed_image: np.ndarray,
    moving_image: np.ndarray,
    fixed: np.ndarray,
    moving: np.ndarray,
    inliers: np.ndarray,
    confidence: np.ndarray,
    maximum: int = 250,
) -> np.ndarray:
    left, scale_left = resize_for_visualization(fixed_image)
    right, scale_right = resize_for_visualization(moving_image)
    target_h = max(left.shape[0], right.shape[0])
    canvas = np.zeros((target_h, left.shape[1] + right.shape[1], 3), dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] :] = right
    ranked = np.argsort(confidence)[::-1][:maximum]
    for index in ranked:
        point_a = tuple(np.rint(fixed[index] * scale_left).astype(int))
        point_b_array = moving[index] * scale_right + np.array([left.shape[1], 0])
        point_b = tuple(np.rint(point_b_array).astype(int))
        color = (60, 220, 60) if inliers[index] else (60, 60, 230)
        cv2.line(canvas, point_a, point_b, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, point_a, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, point_b, 2, color, -1, cv2.LINE_AA)
    cv2.putText(canvas, "T1 fixed", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(
        canvas,
        "moving",
        (left.shape[1] + 15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    return canvas


def make_checkerboard(first: np.ndarray, second: np.ndarray, tile: int = 160) -> np.ndarray:
    rows, columns = np.indices(first.shape[:2])
    use_second = ((rows // tile) + (columns // tile)) % 2 == 1
    output = first.copy()
    output[use_second] = second[use_second]
    return output


def make_masked_checkerboard(
    fixed: np.ndarray,
    registered: np.ndarray,
    valid_mask: np.ndarray,
    tile: int = 160,
) -> np.ndarray:
    """Checkerboard only where the moving period was actually photographed."""

    output = make_checkerboard(fixed, registered, tile)
    output[valid_mask == 0] = (fixed[valid_mask == 0] * 0.35).astype(np.uint8)
    return output


def make_color_overlap(
    fixed: np.ndarray,
    registered: np.ndarray,
    valid_mask: np.ndarray,
    droplet_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Magenta=T1, green=moving; neutral gray means excluded water/NoData."""

    fixed_gray = cv2.cvtColor(fixed, cv2.COLOR_BGR2GRAY)
    moving_gray = cv2.cvtColor(registered, cv2.COLOR_BGR2GRAY)
    valid = valid_mask > 0
    fixed_gray = robust_normalize_uint8(fixed_gray, valid_mask)
    moving_gray = robust_normalize_uint8(moving_gray, valid_mask)
    context = cv2.cvtColor((fixed_gray * 0.32).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    color = np.stack((fixed_gray, moving_gray, fixed_gray), axis=-1)
    output = context
    output[valid] = color[valid]
    if droplet_mask is not None:
        water = (droplet_mask > 0) & valid
        output[water] = (58, 58, 58)
    cv2.putText(
        output,
        "root/soil only: T1 magenta | period green | gray = water/NoData",
        (24, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def make_context_view(
    fixed: np.ndarray, registered: np.ndarray, valid_mask: np.ndarray
) -> np.ndarray:
    """Show registered pixels in their position on a dimmed full T1 canvas."""

    context = (fixed.astype(np.float32) * 0.28).astype(np.uint8)
    context[valid_mask > 0] = registered[valid_mask > 0]
    contours, _ = cv2.findContours(
        (valid_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(context, contours, -1, (0, 215, 255), 5, cv2.LINE_AA)
    cv2.putText(
        context,
        "registered period on T1 canvas (dim area = not observed in this period)",
        (24, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return context


def make_root_change_overlay(
    fixed: np.ndarray,
    fixed_root_mask: np.ndarray,
    moving_root_mask_registered: np.ndarray,
    valid_mask: np.ndarray,
    droplet_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Conservative elongated-root view without changing either source image."""

    gray = cv2.cvtColor(fixed, cv2.COLOR_BGR2GRAY)
    output = cv2.cvtColor((gray * 0.25).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    fixed_root = (fixed_root_mask > 0) & (valid_mask > 0)
    moving_root = (moving_root_mask_registered > 0) & (valid_mask > 0)
    distance_to_fixed = cv2.distanceTransform(
        (~fixed_root).astype(np.uint8), cv2.DIST_L2, 3
    )
    distance_to_moving = cv2.distanceTransform(
        (~moving_root).astype(np.uint8), cv2.DIST_L2, 3
    )
    shared = (fixed_root & (distance_to_moving <= 6.0)) | (
        moving_root & (distance_to_fixed <= 6.0)
    )
    output[fixed_root] = (220, 35, 220)  # magenta in BGR
    output[moving_root] = (35, 220, 35)  # green
    output[shared] = (225, 225, 225)
    if droplet_mask is not None:
        output[droplet_mask > 0] = (48, 48, 48)
    cv2.putText(
        output,
        "heuristic root lines: T1 magenta | period green | shared<=6px white",
        (24, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def make_placement_diagram(
    fixed_shape: tuple[int, int],
    moving_shape: tuple[int, int],
    matrix: np.ndarray,
    period: str,
) -> np.ndarray:
    """Draw both measured fields of view in a common coordinate diagram."""

    fixed_h, fixed_w = fixed_shape
    moving_h, moving_w = moving_shape
    fixed_corners = np.array(
        [[0, 0], [fixed_w, 0], [fixed_w, fixed_h], [0, fixed_h]], np.float32
    )
    moving_corners = project_points(
        np.array(
            [[0, 0], [moving_w, 0], [moving_w, moving_h], [0, moving_h]],
            np.float32,
        ),
        matrix,
    )
    all_corners = np.vstack((fixed_corners, moving_corners))
    minimum = all_corners.min(axis=0)
    maximum = all_corners.max(axis=0)
    span = np.maximum(maximum - minimum, 1.0)
    canvas = np.full((1000, 1500, 3), 248, np.uint8)
    scale = min(1320 / span[0], 780 / span[1])
    offset = np.array([90.0, 130.0]) - minimum * scale

    def convert(points: np.ndarray) -> np.ndarray:
        return np.rint(points * scale + offset).astype(np.int32)

    cv2.polylines(canvas, [convert(fixed_corners)], True, (190, 80, 25), 7)
    cv2.polylines(canvas, [convert(moving_corners)], True, (20, 125, 225), 7)
    cv2.putText(
        canvas,
        f"Measured field-of-view placement: T1 (blue) and {period} (orange)",
        (45, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.15,
        (25, 25, 25),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Only the geometric intersection has observations in both periods.",
        (45, 970),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (40, 40, 40),
        2,
        cv2.LINE_AA,
    )
    return canvas


def colorize_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Make a browser-friendly RGB diagnostic (green=valid, red=invalid)."""

    visualization = np.empty((*mask.shape, 3), dtype=np.uint8)
    visualization[mask > 0] = (70, 190, 70)  # BGR green
    visualization[mask == 0] = (70, 70, 220)  # BGR red
    return visualization


def colorize_feature_mask(mask: np.ndarray) -> np.ndarray:
    """Browser-friendly sparse feature mask (green=detected, black=background)."""

    visualization = np.zeros((*mask.shape, 3), dtype=np.uint8)
    visualization[mask > 0] = (45, 220, 45)
    return visualization


def make_exclusion_overlay(image: np.ndarray, excluded_mask: np.ndarray) -> np.ndarray:
    """Overlay excluded water pixels in red for auditability."""

    output = image.copy()
    excluded = excluded_mask > 0
    red = np.zeros_like(output)
    red[..., 2] = 255
    output[excluded] = (
        0.35 * output[excluded].astype(np.float32)
        + 0.65 * red[excluded].astype(np.float32)
    ).astype(np.uint8)
    return output


def make_full_canvas_outputs(
    fixed: np.ndarray,
    moving: np.ndarray,
    moving_to_fixed: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[int, int],
]:
    """Warp both images to their union bounds so neither image is cropped."""

    fixed_h, fixed_w = fixed.shape[:2]
    moving_h, moving_w = moving.shape[:2]
    fixed_corners = np.array(
        [[0, 0], [fixed_w, 0], [fixed_w, fixed_h], [0, fixed_h]], np.float32
    )
    moving_corners = np.array(
        [[0, 0], [moving_w, 0], [moving_w, moving_h], [0, moving_h]], np.float32
    )
    transformed_moving = project_points(moving_corners, moving_to_fixed)
    all_corners = np.vstack((fixed_corners, transformed_moving))
    minimum = np.floor(all_corners.min(axis=0)).astype(np.int64)
    maximum = np.ceil(all_corners.max(axis=0)).astype(np.int64)
    canvas_w, canvas_h = (maximum - minimum).tolist()
    if canvas_w <= 0 or canvas_h <= 0 or canvas_w > 20000 or canvas_h > 20000:
        raise ValueError(f"unsafe expanded canvas size: {canvas_w}x{canvas_h}")
    if canvas_w * canvas_h > 120_000_000:
        raise ValueError(f"expanded canvas is too large: {canvas_w}x{canvas_h}")

    translation = np.array(
        [[1.0, 0.0, -minimum[0]], [0.0, 1.0, -minimum[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    moving_to_canvas = translation @ moving_to_fixed
    output_size = (int(canvas_w), int(canvas_h))
    registered_full = cv2.warpPerspective(moving, moving_to_canvas, output_size)
    reference_full = cv2.warpPerspective(fixed, translation, output_size)
    moving_mask = cv2.warpPerspective(
        np.full((moving_h, moving_w), 255, np.uint8),
        moving_to_canvas,
        output_size,
        flags=cv2.INTER_NEAREST,
    )
    fixed_mask = cv2.warpPerspective(
        np.full((fixed_h, fixed_w), 255, np.uint8),
        translation,
        output_size,
        flags=cv2.INTER_NEAREST,
    )

    overlay_full = np.zeros_like(reference_full)
    fixed_only = (fixed_mask > 0) & (moving_mask == 0)
    moving_only = (moving_mask > 0) & (fixed_mask == 0)
    overlap = (fixed_mask > 0) & (moving_mask > 0)
    overlay_full[fixed_only] = reference_full[fixed_only]
    overlay_full[moving_only] = registered_full[moving_only]
    blended = cv2.addWeighted(reference_full, 0.5, registered_full, 0.5, 0.0)
    overlay_full[overlap] = blended[overlap]
    return (
        registered_full,
        reference_full,
        overlay_full,
        moving_mask,
        fixed_mask,
        moving_to_canvas,
        (int(minimum[0]), int(minimum[1])),
    )


def write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=True)
        handle.write("\n")


def save_pair_outputs(
    output_dir: Path,
    stem: str,
    period: str,
    fixed_image: np.ndarray,
    moving_image: np.ndarray,
    registered: np.ndarray,
    valid_mask: np.ndarray,
    fixed_root_mask: np.ndarray,
    moving_root_mask: np.ndarray,
    fixed_droplet_mask: np.ndarray,
    moving_droplet_mask: np.ndarray,
    moving_droplet_registered: np.ndarray,
    fixed_points: np.ndarray,
    moving_points: np.ndarray,
    confidence: np.ndarray,
    root_score: np.ndarray,
    inliers: np.ndarray,
    matrix: np.ndarray,
    candidate_transforms: dict[str, np.ndarray],
    transform_source: str,
    registration_status: str,
    overlap: FrameOverlap,
    dense_warp: np.ndarray,
    dense_certainty: np.ndarray,
    metrics: QualityMetrics,
) -> None:
    pair_dir = output_dir / stem / period
    pair_dir.mkdir(parents=True, exist_ok=True)
    (
        registered_full,
        reference_full,
        overlay_full,
        moving_mask_full,
        fixed_mask_full,
        moving_to_canvas,
        canvas_origin,
    ) = make_full_canvas_outputs(fixed_image, moving_image, matrix)
    full_size = (registered_full.shape[1], registered_full.shape[0])
    fixed_to_canvas = np.array(
        [
            [1.0, 0.0, -canvas_origin[0]],
            [0.0, 1.0, -canvas_origin[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    fixed_droplet_full = cv2.warpPerspective(
        fixed_droplet_mask,
        fixed_to_canvas,
        full_size,
        flags=cv2.INTER_NEAREST,
    )
    moving_droplet_full = cv2.warpPerspective(
        moving_droplet_mask,
        moving_to_canvas,
        full_size,
        flags=cv2.INTER_NEAREST,
    )
    full_droplets = cv2.bitwise_or(fixed_droplet_full, moving_droplet_full)
    root_soil_full_overlay = overlay_full.copy()
    root_soil_full_overlay[full_droplets > 0] = (58, 58, 58)
    moving_root_registered = cv2.warpPerspective(
        moving_root_mask,
        matrix,
        (fixed_image.shape[1], fixed_image.shape[0]),
        flags=cv2.INTER_NEAREST,
    )
    combined_droplets = cv2.bitwise_or(
        fixed_droplet_mask, moving_droplet_registered
    )
    analysis_mask = valid_mask.copy()
    analysis_mask[combined_droplets > 0] = 0
    rgba = np.dstack((registered, valid_mask))
    rgba_full = np.dstack((registered_full, moving_mask_full))
    blend = (fixed_image.astype(np.float32) * 0.28).astype(np.uint8)
    overlap_blend = cv2.addWeighted(fixed_image, 0.5, registered, 0.5, 0.0)
    blend[analysis_mask > 0] = overlap_blend[analysis_mask > 0]
    blend[combined_droplets > 0] = (58, 58, 58)
    difference = cv2.absdiff(fixed_image, registered)
    difference[analysis_mask == 0] = 0
    difference_gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    difference_color = cv2.applyColorMap(difference_gray, cv2.COLORMAP_TURBO)
    difference_color[analysis_mask == 0] = 0
    root_soil_overlay = make_color_overlap(
        fixed_image, registered, valid_mask, combined_droplets
    )
    checkerboard = make_masked_checkerboard(
        fixed_image, registered, analysis_mask
    )
    checkerboard[combined_droplets > 0] = (58, 58, 58)
    outputs = {
        "registered_to_T1.png": registered,
        "registered_to_T1_rgba.png": rgba,
        "registered_on_T1_context.png": make_context_view(
            fixed_image, registered, valid_mask
        ),
        "overlay_blend.png": blend,
        "overlay_color.png": root_soil_overlay,
        "root_soil_overlay.png": root_soil_overlay,
        "checkerboard.png": checkerboard,
        "root_change_overlay.png": make_root_change_overlay(
            fixed_image,
            fixed_root_mask,
            moving_root_registered,
            analysis_mask,
            combined_droplets,
        ),
        "absolute_difference.png": difference_color,
        "coverage_mask.png": colorize_binary_mask(valid_mask),
        "root_soil_analysis_mask.png": colorize_binary_mask(analysis_mask),
        "root_soil_analysis_mask_gray.png": analysis_mask,
        "droplet_mask_T1_gray.png": fixed_droplet_mask,
        "droplet_mask_registered_gray.png": moving_droplet_registered,
        "droplet_mask_combined_gray.png": combined_droplets,
        "droplet_nuisance_mask.png": colorize_feature_mask(combined_droplets),
        "droplet_exclusion_overlay.png": make_exclusion_overlay(
            fixed_image, combined_droplets
        ),
        "root_line_candidates_T1.png": colorize_feature_mask(fixed_root_mask),
        "root_line_candidates_registered.png": colorize_feature_mask(
            moving_root_registered
        ),
        # Backward-compatible filenames; these now contain the same conservative
        # candidate-line view rather than the previous broad edge mask.
        "root_priority_T1.png": colorize_feature_mask(fixed_root_mask),
        "root_priority_registered.png": colorize_feature_mask(
            moving_root_registered
        ),
        "placement_diagram.png": make_placement_diagram(
            fixed_image.shape[:2], moving_image.shape[:2], matrix, period
        ),
        "matches.png": draw_matches(
            fixed_image, moving_image, fixed_points, moving_points, inliers, confidence
        ),
        "full_extent_registered.png": registered_full,
        "full_extent_registered_rgba.png": rgba_full,
        "full_extent_T1.png": reference_full,
        "full_extent_overlay.png": overlay_full,
        "full_extent_root_soil_overlay.png": root_soil_full_overlay,
        "full_extent_droplet_mask.png": colorize_feature_mask(full_droplets),
        "full_extent_moving_mask.png": colorize_binary_mask(moving_mask_full),
        "full_extent_T1_mask.png": colorize_binary_mask(fixed_mask_full),
    }
    for name, image in outputs.items():
        if not cv2.imwrite(str(pair_dir / name), image):
            raise OSError(f"failed to write {pair_dir / name}")

    inverse = np.linalg.inv(matrix)
    write_json(
        pair_dir / "transform.json",
        {
            "moving_to_T1": matrix.tolist(),
            "T1_to_moving": inverse.tolist(),
            "selected_transform_type": metrics.transform_type,
            "transform_source": transform_source,
            "registration_status": registration_status,
            "candidate_transforms_moving_to_T1": {
                name: value.tolist() for name, value in candidate_transforms.items()
            },
            "overlap": asdict(overlap),
            "T1_canvas_output_size": [fixed_image.shape[1], fixed_image.shape[0]],
            "unobserved_pixel_policy": (
                "transparent/NoData; registered_on_T1_context.png uses dim T1 only "
                "for visual context and never claims it is moving-period data"
            ),
            "comparison_policy": (
                "circular condensation droplets are excluded from transform fitting, "
                "overlays, difference images, and root/soil quality metrics"
            ),
            "moving_to_full_canvas": moving_to_canvas.tolist(),
            "full_canvas_origin_in_T1": list(canvas_origin),
            "full_canvas_size": [registered_full.shape[1], registered_full.shape[0]],
            "coordinate_convention": "pixel coordinates, homogeneous 3x3 matrices",
        },
    )
    np.savez_compressed(
        pair_dir / "roma_warp.npz",
        T1_to_moving_normalized=dense_warp[..., 2:].astype(np.float32),
        T1_normalized_grid=dense_warp[..., :2].astype(np.float32),
        certainty=dense_certainty.astype(np.float32),
        fixed_points_px=fixed_points.astype(np.float32),
        moving_points_px=moving_points.astype(np.float32),
        selected_confidence=confidence.astype(np.float32),
        root_priority=root_score.astype(np.float32),
        ransac_inliers=inliers.astype(np.uint8),
        moving_to_T1=matrix.astype(np.float64),
        T1_valid_mask=valid_mask.astype(np.uint8),
        T1_root_soil_analysis_mask=analysis_mask.astype(np.uint8),
        T1_droplet_mask=fixed_droplet_mask.astype(np.uint8),
        moving_droplet_mask=moving_droplet_mask.astype(np.uint8),
        registered_moving_droplet_mask=moving_droplet_registered.astype(np.uint8),
    )
    write_json(pair_dir / "metrics.json", asdict(metrics))


def register_pair(
    model: torch.nn.Module | None,
    device: torch.device,
    fixed_path: Path,
    moving_path: Path,
    period: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> QualityMetrics:
    pair = f"{fixed_path.name} T1<-{period}"
    try:
        fixed_image = read_color(fixed_path, pair)
        moving_image = read_color(moving_path, pair)
        root_map, _ = root_priority_map(fixed_image)
        # These sparse line masks are used only for the descriptive change view.
        # RoMa + RANSAC still determine geometry from continuous confidence and
        # root/soil context, so real root growth is not optimized away.
        fixed_root_mask = conservative_root_line_mask(fixed_image)
        moving_root_mask = conservative_root_line_mask(moving_image)
        if args.ignore_droplets:
            fixed_droplet_mask, fixed_droplet_count = detect_droplet_mask(
                fixed_image, args.droplet_hough_threshold
            )
            moving_droplet_mask, moving_droplet_count = detect_droplet_mask(
                moving_image, args.droplet_hough_threshold
            )
            print(
                f"[droplets] {pair}: T1={fixed_droplet_count}, "
                f"{period}={moving_droplet_count}",
                flush=True,
            )
        else:
            fixed_droplet_mask = np.zeros(fixed_image.shape[:2], dtype=np.uint8)
            moving_droplet_mask = np.zeros(moving_image.shape[:2], dtype=np.uint8)

        cache_path: Path | None = None
        if args.reuse_warp_dir is not None:
            cache_candidates = (
                args.reuse_warp_dir
                / fixed_path.name
                / period
                / "roma_warp.npz",
                args.reuse_warp_dir
                / fixed_path.stem
                / period
                / "roma_warp.npz",
            )
            cache_path = next((path for path in cache_candidates if path.is_file()), None)
            if cache_path is None:
                print(
                    f"[resume miss] {pair}: no cached warp; running RoMa",
                    flush=True,
                )

        if cache_path is not None:
            try:
                with np.load(cache_path) as cached:
                    warp = np.concatenate(
                        (
                            cached["T1_normalized_grid"],
                            cached["T1_to_moving_normalized"],
                        ),
                        axis=-1,
                    ).astype(np.float32)
                    certainty = cached["certainty"].astype(np.float32)
            except Exception as exc:
                raise RegistrationError("RoMa cache loading", pair, str(exc)) from exc
            print(f"[resume] {pair}: {cache_path}", flush=True)
        else:
            if model is None:
                raise RegistrationError(
                    "RoMa matching", pair, "model was not initialized"
                )
            try:
                warp_tensor, certainty_tensor = model.match(
                    str(fixed_path), str(moving_path), device=device
                )
            except Exception as exc:
                raise RegistrationError("RoMa matching", pair, str(exc)) from exc

            warp = warp_tensor[0].detach().float().cpu().numpy()
            certainty = certainty_tensor[0].detach().float().cpu().numpy()
            del warp_tensor, certainty_tensor
            if device.type == "cuda":
                torch.cuda.empty_cache()

        try:
            (
                fixed_points,
                moving_points,
                confidence,
                root_score,
                candidates,
                raw_candidates,
            ) = select_matches(
                warp,
                certainty,
                fixed_image.shape[:2],
                moving_image.shape[:2],
                root_map,
                fixed_droplet_mask,
                moving_droplet_mask,
                args.min_confidence,
                args.max_matches,
            )
        except Exception as exc:
            raise RegistrationError("match filtering", pair, str(exc)) from exc

        candidate_transforms: dict[str, np.ndarray] = {}
        candidate_inliers: dict[str, np.ndarray] = {}
        model_types = (
            (args.transform,)
            if args.transform != "auto"
            else ("similarity", "affine", "homography")
        )
        estimation_errors: list[str] = []
        for model_type in model_types:
            try:
                candidate_matrix, model_inliers = estimate_transform(
                    moving_points,
                    fixed_points,
                    model_type,
                    args.ransac_threshold,
                )
                minimum_inliers = 8 if model_type == "homography" else 6
                if int(model_inliers.sum()) < minimum_inliers:
                    raise ValueError(
                        f"only {int(model_inliers.sum())} inliers; need {minimum_inliers}"
                    )
                candidate_transforms[model_type] = candidate_matrix
                candidate_inliers[model_type] = model_inliers
            except Exception as exc:
                estimation_errors.append(f"{model_type}: {exc}")

        if not candidate_transforms:
            raise RegistrationError(
                "transform estimation", pair, "; ".join(estimation_errors)
            )

        if args.transform == "auto":
            provisional_type = (
                "homography"
                if "homography" in candidate_transforms
                else next(iter(candidate_transforms))
            )
            provisional_overlap = measure_frame_overlap(
                candidate_transforms[provisional_type],
                fixed_image.shape[:2],
                moving_image.shape[:2],
            )
            if (
                min(
                    provisional_overlap.t1_coverage_fraction,
                    provisional_overlap.moving_visible_fraction,
                )
                < args.min_full_overlap
                and "similarity" in candidate_transforms
            ):
                transform_type = "similarity"
            else:
                transform_type = provisional_type
        else:
            transform_type = args.transform

        matrix = candidate_transforms[transform_type]
        inliers = candidate_inliers[transform_type]
        overlap = measure_frame_overlap(
            matrix, fixed_image.shape[:2], moving_image.shape[:2]
        )
        fixed_support = grid_support_fraction(
            fixed_points[inliers], fixed_image.shape[:2]
        )
        moving_support = grid_support_fraction(
            moving_points[inliers], moving_image.shape[:2]
        )
        is_full_frame = (
            min(overlap.t1_coverage_fraction, overlap.moving_visible_fraction)
            >= args.min_full_overlap
            and min(fixed_support, moving_support) >= 0.65
        )
        overlap_class = "full_frame" if is_full_frame else "partial_overlap"
        registration_status = (
            "verified_full_frame" if is_full_frame else "verified_partial_overlap"
        )
        if args.strict_full_overlap and not is_full_frame:
            raise RegistrationError(
                "transform validation",
                pair,
                f"verified overlap is partial: T1={overlap.t1_coverage_fraction:.1%}, "
                f"moving={overlap.moving_visible_fraction:.1%}; required "
                f"{args.min_full_overlap:.1%} for both",
            )
        transform_source = f"roma_correspondences_ransac_{transform_type}"
        dense_confident_fraction = float((certainty >= args.min_confidence).mean())

        height, width = fixed_image.shape[:2]
        registered = cv2.warpPerspective(
            moving_image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        moving_valid = np.full(moving_image.shape[:2], 255, dtype=np.uint8)
        valid_mask = cv2.warpPerspective(
            moving_valid, matrix, (width, height), flags=cv2.INTER_NEAREST
        )
        moving_droplet_registered = cv2.warpPerspective(
            moving_droplet_mask,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
        )
        try:
            metrics = compute_metrics(
                fixed_path.name,
                period,
                transform_type,
                transform_source,
                registration_status,
                overlap,
                overlap_class,
                dense_confident_fraction,
                fixed_support,
                moving_support,
                args.ignore_droplets,
                fixed_droplet_mask,
                moving_droplet_mask,
                moving_droplet_registered,
                fixed_points,
                moving_points,
                confidence,
                root_score,
                inliers,
                matrix,
                candidates,
                raw_candidates,
                fixed_image,
                moving_image,
                registered,
                valid_mask,
            )
            save_pair_outputs(
                output_dir,
                fixed_path.stem,
                period,
                fixed_image,
                moving_image,
                registered,
                valid_mask,
                fixed_root_mask,
                moving_root_mask,
                fixed_droplet_mask,
                moving_droplet_mask,
                moving_droplet_registered,
                fixed_points,
                moving_points,
                confidence,
                root_score,
                inliers,
                matrix,
                candidate_transforms,
                transform_source,
                registration_status,
                overlap,
                warp,
                certainty,
                metrics,
            )
        except Exception as exc:
            raise RegistrationError("output writing", pair, str(exc)) from exc
        return metrics
    except RegistrationError:
        raise
    except Exception as exc:
        raise RegistrationError("unexpected failure", pair, str(exc)) from exc


def write_temporal_outputs(
    output_dir: Path, fixed_path: Path, completed_periods: Iterable[str]
) -> None:
    """Create an across-period view after every pair has its own audited output."""

    sequence_dir = output_dir / fixed_path.stem
    fixed = read_color(fixed_path, f"{fixed_path.name} temporal summary")
    height, width = fixed.shape[:2]
    fixed_gray = robust_normalize_uint8(cv2.cvtColor(fixed, cv2.COLOR_BGR2GRAY))
    layers: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    fixed_non_droplet = np.ones((height, width), dtype=np.uint8) * 255
    for period in completed_periods:
        rgba_path = sequence_dir / period / "registered_to_T1_rgba.png"
        rgba = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.shape[:2] != (height, width) or rgba.shape[2] != 4:
            raise OSError(f"cannot read a valid registered RGBA layer: {rgba_path}")
        layers[period] = rgba[..., :3]
        analysis_path = sequence_dir / period / "root_soil_analysis_mask_gray.png"
        analysis_mask = cv2.imread(str(analysis_path), cv2.IMREAD_GRAYSCALE)
        if analysis_mask is None or analysis_mask.shape != (height, width):
            raise OSError(f"cannot read root/soil analysis mask: {analysis_path}")
        masks[period] = analysis_mask
        droplet_path = sequence_dir / period / "droplet_mask_T1_gray.png"
        fixed_droplet = cv2.imread(str(droplet_path), cv2.IMREAD_GRAYSCALE)
        if fixed_droplet is not None:
            fixed_non_droplet[fixed_droplet > 0] = 0

    if not cv2.imwrite(str(sequence_dir / "T1_reference.png"), fixed):
        raise OSError(f"failed to write {sequence_dir / 'T1_reference.png'}")

    temporal = np.zeros((height, width, 3), dtype=np.uint8)
    temporal[..., 2] = fixed_gray  # T1 is red in an RGB interpretation.
    temporal[fixed_non_droplet == 0, 2] = 0
    if "T2" in layers:
        gray = cv2.cvtColor(layers["T2"], cv2.COLOR_BGR2GRAY)
        temporal[..., 1] = robust_normalize_uint8(gray, masks["T2"])
        temporal[masks["T2"] == 0, 1] = 0
    if "T3" in layers:
        gray = cv2.cvtColor(layers["T3"], cv2.COLOR_BGR2GRAY)
        temporal[..., 0] = robust_normalize_uint8(gray, masks["T3"])
        temporal[masks["T3"] == 0, 0] = 0
    cv2.putText(
        temporal,
        "root/soil temporal RGB (droplets masked): T1=red | T2=green | T3=blue",
        (24, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(sequence_dir / "temporal_RGB_T1_T2_T3.png"), temporal):
        raise OSError("failed to write temporal RGB overlay")
    if not cv2.imwrite(str(sequence_dir / "temporal_RGB_root_soil.png"), temporal):
        raise OSError("failed to write root/soil temporal RGB overlay")

    preview_images: list[np.ndarray] = []
    preview_sources = [("T1", fixed)]
    for period in completed_periods:
        context_path = sequence_dir / period / "registered_on_T1_context.png"
        context = cv2.imread(str(context_path), cv2.IMREAD_COLOR)
        if context is not None:
            preview_sources.append((period, context))
    for label, source in preview_sources:
        preview, _ = resize_for_visualization(source, max_side=900)
        preview = cv2.copyMakeBorder(
            preview, 55, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25)
        )
        cv2.putText(
            preview,
            label,
            (18, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        preview_images.append(preview)
    if preview_images:
        overview = np.concatenate(preview_images, axis=1)
        if not cv2.imwrite(
            str(sequence_dir / "temporal_overview_preview.png"), overview
        ):
            raise OSError("failed to write temporal overview preview")


def largest_valid_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return the largest axis-aligned rectangle containing only nonzero pixels."""

    if mask.ndim != 2:
        raise ValueError("largest_valid_rectangle expects a 2-D mask")
    height, width = mask.shape
    histogram = np.zeros(width, dtype=np.int32)
    best_area = 0
    best = (0, 0, 0, 0)
    for row_index in range(height):
        row_valid = mask[row_index] > 0
        histogram = np.where(row_valid, histogram + 1, 0)
        stack: list[tuple[int, int]] = []
        for column_index in range(width + 1):
            current_height = int(histogram[column_index]) if column_index < width else 0
            start = column_index
            while stack and stack[-1][1] > current_height:
                left, rectangle_height = stack.pop()
                area = rectangle_height * (column_index - left)
                if area > best_area:
                    best_area = area
                    best = (
                        left,
                        row_index - rectangle_height + 1,
                        column_index - left,
                        rectangle_height,
                    )
                start = left
            if current_height > 0 and (
                not stack or stack[-1][1] < current_height
            ):
                stack.append((start, current_height))
    return best


def add_panel_label(image: np.ndarray, label: str) -> np.ndarray:
    """Add a label above an image without covering measured pixels."""

    output = cv2.copyMakeBorder(
        image, 58, 0, 0, 0, cv2.BORDER_CONSTANT, value=(28, 28, 28)
    )
    cv2.putText(
        output,
        label,
        (18, 39),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def write_common_overlap_outputs(
    output_dir: Path, fixed_path: Path, completed_periods: Iterable[str]
) -> None:
    """Crop every period to one real, all-period field of view.

    This is the preferred view for longitudinal comparison: all panels have the
    same T1 pixel coordinates and scale, and no image content outside the true
    multi-period intersection is stitched into the comparison.
    """

    periods = list(completed_periods)
    if not periods:
        return
    sequence_dir = output_dir / fixed_path.stem
    common_dir = sequence_dir / "common_overlap"
    common_dir.mkdir(parents=True, exist_ok=True)
    fixed = read_color(fixed_path, f"{fixed_path.name} common-overlap crop")
    height, width = fixed.shape[:2]
    images: dict[str, np.ndarray] = {"T1": fixed}
    coverage_masks: list[np.ndarray] = []
    droplet_masks: dict[str, np.ndarray] = {}

    for period in periods:
        rgba_path = sequence_dir / period / "registered_to_T1_rgba.png"
        rgba = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.shape != (height, width, 4):
            raise OSError(f"cannot read registered layer for common crop: {rgba_path}")
        images[period] = rgba[..., :3]
        coverage_masks.append(rgba[..., 3] > 0)
        droplet_path = sequence_dir / period / "droplet_mask_registered_gray.png"
        droplet = cv2.imread(str(droplet_path), cv2.IMREAD_GRAYSCALE)
        if droplet is None or droplet.shape != (height, width):
            raise OSError(f"cannot read registered droplet mask: {droplet_path}")
        droplet_masks[period] = droplet

    fixed_droplet_path = sequence_dir / periods[0] / "droplet_mask_T1_gray.png"
    fixed_droplet = cv2.imread(str(fixed_droplet_path), cv2.IMREAD_GRAYSCALE)
    if fixed_droplet is None or fixed_droplet.shape != (height, width):
        raise OSError(f"cannot read T1 droplet mask: {fixed_droplet_path}")
    droplet_masks["T1"] = fixed_droplet

    common_coverage = np.logical_and.reduce(coverage_masks)
    x_coord, y_coord, crop_width, crop_height = largest_valid_rectangle(
        common_coverage.astype(np.uint8)
    )
    if crop_width < 32 or crop_height < 32:
        raise ValueError(
            "the all-period overlap has no usable rectangular crop "
            f"(largest={crop_width}x{crop_height})"
        )
    crop_slice = np.s_[
        y_coord : y_coord + crop_height,
        x_coord : x_coord + crop_width,
    ]
    water_union = np.logical_or.reduce(
        [droplet_masks[label] > 0 for label in ("T1", *periods)]
    )
    crop_water = water_union[crop_slice]
    comparison_mask = (~crop_water).astype(np.uint8) * 255

    raw_crops: dict[str, np.ndarray] = {}
    root_soil_crops: dict[str, np.ndarray] = {}
    for label in ("T1", *periods):
        crop = images[label][crop_slice].copy()
        raw_crops[label] = crop
        root_soil = crop.copy()
        root_soil[crop_water] = (58, 58, 58)
        root_soil_crops[label] = root_soil
        for name, output in (
            (f"{label}_aligned_crop.png", crop),
            (f"{label}_root_soil_crop.png", root_soil),
        ):
            if not cv2.imwrite(str(common_dir / name), output):
                raise OSError(f"failed to write {common_dir / name}")

    raw_montage = np.hstack(
        [add_panel_label(raw_crops[label], label) for label in ("T1", *periods)]
    )
    root_soil_montage = np.hstack(
        [
            add_panel_label(root_soil_crops[label], f"{label} (water ignored)")
            for label in ("T1", *periods)
        ]
    )
    pairwise_overlays: list[np.ndarray] = []
    pairwise_root_lines: list[np.ndarray] = []
    all_valid_crop = np.full((crop_height, crop_width), 255, dtype=np.uint8)
    for period in periods:
        overlay = make_color_overlap(
            raw_crops["T1"],
            raw_crops[period],
            all_valid_crop,
            crop_water.astype(np.uint8) * 255,
        )
        checkerboard = make_masked_checkerboard(
            raw_crops["T1"], raw_crops[period], comparison_mask
        )
        checkerboard[crop_water] = (58, 58, 58)
        overlay_name = f"overlay_T1_{period}.png"
        checker_name = f"checkerboard_T1_{period}.png"
        if not cv2.imwrite(str(common_dir / overlay_name), overlay):
            raise OSError(f"failed to write {common_dir / overlay_name}")
        if not cv2.imwrite(str(common_dir / checker_name), checkerboard):
            raise OSError(f"failed to write {common_dir / checker_name}")
        pairwise_overlays.append(add_panel_label(overlay, f"T1 vs {period}"))

        root_line_path = sequence_dir / period / "root_change_overlay.png"
        root_line = cv2.imread(str(root_line_path), cv2.IMREAD_COLOR)
        if root_line is not None and root_line.shape[:2] == (height, width):
            root_crop = root_line[crop_slice]
            root_name = f"heuristic_root_lines_T1_{period}.png"
            if not cv2.imwrite(str(common_dir / root_name), root_crop):
                raise OSError(f"failed to write {common_dir / root_name}")
            pairwise_root_lines.append(
                add_panel_label(root_crop, f"T1 vs {period} root-line candidates")
            )

    temporal = np.zeros((crop_height, crop_width, 3), dtype=np.uint8)
    t1_gray = cv2.cvtColor(raw_crops["T1"], cv2.COLOR_BGR2GRAY)
    temporal[..., 2] = robust_normalize_uint8(t1_gray, comparison_mask)
    if "T2" in raw_crops:
        t2_gray = cv2.cvtColor(raw_crops["T2"], cv2.COLOR_BGR2GRAY)
        temporal[..., 1] = robust_normalize_uint8(t2_gray, comparison_mask)
    if "T3" in raw_crops:
        t3_gray = cv2.cvtColor(raw_crops["T3"], cv2.COLOR_BGR2GRAY)
        temporal[..., 0] = robust_normalize_uint8(t3_gray, comparison_mask)
    temporal[crop_water] = (58, 58, 58)

    outputs = {
        "aligned_sequence_montage.png": raw_montage,
        "root_soil_sequence_montage.png": root_soil_montage,
        "temporal_RGB_common_crop.png": temporal,
        "common_comparison_mask.png": colorize_binary_mask(comparison_mask),
        "common_coverage_mask_full.png": colorize_binary_mask(
            common_coverage.astype(np.uint8) * 255
        ),
    }
    if pairwise_overlays:
        outputs["pairwise_root_soil_overlays.png"] = np.hstack(pairwise_overlays)
    if pairwise_root_lines:
        outputs["pairwise_heuristic_root_lines.png"] = np.hstack(
            pairwise_root_lines
        )
    for name, image in outputs.items():
        if not cv2.imwrite(str(common_dir / name), image):
            raise OSError(f"failed to write {common_dir / name}")

    write_json(
        common_dir / "common_crop.json",
        {
            "coordinate_system": "T1 pixels",
            "periods": ["T1", *periods],
            "crop_xywh": [x_coord, y_coord, crop_width, crop_height],
            "crop_bounds_xyxy": [
                x_coord,
                y_coord,
                x_coord + crop_width,
                y_coord + crop_height,
            ],
            "crop_fraction_of_T1": float(
                crop_width * crop_height / (width * height)
            ),
            "root_soil_comparable_fraction_inside_crop": float(
                comparison_mask.mean() / 255.0
            ),
            "policy": (
                "largest axis-aligned rectangle observed in every period; all panels "
                "share T1 scale and coordinates; non-overlap is omitted; detected "
                "droplets are neutral gray only in comparison views"
            ),
            "biological_change_policy": (
                "visualization only; no new/elongated/disappeared-root classification"
            ),
            "orientation_policy": (
                "current inputs required no discrete rotation or mirror correction; "
                "future orientation normalization is recorded before registration"
            ),
        },
    )


def write_summary(output_dir: Path, metrics: Iterable[QualityMetrics]) -> None:
    rows = [asdict(metric) for metric in metrics]
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_dir / "summary.json", rows)
    partial = [row for row in rows if row["overlap_class"] == "partial_overlap"]
    report = [
        "# 同一点位根系时序配准结果\n",
        "原始图像未被修改。每个时期目录中的 `registered_to_T1.png` 与 T1 "
        "宽高完全一致；真实重合范围之外为 NoData，并在 "
        "`registered_to_T1_rgba.png` 中显示为透明。\n",
        "`registered_on_T1_context.png` 在未覆盖区域显示变暗的 T1，只用于帮助"
        "观察位置，不能当作该时期的图像数据。`full_extent_registered.png` 在 T1 "
        "坐标系中保留该时期的完整原图。\n",
        "默认在匹配、指标及变化对比中排除保守检测的水珠，但完整配准图仍保留"
        "原始像素。`root_soil_overlay.png`：T1 为洋红色、配准时期为绿色，水珠"
        "或 NoData 为中性灰；`droplet_nuisance_mask.png` 可审查实际排除范围。\n",
        "`root_change_overlay.png` 只显示启发式候选根线，不能当作定量根分割。"
        "`full_extent_root_soil_overlay.png` 在扩展画布中保留两期完整视野。"
        "`temporal_RGB_root_soil.png`：T1=红、T2=绿、T3=蓝。\n",
    ]
    if partial:
        labels = ", ".join(
            f"{row['moving_period']} ({row['t1_coverage_fraction']:.1%})"
            for row in partial
        )
        report.append(
            "## 部分重合警告\n\n"
            f"以下时期未覆盖完整 T1 视野：{labels}。结果没有通过拉伸、复制或"
            "低置信度稠密场来伪造缺失像素。若需这些时期覆盖完整 T1，必须补充"
            "相邻拍摄帧或核对上传的点位图像。\n"
        )
    with (output_dir / "README.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(report))


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    try:
        jobs = discover_inputs(args.data_dir)
        device = choose_device(args.device)
        try:
            roma_version = importlib.metadata.version("romatch")
        except importlib.metadata.PackageNotFoundError:
            roma_version = "local source"
        print(f"[setup] RoMa {roma_version}; device={device}; jobs={len(jobs)}")
        torch.set_float32_matmul_precision("highest")
        torch_cache = configure_torch_cache()
        print(f"[setup] torch cache={torch_cache}")
        try:
            model: torch.nn.Module | None = roma_outdoor(
                device=device,
                coarse_res=args.coarse_res,
                upsample_res=args.upsample_res,
                symmetric=False,
                use_custom_corr=device.type == "cuda",
                upsample_preds=True,
            )
        except Exception as exc:
            raise RegistrationError(
                "model setup",
                "all",
                "failed to initialize RoMa or load its official weights: " + str(exc),
            ) from exc
        if args.reuse_warp_dir is not None:
            print(
                f"[setup] reusing RoMa warps below {args.reuse_warp_dir} "
                "and computing cache misses"
            )

        all_metrics: list[QualityMetrics] = []
        failures: list[str] = []
        for filename, paths in jobs:
            completed_periods: list[str] = []
            for period in PERIODS:
                if period not in paths:
                    continue
                print(f"[registering] {filename}: T1 <- {period}", flush=True)
                try:
                    metric = register_pair(
                        model,
                        device,
                        paths["T1"],
                        paths[period],
                        period,
                        args.output_dir,
                        args,
                    )
                    all_metrics.append(metric)
                    completed_periods.append(period)
                    rmse_text = (
                        f"{metric.reprojection_rmse_px:.3f}px"
                        if metric.reprojection_rmse_px is not None
                        else "n/a"
                    )
                    print(
                        f"[done] {filename} {period}: "
                        f"inliers={metric.inlier_matches}/{metric.selected_matches}, "
                        f"RMSE={rmse_text}, T1 coverage={metric.t1_coverage_fraction:.1%}, "
                        f"status={metric.registration_status}"
                    )
                except RegistrationError as exc:
                    failures.append(str(exc))
                    print(f"ERROR {exc}", file=sys.stderr)
                    if args.debug:
                        traceback.print_exc()
                    if not args.continue_on_error:
                        write_summary(args.output_dir, all_metrics)
                        return 1

            if completed_periods:
                try:
                    write_temporal_outputs(
                        args.output_dir, paths["T1"], completed_periods
                    )
                    write_common_overlap_outputs(
                        args.output_dir, paths["T1"], completed_periods
                    )
                except Exception as exc:
                    message = f"[temporal/common-crop output writing] {filename}: {exc}"
                    failures.append(message)
                    print(f"ERROR {message}", file=sys.stderr)
                    if args.debug:
                        traceback.print_exc()
                    if not args.continue_on_error:
                        write_summary(args.output_dir, all_metrics)
                        return 1

        write_summary(args.output_dir, all_metrics)
        if failures:
            write_json(args.output_dir / "failures.json", failures)
            print(f"Completed with {len(failures)} failed pair(s).", file=sys.stderr)
            return 1
        print(f"All {len(all_metrics)} pair(s) completed. Results: {args.output_dir}")
        return 0
    except RegistrationError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
