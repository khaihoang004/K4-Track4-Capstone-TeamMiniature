"""Geometry uses local interpolation, preserving curved lane boundaries."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class LaneGeometry:
    left: float
    right: float
    row: float
    vehicle: float
    offset: float


def boundary_at(points, row, max_gap):
    """Interpolate bracketing points; never extrapolate through missing rows."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        return None
    if not np.isfinite(points).all():
        return None
    points = points[np.argsort(points[:, 1])]
    ys = points[:, 1]
    if np.any(np.diff(ys) <= 0) or not ys[0] <= row <= ys[-1]:
        return None
    index = min(max(int(np.searchsorted(ys, row)), 1), len(ys) - 1)
    if ys[index] - ys[index - 1] > max_gap:
        return None
    return float(np.interp(row, ys, points[:, 0]))


def estimate_geometry(lanes, width, height, lookahead=0.80, reference=0.50,
                      min_width=0.15, max_width=0.90):
    """Use UFLD ego-lane slots 1/2, never substitute neighboring lane slots.

    Offset > 0 means camera reference lies RIGHT of the estimated lane center.
    Width bounds and row gap are heuristic uncertainty gates, not confidence.
    """
    if len(lanes) != 4 or width <= 0 or height <= 0:
        return None
    row = lookahead * (height - 1)
    left = boundary_at(lanes[1], row, height * 0.10)
    right = boundary_at(lanes[2], row, height * 0.10)
    if left is None or right is None:
        return None
    lane_width = right - left
    if not (0 <= left < right < width and min_width * width <= lane_width <= max_width * width):
        return None
    vehicle = reference * (width - 1)
    offset = (vehicle - (left + right) / 2) / (lane_width / 2)
    if not math.isfinite(offset):
        return None
    return LaneGeometry(left, right, row, vehicle, offset)
