import pytest

from laneguard.geometry import LaneGeometry, estimate_geometry
from laneguard.monitor import DepartureMonitor, State


def lanes(left=200, right=800):
    return [[], [(left, 600), (left, 800), (left, 900)],
            [(right, 600), (right, 800), (right, 900)], []]


def geometry(offset):
    return LaneGeometry(200, 800, 800, 500, offset)


def test_offset_and_sign():
    # Use closely spaced rows, meeting the missing-row gate.
    points = [[], [(200, 700), (200, 800), (200, 900)],
              [(800, 700), (800, 800), (800, 900)], []]
    centered = estimate_geometry(points, 1000, 1000)
    assert centered.offset == pytest.approx(-0.5 / 300)
    assert estimate_geometry(points, 1000, 1000, reference=0.6).offset > 0
    assert estimate_geometry(points, 1000, 1000, reference=0.4).offset < 0


@pytest.mark.parametrize('left,right', [(800, 200), (500, 500), (490, 510), (-10, 800)])
def test_invalid_boundaries(left, right):
    assert estimate_geometry(lanes(left, right), 1000, 1000) is None


def test_missing_boundary_and_no_extrapolation():
    assert estimate_geometry([[], [], [], []], 1000, 1000) is None
    assert estimate_geometry(lanes(), 1000, 1000, lookahead=0.95) is None


def test_curved_local_interpolation():
    points = [[], [(180, 700), (200, 800), (240, 900)],
              [(780, 700), (800, 800), (840, 900)], []]
    result = estimate_geometry(points, 1000, 1000, lookahead=850/999)
    assert result.left == pytest.approx(220)
    assert result.right == pytest.approx(820)


def test_warning_persistence_clear_and_missing_reset():
    monitor = DepartureMonitor(persistence=3, alpha=1)
    assert monitor.update(geometry(0.8)).state == State.NORMAL
    assert monitor.update(geometry(0.8)).state == State.NORMAL
    assert monitor.update(geometry(0.8)).state == State.WARNING
    assert monitor.update(geometry(0.55)).state == State.WARNING  # hysteresis
    assert monitor.update(geometry(0.1)).state == State.NORMAL
    assert monitor.update(None).state == State.UNRELIABLE
    assert monitor.smoothed is None
    assert monitor.update(geometry(0.8)).state == State.NORMAL


def test_smoothing_and_noise():
    monitor = DepartureMonitor(persistence=2, alpha=0.5)
    monitor.update(geometry(0))
    assert monitor.update(geometry(0.8)).offset == pytest.approx(0.4)
    assert monitor.update(geometry(0)).state == State.NORMAL
    assert monitor.update(geometry(0.8)).state == State.NORMAL
    assert monitor.update(geometry(float('nan'))).state == State.UNRELIABLE


def test_sign_reversal_requires_new_persistence():
    monitor = DepartureMonitor(persistence=2, alpha=1)
    monitor.update(geometry(.8))
    assert monitor.update(geometry(.8)).state == State.WARNING
    assert monitor.update(geometry(-.8)).state == State.NORMAL
    assert monitor.update(geometry(-.8)).state == State.WARNING


@pytest.mark.parametrize('kwargs', [{'persistence': 1}, {'alpha': 0}, {'threshold': .3}])
def test_invalid_monitor_config(kwargs):
    with pytest.raises(ValueError):
        DepartureMonitor(**kwargs)
