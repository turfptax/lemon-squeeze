"""Wilson interval + interval-overlap helpers."""
from lemon_squeeze.stats import Z_95, intervals_disjoint, wilson_interval


def test_wilson_zero_observations_total_uncertainty():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_perfect_pass_low_n_is_wide():
    lo, hi = wilson_interval(3, 3)
    assert hi == 1.0
    # 3/3 lower bound should be well below 0.7 — small samples ≠ confidence.
    assert lo < 0.5


def test_wilson_perfect_pass_high_n_is_tight():
    lo, hi = wilson_interval(100, 100)
    assert lo > 0.95


def test_wilson_half_pass_is_centered_around_half():
    lo, hi = wilson_interval(50, 100)
    center = (lo + hi) / 2
    assert abs(center - 0.5) < 0.02


def test_wilson_returns_clamped_to_unit_interval():
    lo, hi = wilson_interval(0, 5)
    assert lo == 0.0  # never negative
    lo2, hi2 = wilson_interval(5, 5)
    assert hi2 == 1.0


def test_intervals_disjoint_basic():
    assert intervals_disjoint((0.0, 0.3), (0.5, 0.9)) is True
    assert intervals_disjoint((0.5, 0.9), (0.0, 0.3)) is True
    assert intervals_disjoint((0.0, 0.6), (0.5, 0.9)) is False  # overlap


def test_intervals_disjoint_with_slack():
    # 0.4 < 0.5 by 0.1 — disjoint normally, overlapping with slack 0.2.
    assert intervals_disjoint((0.0, 0.4), (0.5, 0.9)) is True
    assert intervals_disjoint((0.0, 0.4), (0.5, 0.9), slack=0.2) is False


def test_z_95_is_standard_value():
    # Don't accidentally use 1.96 (3-digit) — it's slightly off.
    assert abs(Z_95 - 1.96) < 0.001
