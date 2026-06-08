"""PEP 561 — ship `py.typed` so downstream type-checkers see our inline types."""
from pathlib import Path

import lemon_squeeze


def test_py_typed_marker_exists():
    """The PEP 561 marker must be present in the installed package directory.
    Without it, downstream code that imports lemon_squeeze gets no type info
    from mypy/pyright even though the project annotates every public function.
    """
    pkg_dir = Path(lemon_squeeze.__file__).parent
    marker = pkg_dir / "py.typed"
    assert marker.exists(), (
        f"missing PEP 561 marker at {marker} — downstream type-checkers "
        f"won't see lemon_squeeze's inline annotations"
    )


def test_py_typed_marker_is_empty():
    """PEP 561 specifies an empty file. Convention: 0 bytes (or 'partial' for
    partial typing — we're fully typed)."""
    pkg_dir = Path(lemon_squeeze.__file__).parent
    marker = pkg_dir / "py.typed"
    assert marker.stat().st_size == 0, (
        "py.typed should be empty per PEP 561 convention for fully-typed packages"
    )
