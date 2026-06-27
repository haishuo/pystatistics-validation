"""Survival validation drivers (KM, log-rank, Cox PH, discrete-time).

Generators that produce the frozen evidence behind ``reports/survival-v<X.Y.Z>.md``.
They fit a PyPI-installed ``pystatistics`` (never a local checkout — ``require_pypi``
enforces it) and the R ``survival`` reference on the shared ``pystatsval`` harness.
"""
