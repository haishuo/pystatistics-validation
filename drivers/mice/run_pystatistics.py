"""Run and time pystatistics MICE → a canonical validation record.

One job: fit ``pystatistics.mice.mice`` on a matrix (NaN = missing) with a given
backend, under the shared device-aware timing protocol, and return one
``validation-run/v1`` record. Analysis-side reductions (pooling, fidelity vs R)
belong in the study generators, not here.

Consumes the pystatistics library; never modifies it. Timing/memory and the record
envelope come from the shared harness ``pystatsval``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pystatsval.measure import measure
from pystatsval.record import make_record


def run_mice_record(data: NDArray[np.floating], *,
                    backend: str,
                    dataset: str,
                    m: int = 5,
                    maxit: int = 5,
                    method: str = "pmm",
                    seed: int,
                    repeats: int = 5,
                    warmup: int = 1,
                    use_fp64: bool = False,
                    column_kinds: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Fit + time ``mice()`` on ``data`` for ``backend`` ('cpu' or 'gpu').

    For mixed-type data pass ``column_kinds`` (per-column numeric/binary/categorical/
    ordered); a ``MICEDesign`` with ``method='auto'`` is built OUTSIDE the timed
    region so the timing reflects the imputation sweep, not design setup.

    On failure a record with timing nulled and an ``error`` field is returned rather
    than raising — a sweep records the failure and continues.
    """
    from pystatistics.mice import mice
    from pystatistics.mice.design import MICEDesign

    n, p = data.shape
    device = "gpu" if backend == "gpu" else "cpu"
    engine = f"pystatistics:{backend}"
    common = {"m_imputations": int(m), "maxit": int(maxit), "method": method}

    design = (MICEDesign.from_array(data, method="auto", column_kinds=list(column_kinds))
              if column_kinds is not None else None)

    def _call():
        target = design if design is not None else data
        kwargs = {} if design is not None else {"method": method}
        return mice(target, m=m, maxit=maxit, seed=seed, backend=backend,
                    use_fp64=use_fp64, **kwargs)

    try:
        summary, _sol = measure(_call, device=device, repeats=repeats,
                                warmup=warmup if backend == "gpu" else 0)
    except Exception as exc:  # noqa: BLE001 - record and continue the sweep
        return make_record(engine=engine, dataset=dataset, n=n, p=p, wall=None,
                           backend_name=f"mice_{backend}",
                           extra={**common, "error": f"{type(exc).__name__}: {exc}"})

    peak = summary.get("peak_mem_bytes")
    return make_record(
        engine=engine, dataset=dataset, n=n, p=p, wall=summary,
        backend_name=f"mice_{backend}",
        precision=("fp64" if (backend == "cpu" or use_fp64) else "fp32"),
        extra={**common,
               "peak_mem_mb": (round(peak / 1e6, 1) if peak else None)},
    )
