"""Reproduce rejected differentiation overhead experiments (not runtime patches).

Run through tools.platform_suite.run with a bounded process-tree RSS budget.
All experiments preserve SciPy's numerical expressions and iteration loop.
The private-state variants are included to document their lack of benefit.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import timeit

import numpy as np
import scipy
import fastscipy
from fastscipy.testing import identical
from scipy import differentiate
from scipy.differentiate import _differentiate

from fastscipy import _patch


class _ArrayExtra:
    def __init__(self, original):
        self._original = original
        original_at = original.at

        class at(original_at):
            def _op(self, at_op, in_place_op, out_of_place_op, y, /, copy, xp):
                x, idx = self._x, self._idx
                if (
                    _patch.enabled
                    and copy is None
                    and xp is None
                    and type(x) is np.ndarray
                    and x.dtype.kind in "biufc"
                    and x.flags.writeable
                    and (type(y) in (int, float, bool, complex) or (type(y) is np.ndarray and y.dtype.kind in "biufc"))
                    and (
                        (type(idx) is np.ndarray and idx.dtype.kind in "biu")
                        or (type(idx) is tuple and all(type(i) is np.ndarray and i.dtype.kind in "biu" for i in idx))
                    )
                ):
                    _patch.counts["differentiate.at", "fast"] += 1
                    # The same final operation as array-api-extra. Never retry
                    # after an assignment or user-visible warning/exception.
                    x[idx] = in_place_op(x[idx], y) if in_place_op else y
                    return x
                _patch.counts["differentiate.at", "scipy"] += 1
                return super()._op(at_op, in_place_op, out_of_place_op, y, copy=copy, xp=xp)

        self.at = at

    def __getattr__(self, name):
        return getattr(self._original, name)


def _work_factory(original):
    def create(*args, **kwargs):
        result = original(*args, **kwargs)
        if _patch.enabled:
            # All instances constructed by this module are private work state.
            object.__setattr__(result, "__dict__", result)
            _patch.counts["differentiate.work", "fast"] += 1
        return result

    return create


class _WorkState(_differentiate._RichResult):
    def __getattribute__(self, name):
        try:
            return dict.__getitem__(self, name)
        except KeyError:
            return super().__getattribute__(name)


@contextmanager
def _mode(name):
    fastscipy.unpatch()
    if name != "plain":
        fastscipy.patch()
    _patch.current = "differentiate_probe"
    try:
        if name in {"updates", "state_and_updates"}:
            _patch.replace(_differentiate, "xpx", _ArrayExtra(_differentiate.xpx))
        if name in {"state_alias", "state_and_updates"}:
            _patch.replace(_differentiate, "_RichResult", _work_factory(_differentiate._RichResult))
        if name == "state_subclass":
            _patch.replace(_differentiate, "_RichResult", _WorkState)
        yield
    finally:
        _patch.restore("differentiate_probe")
        _patch.current = None
        fastscipy.unpatch()


def _cases():
    yield "derivative_scalar", differentiate.derivative, np.sin, lambda: 0.25
    for n in (32, 4096):
        yield f"derivative_{n}", differentiate.derivative, np.sin, lambda n=n: np.linspace(0.1, 1, n)
    yield "jacobian_8", differentiate.jacobian, lambda x: np.sin(x) + np.sum(x, axis=0), lambda: np.linspace(0.1, 1, 8)
    yield "hessian_3", differentiate.hessian, lambda x: np.sum(np.exp(x), axis=0), lambda: np.linspace(0.1, 1, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--number", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    if args.number <= 0 or args.repeat <= 0:
        parser.error("number and repeat must be positive")
    _patch.require_validated("1.18.")
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "scipy": scipy.__version__,
        "numpy": np.__version__,
        "fastscipy": fastscipy.__version__,
        "threads": fastscipy.get_num_threads(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("FASTSCIPY_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
        },
        "scipy_source": _differentiate.__file__,
        "scipy_source_sha256": hashlib.sha256(Path(_differentiate.__file__).read_bytes()).hexdigest(),
        "number": args.number,
        "repeat": args.repeat,
        "cases": {},
    }
    references = {}
    for mode in ("plain", "previous", "updates", "state_alias", "state_and_updates", "state_subclass"):
        with _mode(mode):
            for name, method, f, make_x in _cases():
                calls = []

                def traced(x):
                    calls.append(np.copy(x))
                    return f(x)

                result = method(traced, make_x())
                if mode == "plain":
                    references[name] = result, calls
                else:
                    assert identical(references[name], (result, calls)), (mode, name)
                times = timeit.repeat(lambda: method(f, make_x()), number=args.number, repeat=args.repeat)
                report["cases"].setdefault(name, {})[mode] = {
                    "seconds_per_call": [t / args.number for t in times],
                    "result_and_callback_trace_identical": True,
                }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
