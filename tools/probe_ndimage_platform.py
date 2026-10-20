"""Small, deterministic diagnostics for ndimage compiler/platform differences."""

import math
import platform
import sys
import traceback

import numpy as np
import scipy
import scipy.ndimage as ndi

import fastscipy
from fastscipy import _ndimage_correlate as corr
from fastscipy._ndimage_seam import core, orig


def compare(name, ref, got):
    a, b = np.asarray(ref), np.asarray(got)
    if a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes():
        return True
    print(name, "MISMATCH", a.dtype, b.dtype, a.shape, b.shape, flush=True)
    if a.dtype == b.dtype and a.shape == b.shape:
        width = a.dtype.itemsize
        av = np.ascontiguousarray(a).view(np.uint8).reshape(-1, width)
        bv = np.ascontiguousarray(b).view(np.uint8).reshape(-1, width)
        ix = np.flatnonzero(np.any(av != bv, axis=1))
        print("different elements:", ix.size, flush=True)
        for i in ix[:4]:
            print(int(i), repr(a.flat[i]), repr(b.flat[i]), av[i].tobytes().hex(), bv[i].tobytes().hex(), flush=True)
    return False


def filters():
    rng = np.random.default_rng(20261015)
    x = rng.standard_normal(61) * 100
    probes = []
    for halfsize in list(range(26)) + [31, 32, 33, 40]:
        half = rng.standard_normal(halfsize)
        probes.append(("sym", np.r_[half, rng.standard_normal(1), half[::-1]]))
        if halfsize:
            probes.append(("anti", np.r_[half, rng.standard_normal(1), -half[::-1]]))
    for size in list(range(2, 26)) + [40, 41]:
        probes.append(("general", rng.standard_normal(size)))
    for scheme in range(4):
        matches = 0
        first = set()
        for kind, weights in probes:
            ref, got = np.zeros_like(x), np.zeros_like(x)
            orig.correlate1d(x, weights, 0, ref, 2, 0.0, 0)
            accepted = core.corr_correlate1d(x, weights, 0, got, 2, 0.0, 0, scheme)
            same = ref.tobytes() == got.tobytes()
            matches += accepted and same
            if not same and kind not in first:
                compare(f"correlate1d scheme={scheme} {kind} size={weights.size} accepted={accepted}", ref, got)
                first.add(kind)
        print("correlate1d scheme", scheme, "matches", matches, "/", len(probes), flush=True)
    if hasattr(math, "fma"):
        matches = 0
        for kind, weights in probes:
            ref = np.zeros_like(x)
            orig.correlate1d(x, weights, 0, ref, 2, 0.0, 0)
            got = windows_arm64_reference(x, weights, kind)
            matches += compare(f"correlate1d Windows arm64 software {kind} size={weights.size}", ref, got)
        print("correlate1d Windows arm64 software matches", matches, "/", len(probes), flush=True)
    for size in [1, 2, 3, 11, 31, 80]:
        ref, got = np.zeros_like(x), np.zeros_like(x)
        orig.uniform_filter1d(x, size, 0, ref, 2, 0.0, 0)
        accepted = core.corr_uniform_filter1d(x, size, 0, got, 2, 0.0, 0)
        print("uniform", size, "accepted", accepted, "same", compare("uniform", ref, got), flush=True)
    x2 = rng.standard_normal((13, 11)) * 100
    for fused in [False, True]:
        for shape in [(3, 3), (5, 5), (2, 4), (1, 7), (6, 1)]:
            w = rng.standard_normal(shape)
            ref, got = np.zeros_like(x2), np.zeros_like(x2)
            orig.correlate(x2, w, ref, 2, 0.0, [0, 0])
            accepted = core.corr_correlate(x2, w, got, 2, 0.0, [0, 0], fused)
            print("nd", fused, shape, "accepted", accepted, "same", compare("nd", ref, got), flush=True)
    x = corr._probe_values()
    for char in sorted(corr._CHARS):
        ref = np.zeros(x.shape, np.dtype(char))
        got = np.zeros_like(ref)
        orig.uniform_filter1d(x, 1, 0, ref, 2, 0.0, 0)
        accepted = core.corr_uniform_filter1d(x, 1, 0, got, 2, 0.0, 0)
        print("cast", char, "accepted", accepted, "same", compare("line cast " + char, ref, got), flush=True)


def windows_arm64_reference(x, weights, kind):
    """Mirror the observed four-tap unfused prefix and fused scalar tail."""
    size, half = len(weights), len(weights) // 2
    padded = np.pad(x, (half, size - half - 1), mode="symmetric")
    iterations = size - 1 if kind == "general" else half
    unfused = iterations & ~3 if iterations >= 6 else 0
    out = np.empty_like(x)
    for j in range(len(x)):
        line = padded[j : j + size]
        center = size - 1 if kind == "general" else half
        acc = float(line[center]) * float(weights[center])
        for i in range(iterations):
            value = float(line[i])
            if kind != "general":
                value = value + float(line[size - 1 - i]) if kind == "sym" else value - float(line[size - 1 - i])
            weight = float(weights[i])
            acc = acc + value * weight if i < unfused else math.fma(value, weight, acc)
        out[j] = acc
    return out


def public_case(name, fn):
    with fastscipy.disabled():
        ref = fn()
    before = fastscipy.stats()
    got = fn()
    if isinstance(ref, tuple):
        same = all([compare(f"{name}[{i}]", a, b) for i, (a, b) in enumerate(zip(ref, got, strict=True))])
    else:
        same = compare(name, ref, got)
    after = fastscipy.stats()
    delta = {str(k): v - before.get(k, 0) for k, v in after.items() if v != before.get(k, 0)}
    print(name, "same", same, "calls", delta, flush=True)


def uniform_nan():
    """Separate running-sum NaN propagation from output conversion."""
    probes = [(f"fullprobe-{n}", np.resize(corr._probe_values(), (n, 9))) for n in (7, 8, 9)]
    probes.append(("nonfinite", np.array([[np.inf, -np.inf, np.nan, 0.0, -0.0, 0.4, -0.4, 0.9999, -0.9999]])))
    bits = np.array([0x7FF8000000001234, 0xFFF8000000005678, 0x3FF0000000000000, 0, 0, 0, 0, 0, 0], np.uint64)
    probes.append(("payloads", bits.view(np.float64)[None, :]))
    for name, x in probes:
        for axis in (0, 1):
            for dtype in (np.float64, np.float32):
                ref, direct, converted = (np.empty(x.shape, dtype) for _ in range(3))
                raw = np.empty(x.shape, np.float64)
                orig.uniform_filter1d(x, 3, axis, ref, 2, 0.0, 0)
                accepted = core.corr_uniform_filter1d(x, 3, axis, direct, 2, 0.0, 0)
                core.corr_uniform_filter1d(x, 3, axis, raw, 2, 0.0, 0)
                orig.correlate1d(raw, np.ones(1), axis, converted, 2, 0.0, 0)
                label = f"uniform-nan {name} axis={axis} dtype={np.dtype(dtype)} shape={x.shape} accepted={accepted}"
                direct_same = compare(label + " direct-native", ref, direct)
                cast_same = compare(label + " native-then-C-cast", ref, converted)
                copy = np.empty_like(ref)
                orig.correlate1d(ref.astype(np.float64), np.ones(1), axis, copy, 2, 0.0, 0)
                copy_same = compare(label + " C-copy-reference", ref, copy)
                print(label, "direct/cast/copy matches", direct_same, cast_same, copy_same, flush=True)


def main():
    print(
        sys.version,
        platform.platform(),
        platform.machine(),
        "numpy",
        np.__version__,
        "scipy",
        scipy.__version__,
        flush=True,
    )
    fastscipy.patch(threads=2)
    try:
        print("raw calibration:", corr._calibrate(), flush=True)
    except Exception:
        traceback.print_exc()
    filters()
    uniform_nan()
    rng = np.random.default_rng(901)
    x = rng.standard_normal((7, 11, 13)) > 0
    for sampling in [None, [0.7, 1.3, 2.1], [1.0, 1.0, 1.0]]:
        public_case(
            f"EDT sampling={sampling}",
            lambda s=sampling: ndi.distance_transform_edt(x, sampling=s, return_indices=True),
        )
    for dtype in [np.float64, np.float32, np.int16, np.uint64]:
        with np.errstate(invalid="ignore"):
            x = (rng.standard_normal((7, 13)) * 100).astype(dtype)
        for order in [0, 1, 2, 3, 4, 5]:
            public_case(f"spline {dtype} order={order}", lambda o=max(order, 2): ndi.spline_filter(x, o))
            public_case(f"shift {dtype} order={order}", lambda o=order: ndi.shift(x, [0.3, -1.2], order=o))
            public_case(
                f"affine {dtype} order={order}",
                lambda o=order: ndi.affine_transform(x, [[0.91, 0.13], [-0.07, 1.1]], [0.3, -1.2], order=o),
            )


if __name__ == "__main__":
    main()
