"""Draw the README charts from ``bench/results/*.json`` (needs matplotlib).

python bench/plot.py      -> docs/speedup-by-area.svg, docs/speedup-highlights.svg
"""

import json
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

HERE = pathlib.Path(__file__).parent
DOCS = HERE.parent / "docs"
ONE, MANY = "#4C78A8", "#F58518"


def load():
    areas = {}
    for f in sorted((HERE / "results").glob("*.json")):
        data = json.loads(f.read_text())
        areas[f.stem] = (data["meta"]["threads"], data["cases"])
    return areas


def geomean(xs):
    return math.exp(sum(map(math.log, xs)) / len(xs))


def is_control(name):
    """Cases the fast path is meant to decline (already-ordered queries, tiny calls)."""
    return "(control)" in name or "sorted" in name or "raster" in name


def bars(ax, labels, one, many, threads):
    y = range(len(labels))
    ax.barh([i + 0.2 for i in y], one, 0.4, color=ONE, label="1 thread")
    ax.barh([i - 0.2 for i in y], many, 0.4, color=MANY, label=f"{threads} threads")
    fmt = lambda v: f"{v:,.0f}×" if v >= 100 else f"{v:.3g}×"
    for i, (a, b) in enumerate(zip(one, many)):
        ax.text(a * 1.08, i + 0.2, fmt(a), va="center", fontsize=8)
        ax.text(b * 1.08, i - 0.2, fmt(b), va="center", fontsize=8)
    ax.set_yticks(list(y), labels)
    ax.set_xscale("log")
    ax.set_xlim(0.7, max(many + one) * 3)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.axvline(1, color="#888", lw=0.8)
    ax.set_xlabel("speedup over SciPy (log scale; 1× = same speed)")
    ax.legend(loc="lower right", frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(DOCS / name, facecolor="white")
    print("wrote", DOCS / name)


def main():
    areas = load()
    threads = max(t for t, _ in areas.values())
    rows = []
    for area, (_, cases) in areas.items():
        real = [r for n, r in cases.items() if not is_control(n)]
        rows.append((area, geomean([r["x1t"] for r in real]), geomean([r["xnt"] for r in real]), len(real)))
    rows.sort(key=lambda r: r[2])
    fig, ax = plt.subplots(figsize=(8, 5.2))
    bars(ax, [f"{a} ({n})" for a, _, _, n in rows], [r[1] for r in rows], [r[2] for r in rows], threads)
    ax.set_title("Geometric-mean speedup by area (number of cases)", fontsize=11)
    save(fig, "speedup-by-area.svg")

    top = sorted(
        ((r["xnt"], r["x1t"], area, name) for area, (_, cases) in areas.items() for name, r in cases.items()),
        reverse=True,
    )
    picked, seen = [], set()
    for xn, x1, area, name in top:
        if area not in seen or len(picked) < 6:
            picked.append((xn, x1, f"{area}: {name}"))
            seen.add(area)
        if len(picked) == 12:
            break
    picked.reverse()
    fig, ax = plt.subplots(figsize=(11, 6))
    bars(ax, [p[2] for p in picked], [p[1] for p in picked], [p[0] for p in picked], threads)
    ax.set_title("Largest speedups (byte-identical results)", fontsize=11)
    save(fig, "speedup-highlights.svg")


if __name__ == "__main__":
    main()
