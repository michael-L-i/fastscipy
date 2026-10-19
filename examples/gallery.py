"""Measure and render the eight-task gallery. Run from the repository root.

pip install 'pillow>=10.1' matplotlib
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python examples/gallery.py

Inputs are synthetic and deterministic. Each task runs serially with a bounded
thread budget. Timings include the shown SciPy calls and their result allocation;
setup, warmup, byte comparison and plot/GIF rendering are outside the timer.
"""

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import platform
import resource
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
import scipy  # noqa: E402
from scipy import sparse  # noqa: E402

import fastscipy  # noqa: E402
from fastscipy.testing import fast_calls, identical  # noqa: E402
from _gallery_tasks import TASKS  # noqa: E402

HERE = Path(__file__).parent
BG, PANEL, INK, MUTED, ACCENT = "#101824", "#192536", "#f1f5fa", "#a7b8ca", "#68e2c2"


def measure(jobs, plain):
    best = None
    for _ in range(3):
        with fastscipy.disabled() if plain else nullcontext():
            results, times = [], []
            start = perf_counter()
            for job in jobs:
                results.append(job())
                times.append(perf_counter() - start)
        if best is None or times[-1] < best[1][-1]:
            best = results, times
    return best


def compare(a, b):
    if sparse.issparse(a):
        return (
            type(a) is type(b)
            and a.shape == b.shape
            and identical((a.data, a.indices, a.indptr), (b.data, b.indices, b.indptr))
        )
    return identical(a, b)


def previews(task, results):
    images = []
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.facecolor": PANEL,
            "figure.facecolor": PANEL,
            "axes.edgecolor": "#506179",
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
        }
    )
    for completed in range(len(results) + 1):
        fig, ax = plt.subplots(figsize=(4.4, 2.65), dpi=100)
        task.plot(ax, results[:completed])
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(pad=0.8)
        fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()))
        plt.close(fig)
    return images


def render(task, results, record, path):
    pictures = previews(task, results)
    fonts = {n: ImageFont.load_default(size=n) for n in (15, 17, 19, 23, 29)}
    totals = record["scipy_seconds"], record["fastscipy_seconds"]
    # All animations last 3.6 s plus a final hold. Both lanes share one time scale.
    slow = 3.6 / max(totals)
    if slow < 1:
        raise RuntimeError("Task exceeds the 3.6-second replay window; increase the window instead of speeding it up")
    frames = []
    for index in range(46):
        elapsed = max(totals) * index / 45
        image = Image.new("RGB", (1000, 550), BG)
        draw = ImageDraw.Draw(image)
        draw.text((32, 23), task.title, font=fonts[29], fill=INK)
        draw.text((32, 67), f"scipy.{task.area}  /  same inputs, same SciPy calls", font=fonts[17], fill=MUTED)
        for left, label, key in zip(
            (24, 512),
            ("SciPy", f"fastscipy / {record['threads']} threads"),
            ("scipy_completed_seconds", "fastscipy_completed_seconds"),
        ):
            done = sum(t <= elapsed for t in record[key])
            draw.rounded_rectangle((left, 105, left + 464, 445), radius=14, fill=PANEL)
            draw.text((left + 16, 118), label, font=fonts[23], fill=INK)
            draw.text((left + 16, 153), f"{done}/8 {task.unit}", font=fonts[17], fill=ACCENT if done == 8 else MUTED)
            image.paste(pictures[done], (left + 12, 179))
        draw.text((32, 463), f"{record['speedup']:.1f}x faster", font=fonts[29], fill=ACCENT)
        draw.text((285, 472), f"{totals[0] * 1000:,.2f} ms  ->  {totals[1] * 1000:,.2f} ms", font=fonts[19], fill=INK)
        draw.text(
            (32, 509),
            f"Measured completion times replayed {slow:,.1f}x slower on a shared clock. Identical outputs.",
            font=fonts[15],
            fill=MUTED,
        )
        frames.append(image)
    # One palette for stable plot colors and legible text across the loop.
    sample = Image.new("RGB", (1000, 1650))
    for i, frame in enumerate((frames[0], frames[23], frames[-1])):
        sample.paste(frame, (0, i * 550))
    palette = sample.quantize(colors=128)
    indexed = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    indexed[0].save(path, save_all=True, append_images=indexed[1:], duration=[80] * 45 + [1400], loop=0, optimize=True)
    record["replay_slowdown"] = slow
    record["gif_bytes"] = path.stat().st_size


def readme(records):
    rows = [
        "# Eight everyday SciPy tasks, faster",
        "",
        "Same SciPy calls. Less waiting. Realistic tasks on reproducible synthetic data.",
        "",
        "*Run on a MacBook Pro with M5 Pro, using two fastscipy threads.*",
        "",
        "| Task | SciPy area | Speedup |",
        "|---|---|---:|",
    ]
    for factory in TASKS:
        r = records[factory.__name__]
        rows.append(f"| [{r['title']}](#{r['slug']}) | `{r['area']}` | **{r['speedup']:.1f}×** |")
    for i, factory in enumerate(TASKS, 1):
        r = records[factory.__name__]
        rows += [
            "",
            f'<a id="{r["slug"]}"></a>',
            "",
            f"## {i}. {r['title']}",
            "",
            r["description"],
            "",
            f"```python\n{r['code']}\n```",
            "",
            f"![{r['title']}: SciPy and fastscipy](assets/{r['slug']}.gif)",
        ]
    rows += [
        "",
        "---",
        "",
        "Synthetic inputs, real timings. Each GIF replays measured completion times on a",
        "shared clock; the slowdown is shown in the frame. Byte-identical outputs.",
        "",
        "[Measured timings](measurements.json) · [Runnable source](gallery.py) · [Back to fastscipy](../README.md)",
        "",
        "<details>",
        "<summary>Run the gallery</summary>",
        "",
        "From the repository root, with fastscipy installed:",
        "",
        "```sh",
        'pip install "pillow>=10.1" matplotlib',
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python examples/gallery.py",
        "```",
        "",
        "Each lane uses the fastest of three warmed batches. Input setup and rendering",
        "are outside the timer. Tasks run one at a time; results depend on hardware and inputs.",
        "",
        "</details>",
        "",
    ]
    (HERE / "README.md").write_text("\n".join(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=[f.__name__ for f in TASKS])
    args = parser.parse_args()
    database = HERE / "measurements.json"
    data = json.loads(database.read_text()) if database.exists() else {"meta": {}, "tasks": {}}
    data["meta"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "threads": 2,
        "repeats": 3,
        "thread_environment": {
            name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
        },
        "method": (
            "Fastest of three warmed sequential batches per implementation; full batch output allocation timed. "
            "Setup, identity checking and rendering excluded. Per-job completion timestamps come from the chosen batch."
        ),
    }
    for factory in TASKS:
        if args.only and factory.__name__ != args.only:
            continue
        status = fastscipy.patch(threads=2)
        if not all(s == "on" for s in status.values()):
            parser.error(f"Gallery needs a validated fastscipy installation: {status}")
        task = factory()
        with fastscipy.disabled():
            task.jobs[0]()
        task.jobs[0]()
        reference, baseline = measure(task.jobs, plain=True)
        before = fast_calls()
        result, accelerated = measure(task.jobs, plain=False)
        calls = fast_calls() - before
        if not all(compare(a, b) for a, b in zip(reference, result)):
            raise RuntimeError(f"{task.slug}: outputs differ; no GIF generated")
        if calls < len(task.jobs) * 3:
            raise RuntimeError(f"{task.slug}: expected every job to use a fast path")
        record = {
            "slug": task.slug,
            "title": task.title,
            "area": task.area,
            "description": task.description,
            "code": task.code,
            "threads": 2,
            "jobs": len(task.jobs),
            "byte_identical": True,
            "fast_calls": calls,
            "measured_at_utc": datetime.now(timezone.utc).isoformat(),
            "scipy_completed_seconds": baseline,
            "fastscipy_completed_seconds": accelerated,
            "scipy_seconds": baseline[-1],
            "fastscipy_seconds": accelerated[-1],
            "speedup": baseline[-1] / accelerated[-1],
        }
        del reference
        render(task, result, record, HERE / "assets" / f"{task.slug}.gif")
        data["tasks"][factory.__name__] = record
        database.write_text(json.dumps(data, indent=2) + "\n")
        print(
            f"{task.slug}: {baseline[-1] * 1000:.2f} -> {accelerated[-1] * 1000:.2f} ms / {record['speedup']:.1f}x",
            flush=True,
        )
        del result, task
        gc.collect()
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin" and rss > 1024**3:
            raise MemoryError("Gallery exceeded the 1 GiB peak-RSS guard")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    data["meta"]["max_rss_bytes"] = peak if platform.system() == "Darwin" else peak * 1024
    database.write_text(json.dumps(data, indent=2) + "\n")
    if len(data["tasks"]) == len(TASKS):
        readme(data["tasks"])
    print("Peak RSS:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, flush=True)


if __name__ == "__main__":
    main()
