"""Compare an image-cleanup batch; optionally replay measured timings as a GIF.

python examples/median_filter.py
python examples/median_filter.py --gif docs/median-filter-demo.gif

Timing needs only fastscipy's normal dependencies. GIF rendering needs Pillow.
Images are synthetic; no downloads. Rendering and validation are outside timing.
"""

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import scipy
from scipy import ndimage

import fastscipy


def images(count, size):
    rng = np.random.default_rng(42)
    y, x = np.mgrid[:size, :size].astype(np.float32) / size
    base = np.full((size, size), 15, dtype=np.float32)
    for cx, cy, radius in rng.uniform([0, 0, 0.025], [1, 1, 0.07], (40, 3)):
        base += 150 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * radius**2))
    base = np.clip(base, 0, 240).astype(np.uint8)
    result = []
    for _ in range(count):
        image = np.roll(base, tuple(rng.integers(0, size, 2)), axis=(0, 1))
        noise = rng.random(image.shape)
        image[noise < 0.04] = 0
        image[noise > 0.96] = 255
        result.append(image)
    return result


def run_batch(batch, plain):
    output, completed = [], []
    # One identical SciPy call per image; only the patch state changes.
    with fastscipy.disabled() if plain else nullcontext():
        start = perf_counter()
        for image in batch:
            output.append(ndimage.median_filter(image, size=3))
            completed.append(perf_counter() - start)
    return output, completed


def render(path, batch, output, data):
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    font = lambda size: ImageFont.load_default(size=size)
    bg, panel, ink, muted, accent = "#101824", "#192536", "#f1f5fa", "#a7b8ca", "#68e2c2"
    count = len(batch)
    before, after = [], []
    for source, target in zip(batch, output):
        for value, dest in ((source, before), (target, after)):
            tile = Image.fromarray(value).resize((96, 96), Image.Resampling.NEAREST)
            dest.append(ImageOps.colorize(tile, "#102633", "#bdece2"))
    totals = data["scipy_seconds"], data["fastscipy_seconds"]
    thread_label = f"{data['threads']} thread" + ("s" if data["threads"] != 1 else "")
    replay_seconds = max(3.6, max(totals))
    slow = replay_seconds / max(totals)
    frames, duration = [], 60
    frame_count = int(replay_seconds * 1000 / duration) + 1
    # A shared palette prevents colors flickering between frames.
    template = Image.new("RGB", (1000, 550), bg)
    palette = Image.new("RGB", (256, 1))
    colors = [bg, panel, ink, muted, accent, "#34475c"]
    colors += [tuple(round(a + (b - a) * i / 249) for a, b in zip((16, 38, 51), (189, 236, 226))) for i in range(250)]
    palette.putdata([Image.new("RGB", (1, 1), c).getpixel((0, 0)) for c in colors])
    palette = palette.quantize(colors=256)
    for index in range(frame_count):
        elapsed = index / (frame_count - 1) * max(totals)
        canvas = template.copy()
        draw = ImageDraw.Draw(canvas)
        draw.text((32, 24), "One line. A batch of cleaner images.", fill=ink, font=font(30))
        size = batch[0].shape[0]
        draw.text(
            (32, 72), f"{count} images  /  {size} x {size} pixels  /  3 x 3 median filter", fill=muted, font=font(17)
        )
        for left, name, stamps, total in zip(
            (24, 512),
            ("SciPy", f"fastscipy  /  {thread_label}"),
            (data["scipy_completed_seconds"], data["fastscipy_completed_seconds"]),
            totals,
        ):
            done = sum(t <= elapsed for t in stamps)
            draw.rounded_rectangle((left, 112, left + 464, 440), radius=14, fill=panel)
            draw.text((left + 16, 128), name, fill=ink, font=font(24))
            status = f"{done}/{count} cleaned" if done < count else f"{count}/{count} cleaned  /  done"
            draw.text((left + 16, 165), status, fill=accent if done == count else muted, font=font(17))
            draw.text((left + 16, 409), f"Measured batch: {total * 1000:,.2f} ms", fill=muted, font=font(15))
            for item in range(count):
                x, y = left + 16 + (item % 4) * 112, 200 + (item // 4) * 104
                tile = (after if item < done else before)[item]
                canvas.paste(tile, (x, y))
                draw.rectangle((x, y, x + 95, y + 95), outline=accent if item < done else "#34475c")
        draw.text((32, 458), f"{data['speedup']:.1f}x faster  /  byte-identical output", fill=accent, font=font(24))
        draw.text(
            (32, 501),
            f"Measured completion times replayed {slow:.1f}x slower. Waiting tiles show noisy input.",
            fill=muted,
            font=font(16),
        )
        frames.append(canvas.quantize(palette=palette, dither=Image.Dither.NONE))
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=[duration] * (len(frames) - 1) + [1400],
        loop=0,
        optimize=True,
    )
    data["replay_slowdown"] = slow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--gif", type=Path)
    parser.add_argument("--json", type=Path, help="Save measured completion times and environment")
    args = parser.parse_args()
    if args.threads < 1 or not 64 <= args.size <= 2048:
        parser.error("Use at least one thread and an image size from 64 to 2048")
    status = fastscipy.patch(components=["ndimage"], threads=args.threads)
    if status.get("ndimage") != "on":
        parser.error(f"No validated fast path on this installation: {status}")
    batch = images(8, args.size)
    # Warm both implementations on the same shape. No setup enters the timer.
    for plain in (True, False):
        with fastscipy.disabled() if plain else nullcontext():
            ndimage.median_filter(batch[0], size=3)
    plain, scipy_times = run_batch(batch, plain=True)
    before = fastscipy.stats()
    patched, fast_times = run_batch(batch, plain=False)
    for a, b in zip(plain, patched):
        if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
            raise RuntimeError("Outputs differ; refusing to report a speedup")
    fast_calls = sum(v - before.get(k, 0) for k, v in fastscipy.stats().items() if k[1] == "fast")
    if fast_calls < len(batch):
        raise RuntimeError("Expected a fast-path call for every image")
    data = {
        "task": "3x3 median filtering, reflect mode, synthetic noisy uint8 images",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "images": len(batch),
        "shape": list(batch[0].shape),
        "threads": args.threads,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "scipy_completed_seconds": scipy_times,
        "fastscipy_completed_seconds": fast_times,
        "scipy_seconds": scipy_times[-1],
        "fastscipy_seconds": fast_times[-1],
        "speedup": scipy_times[-1] / fast_times[-1],
        "byte_identical": True,
        "fast_calls": fast_calls,
        "method": (
            "One warmed sequential batch per implementation; timers include calls and output allocation, "
            "exclude inputs, warmup, validation and rendering. Illustrative workload, not a benchmark-suite aggregate."
        ),
    }
    print(f"SciPy:     {scipy_times[-1] * 1000:9.2f} ms")
    print(f"fastscipy: {fast_times[-1] * 1000:9.2f} ms  ({data['speedup']:.1f}x, {args.threads} thread(s))")
    print("All 8 outputs are byte-identical; all 8 calls used a fast path.")
    if args.gif:
        render(args.gif, batch, patched, data)
        print(f"GIF: {args.gif}")
    output = args.json or (args.gif.with_suffix(".json") if args.gif else None)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
