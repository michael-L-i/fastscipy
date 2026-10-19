"""Deterministic workloads and output plots for the visual gallery."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import integrate, interpolate, linalg, signal, sparse, stats
from scipy.sparse import csgraph
from scipy.cluster import hierarchy


@dataclass
class Task:
    slug: str
    title: str
    area: str
    description: str
    code: str
    unit: str
    jobs: list
    plot: Callable


def heatmap(ax, data, **kw):
    ax.imshow(data, origin="lower", aspect="auto", interpolation="nearest", **kw)


def bellman():
    height, width = 200, 250
    ids = np.arange(height * width).reshape(height, width)
    x, y = np.meshgrid(np.linspace(0, 1, width, endpoint=False), np.linspace(0, 1, height, endpoint=False))
    potential = np.rint(150 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)).astype(np.int64).ravel()
    rows = np.repeat(ids.ravel(), 2)
    cols = np.stack((np.roll(ids, -1, axis=1), np.roll(ids, -1, axis=0)), axis=-1).ravel()
    base_cost = np.random.default_rng(79).integers(1, 5, cols.size)
    # The potential differences telescope to zero around every cycle. Positive
    # base costs therefore rule out negative cycles, despite negative edges.
    cost = base_cost + potential[cols] - potential[rows]
    graph = sparse.csr_matrix((cost.astype(float), (rows, cols)), shape=(ids.size, ids.size))
    origins = np.array([[0, 0], [135, 165]])
    jobs = [lambda source=int(ids[y, x]): csgraph.bellman_ford(graph, indices=source) for y, x in origins]

    def plot(ax, results):
        if results:
            costs = np.minimum.reduce(results).reshape(height, width)
            heatmap(ax, costs, cmap="magma", vmin=-100, vmax=650)
            ax.scatter(origins[: len(results), 1], origins[: len(results), 0], c="#68e2c2", s=32, edgecolors="#101824")
        else:
            ax.scatter(
                np.tile(np.arange(width), height)[::10],
                np.repeat(np.arange(height), width)[::10],
                c="#506179",
                s=0.25,
                linewidths=0,
            )
        ax.set(
            xlim=(0, width - 1),
            ylim=(0, height - 1),
            xticks=[],
            yticks=[],
            xlabel="50,000 nodes / 100,000 directed edges",
            ylabel="Minimum route cost, including credits",
        )
        ax.text(
            0.03,
            0.94,
            "11,571 negative edges / no negative cycles",
            transform=ax.transAxes,
            color="#f1f5fa",
            fontsize=8,
            va="top",
            bbox={"facecolor": "#101824", "alpha": 0.7, "edgecolor": "none"},
        )

    return Task(
        "bellman",
        "Bellman-Ford on a 50,000-node graph",
        "csgraph",
        "Find routes from two origins through a 50,000-node, 100,000-edge network "
        "with negative costs and no negative cycles.",
        "csgraph.bellman_ford(network, indices=origin)",
        "origins",
        jobs,
        plot,
    )


def temperature():
    axis = np.linspace(0, 1, 64)
    x, y = np.meshgrid(axis, axis)
    values = 16 + 6 * np.sin(4 * x) * np.cos(3 * y) + 8 * np.exp(-((x - 0.6) ** 2 + (y - 0.5) ** 2) / 0.04)
    model = interpolate.RegularGridInterpolator((axis, axis), values, method="pchip")
    xx, yy = np.meshgrid(np.linspace(0, 1, 160), np.linspace(0, 1, 160))
    queries = np.stack((yy, xx), axis=-1).reshape(8, -1, 2)
    jobs = [lambda q=q: model(q) for q in queries]

    def plot(ax, results):
        heatmap(ax, values, cmap="magma", vmin=10, vmax=30, extent=(0, 160, 0, 160), alpha=0.25)
        if results:
            filled = np.full((160, 160), np.nan)
            filled[: len(results) * 20] = np.concatenate(results).reshape(-1, 160)
            heatmap(ax, filled, cmap="magma", vmin=10, vmax=30, extent=(0, 160, 0, 160))
        ax.set(xlabel="Longitude", ylabel="Latitude", xticks=[], yticks=[])
        ax.text(0.03, 0.95, "Temperature / degrees C", transform=ax.transAxes, va="top", color="white", fontsize=9)

    return Task(
        "temperature",
        "Fill in a temperature map",
        "interpolate",
        "Resample a 64 x 64 weather grid to 160 x 160 with shape-preserving PCHIP interpolation.",
        'model(points)  # prebuilt RegularGridInterpolator(method="pchip")',
        "map strips",
        jobs,
        plot,
    )


def sensors():
    rng = np.random.default_rng(71)
    time = np.arange(1000) / 250
    recordings = np.asarray(
        [
            np.sin(2 * np.pi * 2 * time + i / 8) + 0.35 * np.sin(2 * np.pi * 37 * time) + rng.normal(0, 0.15, 1000)
            for i in range(128)
        ],
        dtype=np.float32,
    ).reshape(8, 16, 1000)
    sos = signal.butter(4, 10, fs=250, output="sos").astype(np.float32)
    jobs = [lambda block=block: np.stack([signal.sosfiltfilt(sos, epoch) for epoch in block]) for block in recordings]

    def plot(ax, results):
        for i, block in enumerate(recordings):
            ax.plot(time, block[0] * 0.3 + i, color="#506179", linewidth=0.5, alpha=0.55)
        for i, result in enumerate(results):
            ax.plot(time, result[0] * 0.3 + i, color="#68e2c2", linewidth=1.3)
        ax.set(xlim=(0, 2), ylim=(-0.7, 7.7), xlabel="Time / seconds", ylabel="Sensor epochs", yticks=[])

    return Task(
        "sensors",
        "Clean noisy sensor recordings",
        "signal",
        "Low-pass filter 128 four-second sensor epochs, preserving phase with forward-backward filtering.",
        "signal.sosfiltfilt(sos, epoch)",
        "epoch batches",
        jobs,
        plot,
    )


def calibration():
    rng = np.random.default_rng(72)
    truth = rng.normal(size=(8, 1000, 3, 1))
    transform = np.array([[2.0, 0.8, 0.2], [0, 0.7, 0.1], [0, 0, 1.3]])
    # Local calibration varies by point, so each sample has its own system.
    matrices = transform + np.triu(rng.normal(0, 0.08, (8, 1000, 3, 3)))
    measured = matrices @ truth
    jobs = [lambda a=a, b=b: linalg.solve_triangular(a, b) for a, b in zip(matrices, measured)]

    def plot(ax, results):
        raw = measured[:, ::10, :2, 0].reshape(-1, 2)
        ax.scatter(raw[:, 0], raw[:, 1], s=3, c="#506179", alpha=0.45, linewidths=0)
        if results:
            corrected = np.concatenate(results)[::10, :2, 0]
            ax.scatter(corrected[:, 0], corrected[:, 1], s=6, c="#68e2c2", alpha=0.7, linewidths=0)
        ax.set(xlim=(-6, 6), ylim=(-3, 3), xlabel="X coordinate", ylabel="Y coordinate")
        ax.text(0.04, 0.92, "Raw / gray     Calibrated / mint", transform=ax.transAxes, color="#a7b8ca", fontsize=9)

    return Task(
        "calibration",
        "Calibrate 8,000 3D points",
        "linalg",
        "Correct scale and cross-axis coupling with a different 3 x 3 calibration system for each point.",
        "linalg.solve_triangular(calibration_matrices, observed_points)",
        "point batches",
        jobs,
        plot,
    )


def uncertainty():
    rng = np.random.default_rng(73)
    x = np.arange(128)
    samples = rng.normal(0, 0.45, (30, 128)) + np.sin(x / 23) + x / 160
    blocks = np.split(samples, 8, axis=1)
    mean = samples.mean(axis=0)
    jobs = [lambda block=block: stats.sem(block, axis=0) for block in blocks]

    def plot(ax, results):
        ax.plot(x, mean, color="#506179", linewidth=1)
        if results:
            error = np.concatenate(results)
            end = error.size
            ax.fill_between(x[:end], mean[:end] - 2 * error, mean[:end] + 2 * error, color="#68e2c2", alpha=0.25)
            ax.plot(x[:end], mean[:end], color="#68e2c2", linewidth=1.3)
        ax.set(xlim=(0, 127), ylim=(-0.8, 2.0), xlabel="Sensor channel", ylabel="Mean +/- 2 SEM")

    return Task(
        "uncertainty",
        "Put uncertainty around measurements",
        "stats",
        "Compute standard errors for 128 sensor channels, each measured 30 times.",
        "stats.sem(repeated_measurements, axis=0)",
        "channel groups",
        jobs,
        plot,
    )


def expression():
    rng = np.random.default_rng(74)
    samples = np.linspace(0, 2 * np.pi, 32)
    profiles = np.array([np.sin(samples + phase) for phase in (0, 1.5, 3, 4.5)])
    panels, trees = [], []
    for _ in range(8):
        panel = np.repeat(profiles, 32, axis=0) + rng.normal(0, 0.15, (128, 32))
        panel = panel[rng.permutation(128)]
        panels.append(panel)
        trees.append(hierarchy.linkage(panel, method="average"))
    jobs = [lambda tree=tree: hierarchy.leaves_list(tree) for tree in trees]

    def plot(ax, results):
        canvas = np.concatenate(
            [panels[i][results[i]] if i < len(results) else np.full((128, 32), np.nan) for i in range(8)], axis=1
        )
        heatmap(ax, canvas, cmap="magma", vmin=-1.4, vmax=1.4)
        for col in range(32, 256, 32):
            ax.axvline(col - 0.5, color="#506179", linewidth=1)
        ax.set(xticks=[], yticks=[], xlabel="Eight expression panels / samples", ylabel="Genes in cluster order")

    return Task(
        "expression",
        "Order clustered gene-expression panels",
        "cluster",
        "Arrange 128 genes in each of eight expression panels using precomputed hierarchical clusters.",
        "hierarchy.leaves_list(linkage)",
        "panels",
        jobs,
        plot,
    )


def peaks():
    rng = np.random.default_rng(75)
    groups = []
    for group in range(8):
        traces = []
        for replicate in range(16):
            time = np.cumsum(rng.uniform(0.05, 0.15, 100 + replicate % 9))
            time = time / time[-1] * 10
            intensity = (1 + group * 0.1) * np.exp(-0.5 * ((time - 3 - group * 0.45) / 0.8) ** 2)
            traces.append((time, intensity))
        groups.append(traces)
    jobs = [lambda traces=traces: np.array([integrate.simpson(y, x=x) for x, y in traces]) for traces in groups]

    def plot(ax, results):
        for traces in groups:
            time, intensity = traces[0]
            ax.plot(time, intensity, color="#506179", linewidth=0.7, alpha=0.5)
        for i, _ in enumerate(results):
            time, intensity = groups[i][0]
            color = (0.4 + i * 0.04, 0.88, 0.76 - i * 0.05)
            ax.fill_between(time, 0, intensity, color=color, alpha=0.12)
            ax.plot(time, intensity, color=color, linewidth=1.2)
        total = sum(result.sum() for result in results)
        ax.text(0.04, 0.91, f"Integrated area: {total:.1f}", transform=ax.transAxes, color="#a7b8ca", fontsize=9)
        ax.set(xlim=(0, 10), ylim=(0, 2), xlabel="Retention time / minutes", ylabel="Peak intensity")

    return Task(
        "peaks",
        "Measure chromatography peak areas",
        "integrate",
        "Integrate 128 chromatograms with different, irregularly sampled time coordinates.",
        "integrate.simpson(intensity, x=retention_time)",
        "trace batches",
        jobs,
        plot,
    )


def systems():
    rng = np.random.default_rng(76)
    groups = []
    for _ in range(8):
        blocks = []
        for _ in range(24):
            value = rng.normal(size=(16, 16)) * (rng.random((16, 16)) < 0.12) + 3 * np.eye(16)
            blocks.append(sparse.csr_array(value))
        groups.append(blocks)
    jobs = [lambda blocks=blocks: sparse.block_diag(blocks, format="csr") for blocks in groups]

    def plot(ax, results):
        canvas = np.full((128, 256), np.nan)
        for i, result in enumerate(results):
            row, col = (i // 4) * 64, (i % 4) * 64
            canvas[row : row + 64, col : col + 64] = result[:64, :64].toarray()
        ax.imshow(canvas, cmap="magma", vmin=-1, vmax=4, interpolation="nearest", aspect="auto")
        for col in (63.5, 127.5, 191.5):
            ax.axvline(col, color="#506179", linewidth=1)
        ax.axhline(63.5, color="#506179", linewidth=1)
        ax.set(xticks=[], yticks=[], xlabel="First 64 rows and columns of each system")

    return Task(
        "systems",
        "Build independent sparse systems",
        "sparse",
        "Assemble eight block-diagonal models from 192 small local coupling matrices.",
        'sparse.block_diag(local_models, format="csr")',
        "systems",
        jobs,
        plot,
    )


def routes():
    side = 120
    blocked = np.zeros((side, side), dtype=bool)
    for col in (25, 55, 85):
        blocked[:, col : col + 2] = True
        blocked[12 + col // 2 : 26 + col // 2, col : col + 2] = False
    ids = np.arange(side * side).reshape(side, side)
    rows = np.concatenate((ids[:, :-1].ravel(), ids[:-1, :].ravel()))
    cols = np.concatenate((ids[:, 1:].ravel(), ids[1:, :].ravel()))
    keep = ~blocked.ravel()[rows] & ~blocked.ravel()[cols]
    graph = sparse.csr_matrix((np.ones(keep.sum()), (rows[keep], cols[keep])), shape=(side * side,) * 2)
    depots = np.array([[8, 8], [110, 110], [15, 110], [105, 8], [45, 40], [70, 70], [8, 65], [110, 45]])
    jobs = [
        lambda source=int(ids[y, x]): csgraph.shortest_path(graph, directed=False, unweighted=True, indices=source)
        for y, x in depots
    ]

    def plot(ax, results):
        heatmap(ax, blocked.astype(float), cmap="gray", vmin=0, vmax=1)
        if results:
            costs = np.min(results, axis=0).reshape(side, side)
            heatmap(ax, np.ma.masked_where(blocked | ~np.isfinite(costs), costs), cmap="magma", vmin=0, vmax=140)
            ax.scatter(depots[: len(results), 1], depots[: len(results), 0], c="#68e2c2", s=25, edgecolors="#101824")
        ax.set(xlabel="Warehouse aisle", ylabel="Walking distance to nearest depot", xticks=[], yticks=[])

    return Task(
        "routes",
        "Map walking distances to depots",
        "csgraph",
        "Find unweighted shortest paths from eight depots through a 120 x 120 warehouse with aisle barriers.",
        "csgraph.shortest_path(aisles, directed=False, unweighted=True, indices=depot)",
        "depots",
        jobs,
        plot,
    )


TASKS = (bellman, temperature, sensors, calibration, uncertainty, expression, peaks, systems, routes)
