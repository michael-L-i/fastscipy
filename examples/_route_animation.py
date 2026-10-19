"""Route-planner replay of measured Bellman-Ford completion times.

Distances come from the actual solver. Route extraction and moving markers are
presentation only, outside the benchmark; they do not depict algorithm progress.
"""

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG, PANEL, INK, MUTED = "#101824", "#192536", "#f1f5fa", "#a7b8ca"
ROAD, CREDIT, ROUTE, TARGET = "#34475c", "#eab56b", "#68e2c2", "#f28e8e"


def endpoints(draw, route, font):
    for label, position, color in (("S", route[0], ROUTE), ("D", route[-1], TARGET)):
        x, y = position
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color)
        draw.text((x, y), label, fill=BG, font=font, anchor="mm")


def shortest_route(incoming, distances, start, target):
    """Recover a route using tight incoming edges, without rerunning a solver."""
    route = [target]
    while route[-1] != start:
        node = route[-1]
        lo, hi = incoming.indptr[node : node + 2]
        candidates = incoming.indices[lo:hi]
        tight = distances[candidates] + incoming.data[lo:hi] == distances[node]
        if not tight.any() or len(route) > incoming.shape[0]:
            raise RuntimeError("Could not recover the measured shortest route")
        route.append(int(candidates[np.flatnonzero(tight)[0]]))
    return np.array(route[::-1])


def route_cards(graph, origins, destinations, shape, results, fonts):
    incoming = graph.tocsc()
    cards = []
    for origin, target, distances in zip(origins, destinations, results):
        start_id, target_id = (int(np.ravel_multi_index(p, shape)) for p in (origin, target))
        route = shortest_route(incoming, distances, start_id, target_id)
        # Check each displayed connection and its cost against the actual graph.
        weights = np.array([graph[a, b] for a, b in zip(route, route[1:])])
        if float(weights.sum()) != distances[target_id]:
            raise RuntimeError("Displayed route cost differs from the solver result")
        rows, cols = np.unravel_index(route, shape)
        x0, x1 = max(0, cols.min() - 2), min(shape[1] - 1, cols.max() + 2)
        y0, y1 = max(0, rows.min() - 2), min(shape[0] - 1, rows.max() + 2)
        scale = min(176 / max(1, x1 - x0), 140 / max(1, y1 - y0))
        offset = (106 - (x1 - x0) * scale / 2, 91 + (y1 - y0) * scale / 2)

        def point(row, col):
            return offset[0] + (col - x0) * scale, offset[1] - (row - y0) * scale

        card = Image.new("RGB", (212, 200), "#142030")
        draw = ImageDraw.Draw(card)
        for row in range(int(y0), int(y1) + 1):
            for col in range(int(x0), int(x1) + 1):
                node = row * shape[1] + col
                for at in range(graph.indptr[node], graph.indptr[node + 1]):
                    other_row, other_col = divmod(int(graph.indices[at]), shape[1])
                    if y0 <= other_row <= y1 and x0 <= other_col <= x1:
                        draw.line(
                            (point(row, col), point(other_row, other_col)),
                            fill=CREDIT if graph.data[at] < 0 else ROAD,
                            width=1,
                        )
        path = [point(row, col) for row, col in zip(rows, cols)]
        endpoints(draw, path, fonts[12])
        draw.text((12, 183), "Local street detail", fill=MUTED, font=fonts[12])
        cards.append((card, path, float(distances[target_id])))
    return cards


def render_route_planner(graph, origins, destinations, shape, results, record, path):
    fonts = {n: ImageFont.load_default(size=n) for n in (12, 14, 15, 16, 17, 19, 23, 29, 40)}
    cards = route_cards(graph, origins, destinations, shape, results, fonts)
    end_ms = math.ceil(max(record["scipy_seconds"], record["fastscipy_seconds"]) * 100) * 10
    events = record["scipy_completed_seconds"] + record["fastscipy_completed_seconds"]
    ticks = sorted({0, end_ms, *range(100, end_ms, 100), *(math.ceil(t * 100) * 10 for t in events)})
    durations = np.diff(ticks).tolist() + [1400]

    def frame(tick):
        elapsed = tick / 1000
        image = Image.new("RGB", (1000, 650), BG)
        draw = ImageDraw.Draw(image)
        draw.text((32, 22), "Plan routes across 50,000 nodes", fill=INK, font=fonts[29])
        draw.text(
            (32, 66), "Bellman-Ford  /  100,000 directed edges  /  negative edge costs", fill=MUTED, font=fonts[17]
        )
        for left, label, key in ((24, "SciPy", "scipy"), (512, "fastscipy / 2-thread budget", "fastscipy")):
            stamps = record[key + "_completed_seconds"]
            done = sum(t <= elapsed for t in stamps)
            ready = done == len(cards)
            draw.rounded_rectangle((left, 105, left + 464, 560), radius=14, fill=PANEL)
            draw.text((left + 16, 119), label, fill=INK, font=fonts[23])
            clock = record[key + "_seconds"] if ready else elapsed
            time_label = f"{clock * 1000:.2f} ms" if ready and clock < 1 else f"{clock:.2f} s"
            draw.text((left + 16, 153), time_label, fill=ROUTE if ready else INK, font=fonts[40])
            status = "ROUTES READY" if ready else "FINDING ROUTES"
            draw.text((left + 274, 165), status, fill=ROUTE if ready else MUTED, font=fonts[14])
            draw.text((left + 274, 186), f"{done} / {len(cards)} origins solved", fill=MUTED, font=fonts[14])
            if not ready:
                angle = elapsed * 240 % 360
                draw.arc((left + 246, 164, left + 264, 182), angle, angle + 240, fill=CREDIT, width=2)
            for index, (base, route, cost) in enumerate(cards):
                x, y = left + 12 + index * 228, 240
                draw.text((x + 2, 216), f"Route {'AB'[index]}  /  S to D", fill=INK, font=fonts[16])
                card = base.copy()
                if index < done:
                    overlay = ImageDraw.Draw(card)
                    overlay.line(route, fill=ROUTE, width=3)
                    # Moving markers illustrate a completed route, not compute progress.
                    position = max(0, elapsed - stamps[index]) * 12 % (len(route) - 1)
                    step = int(position)
                    blend = position - step
                    px, py = ((1 - blend) * a + blend * b for a, b in zip(route[step], route[step + 1]))
                    overlay.ellipse((px - 4, py - 4, px + 4, py + 4), fill=INK, outline=ROUTE, width=2)
                    endpoints(overlay, route, fonts[12])
                image.paste(card, (x, y))
                draw.text(
                    (x + 2, y + 210),
                    f"Net cost: {cost:+.0f}" if index < done else "Waiting for solver...",
                    fill=ROUTE if index < done else MUTED,
                    font=fonts[14],
                )
            # A full-network locator anchors both zoomed street views.
            ox, oy = left + 18, 487
            draw.rectangle((ox, oy, ox + 80, oy + 58), fill="#142030", outline=ROAD)
            for col in range(10, 80, 10):
                draw.line((ox + col, oy, ox + col, oy + 58), fill=ROAD)
            for row in range(10, 58, 10):
                draw.line((ox, oy + row, ox + 80, oy + row), fill=ROAD)
            for row, col in origins:
                px, py = ox + col / shape[1] * 80, oy + (1 - row / shape[0]) * 58
                draw.rectangle((px - 3, py - 3, px + 4, py + 4), outline=ROUTE, width=1)
            draw.text((left + 114, 489), "Full 50,000-node network", fill=MUTED, font=fonts[14])
            distance_status = "All-node distances calculated" if ready else "Solving all-node distances"
            draw.text((left + 114, 512), distance_status, fill=MUTED, font=fonts[14])
            draw.text((left + 114, 534), "Amber streets carry negative costs", fill=CREDIT, font=fonts[12])
        speed = f"{record['speedup']:,.1f}x faster"
        draw.text((32, 581), speed, fill=ROUTE, font=fonts[29])
        draw.text(
            (365, 590),
            f"{record['scipy_seconds']:.2f} s  ->  {record['fastscipy_seconds'] * 1000:.2f} ms",
            fill=INK,
            font=fonts[19],
        )
        draw.text(
            (32, 625),
            "Recorded solver clocks, replayed at real time. Markers illustrate completed routes.",
            fill=MUTED,
            font=fonts[15],
        )
        return image

    sample = Image.new("RGB", (1000, 1950))
    for i, tick in enumerate((0, end_ms // 2, end_ms)):
        sample.paste(frame(tick), (0, i * 650))
    palette = sample.quantize(colors=128)
    frames = [frame(tick).quantize(palette=palette, dither=Image.Dither.NONE) for tick in ticks]
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    record.update(
        {
            "replay_slowdown": end_ms / 1000 / record["scipy_seconds"],
            "replay_seconds": end_ms / 1000,
            "replay_quantum_ms": 10,
            "gif_bytes": path.stat().st_size,
            "visualization": (
                "Real-time route planner. Displayed paths are recovered from measured all-node distances. "
                "Moving markers illustrate completed routes, not solver progress."
            ),
        }
    )
