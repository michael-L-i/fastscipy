"""Run every area serially, saving correctness, timings, and bounded process-tree RSS.

Requires psutil. Example: python tools/platform_suite.py --output artifacts/platform
"""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import psutil
import scipy

ROOT = Path(__file__).resolve().parent.parent


def run(command, log, memory_mb, timeout):
    started, peak, stopped = time.monotonic(), 0, None
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        while process.poll() is None:
            try:
                monitored = psutil.Process(process.pid)
                tree = [monitored, *monitored.children(recursive=True)]
                rss = 0
                for child in tree:
                    try:
                        rss += child.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                peak = max(peak, rss)
                if rss > memory_mb * 1024**2 or time.monotonic() - started > timeout:
                    stopped = "memory limit" if rss > memory_mb * 1024**2 else "timeout"
                    for child in reversed(tree):
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(0.2)
        code = process.wait()
    result = {"returncode": code, "seconds": time.monotonic() - started, "peak_rss_bytes": peak, "stopped": stopped}
    print(f"{log.stem}: exit={code}, {result['seconds']:.1f}s, peak={peak / 1024**2:.0f} MiB", flush=True)
    return result


def mean(values):
    return math.exp(sum(math.log(x) for x in values) / len(values)) if values else None


def revision():
    if value := os.environ.get("FASTSCIPY_BUILD_REVISION"):
        return value
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=0.1)
    parser.add_argument("--max-rss-mb", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=2400, help="Seconds per test run or benchmark area")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--bench-only", action="store_true")
    mode.add_argument("--tests-only", action="store_true")
    parser.add_argument("--upstream", action="store_true")
    args = parser.parse_args()
    if args.budget <= 0 or args.max_rss_mb <= 0 or args.timeout <= 0:
        parser.error("Timing, memory, and timeout budgets must be positive")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Choose an empty output directory so stale results cannot enter this report")
    output.mkdir(parents=True, exist_ok=True)
    for key in ("FASTSCIPY_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(key, "2" if key == "FASTSCIPY_NUM_THREADS" else "1")
    report = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_commit": revision(),
        "fastscipy": importlib.metadata.version("fastscipy"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numpy_config": np.__config__.show(mode="dicts"),
        "thread_budget": int(os.environ["FASTSCIPY_NUM_THREADS"]),
        "blas_thread_budget": {
            key: os.environ[key] for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
        },
        "timing_budget_seconds": args.budget,
        "memory_limit_mb": args.max_rss_mb,
        "runs": {},
        "areas": {},
    }

    def execute(name, arguments):
        result = run([sys.executable, *arguments], output / f"{name}.log", args.max_rss_mb, args.timeout)
        report["runs"][name] = result
        (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return result

    if not args.bench_only:
        xml = output / "tests.xml"
        execute("tests", ["-m", "pytest", "tests", "-q", "--tb=short", f"--junitxml={xml}"])
        if xml.exists():
            report["tests"] = [dict(suite.attrib) for suite in ET.parse(xml).getroot().iter("testsuite")]
    if not args.tests_only:
        for path in sorted((ROOT / "bench").glob("bench_*.py")):
            area = path.stem.removeprefix("bench_")
            execute(area, ["bench/run.py", area, "--budget", str(args.budget), "--output-dir", str(output / "cases")])
            data = output / "cases" / f"{area}.json"
            if not data.exists():
                continue
            rows = json.loads(data.read_text())["cases"]
            accelerated = [row for row in rows.values() if row["expected_fast_path"] is not False]
            report["areas"][area] = {
                "cases": len(rows),
                "controls": len(rows) - len(accelerated),
                "nonidentical": [name for name, row in rows.items() if not row["identical"]],
                "missing_fast_path": [name for name, row in rows.items() if not row["fast_path"]],
                "speedup_1t": mean([row["x1t"] for row in accelerated]),
                "speedup_nt": mean([row["xnt"] for row in accelerated]),
                "all_case_speedup_1t": mean([row["x1t"] for row in rows.values()]),
                "all_case_speedup_nt": mean([row["xnt"] for row in rows.values()]),
            }
    if args.upstream:
        execute("upstream", ["tools/scipy_suite.py", "--pyargs", "scipy", "-n", "2"])
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    bad = any(run["returncode"] != 0 for run in report["runs"].values())
    bad |= any(area["nonidentical"] or area["missing_fast_path"] for area in report["areas"].values())
    sys.exit(bool(bad))


if __name__ == "__main__":
    main()
