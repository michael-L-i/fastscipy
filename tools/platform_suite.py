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
AREAS = {
    "cluster",
    "csgraph",
    "fft",
    "integrate",
    "interpolate",
    "linalg",
    "ndimage",
    "optimize",
    "signal",
    "sparse",
    "sparse_linalg",
    "spatial",
    "special",
    "stats",
}


def run(command, log, memory_mb, timeout, *, allow_privileged_installer=False):
    """Monitor readable RSS; only opted-in Darwin installer children may deny it.

    A denied installer sample makes the recorded peak a lower bound. Root,
    compiler, unknown-process and process-enumeration denials fail closed.
    """
    started, peak, stopped = time.monotonic(), 0, None
    denied, cleanup_errors, tree = {}, [], []
    monitor_error = unexpected = None
    installer_names = {"sudo", "installer", "installd", "package_script_service", "system_installd"}
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)

        def access_denied(child, operation):
            nonlocal monitor_error, stopped
            try:
                name = child.name()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                name = None
            allowed = (
                allow_privileged_installer
                and sys.platform == "darwin"
                and child.pid != process.pid
                and operation == "memory_info"
                and name in installer_names
            )
            elapsed = time.monotonic() - started
            record = denied.setdefault(
                child.pid,
                {"pid": child.pid, "name": name, "first_denied_seconds": elapsed, "denied_samples": 0},
            )
            record.update(last_denied_seconds=elapsed, allowed_installer=allowed, operation=operation)
            record["denied_samples"] += 1
            if not allowed:
                stopped = "monitor access denied"
                monitor_error = {"type": "AccessDenied", "pid": child.pid, "name": name, "operation": operation}

        try:
            while process.poll() is None:
                rss = 0
                try:
                    monitored = psutil.Process(process.pid)
                    tree = [monitored]
                    try:
                        tree.extend(monitored.children(recursive=True))
                    except psutil.AccessDenied:
                        access_denied(monitored, "children")
                    for child in tree:
                        try:
                            rss += child.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                        except psutil.AccessDenied:
                            access_denied(child, "memory_info")
                except psutil.NoSuchProcess:
                    pass
                peak = max(peak, rss)
                if rss > memory_mb * 1024**2:
                    stopped = "memory limit"
                elif time.monotonic() - started > timeout and stopped is None:
                    stopped = "timeout"
                if stopped is not None:
                    break
                time.sleep(0.2)
        except BaseException as error:
            # Preserve unexpected failures (including interrupts) after cleanup.
            unexpected = error
            stopped = "monitor error"
            monitor_error = {"type": type(error).__name__, "message": str(error)}

        if stopped is not None:
            # Refresh descendants, retaining the last snapshot if enumeration fails.
            descendants = {child.pid: child for child in tree if child.pid != process.pid}
            try:
                descendants.update({child.pid: child for child in psutil.Process(process.pid).children(recursive=True)})
            except psutil.NoSuchProcess:
                pass
            except Exception as error:
                cleanup_errors.append({"pid": process.pid, "operation": "children", "error": repr(error)})
            for child in reversed(list(descendants.values())):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
                except Exception as error:
                    cleanup_errors.append({"pid": child.pid, "operation": "kill", "error": repr(error)})
            try:
                process.kill()
            except ProcessLookupError:
                pass
            except Exception as error:
                cleanup_errors.append({"pid": process.pid, "operation": "kill", "error": repr(error)})
        try:
            code = process.wait(timeout=10)
        except subprocess.TimeoutExpired as error:
            cleanup_errors.append({"pid": process.pid, "operation": "wait", "error": repr(error)})
            code = 1
        result = {
            "returncode": code if code != 0 or stopped is None else 1,
            "seconds": time.monotonic() - started,
            "peak_rss_bytes": peak,
            "stopped": stopped,
            "rss_scope": "sampled_readable_process_tree",
            "rss_complete": not denied and monitor_error is None,
            "peak_rss_is_lower_bound": bool(denied) or monitor_error is not None,
            "inaccessible_pids": sorted(denied),
            "inaccessible_processes": list(denied.values()),
            "cleanup_errors": cleanup_errors,
        }
        if monitor_error is not None:
            result["monitor_error"] = monitor_error
        if cleanup_errors:
            result["returncode"] = result["returncode"] or 1
        log.with_suffix(log.suffix + ".monitor.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    scope = "measured lower bound" if denied else "sampled tree"
    print(
        f"{log.stem}: exit={result['returncode']}, {result['seconds']:.1f}s, "
        f"peak={peak / 1024**2:.0f} MiB ({scope}), stopped={stopped}",
        flush=True,
    )
    if unexpected is not None:
        raise unexpected.with_traceback(unexpected.__traceback__)
    return result


def validation_record_groups(records):
    """Keep compatibility checks separate from observed reference variation."""
    return {
        "reference_variance": [r for r in records if r.get("policy") == "intel-accelerate-reference-variance"],
        "numerical_compatibility": [
            r for r in records if r.get("policy") == "intel-accelerate-numerical-compatibility"
        ],
        "validation_records": records,
    }


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
    from check_native_build import REBUILD, inspect_build

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=0.1)
    parser.add_argument("--max-rss-mb", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=2400, help="Seconds per test run or benchmark area")
    parser.add_argument(
        "--upstream-timeout", type=int, default=14400, help="Seconds for the complete serial paired upstream suite"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--bench-only", action="store_true")
    mode.add_argument("--tests-only", action="store_true")
    parser.add_argument("--upstream", action="store_true")
    args = parser.parse_args()
    if args.budget <= 0 or args.max_rss_mb <= 0 or args.timeout <= 0 or args.upstream_timeout <= 0:
        parser.error("Timing, memory, and timeout budgets must be positive")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Choose an empty output directory so stale results cannot enter this report")
    output.mkdir(parents=True, exist_ok=True)
    benchmark_files = sorted((ROOT / "bench").glob("bench_*.py"))
    if not args.tests_only and {path.stem.removeprefix("bench_") for path in benchmark_files} != AREAS:
        parser.error(f"Run from a complete source checkout containing benchmarks for all {len(AREAS)} areas")
    for key in ("FASTSCIPY_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(key, "2" if key == "FASTSCIPY_NUM_THREADS" else "1")
    native_build = inspect_build(ROOT)
    (output / "native-build.json").write_text(json.dumps(native_build, indent=2), encoding="utf-8")
    if native_build["status"] != "match":
        parser.error(f"Native build check: {native_build['status']}. {REBUILD}")
    os.environ.setdefault("SCIPY_AVAILABLE_MEM", f"{max(1, args.max_rss_mb // 2)}Mib")
    reference_report = output.resolve() / "reference-variance.jsonl"
    os.environ["FASTSCIPY_REFERENCE_REPORT"] = str(reference_report)
    report = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_commit": revision(),
        "native_build": native_build,
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
        "scipy_available_memory": os.environ["SCIPY_AVAILABLE_MEM"],
        "upstream_timeout_seconds": args.upstream_timeout,
        "runs": {},
        "areas": {},
    }

    def execute(name, arguments, timeout=None):
        result = run(
            [sys.executable, *arguments],
            output / f"{name}.log",
            args.max_rss_mb,
            args.timeout if timeout is None else timeout,
        )
        report["runs"][name] = result
        (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return result

    if not args.bench_only:
        xml = output / "tests.xml"
        execute("tests", ["-m", "pytest", "tests", "-q", "--tb=short", f"--junitxml={xml}"])
        if xml.exists():
            report["tests"] = [dict(suite.attrib) for suite in ET.parse(xml).getroot().iter("testsuite")]
    if not args.tests_only:
        for path in benchmark_files:
            area = path.stem.removeprefix("bench_")
            execute(area, ["bench/run.py", area, "--budget", str(args.budget), "--output-dir", str(output / "cases")])
            data = output / "cases" / f"{area}.json"
            if not data.exists():
                continue
            rows = json.loads(data.read_text())["cases"]
            accelerated = [row for row in rows.values() if row["expected_fast_path"] is not False and row["identical"]]
            report["areas"][area] = {
                "cases": len(rows),
                "controls": sum(row["expected_fast_path"] is False for row in rows.values()),
                "nonidentical": [name for name, row in rows.items() if not row["identical"]],
                "reference_variance": [
                    name for name, row in rows.items() if row.get("validation") == "reference_variance"
                ],
                "unexplained_nonidentical": [
                    name
                    for name, row in rows.items()
                    if not row["identical"] and row.get("validation") != "reference_variance"
                ],
                "missing_fast_path": [name for name, row in rows.items() if not row["fast_path"]],
                "speedup_1t": mean([row["x1t"] for row in accelerated]),
                "speedup_nt": mean([row["xnt"] for row in accelerated]),
                "all_case_speedup_1t": mean([row["x1t"] for row in rows.values()]),
                "all_case_speedup_nt": mean([row["xnt"] for row in rows.values()]),
            }
    if args.upstream:
        execute(
            "upstream",
            [
                "tools/scipy_suite.py",
                "--partitioned",
                "--output-dir",
                str(output / "upstream-junit"),
                "--pyargs",
                "scipy",
            ],
            args.upstream_timeout,
        )
    records = (
        [json.loads(line) for line in reference_report.read_text(encoding="utf-8").splitlines()]
        if reference_report.exists()
        else []
    )
    report.update(validation_record_groups(records))

    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    bad = any(run["returncode"] != 0 for run in report["runs"].values())
    if not args.tests_only:
        bad |= report["areas"].keys() != AREAS
    bad |= any(area["unexplained_nonidentical"] or area["missing_fast_path"] for area in report["areas"].values())
    sys.exit(bool(bad))


if __name__ == "__main__":
    main()
