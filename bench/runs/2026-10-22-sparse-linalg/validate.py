import subprocess
import sys

commands = [
    [sys.executable, "-m", "pytest", "tests/test_sparse_linalg.py", "-q"],
    [
        sys.executable,
        "bench/run.py",
        "sparse_linalg",
        "--budget",
        "0.03",
        "--check",
        "--output-dir",
        "bench/runs/2026-10-22-sparse-linalg/benchmark",
    ],
    [
        sys.executable,
        "tools/scipy_suite.py",
        "--output-dir",
        "bench/runs/2026-10-22-sparse-linalg/upstream",
        "--pyargs",
        "scipy.sparse.linalg.tests.test_interface",
        "scipy.sparse.linalg._isolve.tests.test_iterative",
        "-q",
    ],
]
for command in commands:
    print("RUN", command, flush=True)
    subprocess.run(command, check=True)
