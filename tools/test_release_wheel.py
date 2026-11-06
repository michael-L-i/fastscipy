"""Run the installed-wheel gate selected by the build interpreter.

Invoked with -I by cibuildwheel so the checkout cannot supply fastscipy.
Python 3.9–3.11 retain their scoped compatibility qualification; every newer
interpreter runs the project suite, and Python 3.13 also runs all benchmark
areas and paired upstream suites.
"""

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def commands(output, version):
    prefix = [sys.executable, "-I"]
    if version < (3, 12):
        return [[*prefix, str(ROOT / "tools/compatibility_release.py"), "--output", str(output)]]
    return [
        [*prefix, str(ROOT / "tools/check_release.py"), "--installed"],
        [
            *prefix,
            str(ROOT / "tools/platform_suite.py"),
            "--output",
            str(output),
            "--max-rss-mb",
            "4096",
            "--upstream" if version == (3, 13) else "--tests-only",
        ],
    ]


def main():
    reports = Path(os.environ.get("FASTSCIPY_RELEASE_REPORT_DIR", ROOT / "artifacts"))
    output = reports / sys.implementation.cache_tag
    for command in commands(output, sys.version_info[:2]):
        print("Installed release gate:", command, flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
