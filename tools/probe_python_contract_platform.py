"""Selected Intel Python-source tests with an unchanged reused native binary.

This tests-only diagnostic shares the source/native/import guards. It runs no
benchmarks or upstream comparisons and cannot certify a release wheel.
"""

from probe_python_source_platform import main


if __name__ == "__main__":
    raise SystemExit(main(tests_only=True))
