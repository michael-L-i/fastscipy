"""Partial Intel diagnostic: selected tests and the remaining 95 special/stats cases.

Reuses the shared source/native/import guards. This does not run all project
tests or all 407 benchmark cases and cannot certify a release wheel.
"""

from probe_python_source_platform import main


if __name__ == "__main__":
    raise SystemExit(main(targeted=True))
