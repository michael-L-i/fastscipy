"""Inventory public SciPy callable exports and fastscipy's direct patch hooks.

Run in a fresh process: python tools/coverage_inventory.py --output PATH
An unchanged export may still benefit from patched internals; a changed export
may decline most inputs. These counts measure hook coverage, not speedups.
"""

import argparse
import importlib
import inspect
import json
import pathlib
import warnings

import scipy

import fastscipy
from fastscipy import _patch


def inventory():
    if fastscipy.status():
        raise RuntimeError("Run the inventory in a fresh, unpatched process")
    names = [name for name in scipy.__all__ if not name.startswith("_")]
    names.extend(
        (
            "sparse.linalg",
            "sparse.csgraph",
            "cluster.vq",
            "cluster.hierarchy",
            "spatial.distance",
            "signal.windows",
            "stats.qmc",
            "stats.sampling",
            "linalg.interpolative",
        )
    )
    exports = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for name in sorted(set(names)):
            try:
                module = importlib.import_module(f"scipy.{name}")
            except ModuleNotFoundError:
                continue
            exports[name] = {
                key: value
                for key in getattr(module, "__all__", dir(module))
                if not key.startswith("_") and callable(value := getattr(module, key))
            }
    status = fastscipy.patch(threads=1)
    owners = {owner for _, owner, _, _ in _patch._records if inspect.isclass(owner)}
    areas = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for name, before in exports.items():
            module = importlib.import_module(f"scipy.{name}")
            rows = []
            for key, old in sorted(before.items()):
                new = getattr(module, key)
                cls = old if inspect.isclass(old) else type(old)
                method_hook = any(base in owners for base in cls.__mro__)
                hook = "rebound" if new is not old else "class_method" if method_hook else "unchanged_export"
                kind = "class" if inspect.isclass(old) else "function" if inspect.isroutine(old) else "callable_object"
                rows.append({"name": key, "kind": kind, "hook": hook})
            counts = {
                hook: sum(r["hook"] == hook for r in rows) for hook in ("rebound", "class_method", "unchanged_export")
            }
            areas[name] = {"exports": len(rows), "counts": counts, "callables": rows}
    hooks = {}
    for component, owner, attribute, _ in _patch._records:
        owner_name = getattr(owner, "__module__", "")
        label = getattr(owner, "__qualname__", getattr(owner, "__name__", type(owner).__name__))
        hooks.setdefault(component, set()).add(f"{owner_name + '.' if owner_name else ''}{label}.{attribute}")
    return {
        "scipy_version": scipy.__version__,
        "interpretation": (
            "Public callable exports in the listed namespaces, including aliases and classes. "
            "Direct hooks are not a count of accelerated input variants. Unchanged exports "
            "may benefit from internal hooks; see component_hooks and docs/coverage.md."
        ),
        "patch_status": status,
        "component_hooks": {component: sorted(records) for component, records in hooks.items()},
        "areas": areas,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    data = inventory()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=1) + "\n")
    for name, area in data["areas"].items():
        hooked = area["counts"]["rebound"] + area["counts"]["class_method"]
        print(f"{name}: {area['exports']} public callable exports, {hooked} directly hooked")


if __name__ == "__main__":
    main()
