# Intel Mac cast-selector experiment

[Diagnostic run 39446755156](https://github.com/michael-L-i/fastscipy/actions/runs/39446755156)
used probe checkout `62147de` with the unchanged installed `daf9acd` native
wheel from run 39376214465. The installed native and interpolation Python
files were independently hash-checked against that wheel.

The sole dispatch change was `_SCIPY_INT_CAST=False` to `True`; the probe
restored `False` in `finally`. All 64 existing spline/affine interpolation
tests passed, including all 18 previously missing fast paths. The monitored
run took 4.68 seconds and peaked at 116.4 MiB.

This verifies the selector on these native selected tests. It is not a
rebuilt-wheel or full-project certification and does not resolve independent
numerical failures or the BFGS benchmark difference. Original JUnit, worker
report, monitor, log, and workflow provenance are retained unchanged.
[sources.json](sources.json) records hashes and the exact experiment boundary.
