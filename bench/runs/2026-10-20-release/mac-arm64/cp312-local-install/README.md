# Final CP312 Mac ARM artifact: local installation and application pilots

This clean-environment check installed the actual `v0.1.0a2` CP312 Mac ARM
wheel artifact from [release run 39464216282](https://github.com/michael-L-i/fastscipy/actions/runs/39464216282),
source `1cf6779`. The wheel SHA256 is
`97e6e89b8a0a5f465d9525e76369b16df44cb5acdb973cfa8c73061b95f45278`,
independently matched against the local file and pip's install report.

**This was a direct release-artifact install, not a PyPI download check.**
It does not establish publication, a complete 32-wheel result, or other
platform/Python behavior. The wheel was built on macOS15/CPython3.12.10;
these local checks used macOS26.6.2 on an M5 Pro, Homebrew CPython3.12.12,
NumPy2.5.4 and SciPy1.18.1, with two native and one BLAS thread.

All 13 components activated under default guards. Image cleanup/segmentation
and sensor filtering/peak-feature pipelines passed byte comparisons against
unpatched SciPy and independent properties. The sensor/statistics pilot
checked independent OLS/signal properties, unchanged inputs, sustained
repetitions, and NaN-omit fallback; the custom metric control used no fast
paths. Dependency checking reported no broken requirements. These are
bounded application pilots, not coverage of arbitrary downstream programs.

The monitored installation peaked at 137.95 MiB; the three component/pilot
checks peaked at approximately 95.75–101.59 MiB. All recorded commands
returned zero without resource stops. The configured512MiB cap is identified
as parent-reported; original result JSON preserves sampled peaks. Raw pip
metadata, logs, local monitoring results, and the wheel's cloud build/summary
are retained unchanged. [sources.json](sources.json) records file hashes and
keeps local testing, cloud build, and pending publication evidence distinct.
