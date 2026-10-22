# Clean installed a4 application pilots

[Run 39793298085](https://github.com/michael-L-i/fastscipy/actions/runs/39793298085) passed all eight native platform variants using clean, binary-only PyPI installations of `fastscipy==0.1.0a4` on CPython 3.13. Each platform passed all installed component checks, image/signal and sensor/statistics application pilots, and `pip check`.

The pilot checkout was `1a3eb809c0cd29be421b6f0ea38c247ac3a24d19`; the published release source is `46b33f2882f0b10a126169bed0a088705cf402a0`. Only publication evidence and workflow comments differ. `audit.json` derives wheel hashes and sampled application peaks from the retained per-platform installation and process reports. Linux containers had a 1 GiB limit; application children had a 512 MiB process-tree limit, two native threads and one BLAS thread. Installation had a separate 1 GiB process limit. Alpine installed runtime `libstdc++` without a compiler.

These small application pilots complement the 32-wheel release validation. They are not benchmarks or a new claim of byte identity. Original reports, logs, install manifests and any numerical compatibility records are retained unchanged.
