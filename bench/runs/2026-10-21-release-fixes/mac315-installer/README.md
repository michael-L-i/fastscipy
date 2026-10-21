# Official Mac Python 3.15 installer monitor smoke

[Run 39679702689](https://github.com/michael-L-i/fastscipy/actions/runs/39679702689),
source `e057c24`, passed on both macOS15 architectures. Each job downloaded
the official Python3.15.0 macos11 package and used cibuildwheel4.3.1's exact
`free-threaded-enable-315.xml` installer choices, matching the earlier failed
release command. Both package/choices hashes agree across runners. Probe and
monitor source hashes independently match the recorded Git revision.

Both jobs verified the ordinary framework Python3.15.0 executable at the
expected native architecture, with `SOABI=cpython-315-darwin`, empty ABI flags
and `Py_GIL_DISABLED=0`. No Rust compilation, project tests, benchmarks, or
wheel release validation ran. This proves the installer-monitor fix for these
recorded runs, not a complete release or publication.

| Runner | Elapsed | Readable worker-tree RSS peak | Inaccessible processes |
|---|---:|---:|---|
| arm64 | 30.412 s | 36.453 MiB lower bound | sudo PID1139; installer PID1140 |
| x86_64 | 98.255 s | 38.129 MiB lower bound | sudo PID2074; installer PID2075 |

**Neither number is the total installer memory peak.** Both reports explicitly
set `rss_complete=false` and `peak_rss_is_lower_bound=true`. Privileged `sudo`
and `installer` RSS could not be read: denied samples span approximately
0.719–30.169 seconds on ARM and1.193–97.456 seconds on Intel. The configured
1GiB limit continued to apply to readable sampled worker/descendant RSS; the
600-second timeout remained active. The monitor coordinator's own RSS is
outside that child tree. Total tree and privileged-process peaks are unknown;
these results do not prove total consumption below1GiB. Both commands returned
zero, with no stop condition or cleanup error.

All six original report/log files per architecture are retained byte for byte.
[audit.json](audit.json) records their hashes, actual denied-sample periods and
counts, source checks, package provenance, ABI results, and evidence limits.
The89MB installer itself is not duplicated in this archive; its measured
size and SHA256 are retained in the original inputs records.
