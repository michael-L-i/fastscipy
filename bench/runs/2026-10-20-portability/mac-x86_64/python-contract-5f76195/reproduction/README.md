# Exact linregress input replay

The source and raw stdout reproduce seed 12345, `4t-linregress`, iteration 69
from the failed Intel contract diagnostic. All 30 x values are exactly
0.05016188054896638; the recorded float64 mean is 0.05016188054896637, leaving
a centered spread of 6.938893903907228e-18. SciPy warned about catastrophic
cancellation and potentially unreliable results. The local result's slope
was approximately -17.0667; the earlier Intel failure recorded -19.7333
and -19.2. These observations concern ill-conditioned inputs, not byte identity.

This was one **local Apple Silicon** replay using the installed source-1f48
runtime, not a new Intel execution. The JSON `threads: 4` identifies the
randomized test-loop fixture; the command's native thread environment was 2,
with BLAS limited to 1. Resource figures (0.65166 seconds, 103.55 MiB peak)
are transcribed from the agent's original monitored tool output; no standalone
raw monitor JSON was saved. Original source/stdout bytes and hashes are
retained. [sources.json](sources.json) separates raw evidence from reported
runtime/resource metadata and records the command and 128 MiB/30-second limits.

Candidate `e290714` changes only the test validator to check ill-conditioned
intercepts against an independently computed mean identity. Its source hash
is recorded separately. This replay does not establish that candidate passed
native Intel validation, nor does it fix or certify a production kernel.
