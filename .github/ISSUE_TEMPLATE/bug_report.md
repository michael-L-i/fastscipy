---
name: Bug report
about: A result that differs from SciPy, a crash, or a slowdown
labels: bug
---

**What happened**

<!-- A minimal script. If a result differs, show it with and without fastscipy:

import numpy as np, scipy.ndimage, fastscipy
fastscipy.patch()
x = ...
with fastscipy.disabled():
    expected = scipy.ndimage.gaussian_filter(x, 2)
got = scipy.ndimage.gaussian_filter(x, 2)
print(np.array_equal(expected, got))
-->

**Environment**

<!-- Output of:
python -c "import fastscipy, scipy, numpy, platform, sys; print(fastscipy.__version__, scipy.__version__, numpy.__version__, platform.platform(), sys.version); print(fastscipy.patch())"
-->
