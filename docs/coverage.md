# What fastscipy accelerates

This page describes the main source branch. The
[latest development improvements](../bench/runs/2026-10-22-review/README.md),
including ND FFT dispatch, small `pdist` calls and scalar optimization bounds,
are not yet in published 0.1.0a4.

Fast paths are checked against SciPy 1.18, byte for byte where its reference
is reproducible. Intel Mac's Accelerate reductions can vary even in plain
SciPy; [platform details](platforms.md) describe that measured limitation and
the explicit numerical checks for those cases. Each path checks its arguments
first and hands unsupported calls to SciPy. The lists below describe coverage
and the main cases handed back.

"Threads" means the work is split across the thread budget
(`fastscipy.set_num_threads`, default: all cores). A path marked
*threads only* is the same speed as SciPy on one thread.

## ndimage

Rust kernels replace SciPy's private `_nd_image` C module. SciPy's own
Python wrappers stay in place, so argument handling is unchanged. The
kernels reproduce SciPy's arithmetic, including where clang fuses
multiply-adds.

| Functions | Notes |
|---|---|
| `correlate`, `convolve`, `correlate1d`, `convolve1d`, `gaussian_filter(1d)`, `uniform_filter(1d)`, `sobel`, `prewitt`, `laplace`, `gaussian_laplace`, `gaussian_gradient_magnitude` | SIMD line kernels, threads |
| `median_filter`, `rank_filter`, `percentile_filter`, `minimum_filter`, `maximum_filter` (N-D) | sorting networks and incremental rank kernels; 1-D calls use SciPy's own `_rank_filter_1d` |
| `zoom`, `shift`, `rotate`, `affine_transform`, `map_coordinates`, `spline_filter(1d)` | spline interpolation, all modes |
| `binary_erosion`, `binary_dilation`, `binary_opening`, `binary_closing`, `binary_hit_or_miss`, `binary_fill_holes`, `binary_propagation` | bit-level erosion; single-pass fixed-point dilation |
| `label`, `find_objects`, `watershed_ift` | Rust ports; `label` is a parallel union-find that numbers objects as SciPy does |
| `minimum`, `maximum`, `extrema`, `minimum_position`, `maximum_position`, `center_of_mass` | single pass over labelled input |
| `distance_transform_edt` | one fused pass |
| `generic_filter` with `np.mean`, `sum`, `min`, `max`, `median`, `std`, `var`, `ptp` | reducer kernels instead of a Python callback per pixel |

The native seam declines complex, float16, bool or byte-swapped buffers for
filters and interpolation. Some SciPy wrappers split complex inputs into real
and imaginary arrays that can use the existing kernels. Other cases handed
to SciPy include Python callbacks other than the reducers above; ties in
labelled position queries; structures with a false centre; `label` with a
non-contiguous output; `distance_transform_edt` with `distances=` or
`indices=`.

Very small arrays can still be slower than plain SciPy because Python
dispatch costs more than the compiled operation; the gains are workload dependent.
Labelled `center_of_mass` includes int16 and uint16 images. Zero-mass labels
use SciPy's own division so warnings and error handlers run once.

## special

Every `scipy.special` ufunc (322) is wrapped. Large calls are split into
chunks that run SciPy's own loops on several threads (*threads only*); a
short probe sets each function's size threshold. Calls with keyword
arguments (`out=`, `where=`, `dtype=`, ...), unsupported strided or object
inputs, or small sizes go straight to SciPy. So does any call while
`scipy.special.errstate` reports errors, or that raises a floating-point
event, so warnings come from SciPy exactly once.
Unary calls also support Fortran-contiguous inputs and preserve the output's
Fortran layout, including functions with several outputs.

## stats

| Functions | How |
|---|---|
| `ttest_1samp`, `ttest_ind`, `ttest_rel`, `pearsonr`, `zscore`, `skew`, `kurtosis`, `sem`, `entropy`, `linregress`, `describe`, `mannwhitneyu` (asymptotic) | lean versions of SciPy's own arithmetic, without the generic dispatch layers. For 1-D float64, NaN-free input with default options; any floating-point event, odd option or unknown keyword goes to SciPy |
| `pdf`, `logpdf`, `cdf`, `logcdf`, `sf`, `logsf`, `ppf`, `isf` on every continuous and discrete distribution; creating frozen distributions | skips per-call argument re-parsing and re-validation |
| `rankdata`, and `spearmanr`, `mannwhitneyu`, `wilcoxon`, `kruskal` through it | Rust radix ranking with ties, threads for large inputs |
| discrete `pmf`/`cdf`/`ppf` (for example `poisson.ppf`, `zipf.cdf`) | table lookups |
| `mode` | counting instead of sorting, for small integer ranges |
| `gaussian_kde.evaluate` | threads only |

`sem` additionally supports float32 and complex128 arrays, arbitrary single
axes and `axis=None`, with SciPy's same standard-deviation arithmetic.

## signal

| Functions | How |
|---|---|
| `sosfilt`, `sosfiltfilt`, `lfilter`, `filtfilt`, `detrend` (linear), `hilbert`, `welch`, `csd` (and `coherence`, `periodogram` through them) | small-call paths that skip SciPy's Python layers; cached `lfilter_zi`, `sosfilt_zi` and `iirfilter` (so `butter`, `cheby1`, ...) |
| `lfilter`, `sosfilt`, `filtfilt`, `sosfiltfilt`, `decimate`, `resample_poly`, `upfirdn`, `medfilt2d` on many channels | channels split across threads (threads only) |
| `find_peaks`, `peak_prominences`, `peak_widths` | Rust kernels |
| `dlsim` | Rust kernel; checked at run time against SciPy's BLAS summation order and used only where it matches |
| `convolve`, `correlate` (direct 2-D float64), `convolve2d`, `correlate2d` | Rust kernels |

The 1-D float64 `lfilter` path also covers FIR filters (`len(a) == 1`),
including initial state. `hilbert` handles float32 and float64 input,
any valid axis, and an explicit positive integer FFT length.

IIR `lfilter`, linear `detrend`, and last-axis 1-D/2-D `sosfilt` additionally
support float32/float64 and complex64/complex128 inputs. Filter coefficient
and initial-state promotion uses SciPy's same dtype rules. The detrending
design-matrix cache retains at most 8 MiB, with entries above 1 MiB uncached.
`sosfiltfilt` supports matching float32 or complex coefficients and data,
using SciPy's own dtype-specific cached initial conditions. Real float32
`welch`/`csd` use typed padding and the same per-frame FFT pipeline; unsupported
non-C-contiguous 2-D float32 spectra and complex two-sided spectra use SciPy.
Other forward/backward filter shortcuts retain narrower float64/default-axis
guards. Unsupported padding, `filtfilt(method='gust')`, and exceptional inputs
use SciPy.
`welch`, `csd`, `coherence` and `stft` are never threaded: on a build with
the GIL that is slower.

## FFT / FFTPACK

`fftn`, `ifftn`, `dctn`, `idctn`, `dstn`, `idstn`, and the corresponding
2-D FFT aliases skip redundant Python preparation for unresized NumPy
arrays. Complex FFTs accept native float32/float64 and complex64/complex128;
real transforms accept float32/float64. They call the same SciPy DUCC kernels.
This mainly helps small transforms; large transforms are approximately unchanged.

Resizing, in-place requests, unsupported layouts/dtypes/options and alternate
backends retain SciPy's dispatch. SciPy's explicit `workers` and worker
contexts are preserved; changing fastscipy's thread budget does not override them.

## interpolate

| Functions | How |
|---|---|
| `BSpline.__call__`, `splev`, `interp1d` (spline kinds) | Rust de Boor kernel with binary search for queries in random order; threads |
| `PPoly`, `BPoly`, and derived spline evaluation | NumPy numeric queries, including strided/float32 arrays and floating scalars, bypass namespace adapters and use SciPy's own evaluator |
| `Delaunay.find_simplex`, `LinearNDInterpolator`, `CloughTocher2DInterpolator`, `griddata` | queries sorted into grid-cell order, then SciPy's own search on chunks (threads); Rust barycentric kernel for real-valued `LinearNDInterpolator`. Points near a face are replayed in the original order, so ties resolve as in SciPy |
| `RegularGridInterpolator(method="pchip")`, `interpn(method="pchip")` | local-window Rust kernel for 1 to 3 dimensions |

For the spline and triangulation kernels, ordered queries, fewer than about
300 queries, `tol=`, `bruteforce=True`, NaN queries and periodic extrapolation
go to SciPy. The `PPoly`/`BPoly` dispatch shortcut also handles ordered,
small and nonfinite NumPy queries; periodic extrapolation uses SciPy's full dispatch.

## linalg

`lu_factor`, `lu_solve`, `cho_solve`, `solve_triangular`, `eigh`, `eigvalsh`
on stacks of float64 matrices call the same LAPACK routines as SciPy
(through `scipy.linalg.cython_lapack`), without SciPy's per-matrix Python
loop, and spread the batch over threads. A result is used only if every
matrix succeeded; otherwise SciPy reruns the whole call and raises or warns.
`null_space` skips computing the unused half of the SVD.

Single-matrix `solve_triangular`, `cho_solve`, and `lu_solve` also support
float32/float64 and complex64/complex128 NumPy inputs through SciPy's own
LAPACK dispatcher and validation. Single `lu_solve` accepts int32 pivots.
Other single matrices, unsupported dtypes, `overwrite_*`, generalized
`eigh` and eigenvalue subsets go to SciPy.

## sparse linear algebra

The `sparse_linalg` component reduces dispatch around unbatched NumPy
`LinearOperator.matvec` and `rmatvec` calls. CSR/CSC matrix operators call
the same SciPy matrix-vector kernel directly instead of inserting and
removing a column axis. This speeds small iterative solves, including
`cg`, `bicgstab`, and `gmres`; exact adjoint matrix operators use the same
shortcut for transpose products inside `lsqr` and `lsmr`. SciPy continues to
run the numerical algorithms, stopping criteria, and callbacks. Intel
Accelerate can vary its reductions even in plain SciPy; see the
[reference reproducibility details](platforms.md#intel-accelerate-reference-reproducibility).

Exact ndarray operator outputs are reshaped directly, retaining view ownership
and writeability; array subclasses retain NumPy dispatch. This also reduces
overhead in composed operators and preconditioned iterative solves.

Custom operator subclasses, non-NumPy backends, batched vectors, and
deprecated column-vector inputs use SciPy's original dispatch.

## sparse and csgraph

| Functions | How |
|---|---|
| `argmax`, `argmin` with `axis` | Rust, threads above 1e6 stored entries |
| COO `sum_duplicates` | counting sort in Rust, threads |
| `csr_array`/`csc_array`/`coo_array` (and `*_matrix`) from dense arrays | Rust scan, threads |
| `find` | canonicalises through CSR instead of a lexsort |
| `block_diag` with canonical CSR/CSC blocks and `format="csr"` or `"csc"` | converts only blocks in the opposite format, then joins compressed buffers, preserving sparse classes, explicit zeros and index dtypes |
| Two-block CSR `hstack` and CSC `vstack` | same SciPy stacking kernel without generic block dispatch and an extra compressed-buffer copy |
| small constructors and format checks | less per-call overhead |
| `csgraph.dijkstra`, `bellman_ford`, `johnson`, `shortest_path`, `connected_components` | Rust ports with SciPy's tie-breaking; many sources run in parallel; Bellman-Ford stops when a pass changes nothing |

Handed to SciPy: float16, bool, complex and small integer dtypes for the
sparse kernels; negative weights in Dijkstra; negative cycles; Floyd-Warshall.
Unweighted Dijkstra on supported canonical integer/bool CSR graphs omits
unused float64 weight buffers, including the reverse graph's weights.

## spatial

Large `cdist` (19 metrics) and `pdist` (18) calls split rows across threads (*threads
only*; custom callables go to SciPy). `mahalanobis` and `seuclidean` use
SciPy's own whole-input covariance or variance calculation once, before
splitting. Explicit float64 NumPy `VI` matrices and `V` vectors are supported;
exceptional statistics and unsupported options keep SciPy's own path.
Large float32/float64 Fortran-order and strided inputs are converted once to
contiguous float64 storage before splitting. Compiled weighted metrics accept
float64 NumPy weights; Minkowski also
accepts positive scalar `p`. Python-based weighted cosine/correlation keep
SciPy's own path.
Small native float64 `pdist` calls with a plain supported metric name use
SciPy's own condensed-distance kernel directly. This saves dispatch overhead
on one thread too. Weighted calls, custom metrics, other dtypes and metrics
requiring Python preprocessing retain SciPy's path.
`KDTree`/`cKDTree` `query` and `query_ball_point` use several workers when
the caller left `workers` at its default and there are at least 2000
queries; SciPy's results do not depend on the worker count.

## cluster

| Functions | How |
|---|---|
| `vq`, including calls inside `kmeans` and `kmeans2` | float32/float64 NumPy inputs of matching dtype go to SciPy's own compiled vector-quantization kernel |
| `whiten` | float32/float64 and complex64/complex128 NumPy standard deviation and scaling along the observation axis, including 1-D and N-D arrays; warnings and exceptional cases go to SciPy |
| `is_valid_linkage`, and validation inside `leaves_list`, `fcluster`, `cophenet`, and other hierarchy operations | valid NumPy float64 linkage tables use direct checks and counting rather than array-API sorting |

Integer or mixed-dtype quantization, other backends, and invalid or unusual
linkage tables use SciPy. Linkage construction and the clustering algorithms
themselves remain SciPy's implementations.

## integrate and optimize

`integrate.trapezoid` on float64 arrays and `cumulative_trapezoid` on
float32, float64 and complex128 arrays skip the array-API dispatch.
Cumulative trapezoid integration supports arbitrary axes, uniform spacing,
or float32/float64 coordinates matching the integration axis or full input
shape, with `initial=None` or zero. Unsupported options go to SciPy.

`simpson` covers uniformly spaced float32/float64 and complex64/complex128
arrays with at least two samples, including the even-length end correction,
along any valid axis. Real float32/float64 arrays also support bounded,
strictly increasing or decreasing real coordinate arrays; mixed coordinate
dtypes follow SciPy's same promotion before taking differences. Zero or
mixed-sign coordinate intervals and complex explicit coordinates use SciPy.
Gaussian `fixed_quad` skips generic
dispatch for ordinary finite scalar limits; it uses SciPy's cached nodes
and weights and calls the integrand exactly once. Callback exceptions and
malformed results propagate without retrying the callback.

`cumulative_simpson` on float64 arrays evaluates only the interval estimates
SciPy retains, for uniform spacing or supported strictly increasing float64
coordinates. Extreme inputs and unusual underflow
reporting use SciPy so discarded arithmetic still produces its warnings.
`romb` on float32/float64 and complex128 arrays retains two rows of the same
Richardson table, preserving its extrapolation arithmetic; `show=True` uses SciPy.

`optimize.lsq_linear` covers dense float64 inputs with unbounded limits and
the exact solver, retaining the complete SciPy result object. `rosen_hess`
fills the Hessian in one output allocation for ordinary 1-D float64 inputs.
Constrained least squares and unusual dtypes or options go to SciPy.
`rosen_der` and `rosen_hess_prod` use the same NumPy expressions without
namespace dispatch; accepted numerical errors and handlers propagate once.
This also reduces overhead in Newton workflows using these functions.

`optimize`'s numerical differentiation (used by `approx_fprime` and by
`minimize` and `least_squares` without an analytic gradient) skips a
per-call namespace lookup and copy. Numerical differentiation still spends
most of its time in the user's own Python function.

Scalar bounds for `least_squares` and numerical differentiation are filled
directly instead of repeating scalars through `np.resize`. This helps large
sparse and matrix-free parameter vectors; arrays and unusual bound types
retain SciPy's preparation. Solver arithmetic, stopping criteria and callback
ordering are preserved. Small problems may see little benefit or slight overhead.

## Not accelerated

Large dense linear algebra and the numerical FFT kernels already run in
tuned libraries (Accelerate or OpenBLAS, and DUCC). `linprog`/HiGHS, scientific
`io`, `differentiate`, and most other functions still use SciPy. Differentiation
prototypes with small gains were measured and rejected. Callback-heavy
`minimize`, `solve_ivp`,
`quad` and `root` workloads often spend their time in the user's function;
the helper shortcuts above do not accelerate arbitrary user callbacks.

## Coverage inventory

`python tools/coverage_inventory.py --output inventory.json` lists public
callable exports in the selected SciPy namespaces and the actual patch hooks.
An unchanged public export may benefit from patched internals; a patched
export can still decline most input variants. These counts are not a count
of guaranteed speedups or a percentage of SciPy covered. Benchmark run
artifacts also record measured rejected candidates and remaining algorithmic
opportunities, so unsupported variants are not mistaken for proven wins.

### Downstream application pilots

Patching before downstream imports can accelerate libraries that call supported
SciPy paths. The [scikit-image, scikit-learn, and statsmodels pilots](../bench/runs/2026-10-26-downstream/README.md)
compare fresh-process unpatched baselines, exact returned arrays, warnings,
unchanged inputs, and patch hit counts. On the recorded local setup, image cleanup
and segmentation improved about 30×; a small Gaussian-process fit/predict improved
about 7%; seasonal decomposition reached no patched calls. These bounded workflows
are evidence for specific call paths, not whole-library compatibility guarantees.
