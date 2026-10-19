# What fastscipy accelerates

Every fast path returns exactly what SciPy 1.18 returns, byte for byte. Each
one also checks its arguments first and hands anything outside what it
covers to SciPy unchanged, so SciPy produces every warning and error. The
lists below give what is covered, how, and the main cases it hands back.

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

Handed to SciPy: complex, float16, bool or byte-swapped inputs to filters and
interpolation; Python callbacks other than the reducers above; ties in
labelled position queries; structures with a false centre; `label` with a
non-contiguous output; `distance_transform_edt` with `distances=` or
`indices=`.

Very small arrays can still be slower than plain SciPy because Python
dispatch costs more than the compiled operation; the gains are workload dependent.

## special

Every `scipy.special` ufunc (322) is wrapped. Large calls are split into
chunks that run SciPy's own loops on several threads (*threads only*); a
short probe sets each function's size threshold. Calls with keyword
arguments (`out=`, `where=`, `dtype=`, ...), non-contiguous or object
inputs, or small sizes go straight to SciPy. So does any call while
`scipy.special.errstate` reports errors, or that raises a floating-point
event, so warnings come from SciPy exactly once.

## stats

| Functions | How |
|---|---|
| `ttest_1samp`, `ttest_ind`, `ttest_rel`, `pearsonr`, `zscore`, `skew`, `kurtosis`, `sem`, `entropy`, `linregress`, `describe`, `mannwhitneyu` (asymptotic) | lean versions of SciPy's own arithmetic, without the generic dispatch layers. For 1-D float64, NaN-free input with default options; any floating-point event, odd option or unknown keyword goes to SciPy |
| `pdf`, `logpdf`, `cdf`, `logcdf`, `sf`, `logsf`, `ppf`, `isf` on every continuous and discrete distribution; creating frozen distributions | skips per-call argument re-parsing and re-validation |
| `rankdata`, and `spearmanr`, `mannwhitneyu`, `wilcoxon`, `kruskal` through it | Rust radix ranking with ties, threads for large inputs |
| discrete `pmf`/`cdf`/`ppf` (for example `poisson.ppf`, `zipf.cdf`) | table lookups |
| `mode` | counting instead of sorting, for small integer ranges |
| `gaussian_kde.evaluate` | threads only |

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

For filter paths, handed to SciPy: float32 or complex coefficients, `axis` other than -1,
`padtype` other than `'odd'`, NaN or inf, `filtfilt(method='gust')`.
`welch`, `csd`, `coherence` and `stft` are never threaded: on a build with
the GIL that is slower.

## interpolate

| Functions | How |
|---|---|
| `BSpline.__call__`, `splev`, `interp1d` (spline kinds) | Rust de Boor kernel with binary search for queries in random order; threads |
| `Delaunay.find_simplex`, `LinearNDInterpolator`, `CloughTocher2DInterpolator`, `griddata` | queries sorted into grid-cell order, then SciPy's own search on chunks (threads); Rust barycentric kernel for real-valued `LinearNDInterpolator`. Points near a face are replayed in the original order, so ties resolve as in SciPy |
| `RegularGridInterpolator(method="pchip")`, `interpn(method="pchip")` | local-window Rust kernel for 1 to 3 dimensions |

Queries that are already ordered, fewer than about 300 queries, `tol=`,
`bruteforce=True`, NaN queries and periodic extrapolation go to SciPy.

## linalg

`lu_factor`, `lu_solve`, `cho_solve`, `solve_triangular`, `eigh`, `eigvalsh`
on stacks of float64 matrices call the same LAPACK routines as SciPy
(through `scipy.linalg.cython_lapack`), without SciPy's per-matrix Python
loop, and spread the batch over threads. A result is used only if every
matrix succeeded; otherwise SciPy reruns the whole call and raises or warns.
`null_space` skips computing the unused half of the SVD.

Single matrices, other dtypes, `overwrite_*`, generalized `eigh` and
eigenvalue subsets go to SciPy.

## sparse linear algebra

The `sparse_linalg` component reduces dispatch around unbatched NumPy
`LinearOperator.matvec` and `rmatvec` calls. CSR/CSC matrix operators call
the same SciPy matrix-vector kernel directly instead of inserting and
removing a column axis. This speeds small iterative solves, including
`cg`, `bicgstab`, and `gmres`, without changing their numerical algorithms,
stopping criteria, callback calls or iteration histories.

Custom operator subclasses, non-NumPy backends, batched vectors, and
deprecated column-vector inputs use SciPy's original dispatch.

## sparse and csgraph

| Functions | How |
|---|---|
| `argmax`, `argmin` with `axis` | Rust, threads above 1e6 stored entries |
| COO `sum_duplicates` | counting sort in Rust, threads |
| `csr_array`/`csc_array`/`coo_array` (and `*_matrix`) from dense arrays | Rust scan, threads |
| `find` | canonicalises through CSR instead of a lexsort |
| small constructors and format checks | less per-call overhead |
| `csgraph.dijkstra`, `bellman_ford`, `johnson`, `shortest_path`, `connected_components` | Rust ports with SciPy's tie-breaking; many sources run in parallel; Bellman-Ford stops when a pass changes nothing |

Handed to SciPy: float16, bool, complex and small integer dtypes for the
sparse kernels; negative weights in Dijkstra; negative cycles; Floyd-Warshall.

## spatial

`cdist` (17 metrics) and `pdist` (16) split rows across threads (*threads
only*; `mahalanobis`, `seuclidean` and custom callables go to SciPy).
Large float64 Fortran-order and strided inputs are copied once to contiguous
storage before splitting; small inputs keep SciPy's own path.
`KDTree`/`cKDTree` `query` and `query_ball_point` use several workers when
the caller left `workers` at its default and there are at least 2000
queries; SciPy's results do not depend on the worker count.

## cluster

| Functions | How |
|---|---|
| `vq`, including calls inside `kmeans` and `kmeans2` | float32/float64 NumPy inputs of matching dtype go to SciPy's own compiled vector-quantization kernel |
| `whiten` | 2-D float32/float64 NumPy standard deviation and scaling without array-API dispatch; warnings and exceptional cases go to SciPy |
| `is_valid_linkage`, and validation inside `leaves_list`, `fcluster`, `cophenet`, and other hierarchy operations | valid NumPy float64 linkage tables use direct checks and counting rather than array-API sorting |

Integer or mixed-dtype quantization, other backends, and invalid or unusual
linkage tables use SciPy. Linkage construction and the clustering algorithms
themselves remain SciPy's implementations.

## integrate and optimize

`integrate.trapezoid` on float64 arrays and `cumulative_trapezoid` on 1-D
float64 arrays skip the array-API dispatch. Cumulative integration covers
uniform spacing or matching 1-D float64 coordinates, with `initial=None`
or zero. Other dtypes, shapes and options go to SciPy.

`simpson` covers uniformly spaced float64 arrays with an odd sample count
of at least three along any valid axis. Gaussian `fixed_quad` skips generic
dispatch for ordinary finite scalar limits; it uses SciPy's cached nodes
and weights and calls the integrand exactly once. Callback exceptions and
malformed results propagate without retrying the callback.

`optimize.lsq_linear` covers dense float64 inputs with unbounded limits and
the exact solver, retaining the complete SciPy result object. `rosen_hess`
fills the Hessian in one output allocation for ordinary 1-D float64 inputs.
Constrained least squares and unusual dtypes or options go to SciPy.

`optimize`'s numerical differentiation (used by `approx_fprime` and by
`minimize` and `least_squares` without an analytic gradient) skips a
per-call namespace lookup and copy. Numerical differentiation still spends
most of its time in the user's own Python function.

## Not accelerated

Large dense linear algebra and FFT already run in tuned libraries
(Accelerate or OpenBLAS, and DUCC). `linprog`/HiGHS, scientific `io`, and
most other functions still use SciPy. Solvers such as
`minimize`, `solve_ivp`, `quad` and `root` spend their time in the user's
function, which only compiling that function would speed up.
