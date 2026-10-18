# Benchmark results

Apple M5 Pro (18 cores: 6 Super + 12 Performance), macOS, Python 3.13, NumPy 2.5.4, SciPy 1.18.1 (Accelerate), release build (thin LTO). Run 2026-10-18 with `python bench/run.py --check`, one job on an otherwise idle machine. Times are the best of repeated calls; every row's patched result is byte-identical to SciPy's. Controls are small calls the fast path should hand back to SciPy.

| case | SciPy ms | 1 thread ms | 18 threads ms | x 1T | x 18T | identical |
|---|---:|---:|---:|---:|---:|---|
| csgraph: shortest_path unweighted BFS, grid 600^2, 1 source | 15.891 | 2.332 | 2.288 | 6.8 | 6.9 | yes |
| csgraph: dijkstra unweighted + predecessors, grid 600^2 | 15.713 | 2.333 | 2.330 | 6.7 | 6.7 | yes |
| csgraph: dijkstra weighted, 18 sources, grid 300^2 | 66.308 | 47.464 | 5.172 | 1.4 | 12.8 | yes |
| csgraph: dijkstra weighted + predecessors, 12 sources, undirected | 96.055 | 64.071 | 6.973 | 1.5 | 13.8 | yes |
| csgraph: bellman_ford 2000 nodes, 40k edges, 20 sources | 1356.401 | 4.357 | 0.557 | 311.3 | 2435.0 | yes |
| csgraph: johnson 2000 nodes, 40k edges, 100 sources | 98.424 | 26.871 | 2.382 | 3.7 | 41.3 | yes |
| csgraph: connected_components weak, 1M nodes | 78.861 | 13.403 | 13.434 | 5.9 | 5.9 | yes |
| csgraph: connected_components strong, 1M nodes | 61.462 | 34.284 | 35.262 | 1.8 | 1.7 | yes |
| csgraph: connected_components, 200 nodes x100 calls | 1.743 | 0.285 | 0.284 | 6.1 | 6.1 | yes |
| **csgraph geomean** | | | | **5.8** | **15.5** | |
| integrate: trapezoid 100 points with x, x100 calls | 0.295 | 0.151 | 0.154 | 2.0 | 1.9 | yes |
| integrate: trapezoid 100 points dx, x100 calls | 0.258 | 0.134 | 0.136 | 1.9 | 1.9 | yes |
| integrate: trapezoid (1000, 30) axis=0 x100 calls | 2.005 | 1.909 | 1.911 | 1.1 | 1.0 | yes |
| **integrate geomean** | | | | **1.6** | **1.6** | |
| interpolate: BSpline 1000 knots, 1e6 random queries | 105.150 | 30.422 | 2.612 | 3.5 | 40.3 | yes |
| interpolate: BSpline 1000 knots, 1e6 sorted queries | 11.789 | 11.840 | 2.591 | 1.0 | 4.5 | yes |
| interpolate: splev 1000 knots, 1e6 random queries | 100.971 | 30.470 | 2.650 | 3.3 | 38.1 | yes |
| interpolate: interp1d cubic, 1e6 random queries | 105.098 | 30.784 | 3.060 | 3.4 | 34.3 | yes |
| interpolate: BSpline 3 values, 2e5 random queries | 21.174 | 6.611 | 0.807 | 3.2 | 26.2 | yes |
| interpolate: BSpline 2e5 queries, derivative | 20.720 | 12.560 | 12.687 | 1.6 | 1.6 | yes |
| interpolate: Delaunay.find_simplex 1e5 points, 1e5 random queries | 485.817 | 7.088 | 3.041 | 68.5 | 159.7 | yes |
| interpolate: Delaunay.find_simplex 1e5 points, 1e5 raster-ordered queries | 4.730 | 4.643 | 4.623 | 1.0 | 1.0 | yes |
| interpolate: LinearNDInterpolator 1e5 points, 1e5 random queries | 494.258 | 8.126 | 3.454 | 60.8 | 143.1 | yes |
| interpolate: LinearNDInterpolator raster-ordered queries | 4.991 | 4.861 | 5.065 | 1.0 | 1.0 | yes |
| interpolate: CloughTocher2DInterpolator 1e5 points, 1e5 random queries | 509.173 | 17.608 | 5.397 | 28.9 | 94.3 | yes |
| interpolate: griddata linear 2e4 points, 1e5 queries | 204.727 | 42.409 | 40.088 | 4.8 | 5.1 | yes |
| interpolate: griddata cubic 2e4 points, 1e5 queries | 217.691 | 54.566 | 47.432 | 4.0 | 4.6 | yes |
| interpolate: RegularGridInterpolator pchip 512x512, 5000 queries | 350.295 | 0.532 | 0.478 | 659.1 | 732.2 | yes |
| interpolate: RegularGridInterpolator pchip 64^3, 2000 queries | 390.597 | 0.575 | 0.574 | 679.8 | 680.2 | yes |
| **interpolate geomean** | | | | **9.1** | **23.0** | |
| linalg: lu_factor 10000x3x3 | 17.496 | 0.614 | 0.084 | 28.5 | 208.0 | yes |
| linalg: lu_solve 10000x3x3 | 21.641 | 0.339 | 0.071 | 63.9 | 305.3 | yes |
| linalg: cho_solve 10000x3x3 | 27.555 | 0.302 | 0.063 | 91.3 | 437.4 | yes |
| linalg: solve_triangular 10000x3x3 | 33.308 | 0.227 | 0.057 | 146.7 | 585.6 | yes |
| linalg: eigh 10000x3x3 | 69.209 | 15.389 | 1.256 | 4.5 | 55.1 | yes |
| linalg: eigvalsh 10000x3x3 | 78.989 | 3.543 | 0.280 | 22.3 | 282.5 | yes |
| linalg: lu_factor 10000x8x8 | 20.288 | 1.744 | 0.195 | 11.6 | 103.8 | yes |
| linalg: lu_solve 10000x8x8 | 22.722 | 0.926 | 0.114 | 24.5 | 199.1 | yes |
| linalg: cho_solve 10000x8x8 | 27.906 | 0.946 | 0.119 | 29.5 | 234.4 | yes |
| linalg: solve_triangular 10000x8x8 | 34.787 | 0.724 | 0.104 | 48.1 | 335.7 | yes |
| linalg: eigh 10000x8x8 | 108.752 | 51.109 | 3.600 | 2.1 | 30.2 | yes |
| linalg: eigvalsh 10000x8x8 | 90.639 | 13.956 | 1.018 | 6.5 | 89.1 | yes |
| linalg: lu_factor 5000x32x32 | 30.246 | 16.523 | 8.353 | 1.8 | 3.6 | yes |
| linalg: lu_solve 5000x32x32 | 14.643 | 3.842 | 0.340 | 3.8 | 43.0 | yes |
| linalg: cho_solve 5000x32x32 | 19.934 | 5.092 | 0.455 | 3.9 | 43.8 | yes |
| linalg: solve_triangular 5000x32x32 | 20.200 | 3.545 | 0.329 | 5.7 | 61.4 | yes |
| linalg: eigh 5000x32x32 | 241.331 | 208.329 | 13.767 | 1.2 | 17.5 | yes |
| linalg: eigvalsh 5000x32x32 | 104.137 | 65.011 | 4.267 | 1.6 | 24.4 | yes |
| linalg: null_space 2000x500 | 105.929 | 39.949 | 40.042 | 2.7 | 2.6 | yes |
| **linalg geomean** | | | | **10.0** | **75.9** | |
| ndimage: gaussian_filter f64 2048^2 s=2 | 54.548 | 8.127 | 1.198 | 6.7 | 45.5 | yes |
| ndimage: gaussian_filter f32 2048^2 s=2 | 48.943 | 6.930 | 0.911 | 7.1 | 53.7 | yes |
| ndimage: gaussian_filter u8 2048^2 s=2 | 39.630 | 8.069 | 0.890 | 4.9 | 44.5 | yes |
| ndimage: gaussian_filter f32 256^3 s=1 | 267.129 | 27.454 | 3.726 | 9.7 | 71.7 | yes |
| ndimage: uniform_filter f64 2048^2 k=5 | 36.650 | 4.433 | 0.989 | 8.3 | 37.1 | yes |
| ndimage: sobel f32 2048^2 | 30.155 | 2.571 | 0.513 | 11.7 | 58.7 | yes |
| ndimage: convolve f64 2048^2 5x5 | 40.949 | 6.749 | 0.837 | 6.1 | 48.9 | yes |
| ndimage: correlate f32 256^3 3x3x3 | 185.751 | 32.820 | 3.348 | 5.7 | 55.5 | yes |
| ndimage: median_filter f32 1024^2 5x5 | 229.458 | 2.209 | 0.369 | 103.9 | 622.5 | yes |
| ndimage: median_filter u8 2048^2 3x3 | 258.713 | 0.521 | 0.209 | 496.3 | 1239.6 | yes |
| ndimage: rank_filter f64 1024^2 5x5 | 212.546 | 4.300 | 0.547 | 49.4 | 388.5 | yes |
| ndimage: percentile_filter f32 1024^2 7x7 | 389.353 | 6.119 | 0.728 | 63.6 | 534.8 | yes |
| ndimage: minimum_filter u8 2048^2 5x5 | 53.920 | 0.295 | 0.190 | 182.9 | 284.2 | yes |
| ndimage: maximum_filter f64 2048^2 7x7 | 67.489 | 4.146 | 0.806 | 16.3 | 83.7 | yes |
| ndimage: maximum_filter f32 256^3 3^3 | 425.248 | 15.122 | 2.948 | 28.1 | 144.3 | yes |
| ndimage: zoom f64 1024^2 x1.5 o3 | 77.196 | 6.212 | 1.261 | 12.4 | 61.2 | yes |
| ndimage: zoom f32 1024^2 x0.5 o1 | 2.137 | 0.277 | 0.071 | 7.7 | 30.3 | yes |
| ndimage: rotate f64 1024^2 o1 | 18.072 | 1.980 | 0.357 | 9.1 | 50.6 | yes |
| ndimage: rotate u8 2048^2 o3 | 207.014 | 47.276 | 4.930 | 4.4 | 42.0 | yes |
| ndimage: affine_transform f64 1024^2 o3 | 43.762 | 7.567 | 1.260 | 5.8 | 34.7 | yes |
| ndimage: map_coordinates f64 1024^2 o3 | 35.498 | 9.076 | 1.572 | 3.9 | 22.6 | yes |
| ndimage: spline_filter f64 2048^2 o3 | 73.745 | 19.345 | 2.655 | 3.8 | 27.8 | yes |
| ndimage: binary_erosion 2048^2 cross | 14.299 | 0.597 | 0.122 | 23.9 | 116.8 | yes |
| ndimage: binary_dilation 2048^2 3x3 | 18.364 | 0.802 | 0.163 | 22.9 | 112.5 | yes |
| ndimage: binary_erosion 256^3 | 71.045 | 4.927 | 0.633 | 14.4 | 112.2 | yes |
| ndimage: binary_opening 2048^2 it=3 | 47.574 | 15.258 | 2.569 | 3.1 | 18.5 | yes |
| ndimage: binary_fill_holes 2048^2 | 46.144 | 4.256 | 1.184 | 10.8 | 39.0 | yes |
| ndimage: binary_propagation 2048^2 | 28.032 | 2.133 | 0.658 | 13.1 | 42.6 | yes |
| ndimage: binary_fill_holes 256^3 | 253.704 | 21.831 | 4.559 | 11.6 | 55.7 | yes |
| ndimage: label 2048^2 | 9.306 | 3.042 | 0.559 | 3.1 | 16.6 | yes |
| ndimage: label 2048^2 8-conn | 10.677 | 3.074 | 0.522 | 3.5 | 20.4 | yes |
| ndimage: label 256^3 | 72.096 | 19.998 | 3.042 | 3.6 | 23.7 | yes |
| ndimage: find_objects 2048^2 | 7.583 | 3.119 | 0.749 | 2.4 | 10.1 | yes |
| ndimage: maximum 2048^2 labels | 332.429 | 6.171 | 1.807 | 53.9 | 183.9 | yes |
| ndimage: extrema 2048^2 labels | 357.588 | 6.790 | 2.508 | 52.7 | 142.6 | yes |
| ndimage: minimum_position 2048^2 labels | 342.098 | 6.532 | 2.151 | 52.4 | 159.0 | yes |
| ndimage: center_of_mass 2048^2 labels | 77.472 | 8.114 | 8.046 | 9.5 | 9.6 | yes |
| ndimage: distance_transform_edt 2048^2 | 153.065 | 117.453 | 16.003 | 1.3 | 9.6 | yes |
| ndimage: distance_transform_edt 128^3 | 155.001 | 116.055 | 11.504 | 1.3 | 13.5 | yes |
| ndimage: generic_filter np.mean 512^2 3x3 | 273.124 | 1.121 | 0.403 | 243.7 | 677.4 | yes |
| ndimage: generic_filter np.std 512^2 5x5 | 953.123 | 5.741 | 1.000 | 166.0 | 953.6 | yes |
| **ndimage geomean** | | | | **13.3** | **68.2** | |
| optimize: approx_fprime rosen n=1000 | 9.652 | 8.741 | 8.877 | 1.1 | 1.1 | yes |
| optimize: approx_fprime cheap sum-of-squares n=1000 | 3.835 | 2.971 | 2.956 | 1.3 | 1.3 | yes |
| optimize: minimize BFGS rosen n=10 (numerical gradient) | 12.986 | 12.256 | 12.295 | 1.1 | 1.1 | yes |
| **optimize geomean** | | | | **1.1** | **1.1** | |
| signal: butter + sosfiltfilt, 200 samples | 0.320 | 0.010 | 0.010 | 31.3 | 32.0 | yes |
| signal: filtfilt, 1000 samples | 0.039 | 0.016 | 0.016 | 2.4 | 2.4 | yes |
| signal: sosfiltfilt, 1000 samples | 0.080 | 0.012 | 0.012 | 6.7 | 6.8 | yes |
| signal: detrend, 1000 samples | 0.043 | 0.020 | 0.021 | 2.1 | 2.1 | yes |
| signal: decimate q=4, 1000 samples | 0.325 | 0.018 | 0.018 | 18.1 | 18.1 | yes |
| signal: hilbert, 1000 samples | 0.016 | 0.014 | 0.013 | 1.2 | 1.2 | yes |
| signal: welch, 1000 samples | 0.091 | 0.040 | 0.040 | 2.3 | 2.3 | yes |
| signal: per-voxel loop, 1000 x sosfiltfilt(150) | 79.920 | 7.570 | 7.350 | 10.6 | 10.9 | yes |
| signal: per-epoch loop, 200 x (detrend, welch) | 28.692 | 12.350 | 11.960 | 2.3 | 2.4 | yes |
| signal: sosfilt 64 x 250k (butter 8) | 71.572 | 71.473 | 7.954 | 1.0 | 9.0 | yes |
| signal: lfilter 64 x 250k (butter 8) | 97.792 | 96.921 | 8.755 | 1.0 | 11.2 | yes |
| signal: filtfilt 64 x 250k (butter 8) | 192.872 | 195.271 | 16.662 | 1.0 | 11.6 | yes |
| signal: sosfiltfilt 64 x 250k (butter 8) | 147.339 | 144.328 | 14.120 | 1.0 | 10.4 | yes |
| signal: decimate q=8, 64 x 250k | 148.427 | 144.788 | 13.843 | 1.0 | 10.7 | yes |
| signal: peak_prominences, random walk 1e6 | 198.375 | 15.044 | 8.349 | 13.2 | 23.8 | yes |
| signal: find_peaks(prominence, width), random walk 1e6 | 210.743 | 22.805 | 13.991 | 9.2 | 15.1 | yes |
| signal: find_peaks(prominence, width), noise 1e6 | 23.851 | 22.115 | 13.443 | 1.1 | 1.8 | yes |
| signal: dlsim, 4 states, 100k steps | 173.751 | 1.657 | 1.626 | 104.8 | 106.9 | yes |
| signal: convolve direct 2-D, 1024^2 * 5x5 | 156.670 | 3.177 | 0.603 | 49.3 | 259.7 | yes |
| signal: correlate direct 2-D same, 1024^2 * 5x5 | 155.246 | 2.837 | 0.618 | 54.7 | 251.1 | yes |
| signal: convolve2d same symm, 1024^2 * 5x5 | 21.740 | 6.553 | 0.534 | 3.3 | 40.7 | yes |
| signal: correlate2d full fill, 512^2 * 7x7 | 9.441 | 2.668 | 0.308 | 3.5 | 30.7 | yes |
| **signal geomean** | | | | **4.7** | **12.4** | |
| sparse: 2000 x csr_array((data, indices, indptr)) 10x10 | 9.945 | 5.867 | 5.738 | 1.7 | 1.7 | yes |
| sparse: 2000 x csr_array((10, 10)) | 13.079 | 8.970 | 8.846 | 1.5 | 1.5 | yes |
| sparse: 2000 x coo->csr_array((v, (i, j))) | 35.216 | 32.092 | 32.078 | 1.1 | 1.1 | yes |
| sparse: 2000 x small A.T + A | 55.727 | 36.194 | 35.213 | 1.5 | 1.6 | yes |
| sparse: csr 1e7 nnz argmax(axis=1) | 75.344 | 6.396 | 0.816 | 11.8 | 92.3 | yes |
| sparse: csr 1e7 nnz argmin(axis=1) explicit | 56.195 | 6.475 | 0.849 | 8.7 | 66.2 | yes |
| sparse: csc 1e7 nnz argmax(axis=0) | 195.680 | 67.675 | 49.151 | 2.9 | 4.0 | yes |
| sparse: coo 2e6 entries sum_duplicates | 529.252 | 21.036 | 7.229 | 25.2 | 73.2 | yes |
| sparse: coo 5e5 entries on 300x300, heavy duplicates | 75.137 | 5.936 | 1.495 | 12.7 | 50.3 | yes |
| sparse: find csr 20000^2, 1.8M nnz | 212.074 | 3.612 | 3.613 | 58.7 | 58.7 | yes |
| sparse: find coo (shuffled, duplicates) 20000^2, 1.8M nnz | 356.476 | 35.755 | 40.169 | 10.0 | 8.9 | yes |
| sparse: csr_array(dense 4000^2, 16%) | 68.174 | 27.220 | 2.652 | 2.5 | 25.7 | yes |
| sparse: csc_array(dense 4000^2, 16%) | 60.316 | 20.800 | 5.622 | 2.9 | 10.7 | yes |
| sparse: coo_array(dense 4000^2, 16%) | 54.241 | 28.508 | 3.685 | 1.9 | 14.7 | yes |
| sparse: csr_matrix(dense 4000^2, 16%) | 68.558 | 27.296 | 2.537 | 2.5 | 27.0 | yes |
| sparse: csr_array(dense 100x100) | 11.341 | 1.680 | 1.682 | 6.8 | 6.7 | yes |
| **sparse geomean** | | | | **4.7** | **11.9** | |
| spatial: cdist euclidean 4000x2000x16 | 19.488 | 19.473 | 1.879 | 1.0 | 10.4 | yes |
| spatial: cdist cityblock 4000x2000x16 | 14.927 | 14.902 | 1.611 | 1.0 | 9.3 | yes |
| spatial: cdist cosine 4000x2000x16 | 38.048 | 37.200 | 2.736 | 1.0 | 13.9 | yes |
| spatial: pdist euclidean 3000x16 | 10.978 | 10.964 | 6.646 | 1.0 | 1.7 | yes |
| spatial: pdist cityblock 3000x16 | 8.412 | 8.402 | 6.418 | 1.0 | 1.3 | yes |
| spatial: cKDTree.query 1M pts, 200k queries, k=1 | 198.919 | 201.737 | 18.181 | 1.0 | 10.9 | yes |
| spatial: cKDTree.query 1M pts, 200k queries, k=8 | 399.855 | 399.916 | 34.853 | 1.0 | 11.5 | yes |
| spatial: cKDTree.query_ball_point 200k, r=0.01 | 304.297 | 305.126 | 28.857 | 1.0 | 10.5 | yes |
| spatial: cdist 300x100x16 (control) | 0.074 | 0.070 | 0.070 | 1.1 | 1.0 | yes |
| spatial: pdist 100x16 (control) | 0.015 | 0.016 | 0.016 | 1.0 | 1.0 | yes |
| spatial: cKDTree.query 100 queries (control) | 0.051 | 0.051 | 0.053 | 1.0 | 1.0 | yes |
| **spatial geomean** | | | | **1.0** | **4.0** | |
| special: erf 1e+05 float64 | 0.535 | 0.537 | 0.203 | 1.0 | 2.6 | yes |
| special: gammaln 1e+05 float64 | 1.153 | 1.140 | 0.281 | 1.0 | 4.1 | yes |
| special: i0 1e+05 float64 | 1.044 | 1.045 | 0.285 | 1.0 | 3.7 | yes |
| special: expit 1e+05 float64 | 0.200 | 0.183 | 0.138 | 1.1 | 1.4 | yes |
| special: jv 1e+05 float64 | 23.019 | 23.064 | 2.016 | 1.0 | 11.4 | yes |
| special: betainc 1e+05 float64 | 9.343 | 9.360 | 1.121 | 1.0 | 8.3 | yes |
| special: erf 1e+05 float32 | 0.584 | 0.574 | 0.205 | 1.0 | 2.8 | yes |
| special: gammaln 1e+05 float32 | 1.208 | 1.176 | 0.286 | 1.0 | 4.2 | yes |
| special: i0 1e+05 float32 | 1.096 | 1.087 | 0.287 | 1.0 | 3.8 | yes |
| special: expit 1e+05 float32 | 0.133 | 0.134 | 0.108 | 1.0 | 1.2 | yes |
| special: jv 1e+05 float32 | 23.013 | 23.167 | 2.003 | 1.0 | 11.5 | yes |
| special: betainc 1e+05 float32 | 5.965 | 5.957 | 0.823 | 1.0 | 7.2 | yes |
| special: erf 1e+06 float64 | 5.660 | 5.655 | 0.792 | 1.0 | 7.1 | yes |
| special: gammaln 1e+06 float64 | 13.255 | 13.092 | 1.431 | 1.0 | 9.3 | yes |
| special: i0 1e+06 float64 | 10.870 | 10.881 | 1.371 | 1.0 | 7.9 | yes |
| special: expit 1e+06 float64 | 1.852 | 1.876 | 0.506 | 1.0 | 3.7 | yes |
| special: jv 1e+06 float64 | 234.100 | 233.824 | 15.303 | 1.0 | 15.3 | yes |
| special: betainc 1e+06 float64 | 94.258 | 94.387 | 7.511 | 1.0 | 12.5 | yes |
| special: erf 1e+06 float32 | 6.072 | 5.995 | 0.750 | 1.0 | 8.1 | yes |
| special: gammaln 1e+06 float32 | 13.986 | 14.059 | 1.446 | 1.0 | 9.7 | yes |
| special: i0 1e+06 float32 | 11.297 | 11.274 | 1.369 | 1.0 | 8.3 | yes |
| special: expit 1e+06 float32 | 1.371 | 1.383 | 0.383 | 1.0 | 3.6 | yes |
| special: jv 1e+06 float32 | 233.673 | 233.205 | 15.323 | 1.0 | 15.2 | yes |
| special: betainc 1e+06 float32 | 59.759 | 60.923 | 4.903 | 1.0 | 12.2 | yes |
| special: erf 1e+07 float64 | 61.739 | 67.496 | 4.601 | 0.9 | 13.4 | yes |
| special: gammaln 1e+07 float64 | 139.390 | 139.395 | 9.476 | 1.0 | 14.7 | yes |
| special: jv 1e+07 float64 | 2335.507 | 2315.045 | 143.163 | 1.0 | 16.3 | yes |
| special: betainc 1e+07 float64 | 950.945 | 941.057 | 72.204 | 1.0 | 13.2 | yes |
| special: erf 1e+07 float32 | 61.338 | 66.690 | 4.751 | 0.9 | 12.9 | yes |
| special: gammaln 1e+07 float32 | 142.865 | 140.143 | 9.593 | 1.0 | 14.9 | yes |
| special: jv 1e+07 float32 | 2319.289 | 2310.143 | 148.980 | 1.0 | 15.6 | yes |
| special: betainc 1e+07 float32 | 612.226 | 605.985 | 45.423 | 1.0 | 13.5 | yes |
| special: erf 1e3 (control) | 0.006 | 0.005 | 0.005 | 1.0 | 1.1 | yes |
| special: jv 1e3 (control) | 0.220 | 0.221 | 0.221 | 1.0 | 1.0 | yes |
| **special geomean** | | | | **1.0** | **6.6** | |
| stats: ttest_1samp n=10 | 0.119 | 0.012 | 0.012 | 9.6 | 9.6 | yes |
| stats: ttest_ind n=10 | 0.178 | 0.020 | 0.020 | 8.8 | 8.8 | yes |
| stats: ttest_rel n=10 | 0.105 | 0.014 | 0.014 | 7.3 | 7.3 | yes |
| stats: pearsonr n=10 | 0.062 | 0.018 | 0.018 | 3.5 | 3.5 | yes |
| stats: linregress n=10 | 0.108 | 0.013 | 0.013 | 8.5 | 8.5 | yes |
| stats: zscore n=10 | 0.037 | 0.008 | 0.008 | 4.6 | 4.6 | yes |
| stats: skew n=10 | 0.091 | 0.009 | 0.009 | 10.6 | 10.7 | yes |
| stats: kurtosis n=10 | 0.094 | 0.009 | 0.009 | 11.0 | 10.9 | yes |
| stats: sem n=10 | 0.058 | 0.004 | 0.004 | 12.9 | 13.1 | yes |
| stats: entropy n=10 | 0.043 | 0.003 | 0.003 | 12.9 | 12.9 | yes |
| stats: describe n=10 | 0.222 | 0.028 | 0.028 | 8.0 | 8.0 | yes |
| stats: mannwhitneyu n=10 | 0.125 | 0.006 | 0.006 | 20.0 | 20.0 | yes |
| stats: ttest_1samp n=1000 | 0.121 | 0.014 | 0.014 | 8.7 | 8.7 | yes |
| stats: ttest_ind n=1000 | 0.181 | 0.023 | 0.023 | 7.9 | 8.0 | yes |
| stats: ttest_rel n=1000 | 0.106 | 0.016 | 0.016 | 6.4 | 6.5 | yes |
| stats: pearsonr n=1000 | 0.069 | 0.023 | 0.024 | 3.0 | 2.9 | yes |
| stats: linregress n=1000 | 0.115 | 0.016 | 0.017 | 7.0 | 6.9 | yes |
| stats: zscore n=1000 | 0.039 | 0.009 | 0.009 | 4.3 | 4.2 | yes |
| stats: skew n=1000 | 0.098 | 0.014 | 0.015 | 6.8 | 6.7 | yes |
| stats: kurtosis n=1000 | 0.102 | 0.014 | 0.014 | 7.1 | 7.1 | yes |
| stats: sem n=1000 | 0.059 | 0.005 | 0.005 | 11.8 | 11.8 | yes |
| stats: entropy n=1000 | 0.045 | 0.006 | 0.006 | 7.4 | 7.5 | yes |
| stats: describe n=1000 | 0.236 | 0.041 | 0.041 | 5.8 | 5.8 | yes |
| stats: mannwhitneyu n=1000 | 0.166 | 0.030 | 0.031 | 5.5 | 5.4 | yes |
| stats: loop: 1000 ttest_ind + pearsonr n=20 | 270.836 | 42.908 | 43.001 | 6.3 | 6.3 | yes |
| stats: norm.cdf scalar | 0.012 | 0.001 | 0.001 | 8.2 | 8.2 | yes |
| stats: norm.ppf scalar | 0.017 | 0.003 | 0.003 | 6.5 | 6.5 | yes |
| stats: t.sf scalar | 0.013 | 0.002 | 0.002 | 5.7 | 5.7 | yes |
| stats: poisson.pmf scalar | 0.012 | 0.004 | 0.004 | 3.0 | 3.1 | yes |
| stats: frozen norm(1, 2) + cdf | 0.094 | 0.007 | 0.007 | 13.3 | 13.4 | yes |
| stats: loop: 1000 frozen t(5).cdf | 90.146 | 10.794 | 10.584 | 8.4 | 8.5 | yes |
| stats: norm.cdf 1e6 | 11.284 | 8.352 | 1.597 | 1.4 | 7.1 | yes |
| stats: norm.ppf 1e6 | 10.406 | 6.904 | 1.878 | 1.5 | 5.5 | yes |
| stats: t.cdf 1e6 (df=7) | 92.458 | 90.233 | 7.345 | 1.0 | 12.6 | yes |
| stats: poisson.ppf 3e5 | 195.407 | 7.022 | 7.021 | 27.8 | 27.8 | yes |
| stats: poisson.pmf 1e6 ints | 15.991 | 0.636 | 0.631 | 25.1 | 25.3 | yes |
| stats: zipf.cdf 1e5 ints | 311.001 | 0.570 | 0.563 | 545.5 | 552.5 | yes |
| stats: rankdata 1e6 | 72.749 | 8.475 | 4.261 | 8.6 | 17.1 | yes |
| stats: rankdata 1e6 ties | 47.117 | 7.165 | 4.267 | 6.6 | 11.0 | yes |
| stats: rankdata 3000x300 axis=1 | 27.654 | 8.353 | 0.817 | 3.3 | 33.9 | yes |
| stats: spearmanr 1e6 | 160.545 | 27.519 | 19.492 | 5.8 | 8.2 | yes |
| stats: mannwhitneyu 1e6 | 76.462 | 10.394 | 6.127 | 7.4 | 12.5 | yes |
| stats: wilcoxon 1e6 | 78.257 | 12.972 | 8.745 | 6.0 | 8.9 | yes |
| stats: kruskal 3 x 1e5 | 17.957 | 3.369 | 2.048 | 5.3 | 8.8 | yes |
| stats: mode 1e7 ints | 45.196 | 12.070 | 12.075 | 3.7 | 3.7 | yes |
| stats: gaussian_kde 2-D 2000 x 20000 | 88.968 | 87.698 | 9.479 | 1.0 | 9.4 | yes |
| **stats geomean** | | | | **7.0** | **9.3** | |
