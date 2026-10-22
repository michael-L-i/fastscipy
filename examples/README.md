# Eight everyday SciPy tasks, faster

Same SciPy calls. Less waiting. Realistic tasks on reproducible synthetic data.

**Try it yourself:** [Image-cleanup notebook](median_filter.ipynb) — a memory-bounded
batch targeting about five seconds of ordinary SciPy filtering, with live progress,
fast-path checks, and byte-identical outputs.

*Run on a MacBook Pro with M5 Pro, using two fastscipy threads.*

| Task | SciPy area | Speedup |
|---|---|---:|
| [Fill in a temperature map](#temperature) | `interpolate` | **779.3×** |
| [Clean noisy sensor recordings](#sensors) | `signal` | **6.4×** |
| [Calibrate 8,000 3D points](#calibration) | `linalg` | **105.9×** |
| [Put uncertainty around measurements](#uncertainty) | `stats` | **13.1×** |
| [Order clustered gene-expression panels](#expression) | `cluster` | **3.0×** |
| [Measure chromatography peak areas](#peaks) | `integrate` | **4.0×** |
| [Build independent sparse systems](#systems) | `sparse` | **3.3×** |
| [Map walking distances to depots](#routes) | `csgraph` | **5.2×** |

<a id="temperature"></a>

## 1. Fill in a temperature map

Resample a 64 x 64 weather grid to 160 x 160 with shape-preserving PCHIP interpolation.

```python
model(points)  # prebuilt RegularGridInterpolator(method="pchip")
```

![Fill in a temperature map: SciPy and fastscipy](assets/temperature.gif)

<a id="sensors"></a>

## 2. Clean noisy sensor recordings

Low-pass filter 128 four-second sensor epochs, preserving phase with forward-backward filtering.

```python
signal.sosfiltfilt(sos, epoch)
```

![Clean noisy sensor recordings: SciPy and fastscipy](assets/sensors.gif)

<a id="calibration"></a>

## 3. Calibrate 8,000 3D points

Correct scale and cross-axis coupling with a different 3 x 3 calibration system for each point.

```python
linalg.solve_triangular(calibration_matrices, observed_points)
```

![Calibrate 8,000 3D points: SciPy and fastscipy](assets/calibration.gif)

<a id="uncertainty"></a>

## 4. Put uncertainty around measurements

Compute standard errors for 128 sensor channels, each measured 30 times.

```python
stats.sem(repeated_measurements, axis=0)
```

![Put uncertainty around measurements: SciPy and fastscipy](assets/uncertainty.gif)

<a id="expression"></a>

## 5. Order clustered gene-expression panels

Arrange 128 genes in each of eight expression panels using precomputed hierarchical clusters.

```python
hierarchy.leaves_list(linkage)
```

![Order clustered gene-expression panels: SciPy and fastscipy](assets/expression.gif)

<a id="peaks"></a>

## 6. Measure chromatography peak areas

Integrate 128 chromatograms with different, irregularly sampled time coordinates.

```python
integrate.simpson(intensity, x=retention_time)
```

![Measure chromatography peak areas: SciPy and fastscipy](assets/peaks.gif)

<a id="systems"></a>

## 7. Build independent sparse systems

Assemble eight block-diagonal models from 192 small local coupling matrices.

```python
sparse.block_diag(local_models, format="csr")
```

![Build independent sparse systems: SciPy and fastscipy](assets/systems.gif)

<a id="routes"></a>

## 8. Map walking distances to depots

Find unweighted shortest paths from eight depots through a 120 x 120 warehouse with aisle barriers.

```python
csgraph.shortest_path(aisles, directed=False, unweighted=True, indices=depot)
```

![Map walking distances to depots: SciPy and fastscipy](assets/routes.gif)

---

Synthetic inputs, real timings. Each GIF replays measured completion times on a
shared clock; the slowdown is shown in the frame. Byte-identical outputs.

[Measured timings](measurements.json) · [Runnable source](gallery.py) · [Back to fastscipy](../README.md)

<details>
<summary>Run the gallery</summary>

From the repository root, with fastscipy installed:

```sh
pip install "pillow>=10.1" matplotlib
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python examples/gallery.py
```

Each lane uses the fastest of three warmed batches. Input setup and rendering
are outside the timer. Tasks run one at a time; results depend on hardware and inputs.

</details>
