"""scipy.ndimage at image sizes: 2048x2048 images and a 256^3 volume."""
import numpy as np
import scipy.ndimage as ndi


def cases():
    rng = np.random.default_rng(0)
    img = rng.random((2048, 2048))
    img32 = img.astype(np.float32)
    u8 = (img * 255).astype(np.uint8)
    mid = img[:1024, :1024].copy()
    mid32 = mid.astype(np.float32)
    vol = rng.random((256, 256, 256)).astype(np.float32)
    k5 = rng.random((5, 5))
    co = np.mgrid[:1024, :1024].astype(np.float64) * 1.3 + 0.37

    yield "gaussian_filter f64 2048^2 s=2", lambda: ndi.gaussian_filter(img, 2)
    yield "gaussian_filter f32 2048^2 s=2", lambda: ndi.gaussian_filter(img32, 2)
    yield "gaussian_filter u8 2048^2 s=2", lambda: ndi.gaussian_filter(u8, 2)
    yield "gaussian_filter f32 256^3 s=1", lambda: ndi.gaussian_filter(vol, 1)
    yield "uniform_filter f64 2048^2 k=5", lambda: ndi.uniform_filter(img, 5)
    yield "sobel f32 2048^2", lambda: ndi.sobel(img32, 0)
    yield "convolve f64 2048^2 5x5", lambda: ndi.convolve(img, k5)
    yield "correlate f32 256^3 3x3x3", lambda: ndi.correlate(vol, np.ones((3, 3, 3)))
    yield "median_filter f32 1024^2 5x5", lambda: ndi.median_filter(mid32, 5)
    yield "median_filter u8 2048^2 3x3", lambda: ndi.median_filter(u8, 3)
    yield "rank_filter f64 1024^2 5x5", lambda: ndi.rank_filter(mid, 6, size=5)
    yield "percentile_filter f32 1024^2 7x7", lambda: ndi.percentile_filter(mid32, 25, size=7)
    yield "minimum_filter u8 2048^2 5x5", lambda: ndi.minimum_filter(u8, 5)
    yield "maximum_filter f64 2048^2 7x7", lambda: ndi.maximum_filter(img, 7)
    yield "maximum_filter f32 256^3 3^3", lambda: ndi.maximum_filter(vol, 3)
    yield "zoom f64 1024^2 x1.5 o3", lambda: ndi.zoom(mid, 1.5)
    yield "zoom f32 1024^2 x0.5 o1", lambda: ndi.zoom(mid32, 0.5, order=1)
    yield "rotate f64 1024^2 o1", lambda: ndi.rotate(mid, 30, order=1)
    yield "rotate u8 2048^2 o3", lambda: ndi.rotate(u8, 17)
    yield "affine_transform f64 1024^2 o3", lambda: ndi.affine_transform(mid, [[0.9, 0.2], [-0.1, 1.1]], offset=[3, 4])
    yield "map_coordinates f64 1024^2 o3", lambda: ndi.map_coordinates(mid, co)
    yield "spline_filter f64 2048^2 o3", lambda: ndi.spline_filter(img)

    # --- binary morphology, labelling, measurements, distance transform, generic_filter
    blob2 = ndi.gaussian_filter(rng.random((2048, 2048)), 4) > 0.5
    blob2[1000:1030, 1000:1030] = False
    blob3 = ndi.gaussian_filter(rng.random((256, 256, 256)).astype(np.float32), 3) > 0.5
    seed = np.zeros_like(blob2)
    seed[500, 500] = True
    lab, n = ndi.label(blob2)
    lab = lab.astype(np.int32)
    ids = np.arange(1, n + 1)
    holes = blob2 & ~ndi.binary_erosion(blob2, iterations=3)
    small = (ndi.gaussian_filter(rng.random((512, 512)), 1.5) > 0.5)

    yield "binary_erosion 2048^2 cross", lambda: ndi.binary_erosion(blob2)
    yield "binary_dilation 2048^2 3x3", lambda: ndi.binary_dilation(blob2, np.ones((3, 3)))
    yield "binary_erosion 256^3", lambda: ndi.binary_erosion(blob3)
    yield "binary_opening 2048^2 it=3", lambda: ndi.binary_opening(blob2, iterations=3)
    yield "binary_fill_holes 2048^2", lambda: ndi.binary_fill_holes(holes)
    yield "binary_propagation 2048^2", lambda: ndi.binary_propagation(seed, mask=blob2)
    yield "binary_fill_holes 256^3", lambda: ndi.binary_fill_holes(blob3)
    yield "label 2048^2", lambda: ndi.label(blob2)
    yield "label 2048^2 8-conn", lambda: ndi.label(blob2, np.ones((3, 3)))
    yield "label 256^3", lambda: ndi.label(blob3)
    yield "find_objects 2048^2", lambda: ndi.find_objects(lab)
    yield "maximum 2048^2 labels", lambda: ndi.maximum(img, lab, ids)
    yield "extrema 2048^2 labels", lambda: ndi.extrema(img, lab, ids)
    yield "minimum_position 2048^2 labels", lambda: ndi.minimum_position(img, lab, ids)
    yield "center_of_mass 2048^2 labels", lambda: ndi.center_of_mass(img, lab, ids)
    yield "distance_transform_edt 2048^2", lambda: ndi.distance_transform_edt(blob2)
    yield "distance_transform_edt 128^3", lambda: ndi.distance_transform_edt(blob3[:128, :128, :128])
    yield "generic_filter np.mean 512^2 3x3", lambda: ndi.generic_filter(small.astype(np.float64), np.mean, size=3)
    yield "generic_filter np.std 512^2 5x5", lambda: ndi.generic_filter(img[:512, :512], np.std, size=5)
