import numpy as np
import scipy.signal as sg


def cases():
    rng = np.random.default_rng(0)
    x200, x1000 = rng.standard_normal(200), rng.standard_normal(1000)
    sos, (b, a) = sg.butter(4, 0.2, output="sos"), sg.butter(4, 0.2)
    voxels = rng.standard_normal((1000, 150))
    epochs = rng.standard_normal((200, 500))
    multi = rng.standard_normal((64, 250_000))
    sos8, (b8, a8) = sg.butter(8, 0.1, output="sos"), sg.butter(8, 0.1)

    yield "butter + sosfiltfilt, 200 samples", lambda: sg.sosfiltfilt(sg.butter(4, [0.05, 0.3], "band", output="sos"), x200)
    yield "filtfilt, 1000 samples", lambda: sg.filtfilt(b, a, x1000)
    yield "sosfiltfilt, 1000 samples", lambda: sg.sosfiltfilt(sos, x1000)
    yield "detrend, 1000 samples", lambda: sg.detrend(x1000)
    yield "decimate q=4, 1000 samples", lambda: sg.decimate(x1000, 4)
    yield "hilbert, 1000 samples", lambda: sg.hilbert(x1000)
    yield "welch, 1000 samples", lambda: sg.welch(x1000, nperseg=256)
    yield "per-voxel loop, 1000 x sosfiltfilt(150)", lambda: [sg.sosfiltfilt(sos, v) for v in voxels]
    yield "per-epoch loop, 200 x (detrend, welch)", lambda: [sg.welch(sg.detrend(e), nperseg=128) for e in epochs]
    yield "sosfilt 64 x 250k (butter 8)", lambda: sg.sosfilt(sos8, multi)
    yield "lfilter 64 x 250k (butter 8)", lambda: sg.lfilter(b8, a8, multi)
    yield "filtfilt 64 x 250k (butter 8)", lambda: sg.filtfilt(b8, a8, multi)
    yield "sosfiltfilt 64 x 250k (butter 8)", lambda: sg.sosfiltfilt(sos8, multi)
    yield "decimate q=8, 64 x 250k", lambda: sg.decimate(multi, 8)
    yield "resample_poly 3/2, 64 x 250k", lambda: sg.resample_poly(multi, 3, 2, axis=-1)
    yield "medfilt2d k=5, 1024 x 1024", lambda: sg.medfilt2d(rng.random((1024, 1024)), 5)
