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

    walk = np.cumsum(rng.standard_normal(1_000_000))
    peaks = sg.find_peaks(walk)[0]
    noise = rng.standard_normal(1_000_000)
    yield "peak_prominences, random walk 1e6", lambda: sg.peak_prominences(walk, peaks)
    yield "find_peaks(prominence, width), random walk 1e6", lambda: sg.find_peaks(walk, prominence=0, width=0)
    yield "find_peaks(prominence, width), noise 1e6", lambda: sg.find_peaks(noise, prominence=0, width=0)
    ss = (np.diag([0.5, 0.3, 0.2, 0.1]), np.ones((4, 1)), np.ones((1, 4)), np.zeros((1, 1)), 1.0)
    drive = rng.standard_normal(100_000)
    yield "dlsim, 4 states, 100k steps", lambda: sg.dlsim(ss, drive)
    kernel7 = rng.standard_normal((7, 7))
    img, kernel = rng.standard_normal((1024, 1024)), rng.standard_normal((5, 5))
    yield "convolve direct 2-D, 1024^2 * 5x5", lambda: sg.convolve(img, kernel, method="direct")
    yield "correlate direct 2-D same, 1024^2 * 5x5", lambda: sg.correlate(img, kernel, mode="same", method="direct")
    yield "convolve2d same symm, 1024^2 * 5x5", lambda: sg.convolve2d(img, kernel, mode="same", boundary="symm")
    yield "correlate2d full fill, 512^2 * 7x7", lambda: sg.correlate2d(img[:512, :512], kernel7)
