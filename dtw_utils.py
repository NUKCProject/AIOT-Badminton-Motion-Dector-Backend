import numpy as np
from fastdtw import fastdtw


def compute_acc_magnitude(waveform):
    ax = np.array([p["ax"] for p in waveform])
    ay = np.array([p["ay"] for p in waveform])
    az = np.array([p["az"] for p in waveform])
    return np.sqrt(ax**2 + ay**2 + az**2)


def dtw_acc_magnitude(input_waveform, reference_waveform):
    input_mag = compute_acc_magnitude(input_waveform)
    ref_mag = compute_acc_magnitude(reference_waveform)
    distance, _ = fastdtw(input_mag, ref_mag)
    return distance


def dtw_abs_mean_six_axis(input_waveform, reference_waveform):
    keys = ["ax", "ay", "az", "gx", "gy", "gz"]
    total = 0
    for key in keys:
        a = np.abs([p[key] for p in input_waveform])
        b = np.abs([p[key] for p in reference_waveform])
        dist, _ = fastdtw(a, b)
        total += dist
    return total / len(keys)


def dtw_distance(seq1, seq2):
    n, m = len(seq1), len(seq2)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(seq1[i - 1] - seq2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return dtw[n, m]
