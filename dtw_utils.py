import numpy as np
from fastdtw import fastdtw

def dtw_abs_six_axis_mean(input_waveform, reference_waveform):
    keys = ["ax", "ay", "az", "gx", "gy", "gz"]
    total = 0
    for key in keys:
        a = np.abs([p[key] for p in input_waveform])
        b = np.abs([p[key] for p in reference_waveform])
        dist, _ = fastdtw(a, b)
        total += dist
    return total / len(keys)

def dtw_raw_six_axis_mean(input_waveform, reference_waveform):
    keys = ["ax", "ay", "az", "gx", "gy", "gz"]
    total = 0
    for key in keys:
        a = [p[key] for p in input_waveform]
        b = [p[key] for p in reference_waveform]
        dist, _ = fastdtw(a, b)
        total += dist
    return total / len(keys)
