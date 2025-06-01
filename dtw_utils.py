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

def dtw_abs_six_axis_mean_with_mean_check(input_waveform, reference_waveform, max_mean_ratio_diff=0.5):
    keys = ["ax", "ay", "az", "gx", "gy", "gz"]
    total_dtw = 0
    for key in keys:
        a_raw = [p[key] for p in input_waveform]
        b_raw = [p[key] for p in reference_waveform]

        # 計算平均值差距比例（避免除以 0）
        mean_a = np.mean(np.abs(a_raw))
        mean_b = np.mean(np.abs(b_raw))
        if mean_b == 0:
            return float("inf")  # 視為完全不相似

        ratio = abs(mean_a - mean_b) / mean_b
        if ratio > max_mean_ratio_diff:
            return float("inf")  # 平均值差距太大，排除

        # 符合數值範圍，才繼續計算 DTW
        dist, _ = fastdtw(np.abs(a_raw), np.abs(b_raw))
        total_dtw += dist

    return total_dtw / len(keys)

def dtw_raw_six_axis_mean(input_waveform, reference_waveform):
    keys = ["ax", "ay", "az", "gx", "gy", "gz"]
    total = 0
    for key in keys:
        a = [p[key] for p in input_waveform]
        b = [p[key] for p in reference_waveform]
        dist, _ = fastdtw(a, b)
        total += dist
    return total / len(keys)
