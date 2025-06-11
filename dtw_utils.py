import numpy as np
from fastdtw import fastdtw

# 
def dtw_abs_six_axis_mean(input_waveform, reference_waveform):
    """
    Calculate the mean of the absolute difference of the DTW distances
    of the 6 axes (ax, ay, az, gx, gy, gz) between the input waveform
    and the reference waveform.

    This function takes in two lists of dictionaries, where each dictionary
    represents a point in the waveform and contains the acceleration and
    angular velocity values of the point. The function calculates the DTW
    distance of each axis separately, takes the absolute difference of the
    distance, and then returns the mean of the absolute differences of all
    axes.

    Parameters
    ----------
    input_waveform : list of dictionaries
        The input waveform to compare with the reference waveform.
    reference_waveform : list of dictionaries
        The reference waveform to compare with the input waveform.

    Returns
    -------
    float
        The mean of the absolute differences of the DTW distances of all
        axes between the input waveform and the reference waveform.
    """

    keys = ["ax", "ay", "az", "gx", "gy", "gz"]
    total = 0
    for key in keys:
        a = np.abs([p[key] for p in input_waveform])
        b = np.abs([p[key] for p in reference_waveform])
        dist, _ = fastdtw(a, b)
        total += dist
    return total / len(keys)

def dtw_abs_six_axis_mean_with_mean_check(input_waveform, reference_waveform, max_mean_ratio_diff=0.5):
    """
    Calculate the mean of the absolute DTW distances of the 6 axes 
    (ax, ay, az, gx, gy, gz) between the input and reference waveforms,
    with a check on the mean ratio difference.

    This function computes the DTW distances for each axis separately 
    after verifying that the mean ratio difference between the input 
    and reference waveforms does not exceed a specified threshold. 
    If the mean ratio difference is too large, the function returns 
    infinity to indicate dissimilarity.

    Parameters
    ----------
    input_waveform : list of dictionaries
        The input waveform to compare with the reference waveform.
    reference_waveform : list of dictionaries
        The reference waveform to compare with the input waveform.
    max_mean_ratio_diff : float, optional
        The maximum allowed ratio difference of means between input 
        and reference waveforms. Default is 0.5.

    Returns
    -------
    float
        The mean of the absolute DTW distances of all axes, or infinity 
        if the mean ratio difference exceeds the threshold.
    """

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
    """
    Calculate the mean of the absolute DTW distances of the 6 axes 
    (ax, ay, az, gx, gy, gz) between the input and reference waveforms.

    Parameters
    ----------
    input_waveform : list of dictionaries
        The input waveform to compare with the reference waveform.
    reference_waveform : list of dictionaries
        The reference waveform to compare with the input waveform.

    Returns
    -------
    float
        The mean of the absolute DTW distances of all axes.
    """
    keys = ["ax", "ay", "az", "gx", "gy", "gz"]
    total = 0
    for key in keys:
        a = [p[key] for p in input_waveform]
        b = [p[key] for p in reference_waveform]
        dist, _ = fastdtw(a, b)
        total += dist
    return total / len(keys)
