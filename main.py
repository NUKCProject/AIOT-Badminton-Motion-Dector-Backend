from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
from dtw_utils import dtw_abs_six_axis_mean, dtw_abs_six_axis_mean_with_mean_check
from fastapi.middleware.cors import CORSMiddleware
from db import (
    save_reference_raw_waveforms,
    save_training_raw_waveforms,
    save_reference_waveform,
    get_reference_raw_waveforms,
    record_labeled_window,
    get_filtered_training_waveforms,
    get_filtered_reference_waveforms,
    get_training_raw_waveforms,
    save_training_waveform,
)
from datetime import datetime
import numpy as np
import pickle
from keras.models import load_model
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
app = FastAPI()
# --- CORS設定開始 ---
origins = [
    "http://localhost:5173",  # 允許vite dev server
    "https://badminton-457613.de.r.appspot.com",  # 允許部署後的前端
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 允許誰可以跨域
    allow_credentials=True,
    allow_methods=["*"],  # 允許所有method (GET, POST, PUT, DELETE...)
    allow_headers=["*"],  # 允許所有header
)


# --- CORS設定結束 --
class IMUPoint(BaseModel):
    ts: int
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    mic_level: int
    mic_peak: int


class RawDataRequest(BaseModel):
    waveform: List[IMUPoint]
    action: Optional[str] = None
    device_id: Optional[str] = None


class ReferenceInsertRequest(BaseModel):
    action: str
    waveform: List[IMUPoint]
    window_index: int
    raw_id: str
    speed: Optional[str] = None


# 設定根路由
@app.get("/")
def root():
    """
    Basic health check route. Returns a simple JSON response with a "message" key.
    """
    
    return {"message": "Hello World"}


# 將 raw 資料存入 reference_raw_waveforms
@app.post("/record-reference-raw-waveforms")
def record_reference_raw(req: RawDataRequest):
    """
    Records raw data as reference data in the database.

    Parameters
    ----------
    req : RawDataRequest
        The request containing the waveform data and optional action and device_id.

    Returns
    -------
    dict
        A JSON response with a status key set to "ok".
    """
    save_reference_raw_waveforms(
        [p.dict() for p in req.waveform], req.action, req.device_id
    )
    return {"status": "ok"}


# 將 raw 資料存入 training_raw_waveforms
@app.post("/record-training-raw-waveforms")
def record_training_raw(req: RawDataRequest):
    """
    Records raw data as training data in the database.

    Parameters
    ----------
    req : RawDataRequest
        The request containing the waveform data and optional action and device_id.

    Returns
    -------
    dict
        A JSON response with a status key set to "ok".
    """
    save_training_raw_waveforms(
        [p.dict() for p in req.waveform], req.action, req.device_id
    )
    return {"status": "ok"}


# 插入人工挑選的 reference 小段
@app.post("/insert-reference")
def insert_reference(req: ReferenceInsertRequest):
    """
    Inserts a manually selected reference waveform into the database.

    Parameters
    ----------
    req : ReferenceInsertRequest
        The request containing the waveform data, action, device_id, raw_id, and window_index.

    Returns
    -------
    dict
        A JSON response with a status key set to "reference saved".
    """
    waveform_dicts = [p.dict() for p in req.waveform]
    print("waveform_dicts", waveform_dicts)
    print("req", req)
    save_reference_waveform(req.action, waveform_dicts, speed=req.speed)
    record_labeled_window(req.raw_id, req.window_index)
    # remove_raw_and_labeled_if_complete(req.raw_id)
    return {"status": "reference saved"}


# 抽出 reference raw 資料，切成小段 (for人工挑選)
@app.get("/extract-reference")
def extract_reference(
    action: str = Query(..., description="指定動作類別，例如 smash、drive"),
    device_id: str = Query(..., description="指定裝置ID，例如 test-device"),
):
    """
    Extracts reference raw waveforms from the database, cuts them into windows of size 30,
    and returns them as a JSON response.

    Parameters
    ----------
    action : str
        The action to extract reference raw waveforms for.
    device_id : str
        The device ID to extract reference raw waveforms for.

    Returns
    -------
    dict
        A JSON response with keys "action", "device_id", and "windows". The "windows" key
        contains a list of dictionaries, each with keys "index", "waveform", and "raw_id". The
        "index" key is the index of the window in the list, the "waveform" key is the window
        itself, and the "raw_id" key is the ID of the raw data that the window was extracted
        from.
    """
    raw_data_list = get_reference_raw_waveforms(action, device_id)

    if not raw_data_list:
        return {
            "error": f"No raw data found for action={action} and device_id={device_id}"
        }

    window_size = 30
    stride = 5
    all_windows = []

    for raw in raw_data_list:
        waveform = raw["waveform"]
        raw_id = str(raw["_id"])
        idx = 0
        while idx + window_size <= len(waveform):
            window = waveform[idx : idx + window_size]
            all_windows.append(
                {"index": len(all_windows), "waveform": window, "raw_id": raw_id}
            )
            idx += stride

    return {"action": action, "device_id": device_id, "windows": all_windows}


# 撈 reference waveforms
@app.get("/reference-waveforms")
def get_reference_waveforms(
    action: str = Query(..., description="動作類別"),
):
    """
    Fetches filtered reference waveforms from the database for a specified action.

    Parameters
    ----------
    action : str
        The action category for which to retrieve reference waveforms.

    Returns
    -------
    list
        A list of dictionaries representing the filtered reference waveforms. Each dictionary
        includes waveform data and has its "_id" field converted to a string.
    """

    results = get_filtered_reference_waveforms(action)
    for r in results:
        r["_id"] = str(r["_id"])  # 轉為字串
    return results

# 撈 training waveforms
@app.get("/training-waveforms")
def get_training_waveforms(
    action: str = Query(..., description="動作類別"),
):
    """
    Fetches filtered training waveforms from the database for a specified action.

    Parameters
    ----------
    action : str
        The action category for which to retrieve training waveforms.

    Returns
    -------
    list
        A list of dictionaries representing the filtered training waveforms. Each dictionary
        includes waveform data and has its "_id" field converted to a string.
    """

    results = get_filtered_training_waveforms(action)
    for r in results:
        r["_id"] = str(r["_id"])  # 轉為字串
    return results

# 撈 training raw waveforms
@app.get("/extract-training")
def extract_training(
    action: str = Query(..., description="動作類別"),
    device_id: str = Query(..., description="裝置ID"),
):
    """
    Extracts training raw waveforms from the database, cuts them into windows of size 30,
    and returns them as a JSON response.

    Parameters
    ----------
    action : str
        The action category to extract training raw waveforms for.
    device_id : str
        The device ID to extract training raw waveforms for.

    Returns
    -------
    dict
        A JSON response with keys "action", "device_id", and "windows". The "windows" key
        contains a list of dictionaries, each with keys "index", "waveform", and "raw_id". The
        "index" key is the index of the window in the list, the "waveform" key is the window
        itself, and the "raw_id" key is the ID of the raw data that the window was extracted
        from.
    """
    raw_data_list = get_training_raw_waveforms(action, device_id)

    if not raw_data_list:
        return {
            "error": f"No raw data found for action={action} and device_id={device_id}"
        }

    window_size = 30
    stride = 5
    all_windows = []

    for raw in raw_data_list:
        waveform = raw["waveform"]
        raw_id = str(raw["_id"])
        idx = 0
        while idx + window_size <= len(waveform):
            window = waveform[idx : idx + window_size]
            all_windows.append(
                {"index": len(all_windows), "waveform": window, "raw_id": raw_id}
            )
            idx += stride

    return {"action": action, "device_id": device_id, "windows": all_windows}


# 自動標記訓練資料

def has_significant_acceleration(waveform, threshold=12.0):
    """
    Checks if a waveform contains at least one point with significant acceleration.

    Parameters
    ----------
    waveform : list of dictionaries
        The waveform to check. Each dictionary should contain the keys "ax", "ay", and "az"
        with the acceleration values of the point.
    threshold : float, optional
        The acceleration threshold. If the absolute value of any of the acceleration values
        is greater than this threshold, the function returns True. Default is 12.0.

    Returns
    -------
    bool
        True if the waveform contains at least one point with significant acceleration, False
        otherwise.
    """
    for point in waveform:
        if abs(point["ax"]) > threshold or abs(point["ay"]) > threshold or abs(point["az"]) > threshold:
            return True
    return False

@app.post("/auto-label")
def auto_label(
    action: str = Query(..., description="動作類別"),
    device_id: str = Query(..., description="裝置ID"),
):
    """
    Automatically labels training data by comparing it with reference waveforms 
    using Dynamic Time Warping (DTW) and saves the segments that meet certain criteria.

    This function processes raw training waveform data, divides it into windows, 
    and calculates the DTW score against reference waveforms. If the score satisfies 
    a predefined threshold, it labels the window and saves it into the database.

    Parameters
    ----------
    action : str
        The action category for which to perform auto-labeling.
    device_id : str
        The device ID associated with the training waveforms.

    Returns
    -------
    dict
        A JSON response with a status key set to "done" and the number of 
        accepted windows labeled as "accepted_B".

    Notes
    -----
    The function uses a threshold mechanism based on predefined DTW values 
    for different actions and an alpha parameter to adjust sensitivity.
    """

    references = get_reference_waveforms(action)
    if not references:
        return {"error": "No reference waveforms found."}

    raw_data_list = get_training_raw_waveforms(action, device_id)
    if not raw_data_list:
        return {"error": "No training raw waveforms found."}

    window_size = 30
    stride = 5
    DTWVALUESET = {
    "toss": 2000,
    "drive": 1500,
    "clear": 1900,
    "drop": 1200,
    "smash": 2400
    }
    ALPHA = 0.9  # 調整 DTW 分數的閾值
    threshold_b = DTWVALUESET.get(action, 1200)
    accepted_b = 0

    for raw in raw_data_list:
        all_waveform = raw["waveform"]
        raw_id = str(raw["_id"])
        idx = 0

        while idx + window_size <= len(all_waveform):
            window = all_waveform[idx : idx + window_size]
            if not has_significant_acceleration(window):
                idx += stride
                continue
            # === 方法 B: 使用六軸絕對值平均 DTW ===
            scores_b = []
            for ref in references:
                ref_wave = ref["waveform"]
                score = dtw_abs_six_axis_mean(window, ref_wave)
                scores_b.append(score)

            min_b = min(scores_b)
            avg_b = sum(scores_b) / len(scores_b)

            # 
            if threshold_b*ALPHA < avg_b < threshold_b/ALPHA and (min_b / avg_b) > 0.8:
                save_training_waveform(action, window,method="magnitude", speed=None)
                # record_labeled_window(raw_id, window_index)
                accepted_b += 1
                print(
                    f"Raw ID: {raw_id}, Index: {idx}, Min B: {min_b}, Avg B: {avg_b}"
                )
            idx += stride
            # window_index += 1
        print(
            f"Raw ID: {raw_id}, Accepted B: {accepted_b}"
        )
        # remove_raw_and_labeled_if_complete(raw_id, window_size, stride)

    return {"status": "done", "accepted_B": accepted_b}


@app.post("/auto-label-peaks")
def auto_label_peaks(
    action: str = Query(..., description="動作類別"),
    device_id: str = Query(..., description="裝置ID"),
):
    """
    Auto-labels training raw waveforms by finding peaks in the waveform.

    Parameters
    ----------
    action : str
        The action category to auto-label training raw waveforms for.
    device_id : str
        The device ID to auto-label training raw waveforms for.

    Returns
    -------
    dict
        A JSON response with keys "status" and "accepted". The "status" key is set to
        "done" if the auto-labeling is successful, and the "accepted" key is the number of
        segments that were auto-labeled.
    """
    
    window_size = 30
    half_window = window_size // 2
    THRESHOLDSET = {
        "toss": 12,
        "drive": 10,
        "clear": 12,
        "drop": 2.5,
        "smash": 12
    }
    threshold = THRESHOLDSET.get(action, 10)
    if threshold is None:
        return {"error": f"No threshold set for action: {action}"}
    accepted = 0

    raw_data_list = get_training_raw_waveforms(action, device_id)
    print(f"Raw data list length: {len(raw_data_list)}")
    if not raw_data_list:
        return {"error": "No training raw waveforms found."}

    for raw in raw_data_list:
        waveform = raw["waveform"]
        raw_id = str(raw["_id"])

        for i in range(len(waveform)):
            point = waveform[i]
            if (
                abs(point["ax"]) > threshold
                or abs(point["ay"]) > threshold
                or abs(point["az"]) > threshold
            ):
                start = i - half_window
                end = i + half_window

                if start < 0 or end > len(waveform):
                    continue

                segment = waveform[start:end]
                save_training_waveform(action, segment, method="peak", speed=None)
                accepted += 1
                print(f"[PEAK] Raw ID: {raw_id}, Center Index: {i}, Saved segment.")

    return {"status": "done", "accepted": accepted}

# load model part
# 載入模型與 LabelEncoder
model = load_model("badminton_model_5class.h5")
with open("label_encoder.pkl", "rb") as f:
    label_encoder = joblib.load(f)


class IMUSample(BaseModel):
    sensor_data: list  # List of 30 dicts, each with ax, ay, az, gx, gy, gz

@app.post("/predict")
def predict(sample: IMUSample):
    # print("Received sample data:", sample.sensor_data)
    """
    Predicts the label of IMU sensor data using a pre-trained model.

    This function receives a sample of IMU sensor data, processes it, and predicts
    the label using a pre-trained neural network model. The function expects exactly
    30 data points in the sample, each containing acceleration and gyro data.

    Parameters
    ----------
    sample : IMUSample
        An object containing a list of 30 dictionaries, each with keys "ax", "ay",
        "az", "gx", "gy", and "gz" representing acceleration and gyro data.

    Returns
    -------
    dict
        A JSON response containing the predicted label and the confidence level
        of the prediction. The label is returned as "other" if the confidence is 
        below 0.999.

    Raises
    ------
    HTTPException
        If the number of data points in the sample is not 30, an HTTPException
        is raised with a 400 status code.
    """

    if len(sample.sensor_data) != 30:
        raise HTTPException(status_code=400, detail="需要 30 筆資料")

    # 預處理成 numpy array
    x = np.array([[[
        p["ax"], p["ay"], p["az"], p["gx"], p["gy"], p["gz"]
    ] for p in sample.sensor_data]])

    # 預測
    probs = model.predict(x)[0]
    max_idx = np.argmax(probs)
    max_prob = probs[max_idx]

    if max_prob >= 0.999:
        label = label_encoder.inverse_transform([max_idx])[0]
    else:
        label = "other"
    print(f"Predicted label: {label}, Confidence: {max_prob}")

    return {"prediction": label, "confidence": float(max_prob)}