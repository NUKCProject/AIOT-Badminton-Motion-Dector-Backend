from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
from dtw_utils import dtw_abs_six_axis_mean, dtw_raw_six_axis_mean
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
    return {"message": "Hello World"}


# 將 raw 資料存入 reference_raw_waveforms
@app.post("/record-reference-raw-waveforms")
def record_reference_raw(req: RawDataRequest):
    save_reference_raw_waveforms(
        [p.dict() for p in req.waveform], req.action, req.device_id
    )
    return {"status": "ok"}


# 將 raw 資料存入 training_raw_waveforms
@app.post("/record-training-raw-waveforms")
def record_training_raw(req: RawDataRequest):
    save_training_raw_waveforms(
        [p.dict() for p in req.waveform], req.action, req.device_id
    )
    return {"status": "ok"}


# 插入人工挑選的 reference 小段
@app.post("/insert-reference")
def insert_reference(req: ReferenceInsertRequest):
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
    results = get_filtered_reference_waveforms(action)
    for r in results:
        r["_id"] = str(r["_id"])  # 轉為字串
    return results

# 撈 training waveforms
@app.get("/training-waveforms")
def get_training_waveforms(
    action: str = Query(..., description="動作類別"),
):
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


@app.post("/auto-label")
def auto_label(
    action: str = Query(..., description="動作類別"),
    device_id: str = Query(..., description="裝置ID"),
):
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
    "drive": 1300,
    "clear": 1400,
    "drop": 1200,
    "smash": 1800
    }
    THRESHOLD = 0.8  # 調整 DTW 分數的閾值
    threshold_b = DTWVALUESET.get(action, 1200)
    accepted_b = 0
    accepted_c = 0

    for raw in raw_data_list:
        all_waveform = raw["waveform"]
        raw_id = str(raw["_id"])
        idx = 0

        while idx + window_size <= len(all_waveform):
            window = all_waveform[idx : idx + window_size]

            # === 方法 B: 使用六軸絕對值平均 DTW ===
            scores_b = []
            for ref in references:
                ref_wave = ref["waveform"]
                score = dtw_abs_six_axis_mean(window, ref_wave)
                scores_b.append(score)

            min_b = min(scores_b)
            avg_b = sum(scores_b) / len(scores_b)
            print(
                f"Raw ID: {raw_id}, Index: {idx}, Min B: {min_b}, Avg B: {avg_b}"
            )
            if threshold_b*THRESHOLD < avg_b < threshold_b/THRESHOLD:
                save_training_waveform(action, window,method="magnitude", speed=None)
                # record_labeled_window(raw_id, window_index)
                accepted_b += 1

            # === 方法 C: 使用六軸原始值 DTW ===
            # scores_c = []
            # for ref in references:
            #     ref_wave = ref["waveform"]
            #     score = dtw_raw_six_axis_mean(window, ref_wave)
            #     scores_c.append(score)

            # min_c = min(scores_c)
            # avg_c = sum(scores_c) / len(scores_c)
            # print(
            #     f"Raw ID: {raw_id}, Index: {idx}, Min C: {min_c}, Avg C: {avg_c}"
            # )
            # if threshold_c*THRESHOLD < avg_c < threshold_c/THRESHOLD:
            #     save_training_waveform(action, window, device_id, speed=None)
            #     # record_labeled_window(raw_id, window_index)
            #     accepted_c += 1

            idx += stride
            # window_index += 1
        print(
            f"Raw ID: {raw_id}, Accepted B: {accepted_b}, Accepted C: {accepted_c}"
        )
        # remove_raw_and_labeled_if_complete(raw_id, window_size, stride)

    return {"status": "done", "accepted_B": accepted_b, "accepted_C": accepted_c}
