from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
from dtw_utils import dtw_distance, compute_acc_magnitude
from fastapi.middleware.cors import CORSMiddleware
from db import (
    save_reference_raw_waveforms,
    save_training_raw_waveforms,
    save_reference_waveform,
    get_reference_raw_waveforms,
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
    save_reference_waveform(req.action, [p.dict() for p in req.waveform])
    return {"status": "reference saved"}


# 抽出 reference raw 資料，切成小段 (for人工挑選)
@app.get("/extract-reference")
def extract_reference(
    action_type: str = Query(..., description="指定動作類別，例如 smash、drive"),
    device_id: str = Query(..., description="指定裝置ID，例如 test-device"),
):
    raw_data_list = get_reference_raw_waveforms(action_type, device_id)

    if not raw_data_list:
        return {
            "error": f"No raw data found for action_type={action_type} and device_id={device_id}"
        }

    # 平鋪展開所有資料
    all_waveform = []
    for raw in raw_data_list:
        all_waveform.extend(raw["waveform"])

    all_waveform = sorted(all_waveform, key=lambda x: x["ts"])

    window_size = 25
    stride = 10
    windows = []
    idx = 0
    while idx + window_size <= len(all_waveform):
        window = all_waveform[idx : idx + window_size]
        windows.append({"index": len(windows), "waveform": window})
        idx += stride

    return {"action_type": action_type, "device_id": device_id, "windows": windows}
