from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from dtw_utils import dtw_distance, compute_acc_magnitude
from db import get_reference_waveforms, save_training_data, save_raw_data, save_reference_waveform
from datetime import datetime

app = FastAPI()

class IMUPoint(BaseModel):
    ts: int
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float

class VerifyRequest(BaseModel):
    action_type: str
    waveform: List[IMUPoint]

class RawDataRequest(BaseModel):
    waveform: List[IMUPoint]
    action: Optional[str] = None

class ReferenceInsertRequest(BaseModel):
    action: str
    waveform: List[IMUPoint]

@app.post("/training-label/verify")
def verify_and_store(req: VerifyRequest):
    input_wave = [p.dict() for p in req.waveform]
    input_magnitude = compute_acc_magnitude(input_wave)

    ref_waveforms = get_reference_waveforms(req.action_type)
    if not ref_waveforms:
        return {"error": "No reference waveforms found for this action"}

    scores = []
    for ref in ref_waveforms:
        ref_mag = compute_acc_magnitude(ref["waveform"])
        score = dtw_distance(input_magnitude, ref_mag)
        scores.append(score)

    min_score = min(scores)
    avg_score = sum(scores) / len(scores)

    THRESHOLD = 2.5

    if min_score < THRESHOLD and min_score < avg_score * 1.1:
        save_training_data(req.action_type, input_wave)
        return {
            "status": "accepted",
            "dtw_min_score": min_score,
            "dtw_avg_score": avg_score,
            "saved": True
        }
    else:
        return {
            "status": "rejected",
            "dtw_min_score": min_score,
            "dtw_avg_score": avg_score,
            "saved": False
        }

@app.post("/record-raw-data")
def record_raw(req: RawDataRequest):
    save_raw_data([p.dict() for p in req.waveform], req.action)
    return {"status": "ok"}

@app.post("/insert-reference")
def insert_reference(req: ReferenceInsertRequest):
    save_reference_waveform(req.action, [p.dict() for p in req.waveform])
    return {"status": "reference saved"}

