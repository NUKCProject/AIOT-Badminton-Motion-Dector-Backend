import os
from datetime import datetime
from pymongo import MongoClient

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URL)
db = client["badminton"]

reference_collection = db["reference_waveforms"]
training_collection = db["training_data"]
raw_collection = db["raw_waveforms"]

def get_reference_waveforms(action):
    return list(reference_collection.find({"action": action}))

def save_reference_waveform(action, waveform):
    reference_collection.insert_one({
        "action": action,
        "waveform": waveform,
        "created_at": datetime.utcnow()
    })

def save_training_data(action, waveform, device_id=None):
    doc = {
        "action": action,
        "waveform": waveform,
        "created_at": datetime.utcnow()
    }
    if device_id:
        doc["device_id"] = device_id
    training_collection.insert_one(doc)

def save_raw_data(waveform, action=None, device_id=None):
    doc = {
        "collected_at": datetime.utcnow(),
        "waveform": waveform
    }
    if action:
        doc["action"] = action
    if device_id:
        doc["device_id"] = device_id
    raw_collection.insert_one(doc)