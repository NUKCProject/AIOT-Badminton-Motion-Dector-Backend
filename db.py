import os
from datetime import datetime
from pymongo import MongoClient

# MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017/")
MONGO_URL = "mongodb+srv://sheep5168947:K3gELDD3DalG3quO@cluster0.s55hmhg.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URL)
db = client["badminton"]

# 資料表
reference_raw_collection = db["reference_raw_waveforms"]
training_raw_collection = db["training_raw_waveforms"]
reference_collection = db["reference_waveforms"]
training_collection = db["training_waveforms"]
labeled_windows_collection = db["labeled_windows"]

# 存 reference raw waveforms
def save_reference_raw_waveforms(waveform, action=None, device_id=None):
    doc = {
        "collected_at": datetime.utcnow(),
        "waveform": waveform
    }
    if action:
        doc["action"] = action
    if device_id:
        doc["device_id"] = device_id
    reference_raw_collection.insert_one(doc)

# 存 training raw waveforms
def save_training_raw_waveforms(waveform, action=None, device_id=None):
    doc = {
        "collected_at": datetime.utcnow(),
        "waveform": waveform
    }
    if action:
        doc["action"] = action
    if device_id:
        doc["device_id"] = device_id
    training_raw_collection.insert_one(doc)

# 存人工挑選的 reference 小段
def save_reference_waveform(action, waveform):
    reference_collection.insert_one({
        "action": action,
        "waveform": waveform,
        "created_at": datetime.utcnow()
    })

# 存 auto-label完的 training 小段
def save_training_waveform(action, waveform, device_id=None, speed=None):
    doc = {
        "action": action,
        "waveform": waveform,
        "created_at": datetime.utcnow()
    }
    if device_id:
        doc["device_id"] = device_id
    if speed is not None:
        doc["speed"] = speed
    training_collection.insert_one(doc)

# 撈 reference raw waveforms
def get_reference_raw_waveforms(action, device_id):
    return list(reference_raw_collection.find({
        "action": action,
        "device_id": device_id
    }))

# 撈 training raw waveforms
def get_training_raw_waveforms(action, device_id):
    return list(training_raw_collection.find({
        "action": action,
        "device_id": device_id
    }))
