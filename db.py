import os
from datetime import datetime
from pymongo import MongoClient
from bson import ObjectId

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017/")
# MONGO_URL = "mongodb+srv://sheep5168947:K3gELDD3DalG3quO@cluster0.s55hmhg.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
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
    doc = {"collected_at": datetime.utcnow(), "waveform": waveform}
    if action:
        doc["action"] = action
    if device_id:
        doc["device_id"] = device_id
    reference_raw_collection.insert_one(doc)


# 存 training raw waveforms
def save_training_raw_waveforms(waveform, action=None, device_id=None):
    doc = {"collected_at": datetime.utcnow(), "waveform": waveform}
    if action:
        doc["action"] = action
    if device_id:
        doc["device_id"] = device_id
    training_raw_collection.insert_one(doc)


# 存人工挑選的 reference 小段（支援 speed）
def save_reference_waveform(action, waveform, speed=None):
    doc = {"action": action, "waveform": waveform, "created_at": datetime.utcnow()}
    if speed:
        doc["speed"] = speed
    reference_collection.insert_one(doc)


# 存 auto-label完的 training 小段
def save_training_waveform(action, waveform, method,speed=None):
    doc = {"action": action, "waveform": waveform,method:method, "created_at": datetime.utcnow()}
    if speed is not None:
        doc["speed"] = speed
    training_collection.insert_one(doc)


# 撈 reference raw waveforms
def get_reference_raw_waveforms(action, device_id):
    return list(
        reference_raw_collection.find({"action": action, "device_id": device_id})
    )


# 撈 training raw waveforms
def get_training_raw_waveforms(action, device_id):
    return list(
        training_raw_collection.find({"action": action, "device_id": device_id})
    )


# 紀錄已標記的 window index
def record_labeled_window(raw_id: str, window_index: int):
    labeled_windows_collection.update_one(
        {"raw_id": ObjectId(raw_id)},
        {"$addToSet": {"window_indices": window_index}},
        upsert=True,
    )


# 計算一筆 raw data 可切出幾個 window
def count_total_windows_in_raw(raw_data, window_size=20, stride=5):
    return (len(raw_data["waveform"]) - window_size) // stride + 1


# 若全部 window 已標記完就刪除該筆 raw data 與紀錄
def remove_raw_and_labeled_if_complete(raw_id: str, window_size=20, stride=5):
    raw = reference_raw_collection.find_one({"_id": ObjectId(raw_id)})
    if not raw:
        return
    total_windows = count_total_windows_in_raw(raw, window_size, stride)
    record = labeled_windows_collection.find_one({"raw_id": ObjectId(raw_id)})
    if record and len(set(record.get("window_indices", []))) >= total_windows:
        reference_raw_collection.delete_one({"_id": ObjectId(raw_id)})
        labeled_windows_collection.delete_one({"raw_id": ObjectId(raw_id)})

# 撈 reference waveforms
def get_filtered_reference_waveforms(action):
    return list(reference_collection.find({"action": action}))

# 撈 training waveforms
def get_filtered_training_waveforms(action):
    return list(training_collection.find({"action": action}))

# 撈 training raw waveforms
def get_training_raw_waveforms(action, device_id):
    return list(
        training_raw_collection.find({"action": action, "device_id": device_id})
    )