# AIOT-Badminton-Motion-Dector-Backend


## How to use
# 1. Clone this repo: `git clone https://github.com/sheep5168947/AIOT-Badminton-Motion-Dector-Backend.git`
# 2. Install the dependencies: `pip install -r requirements.txt`
# 1. Run the server: `uvicorn main:app --reload`

## Database

The database is stored in MongoDB, which is a NoSQL database.

The database is divided into four collections: `reference_raw_waveforms`, `training_raw_waveforms`, `reference_waveforms`, and `training_waveforms`.

url = "mongodb+srv://sheep5168947:K3gELDD3DalG3quO@cluster0.s55hmhg.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"  


## API

The API is defined in `main.py`.
@app.get("/")
@app.post("/record-reference-raw-waveforms")
@app.post("/record-training-raw-waveforms")
@app.post("/insert-reference")
@app.get("/extract-reference")
@app.get("/reference-waveforms")
@app.get("/training-waveforms")
@app.get("/extract-training")
@app.post("/auto-label")
@app.post("/auto-label-peaks")
@app.post("/predict")
