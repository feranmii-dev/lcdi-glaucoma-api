from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import joblib, numpy as np

bundle = joblib.load("glaucoma_model.joblib")
model, FEATURES = bundle["model"], bundle["features"]

app = FastAPI(title="LCDI Glaucoma Screening API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your Vercel URL after deploying
    allow_methods=["*"],
    allow_headers=["*"],
)

# All optional -> a missing field becomes NaN and the imputer fills it
class PatientData(BaseModel):
    avg_dev:   Optional[float] = None
    pat_dev:   Optional[float] = None
    ght1:      Optional[float] = None
    ght2:      Optional[float] = None
    ght3:      Optional[float] = None
    ght4:      Optional[float] = None
    ght5:      Optional[float] = None
    lost_fix:  Optional[float] = None
    false_pos: Optional[float] = None
    false_neg: Optional[float] = None
    lf_qual:   Optional[float] = None
    age:       Optional[float] = None
    cdr:       Optional[float] = None
    iop:       Optional[float] = None
    cct:       Optional[float] = None

@app.post("/predict")
def predict(data: PatientData):
    d = data.model_dump()
    # Build the row in the exact training order; None -> NaN for the imputer
    row = [[np.nan if d[f] is None else d[f] for f in FEATURES]]
    X = np.array(row, dtype=float)

    proba = float(model.predict_proba(X)[0, 1])   # P(glaucoma)
    label = proba >= 0.5
    return {
        "prediction": "Glaucoma" if label else "No glaucoma",
        "probability": round(proba, 3),
    }