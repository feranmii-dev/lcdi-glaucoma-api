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
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# All optional
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

# Readable labels for each feature
LABELS = {
    "avg_dev": "Mean deviation", "pat_dev": "Pattern std deviation",
    "ght1": "Hemifield 1", "ght2": "Hemifield 2", "ght3": "Hemifield 3",
    "ght4": "Hemifield 4", "ght5": "Hemifield 5",
    "lost_fix": "Lost fixation", "false_pos": "False positives",
    "false_neg": "False negatives", "lf_qual": "Fixation quality",
    "age": "Age", "cdr": "Cup-to-disc ratio",
    "iop": "Intraocular pressure", "cct": "Corneal thickness",
}

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: PatientData):
    d = data.model_dump()
    # Build the row in the exact training order; None -> NaN for the imputer
    row = [[np.nan if d[f] is None else d[f] for f in FEATURES]]
    X = np.array(row, dtype=float)

    proba = float(model.predict_proba(X)[0, 1])   # P(glaucoma)
    label = proba >= 0.5

    # Per-feature contribution: coef * (imputed, scaled value)
    imputed = model.named_steps["impute"].transform(X)
    scaled  = model.named_steps["scale"].transform(imputed)
    coefs   = model.named_steps["model"].coef_[0]
    contribs = scaled[0] * coefs

    signals = sorted(
        (
            {
                "feature": f,
                "label": LABELS[f],
                "value": None if d[f] is None else d[f],
                "contribution": round(float(c), 3),
                "direction": "raises" if c > 0 else "lowers",
            }
            for f, c in zip(FEATURES, contribs)
        ),
        key=lambda s: abs(s["contribution"]),
        reverse=True,
    )

    return {
        "prediction": "Glaucoma" if label else "No glaucoma",
        "probability": round(proba, 3),
        "signals": signals[:4],   # top 4 drivers
    }