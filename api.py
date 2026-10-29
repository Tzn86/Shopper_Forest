"""
STEP 3: A small API that lets the web app ask the model questions.

Run:  uvicorn api:app --reload
Then open http://localhost:8000 in your browser.

Endpoints:
  GET  /api/meta     -> model info (metrics, feature importance, dropdown options)
  POST /api/predict  -> send a visit, get back the buy probability + every tree's vote
"""
import json
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model" / "shopper_forest.joblib"
META_PATH = ROOT / "model" / "meta.json"

if not MODEL_PATH.exists():
    raise RuntimeError("No trained model found. Run `python train.py` first.")

model = joblib.load(MODEL_PATH)
meta = json.loads(META_PATH.read_text())
prep = model.named_steps["prep"]
forest = model.named_steps["forest"]

app = FastAPI(title="Will this visitor buy?")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Visit(BaseModel):
    """One website visit. Every field is optional: missing ones use typical values."""
    Administrative: Optional[int] = Field(None, ge=0, description="Account/login pages viewed")
    Administrative_Duration: Optional[float] = Field(None, ge=0, description="Seconds on account pages")
    Informational: Optional[int] = Field(None, ge=0, description="Info pages viewed (shipping, FAQ...)")
    Informational_Duration: Optional[float] = Field(None, ge=0)
    ProductRelated: Optional[int] = Field(None, ge=0, description="Product pages viewed")
    ProductRelated_Duration: Optional[float] = Field(None, ge=0, description="Seconds on product pages")
    BounceRates: Optional[float] = Field(None, ge=0, le=1)
    ExitRates: Optional[float] = Field(None, ge=0, le=1)
    PageValues: Optional[float] = Field(None, ge=0)
    SpecialDay: Optional[float] = Field(None, ge=0, le=1, description="Closeness to a holiday (0-1)")
    OperatingSystems: Optional[int] = None
    Browser: Optional[int] = None
    Region: Optional[int] = None
    TrafficType: Optional[int] = None
    Weekend: Optional[bool] = None
    Month: Optional[str] = None
    VisitorType: Optional[str] = None


def build_row(visit: Visit) -> dict:
    """Fill in missing values so the model always gets a complete row."""
    given = visit.model_dump(exclude_none=True)

    for col in ("Month", "VisitorType"):
        if col in given and given[col] not in meta["options"][col]:
            raise HTTPException(422, f"{col} must be one of {meta['options'][col]}")

    row = dict(meta["defaults"])
    row.update(given)

    # If page counts were sent without durations, estimate a typical time per page
    for count_col, sec in meta["sec_per_page"].items():
        dur_col = f"{count_col}_Duration"
        if count_col in given and dur_col not in given:
            row[dur_col] = given[count_col] * sec

    row["Weekend"] = int(bool(row["Weekend"]))
    return row


@app.get("/api/meta")
def get_meta():
    return meta


@app.post("/api/predict")
def predict(visit: Visit):
    row = build_row(visit)
    X = pd.DataFrame([row])

    probability = float(model.predict_proba(X)[0][1])

    # Ask every single tree for its vote (1 = "will buy", 0 = "won't buy")
    X_numbers = prep.transform(X)
    tree_votes = [int(tree.predict(X_numbers)[0]) for tree in forest.estimators_]

    return {
        "probability": round(probability, 4),
        "label": "will_buy" if probability >= 0.5 else "wont_buy",
        "votes_buy": sum(tree_votes),
        "votes_total": len(tree_votes),
        "tree_votes": tree_votes,
        "used_values": row,
    }


# Serve the web app (frontend/index.html) at http://localhost:8000
app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
