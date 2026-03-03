from datetime import datetime
from typing import Optional, List, Dict, Any
from math import radians, sin, cos, asin, sqrt
import re
import math
import os
import logging

import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

# ======================================================
# FastAPI App Setup
# ======================================================
app = FastAPI(
    title="Spark Parking Recommender API",
    version="1.1.0",
    description="API for recommending parking areas and analyzing user feedback",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

# ======================================================
# Helper Functions
# ======================================================
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return R * c

def yn_to_int(val: Any) -> int:
    if isinstance(val, str):
        s = val.strip().upper()
        if s.startswith("Y"):
            return 1
        if s.startswith("N"):
            return 0
    if isinstance(val, (int, float)):
        return int(bool(val))
    return 0

def discount_to_int(val: Any) -> int:
    if isinstance(val, str):
        s = val.strip().upper()
        if "EXEMPT" in s or "DISCOUNT" in s or "YES" in s:
            return 1
    return 0

def rate_to_float(val: Any) -> float:
    if isinstance(val, (int, float)):
        try:
            if np.isnan(val):
                return 0.0
        except Exception:
            pass
        return float(val)
    if isinstance(val, str):
        m = re.search(r"(\d+(\.\d+)?)", val.replace(",", ""))
        if m:
            return float(m.group(1))
    return 0.0

def parse_hour_from_str(s: Any) -> Optional[int]:
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s or s.upper() == "N/A":
        return None
    if "24/7" in s.upper():
        return 0
    try:
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            return None
        return int(dt.hour)
    except Exception:
        return None

def compute_open_now(opening: Any, closing: Any, hour: int) -> int:
    if isinstance(opening, str) and "24/7" in opening.upper():
        return 1
    if isinstance(closing, str) and "24/7" in closing.upper():
        return 1

    open_h = parse_hour_from_str(opening)
    close_h = parse_hour_from_str(closing)

    if open_h is None or close_h is None:
        return 1

    if open_h == close_h:
        return 1

    if open_h < close_h:
        return int(open_h <= hour < close_h)
    else:
        return int(hour >= open_h or hour < close_h)

# ======================================================
# Load Parking Excel Data
# ======================================================
def load_parking_excel(path: str) -> List[Dict[str, Any]]:
    try:
        xls = pd.ExcelFile(path)
    except Exception as e:
        logging.error(f"Error opening Excel file: {e}")
        return []

    all_rows = []
    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet)
            df.columns = [c.strip().upper() for c in df.columns]
            rename_map = {
                "PARKING NAME": "name",
                "DETAILS": "details",
                "ADDRESS": "address",
                "OPENING": "opening",
                "CLOSING": "closing",
                "LATITUDE": "lat",
                "LONGITUDE": "lng",
                "GUARDS": "guards_raw",
                "CCTVS": "cctvs_raw",
                "INITIAL RATE": "initial_rate_raw",
                "PWD/SC DISCOUNT": "discount_raw",
                "STREET PARKING": "street_raw",
            }
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
            df = df.dropna(subset=["lat", "lng"])
            df["city"] = sheet
            all_rows.extend(df.to_dict(orient="records"))
        except Exception as e:
            logging.warning(f"Error reading sheet {sheet}: {e}")

    logging.info(f"Loaded {len(all_rows)} parking rows")
    return all_rows

# ======================================================
# Paths & Model Loading
# ======================================================
MODULE_DIR = os.path.dirname(__file__)

EXCEL_PATH = os.path.join(MODULE_DIR, "PARKING.xlsx")
RECOMMENDER_MODEL_PATH = os.path.join(MODULE_DIR, "parking_recommender_model_v6.joblib")
NLP_MODEL_PATH = os.path.join(MODULE_DIR, "nlp_feedback_classifier.joblib")

PARKINGS = load_parking_excel(EXCEL_PATH)

try:
    recommender_model = joblib.load(RECOMMENDER_MODEL_PATH)
    logging.info("Parking recommender model loaded")
except Exception as e:
    logging.error(f"Failed to load recommender model: {e}")
    recommender_model = None

try:
    nlp_model = joblib.load(NLP_MODEL_PATH)
    logging.info("NLP feedback model loaded")
except Exception as e:
    logging.warning(f"NLP model not loaded: {e}")
    nlp_model = None

# ======================================================
# Request / Response Schemas
# ======================================================
class ParkingRequest(BaseModel):
    user_lat: float
    user_lng: float
    time_of_day: int
    day_of_week: Optional[int] = None

class FeedbackNLPRequest(BaseModel):
    text: str
    concern: Optional[str] = None

class FeedbackNLPResponse(BaseModel):
    sentiment: str
    category: str

# ======================================================
# Endpoints
# ======================================================
@app.get("/")
def home():
    return {"message": "Spark Parking API running!"}

# ---------- NLP FEEDBACK ENDPOINT ----------
@app.post("/analyze-feedback", response_model=FeedbackNLPResponse)
def analyze_feedback(req: FeedbackNLPRequest):
    if nlp_model is None:
        raise HTTPException(status_code=500, detail="NLP model not loaded")

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Feedback text is empty")

    try:
        prediction = nlp_model.predict([text])[0]
        result = {
            "sentiment": prediction.get("sentiment", "Unknown"),
            "category": prediction.get("category", "General"),
        }

        logging.info({
            "feedback": text,
            "sentiment": result["sentiment"],
            "category": result["category"],
            "timestamp": datetime.utcnow().isoformat()
        })

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NLP analysis failed: {str(e)}")

# ---------- PARKING RECOMMENDATION ----------
@app.post("/recommend")
def recommend(req: ParkingRequest, top_k: int = 5):
    if recommender_model is None:
        raise HTTPException(status_code=500, detail="Recommender model not loaded")

    feature_rows = []
    parking_info = []

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for p in PARKINGS:
        try:
            lat = float(p["lat"])
            lng = float(p["lng"])
            dist_km = haversine_km(req.user_lat, req.user_lng, lat, lng)

            open_now = compute_open_now(
                p.get("opening"), p.get("closing"), req.time_of_day
            )

            features = [
                dist_km,
                open_now,
                yn_to_int(p.get("cctvs_raw")),
                yn_to_int(p.get("guards_raw")),
                rate_to_float(p.get("initial_rate_raw")),
                discount_to_int(p.get("discount_raw")),
                yn_to_int(p.get("street_raw")),
            ]

            feature_rows.append(features)
            parking_info.append({
                "name": p.get("name"),
                "address": p.get("address"),
                "lat": lat,
                "lng": lng,
                "opening": p.get("opening"),
                "closing": p.get("closing"),
                "initial_rate": features[4],
                "distance_km": dist_km,
                "open_now": open_now,
            })

        except Exception as e:
            logging.warning(f"Skipping parking due to error: {e}")

    X = np.array(feature_rows, dtype=float)
    scores = recommender_model.predict(X)

    results = [
        {**info, "score": score}
        for info, score in zip(parking_info, scores)
    ]

    results = sorted(results, key=lambda r: r["score"], reverse=True)[:top_k]

    response = {
        "recommendations": results,
        "current_time": current_time,
    }

    return sanitize_for_json(jsonable_encoder(response))

# ======================================================
# JSON Sanitizer
# ======================================================
def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj
