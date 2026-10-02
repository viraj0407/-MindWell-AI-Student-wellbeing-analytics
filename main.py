from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "Mental_Health_score_Preidiction_Model.pkl"

model = joblib.load(MODEL_PATH)

# Same 10 values the notebook used to build `Group_Country`
TOP_COUNTRIES = ["Other", "India", "USA", "Canada", "Australia",
                 "UK", "Germany", "Mexico", "Turkey", "France"]

app = FastAPI(title="Student Mental Health Score Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class StudentData(BaseModel):
    age: int = Field(..., ge=10, le=100)
    gender: Literal["Male", "Female"]
    country: str = "Other"
    academic_level: Literal["Undergraduate", "Graduate", "High School"]
    most_used_platform: Literal[
        "Facebook", "LinkedIn", "Instagram", "Snapchat", "Twitter", "YouTube",
        "TikTok", "LINE", "KakaoTalk", "VKontakte", "WhatsApp", "WeChat",
    ]
    purpose_of_use: Literal["Networking", "Education", "Entertainment", "News"]
    avg_daily_usage_hours: float = Field(..., ge=0, le=24)
    daily_unlocks: int = Field(..., ge=0)
    study_hours: float = Field(..., ge=0, le=24)
    physical_activity_hours: float = Field(..., ge=0, le=24)
    sleep_hours_per_night: float = Field(..., ge=0, le=24)
    stress_level: Literal["Low", "Medium", "High", "Very High"]


class PredictionResponse(BaseModel):
    predicted_mental_health_score: float


@app.get("/api/meta")
def meta():
    """Model facts shown on the portfolio page (from the notebook results)."""
    return {
        "model": "Random Forest Regressor (scikit-learn pipeline)",
        "r2_test": 0.878,
        "mae": 0.35,
        "rows": 4998,
        "score_range": [3.6, 9.4],
        "countries": TOP_COUNTRIES,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: StudentData):
    # The pipeline was trained WITHOUT `Country`; it expects `Group_Country`.
    group_country = data.country if data.country in TOP_COUNTRIES else "Other"

    row = pd.DataFrame([{
        "Age": data.age,
        "Gender": data.gender,
        "Academic_Level": data.academic_level,
        "Most_Used_Platform": data.most_used_platform,
        "Purpose_Of_Use": data.purpose_of_use,
        "Avg_Daily_Usage_Hours": data.avg_daily_usage_hours,
        "Daily_Unlocks": data.daily_unlocks,
        "Study_Hours": data.study_hours,
        "Physical_Activity_Hours": data.physical_activity_hours,
        "Sleep_Hours_Per_Night": data.sleep_hours_per_night,
        "Stress_Level": data.stress_level,
        "Group_Country": group_country,
    }])

    score = float(model.predict(row)[0])
    return PredictionResponse(predicted_mental_health_score=round(score, 2))


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")
