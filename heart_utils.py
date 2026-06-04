import re
from typing import Dict, List

import pandas as pd

FEATURES: List[str] = [
    "age",
    "gender",
    "chestpain",
    "restingBP",
    "serumcholestrol",
    "fastingbloodsugar",
    "restingrelectro",
    "maxheartrate",
    "exerciseangia",
    "oldpeak",
    "slope",
    "noofmajorvessels",
]


def risk_level(score: float) -> str:
    if score < 25:
        return "Low"
    if score < 50:
        return "Moderate"
    if score < 75:
        return "High"
    return "Critical"


def encode_inputs(
    age: int,
    gender: str,
    chestpain: str,
    resting_bp: int,
    serumcholesterol: int,
    fastingbloodsugar: str,
    restingelectro: str,
    maxheartrate: int,
    exerciseangina: str,
    oldpeak: float,
    slope: str,
    noofmajorvessels: int,
) -> pd.DataFrame:
    row: Dict[str, float] = {
        "age": age,
        "gender": 0 if gender == "Male" else 1,
        "chestpain": ["Atypical Angina", "Non-Anginal Pain", "Asymptomatic", "Typical Angina"].index(chestpain),
        "restingBP": resting_bp,
        "serumcholestrol": serumcholesterol,
        "fastingbloodsugar": 1 if fastingbloodsugar == "> 120 mg/dl" else 0,
        "restingrelectro": ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restingelectro),
        "maxheartrate": maxheartrate,
        "exerciseangia": 1 if exerciseangina == "Yes" else 0,
        "oldpeak": oldpeak,
        "slope": ["Upsloping", "Flat", "Downsloping"].index(slope),
        "noofmajorvessels": noofmajorvessels,
    }
    return pd.DataFrame([row])[FEATURES]


def extract_json_block(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)
    block = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    return block.group(0) if block else stripped


def get_status_for_feature(feature: str, value):
    if feature == "restingBP":
        return "Abnormal" if value > 140 else "Normal"
    if feature == "serumcholestrol":
        return "Abnormal" if value > 240 else "Normal"
    if feature == "oldpeak":
        return "Warning" if value > 2 else "Normal"
    if feature == "noofmajorvessels":
        return "Warning" if value > 1 else "Normal"
    if feature == "exerciseangia":
        return "Warning" if value == 1 else "Normal"
    return "Normal"
