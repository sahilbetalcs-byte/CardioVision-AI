import io
import json
import logging
import os
import pickle
import re
import sqlite3
import base64
from datetime import datetime
from logging.handlers import RotatingFileHandler

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import shap
import streamlit as st
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

try:
    import fitz
except Exception:
    fitz = None

try:
    import google.generativeai as genai
except Exception:
    genai = None


st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.title("Heart Disease Prediction")

# Session-state hardening: prevent KeyError crashes across tabs.
defaults = {
    "gemini_api_key": "",
    "gemini_connected": False,
    "chat_history": [],
    "last_prediction": None,
    "last_risk_score": None,
    "last_risk_level": None,
    "last_patient_name": "",
    "last_shap_values": None,
    "ocr_extracted": {},
    "autofill_trigger": False,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

DB_PATH = "patients.db"
MODELS_DIR = "models"
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURES = [
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
MEDICAL_LABELS = {
    "age": "Age",
    "gender": "Gender",
    "chestpain": "Chest Pain Type",
    "restingBP": "Resting Blood Pressure",
    "serumcholestrol": "Total Cholesterol",
    "fastingbloodsugar": "Fasting Blood Sugar",
    "restingrelectro": "Resting ECG",
    "maxheartrate": "Max Heart Rate",
    "exerciseangia": "Exercise Angina",
    "oldpeak": "ST Depression (Oldpeak)",
    "slope": "ST Slope",
    "noofmajorvessels": "Major Vessels",
}
NORMAL_RANGES = {
    "age": "18-100 years",
    "gender": "Male/Female",
    "chestpain": "Any category",
    "restingBP": "90-140 mm Hg",
    "serumcholestrol": "< 240 mg/dl",
    "fastingbloodsugar": "< 120 mg/dl ideal",
    "restingrelectro": "Normal preferred",
    "maxheartrate": "60-220 bpm",
    "exerciseangia": "No preferred",
    "oldpeak": "0.0-2.0",
    "slope": "Upsloping preferred",
    "noofmajorvessels": "0-1 preferred",
}

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)
APP_LOG_PATH = os.path.join(LOGS_DIR, "app.log")
logger = logging.getLogger("heart_disease_app")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = RotatingFileHandler(APP_LOG_PATH, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def mask_api_key(api_key: str) -> str:
    if not api_key:
        return "<empty>"
    if len(api_key) <= 8:
        return "***"
    return f"{api_key[:4]}...{api_key[-4:]}"


def init_session_state():
    defaults = {
        # Gemini (shared across OCR + AI assistant)
        "gemini_api_key": "",
        "gemini_connected": False,
        # Back-compat (older key name used in this app)
        "api_key": "",
        # Chat
        "chat_history": [],
        # Prediction context for AI assistant
        "last_prediction": None,
        "last_risk_score": None,
        "last_risk_level": None,
        "last_patient_name": "",
        "last_shap_values": None,
        # OCR context
        "ocr_extracted": {},
        "autofill_trigger": False,
        # Back-compat keys used elsewhere in this app
        "ocr_data": {},
        "latest_prediction": None,
        "latest_shap_factors": [],
        "form_age": 50,
        "form_gender": "Male",
        "form_chestpain": "Typical Angina",
        "form_resting_bp": 120,
        "form_serumcholesterol": 200,
        "form_fastingbloodsugar": "<= 120 mg/dl",
        "form_restingelectro": "Normal",
        "form_maxheartrate": 150,
        "form_exerciseangina": "No",
        "form_oldpeak": 0.0,
        "form_slope": "Upsloping",
        "form_noofmajorvessels": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS patient_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT,
            created_at TEXT,
            risk_score REAL,
            risk_level TEXT,
            best_model TEXT,
            prediction INTEGER,
            probability REAL,
            input_json TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def load_assets():
    if not os.path.exists(METADATA_PATH) or not os.path.exists(SCALER_PATH):
        st.error("Models not found. Run `python train_models.py` first.")
        st.stop()
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    model_registry = {}
    for model_name, filename in metadata["models"].items():
        with open(os.path.join(MODELS_DIR, filename), "rb") as f:
            model_registry[model_name] = pickle.load(f)
    return metadata, scaler, model_registry


def risk_level(score: float) -> str:
    if score < 25:
        return "Low"
    if score < 50:
        return "Moderate"
    if score < 75:
        return "High"
    return "Critical"


def encode_inputs(
    age,
    gender,
    chestpain,
    resting_bp,
    serumcholesterol,
    fastingbloodsugar,
    restingelectro,
    maxheartrate,
    exerciseangina,
    oldpeak,
    slope,
    noofmajorvessels,
) -> pd.DataFrame:
    row = {
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


def predict_all_models(model_registry, scaler, input_df):
    scaled = scaler.transform(input_df.values)
    rows = []
    for model_name, model in model_registry.items():
        pred = int(model.predict(scaled)[0])
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(scaled)[0][1])
        else:
            score = float(model.decision_function(scaled)[0])
            proba = float(1.0 / (1.0 + np.exp(-score)))
        rows.append({"Model": model_name, "Prediction": pred, "Probability": proba})
    return pd.DataFrame(rows).sort_values("Probability", ascending=False)


def render_risk_gauge(score: float):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Risk Score (0-100)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0, 25], "color": "#2ECC71"},
                    {"range": [25, 50], "color": "#F1C40F"},
                    {"range": [50, 75], "color": "#E67E22"},
                    {"range": [75, 100], "color": "#E74C3C"},
                ],
            },
        )
    )
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, use_container_width=True)


def render_shap(best_model_name, model, scaler, input_df):
    st.subheader("SHAP Explainability")
    try:
        if plt is None:
            st.info("SHAP waterfall is unavailable because matplotlib is not installed.")
            st.session_state.latest_shap_factors = []
            return

        background = pd.read_csv("Heart_Disease_Dataset.csv")[FEATURES].sample(100, random_state=42)
        background_scaled = scaler.transform(background.values)
        patient_scaled = scaler.transform(input_df.values)

        if best_model_name in ["Decision Tree", "Random Forest", "XGBoost"]:
            explainer = shap.TreeExplainer(model, background_scaled)
            shap_values = explainer.shap_values(patient_scaled)
            values = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
            base = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
        else:
            explainer = shap.Explainer(model.predict_proba, background_scaled)
            explanation = explainer(patient_scaled)
            values = explanation.values[0, :, 1]
            base = explanation.base_values[0, 1]

        exp = shap.Explanation(values=values, base_values=base, data=input_df.iloc[0].values, feature_names=FEATURES)
        fig = plt.figure(figsize=(10, 5))
        shap.plots.waterfall(exp, max_display=10, show=False)
        st.pyplot(fig)
        plt.close(fig)

        factor_df = pd.DataFrame({"feature": FEATURES, "value": input_df.iloc[0].values, "shap": values})
        factor_df["direction"] = factor_df["shap"].apply(lambda x: "increases risk" if x > 0 else "decreases risk")
        st.session_state.latest_shap_factors = (
            factor_df.reindex(factor_df["shap"].abs().sort_values(ascending=False).index).head(5).to_dict(orient="records")
        )
    except Exception as exc:
        st.info(f"SHAP is unavailable for this prediction: {exc}")
        st.session_state.latest_shap_factors = []


def save_patient_record(name, score, level, best_model, pred, prob, input_df):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO patient_predictions
        (patient_name, created_at, risk_score, risk_level, best_model, prediction, probability, input_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            name.strip() or "Unknown",
            datetime.now().isoformat(timespec="seconds"),
            float(score),
            level,
            best_model,
            int(pred),
            float(prob),
            input_df.to_json(orient="records"),
        ),
    )
    conn.commit()
    conn.close()


def get_status_for_feature(feature, value):
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


def _draw_footer(canvas_obj, doc):
    canvas_obj.saveState()
    canvas_obj.setFont("Helvetica", 8)
    canvas_obj.drawString(
        30,
        20,
        "This report is generated by an AI screening tool. It is NOT a medical diagnosis. Always consult a qualified healthcare professional.",
    )
    canvas_obj.restoreState()


def make_professional_pdf(report_data):
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        content = []

        content.append(Paragraph("Heart Disease Risk Assessment Report", styles["Title"]))
        content.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
        content.append(Spacer(1, 10))
        summary_rows = [
            ["Patient Name", report_data["patient_name"]],
            ["Age", str(report_data["age"])],
            ["Gender", report_data["gender"]],
            ["Risk Score", f'{report_data["risk_score"]:.1f}'],
            ["Risk Level", report_data["risk_level"]],
            ["Best Model Prediction", report_data["prediction_text"]],
        ]
        summary_table = Table(summary_rows, colWidths=[180, 300])
        risk_color_map = {
            "Low": colors.green,
            "Moderate": colors.yellow,
            "High": colors.orange,
            "Critical": colors.red,
        }
        summary_table.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                    ("BACKGROUND", (1, 4), (1, 4), risk_color_map.get(report_data["risk_level"], colors.white)),
                ]
            )
        )
        content.append(summary_table)
        content.append(Spacer(1, 8))
        recommendation_map = {
            "Low": "Low risk detected. Maintain healthy lifestyle.",
            "Moderate": "Moderate risk. Consider lifestyle improvements and regular checkups.",
            "High": "High risk detected. Consult a cardiologist soon.",
            "Critical": "Critical risk. Seek immediate medical attention.",
        }
        content.append(Paragraph(f"<b>Recommendation:</b> {recommendation_map[report_data['risk_level']]}", styles["BodyText"]))
        content.append(Spacer(1, 20))

        content.append(Paragraph("Page 2 - Model Results", styles["Heading2"]))
        model_rows = [["Model", "Prediction", "Probability %"]]
        for row in report_data["model_rows"]:
            model_rows.append([row["Model"], row["Prediction"], f'{row["Probability"]:.2f}'])
        model_table = Table(model_rows, colWidths=[200, 160, 120])
        model_style = [("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue)]
        for idx, row in enumerate(report_data["model_rows"], start=1):
            if row["Model"] == report_data["best_model"]:
                model_style.append(("BACKGROUND", (0, idx), (-1, idx), colors.lightgreen))
        model_table.setStyle(TableStyle(model_style))
        content.append(model_table)
        content.append(Spacer(1, 10))
        content.append(Paragraph("Top 5 SHAP Features", styles["Heading3"]))
        shap_rows = [["Feature", "Value", "Impact"]]
        for item in report_data["shap_factors"]:
            shap_rows.append([item["feature"], str(round(float(item["value"]), 3)), item["direction"]])
        shap_table = Table(shap_rows, colWidths=[180, 120, 180])
        shap_table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))
        content.append(shap_table)
        content.append(Spacer(1, 20))

        content.append(Paragraph("Page 3 - Patient Input Data", styles["Heading2"]))
        input_rows = [["Feature", "Value", "Normal Range", "Status"]]
        for feature, value in report_data["input_values"].items():
            input_rows.append([MEDICAL_LABELS.get(feature, feature), str(value), NORMAL_RANGES.get(feature, "-"), get_status_for_feature(feature, value)])
        input_table = Table(input_rows, colWidths=[160, 80, 180, 80])
        input_table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))
        content.append(input_table)

        doc.build(content, onFirstPage=_draw_footer, onLaterPages=_draw_footer)
        buffer.seek(0)
        return buffer.getvalue(), None
    except Exception as exc:
        return None, str(exc)


def _extract_json_block(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)
    block = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    return block.group(0) if block else stripped


def _gemini_rest_generate_content(api_key: str, prompt: str, image_b64: str | None = None, mime_type: str | None = None):
    """
    REST fallback that works without depending on google-generativeai internals.
    Uses x-goog-api-key header so newer key flows are supported when available.
    """
    model_name, model_err = _resolve_gemini_model(api_key)
    if not model_name:
        return None, model_err or "No compatible Gemini model found."
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    parts = [{"text": prompt}]
    if image_b64 and mime_type:
        parts.append({"inline_data": {"mime_type": mime_type, "data": image_b64}})
    payload = {"contents": [{"parts": parts}]}
    response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    if response.status_code >= 400:
        try:
            err = response.json()
        except Exception:
            err = response.text
        logger.warning(
            "Gemini REST generateContent failed | status=%s | model=%s | key=%s",
            response.status_code,
            model_name,
            mask_api_key(api_key),
        )
        return None, f"Gemini REST error ({response.status_code}): {err}"
    try:
        data = response.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return text, None
    except Exception:
        logger.exception("Gemini REST response parsing failed | model=%s", model_name)
        return None, f"Gemini REST response parse failed: {response.text}"


def _resolve_gemini_model(api_key: str):
    """
    Resolve a model that supports generateContent for the caller's key/project.
    """
    url = "https://generativelanguage.googleapis.com/v1beta/models"
    headers = {
        "x-goog-api-key": api_key,
    }
    try:
        response = requests.get(url, headers=headers, timeout=30)
    except Exception as exc:
        logger.exception("Model discovery request failed | key=%s", mask_api_key(api_key))
        return None, f"Model discovery failed: {exc}"

    if response.status_code >= 400:
        try:
            err = response.json()
        except Exception:
            err = response.text
        logger.warning("Model discovery failed | status=%s | key=%s", response.status_code, mask_api_key(api_key))
        return None, f"Model discovery error ({response.status_code}): {err}"

    try:
        data = response.json()
        models = data.get("models", [])
    except Exception:
        logger.exception("Model discovery parse failed")
        return None, f"Model discovery parse failed: {response.text}"

    candidates = []
    for model in models:
        methods = model.get("supportedGenerationMethods", []) or []
        if "generateContent" in methods:
            name = model.get("name", "")
            if name:
                candidates.append(name)

    if not candidates:
        return None, "No models with generateContent support are available for this key."

    preferred = [
        "models/gemini-2.5-flash",
        "models/gemini-2.0-flash",
        "models/gemini-1.5-flash",
        "models/gemini-1.5-pro",
        "models/gemini-pro",
    ]
    for item in preferred:
        if item in candidates:
            return item, None
    return candidates[0], None


def validate_gemini_key(api_key: str):
    # Try REST first; this is more future-proof across key handling changes.
    rest_text, rest_err = _gemini_rest_generate_content(api_key, "Reply with OK only.")
    if rest_text:
        return True, None

    if genai is None:
        return False, rest_err or "google-generativeai package not available."
    try:
        model_name, model_err = _resolve_gemini_model(api_key)
        if not model_name:
            return False, model_err or "No compatible Gemini model found."
        genai.configure(api_key=api_key)
        sdk_model_name = model_name.replace("models/", "")
        model = genai.GenerativeModel(sdk_model_name)
        _ = model.generate_content("Reply with OK only.")
        return True, None
    except Exception as exc:
        logger.exception("Gemini SDK validation failed | key=%s", mask_api_key(api_key))
        sdk_err = f"SDK validation failed: {exc}"
        if rest_err:
            return False, f"{rest_err} | {sdk_err}"
        return False, sdk_err


def run_ocr(uploaded_file, api_key: str):
    if not api_key:
        return None, "Please enter your Gemini API key in the AI Assistant tab first"
    if genai is None:
        return None, "google-generativeai package is not installed."
    try:
        file_name = uploaded_file.name.lower()
        file_bytes = uploaded_file.read()

        if file_name.endswith(".pdf"):
            if fitz is None:
                return None, "PDF conversion dependency missing. Install PyMuPDF."
            try:
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                if doc.page_count == 0:
                    return None, "No page found in uploaded PDF."
                page = doc.load_page(0)
                pix = page.get_pixmap()
                image_bytes = pix.tobytes("png")
                mime_type = "image/png"
                doc.close()
            except Exception as exc:
                logger.exception("PDF conversion failed in OCR | file=%s", file_name)
                return None, f"PDF conversion failed. Details: {exc}"
        else:
            image_bytes = file_bytes
            ext = file_name.split(".")[-1]
            mime_type = "image/png" if ext == "png" else "image/jpeg"

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        prompt = (
            "This is a medical report image. Extract ONLY these values "
            "if present and return as JSON:\n"
            "{\n"
            "  'age': number or null,\n"
            "  'restingBP': systolic blood pressure number or null,\n"
            "  'serumcholestrol': total cholesterol number or null,\n"
            "  'maxheartrate': heart rate number or null,\n"
            "  'fastingbloodsugar': blood sugar number or null,\n"
            "  'oldpeak': ST depression number or null\n"
            "}\n"
            "Return ONLY valid JSON, nothing else."
        )

        text, rest_err = _gemini_rest_generate_content(api_key, prompt, image_b64=image_b64, mime_type=mime_type)
        if not text:
            # Fallback to SDK for compatibility with older environments.
            if genai is None:
                return None, rest_err or "google-generativeai package is not installed."
            model_name, model_err = _resolve_gemini_model(api_key)
            if not model_name:
                return None, model_err or rest_err or "No compatible Gemini model found."
            genai.configure(api_key=api_key)
            sdk_model_name = model_name.replace("models/", "")
            model = genai.GenerativeModel(sdk_model_name)
            response = model.generate_content(
                [
                    prompt,
                    {
                        "mime_type": mime_type,
                        "data": image_b64,
                    },
                ]
            )
            text = getattr(response, "text", "") or ""
        parsed = json.loads(_extract_json_block(text))
        normalized = {
            "age": parsed.get("age"),
            "restingBP": parsed.get("restingBP"),
            "serumcholestrol": parsed.get("serumcholestrol"),
            "maxheartrate": parsed.get("maxheartrate"),
            "fastingbloodsugar_raw": parsed.get("fastingbloodsugar"),
            "oldpeak": parsed.get("oldpeak"),
        }
        return {"parsed": normalized}, None
    except Exception as exc:
        logger.exception("OCR pipeline failed | file=%s", getattr(uploaded_file, "name", "<unknown>"))
        return None, f"OCR failed: {exc}"


def add_health_tips(input_df):
    row = input_df.iloc[0]
    tips = []
    if row["restingBP"] > 140:
        tips.append(("warning", "Your blood pressure is high. Reduce salt intake, exercise regularly, avoid stress."))
    if row["serumcholestrol"] > 240:
        tips.append(("warning", "Your cholesterol is high. Eat less saturated fat, more fiber, consider medication."))
    if row["exerciseangia"] == 1:
        tips.append(("error", "Exercise-induced chest pain detected. Avoid strenuous activity until checked by a doctor."))
    if row["oldpeak"] > 2:
        tips.append(("error", "ST depression is elevated. This is a significant cardiac risk indicator."))
    if row["noofmajorvessels"] > 1:
        tips.append(("error", "Multiple blocked vessels detected. Cardiology consultation strongly recommended."))
    if tips:
        st.subheader("Personalized Health Tips")
        for level, msg in tips:
            if level == "warning":
                st.warning(msg)
            else:
                st.error(msg)


def stream_chat_with_gemini(api_key, user_message, output_placeholder):
    try:
        system_prompt = (
            "You are a helpful medical AI assistant specialized in heart disease and cardiovascular health. "
            "You help patients understand their heart disease prediction results, explain medical terms in simple language, "
            "and provide general lifestyle and wellness advice. Always remind users to consult a real doctor for medical decisions. "
            "Be friendly, empathetic, and clear."
        )
        patient_context = ""
        if st.session_state.get("last_prediction"):
            shap_factors = st.session_state.get("last_shap_values") or []
            shap_text = ", ".join([f"{x['feature']} ({x['direction']})" for x in shap_factors[:4]]) or "No SHAP factors available"
            patient_context = (
                f"Patient context: Risk Score = {st.session_state.get('last_risk_score'):.1f}, "
                f"Risk Level = {st.session_state.get('last_risk_level')}, "
                f"Best Model Prediction = {st.session_state.get('last_prediction')}, "
                f"Key SHAP factors = {shap_text}."
            )
        full_prompt = f"{system_prompt}\n{patient_context}\nUser question: {user_message}"
        output_placeholder.markdown("🤖 AI is thinking...")
        rendered, rest_err = _gemini_rest_generate_content(api_key, full_prompt)
        if rendered:
            output_placeholder.markdown(f'<div class="chat-ai">{rendered}</div>', unsafe_allow_html=True)
            return rendered, None

        if genai is None:
            return None, rest_err or "google-generativeai package not available."

        # SDK fallback path
        model_name, model_err = _resolve_gemini_model(api_key)
        if not model_name:
            return None, model_err or rest_err or "No compatible Gemini model found."
        genai.configure(api_key=api_key)
        sdk_model_name = model_name.replace("models/", "")
        model = genai.GenerativeModel(sdk_model_name)
        response = model.generate_content(full_prompt, stream=True)
        rendered = ""
        dots = 0
        for chunk in response:
            dots = (dots + 1) % 4
            chunk_text = getattr(chunk, "text", "")
            if chunk_text:
                rendered += chunk_text
                output_placeholder.markdown(f'<div class="chat-ai">{rendered}</div>', unsafe_allow_html=True)
            else:
                output_placeholder.markdown(f"🤖 AI is thinking{'.' * dots}")
        return rendered, None
    except Exception as exc:
        logger.exception("Chat request failed | key=%s", mask_api_key(api_key))
        return None, f"Gemini request failed. Verify API key and quota. Details: {exc}"


def color_risk(val):
    mapping = {
        "Low": "background-color: #d4edda; color: #155724",
        "Moderate": "background-color: #fff3cd; color: #856404",
        "High": "background-color: #ffe5b4; color: #7a3e00",
        "Critical": "background-color: #f8d7da; color: #721c24",
    }
    return mapping.get(val, "")


init_session_state()
init_db()
metadata, scaler, model_registry = load_assets()
tabs = st.tabs(["Predict", "What-If Simulation", "📊 Bulk Predict", "Patient History", "AI Assistant", "Model Metrics"])

with tabs[0]:
    st.header("Single Patient Prediction")

    with st.expander("📄 Scan Medical Report (Auto-fill from report)", expanded=False):
        uploaded = st.file_uploader("Upload JPG/PNG/PDF medical report", type=["jpg", "jpeg", "png", "pdf"])
        if uploaded is not None:
            ocr_result, ocr_error = run_ocr(uploaded, st.session_state.get("gemini_api_key", ""))
            if ocr_error:
                st.error(ocr_error)
            else:
                st.success("Report scanned successfully.")
                st.session_state.ocr_extracted = ocr_result["parsed"]
                st.session_state.ocr_data = ocr_result["parsed"]  # back-compat
                parsed = ocr_result["parsed"]
                rows = []
                for key, label in [
                    ("age", "Age"),
                    ("restingBP", "Resting BP"),
                    ("serumcholestrol", "Cholesterol"),
                    ("maxheartrate", "Max Heart Rate"),
                    ("fastingbloodsugar_raw", "Fasting Blood Sugar"),
                    ("oldpeak", "ST Depression"),
                ]:
                    found = parsed.get(key) is not None
                    rows.append({"Field": label, "Extracted Value": parsed.get(key), "Status": "✅ Found" if found else "⚠️ Not Found"})
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
                st.caption("Please verify auto-filled values before predicting.")
                if st.button("Auto-fill Form"):
                    if parsed.get("age") is not None:
                        st.session_state.form_age = int(min(max(parsed["age"], 18), 100))
                    if parsed.get("restingBP") is not None:
                        st.session_state.form_resting_bp = int(min(max(parsed["restingBP"], 60), 300))
                    if parsed.get("serumcholestrol") is not None:
                        st.session_state.form_serumcholesterol = int(min(max(parsed["serumcholestrol"], 100), 600))
                    if parsed.get("maxheartrate") is not None:
                        st.session_state.form_maxheartrate = int(min(max(parsed["maxheartrate"], 60), 220))
                    if parsed.get("oldpeak") is not None:
                        st.session_state.form_oldpeak = float(min(max(parsed["oldpeak"], 0.0), 6.0))
                    fbs_val = parsed.get("fastingbloodsugar_raw")
                    if fbs_val is not None:
                        st.session_state.form_fastingbloodsugar = "> 120 mg/dl" if fbs_val > 120 else "<= 120 mg/dl"
                    st.rerun()

    patient_name = st.text_input("Patient Name", value=st.session_state.get("patient_name", ""))
    st.session_state["patient_name"] = patient_name

    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", min_value=18, max_value=100, key="form_age")
        gender = st.selectbox("Gender", ["Male", "Female"], key="form_gender")
        resting_bp = st.number_input("Resting Blood Pressure", min_value=60, max_value=300, key="form_resting_bp")
        serumcholesterol = st.number_input("Serum Cholesterol", min_value=100, max_value=600, key="form_serumcholesterol")
        maxheartrate = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, key="form_maxheartrate")
        oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=6.0, step=0.1, key="form_oldpeak")
    with c2:
        chestpain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"], key="form_chestpain")
        fastingbloodsugar = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"], key="form_fastingbloodsugar")
        restingelectro = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"], key="form_restingelectro")
        exerciseangina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"], key="form_exerciseangina")
        slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"], key="form_slope")
        noofmajorvessels = st.number_input("Number of Major Vessels", min_value=0, max_value=4, key="form_noofmajorvessels")

    input_df = encode_inputs(
        age,
        gender,
        chestpain,
        resting_bp,
        serumcholesterol,
        fastingbloodsugar,
        restingelectro,
        maxheartrate,
        exerciseangina,
        oldpeak,
        slope,
        noofmajorvessels,
    )

    if st.button("Predict", type="primary"):
        result_df = predict_all_models(model_registry, scaler, input_df)
        best_model_name = metadata["best_model"]
        best_row = result_df[result_df["Model"] == best_model_name].iloc[0]
        prob = float(best_row["Probability"])
        pred = int(best_row["Prediction"])
        score = prob * 100.0
        level = risk_level(score)
        prediction_text = "Heart Disease" if pred == 1 else "No Heart Disease"

        st.subheader("Model Comparison")
        display_df = result_df.copy()
        display_df["Prediction"] = display_df["Prediction"].map({0: "No Heart Disease", 1: "Heart Disease"})
        display_df["Probability"] = (display_df["Probability"] * 100).round(2)
        st.dataframe(display_df, use_container_width=True)

        st.subheader("Best Model Auto-Selected")
        st.success(f"Best model: {best_model_name} (based on validation F1 and ROC-AUC)")
        render_risk_gauge(score)
        st.info(f"Risk Level: **{level}**")
        render_shap(best_model_name, model_registry[best_model_name], scaler, input_df)
        add_health_tips(input_df)

        save_patient_record(patient_name, score, level, best_model_name, pred, prob, input_df)
        st.session_state.latest_prediction = {
            "patient_name": patient_name or "Unknown",
            "age": age,
            "gender": gender,
            "risk_score": score,
            "risk_level": level,
            "best_model": best_model_name,
            "prediction_text": prediction_text,
            "probability": prob,
            "input_values": input_df.iloc[0].to_dict(),
            "model_rows": display_df.to_dict(orient="records"),
        }
        # Context for AI Assistant
        st.session_state.last_prediction = prediction_text
        st.session_state.last_risk_score = score
        st.session_state.last_risk_level = level
        st.session_state.last_patient_name = patient_name or "Unknown"
        st.session_state.last_shap_values = st.session_state.latest_shap_factors

    if st.session_state.latest_prediction:
        report_data = dict(st.session_state.latest_prediction)
        report_data["shap_factors"] = st.session_state.latest_shap_factors
        pdf_bytes, pdf_error = make_professional_pdf(report_data)
        if pdf_error:
            st.error(f"Report generation failed: {pdf_error}")
        else:
            st.download_button(
                "📥 Download Patient Report (PDF)",
                data=pdf_bytes,
                file_name=f"heart_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
            )

with tabs[1]:
    st.header("What-If Simulation")
    st.caption("Adjust sliders to see real-time score changes from the best model.")

    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        base_age = st.slider("Age", 18, 100, st.session_state.form_age)
        base_bp = st.slider("Resting BP", 60, 300, st.session_state.form_resting_bp)
        base_chol = st.slider("Cholesterol", 100, 600, st.session_state.form_serumcholesterol)
    with sim_col2:
        base_hr = st.slider("Max Heart Rate", 60, 220, st.session_state.form_maxheartrate)
        base_oldpeak = st.slider("Oldpeak", 0.0, 6.0, float(st.session_state.form_oldpeak), 0.1)

    simulated = pd.DataFrame([[base_age, 0, 2, base_bp, base_chol, 0, 0, base_hr, 0, base_oldpeak, 1, 0]], columns=FEATURES)
    sim_results = predict_all_models(model_registry, scaler, simulated)
    best_model_name = metadata["best_model"]
    sim_row = sim_results[sim_results["Model"] == best_model_name].iloc[0]
    sim_score = float(sim_row["Probability"]) * 100
    original_score = st.session_state.latest_prediction["risk_score"] if st.session_state.latest_prediction else sim_score

    m1, m2 = st.columns(2)
    m1.metric("Original Risk Score", f"{original_score:.1f}")
    m2.metric("Simulated Risk Score", f"{sim_score:.1f}")
    render_risk_gauge(sim_score)

    delta = sim_score - original_score
    if delta < 0:
        st.success(f"↓ {delta:.1f} reduced")
    elif delta > 0:
        st.error(f"↑ +{delta:.1f} increased")
    else:
        st.info("→ stable")
    base_chol_ref = st.session_state.latest_prediction["input_values"]["serumcholestrol"] if st.session_state.latest_prediction else base_chol
    st.write(
        f"Reducing your cholesterol from {base_chol_ref} to {base_chol} would change your risk by approximately {abs(delta):.1f} points."
    )
    st.write(f"Simulated risk level: **{risk_level(sim_score)}**")

with tabs[2]:
    st.header("📊 Bulk Prediction — Multiple Patients")
    st.info(
        "Upload a CSV file with patient data. Each row = one patient. "
        "The file must have these exact column names."
    )

    required_columns = [
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

    with st.expander("📋 Required Column Names", expanded=False):
        examples = {
            "age": 55,
            "gender": 0,
            "chestpain": 1,
            "restingBP": 130,
            "serumcholestrol": 250,
            "fastingbloodsugar": 1,
            "restingrelectro": 0,
            "maxheartrate": 150,
            "exerciseangia": 0,
            "oldpeak": 1.5,
            "slope": 1,
            "noofmajorvessels": 2,
        }
        for col in required_columns:
            st.write(f"`{col}` (example: {examples[col]})")

    sample_df = pd.DataFrame(
        [
            {
                "age": 55,
                "gender": 0,
                "chestpain": 1,
                "restingBP": 130,
                "serumcholestrol": 250,
                "fastingbloodsugar": 1,
                "restingrelectro": 0,
                "maxheartrate": 150,
                "exerciseangia": 0,
                "oldpeak": 1.5,
                "slope": 1,
                "noofmajorvessels": 2,
            },
            {
                "age": 65,
                "gender": 1,
                "chestpain": 0,
                "restingBP": 120,
                "serumcholestrol": 180,
                "fastingbloodsugar": 0,
                "restingrelectro": 1,
                "maxheartrate": 130,
                "exerciseangia": 1,
                "oldpeak": 2.5,
                "slope": 2,
                "noofmajorvessels": 1,
            },
            {
                "age": 45,
                "gender": 0,
                "chestpain": 3,
                "restingBP": 140,
                "serumcholestrol": 300,
                "fastingbloodsugar": 1,
                "restingrelectro": 2,
                "maxheartrate": 160,
                "exerciseangia": 1,
                "oldpeak": 3.0,
                "slope": 0,
                "noofmajorvessels": 3,
            },
        ],
        columns=required_columns,
    )
    st.download_button(
        "📥 Download Sample CSV Template",
        sample_df.to_csv(index=False),
        file_name="heart_disease_bulk_sample_template.csv",
        mime="text/csv",
    )

    uploaded_csv = st.file_uploader("Choose a CSV file", type=["csv"])
    bulk_results_df = None

    if uploaded_csv is not None:
        try:
            df_upload = pd.read_csv(uploaded_csv)
            st.subheader("Preview")
            st.dataframe(df_upload.head(), use_container_width=True)

            missing_cols = [c for c in required_columns if c not in df_upload.columns]
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")

    if uploaded_csv is not None and st.button("▶ Run Bulk Prediction", type="primary"):
        try:
            st.session_state.bulk_results_df = None
            progress = st.progress(0)
            progress_text = st.empty()

            df_feat = df_upload[required_columns].copy()
            # Ensure numeric types; coercing errors to NaN then failing fast.
            df_feat = df_feat.apply(pd.to_numeric, errors="coerce")
            if df_feat.isna().any().any():
                st.error("CSV contains non-numeric values in required columns. Please fix and retry.")
                st.stop()

            progress_text.text = "Scaling inputs..."
            X_scaled = scaler.transform(df_feat[required_columns].values)
            progress.progress(20)

            model_order = [
                "XGBoost",
                "Logistic Regression",
                "Random Forest",
                "Decision Tree",
                "SVM",
                "Neural Network",
            ]

            preds = {}
            probs = {}
            for idx, model_name in enumerate(model_order):
                progress_text.text = f"Predicting with {model_name}..."
                model = model_registry[model_name]
                preds[model_name] = model.predict(X_scaled)
                if hasattr(model, "predict_proba"):
                    probs[model_name] = model.predict_proba(X_scaled)[:, 1]
                else:
                    scores = model.decision_function(X_scaled)
                    probs[model_name] = 1.0 / (1.0 + np.exp(-scores))
                progress.progress(20 + int(((idx + 1) / len(model_order)) * 70))

            primary_model_name = "XGBoost" if "XGBoost" in model_registry else metadata["best_model"]
            risk_scores = probs[primary_model_name] * 100.0
            risk_levels = [risk_level(s) for s in risk_scores]
            pred_primary = preds[primary_model_name]
            prediction_texts = ["Heart Disease" if int(p) == 1 else "No Heart Disease" for p in pred_primary]

            results = pd.DataFrame(
                {
                    "Patient#": np.arange(1, len(df_feat) + 1),
                    "Age": df_feat["age"].astype(int).values,
                    "Gender": df_feat["gender"].astype(int).values,
                    "Prediction": prediction_texts,
                    "Risk Score": risk_scores,
                    "Risk Level": risk_levels,
                    "XGBoost%": probs["XGBoost"] * 100.0,
                    "LR%": probs["Logistic Regression"] * 100.0,
                    "RF%": probs["Random Forest"] * 100.0,
                    "DT%": probs["Decision Tree"] * 100.0,
                    "SVM%": probs["SVM"] * 100.0,
                    "NN%": probs["Neural Network"] * 100.0,
                }
            )

            # Round for display
            for col in ["Risk Score", "XGBoost%", "LR%", "RF%", "DT%", "SVM%", "NN%"]:
                results[col] = results[col].astype(float).round(2)

            bulk_results_df = results
            st.session_state.bulk_results_df = results
            progress.progress(100)
            progress_text.text = ""
            st.success("Bulk prediction completed!")

            styled = results.style.applymap(color_risk, subset=["Risk Level"])
            st.dataframe(styled, use_container_width=True)

            csv_bytes = results.to_csv(index=False)
            st.download_button(
                "📥 Download Full Results CSV",
                csv_bytes,
                file_name="heart_disease_bulk_predictions.csv",
                mime="text/csv",
            )

        except Exception as exc:
            st.error(f"Bulk prediction failed: {exc}")

    # Visual analytics (after prediction)
    if st.session_state.get("bulk_results_df") is not None:
        results = st.session_state.bulk_results_df

        risk_category_order = ["Low", "Moderate", "High", "Critical"]
        risk_color_map = {"Low": "#2ECC71", "Moderate": "#F1C40F", "High": "#E67E22", "Critical": "#E74C3C"}
        counts = results["Risk Level"].value_counts().reindex(risk_category_order).fillna(0).astype(int)

        chart1 = go.Figure(
            go.Pie(
                labels=risk_category_order,
                values=counts.values,
                marker_colors=[risk_color_map[c] for c in risk_category_order],
                hole=0.35,
            )
        )
        chart1.update_layout(title="Risk Level Distribution")
        st.plotly_chart(chart1, use_container_width=True)

        hd_count = int((results["Prediction"] == "Heart Disease").sum())
        no_hd_count = int((results["Prediction"] == "No Heart Disease").sum())

        chart2 = go.Figure()
        chart2.add_trace(
            go.Bar(
                x=["No Heart Disease", "Heart Disease"],
                y=[no_hd_count, hd_count],
                marker_color=["#2ECC71", "#E74C3C"],
                text=[no_hd_count, hd_count],
                textposition="outside",
                hovertemplate="%{x}: %{y}<extra></extra>",
            )
        )
        chart2.update_layout(title="Heart Disease vs No Heart Disease", yaxis_title="Number of Patients")
        st.plotly_chart(chart2, use_container_width=True)

        scores = results["Risk Score"].astype(float).values
        bins = np.arange(0, 101, 5)  # 0-100 in steps of 5
        hist, bin_edges = np.histogram(scores, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Gradient color from green (0) to red (100)
        def lerp(a, b, t):
            return a + (b - a) * t

        green_rgb = (46, 204, 113)
        red_rgb = (231, 76, 60)
        bar_colors = []
        for c in bin_centers:
            t = max(0.0, min(1.0, c / 100.0))
            r = int(lerp(green_rgb[0], red_rgb[0], t))
            g = int(lerp(green_rgb[1], red_rgb[1], t))
            b = int(lerp(green_rgb[2], red_rgb[2], t))
            bar_colors.append(f"rgb({r},{g},{b})")

        chart3 = go.Figure(
            go.Bar(
                x=[f"{int(a)}-{int(b)}" for a, b in zip(bin_edges[:-1], bin_edges[1:])],
                y=hist,
                marker_color=bar_colors,
                hovertemplate="Risk Score Range %{x}<br>Patients: %{y}<extra></extra>",
            )
        )
        chart3.update_layout(title="Risk Score Distribution Across All Patients", xaxis_title="Risk Score (bins)", yaxis_title="Number of Patients")
        st.plotly_chart(chart3, use_container_width=True)

        chart4 = go.Figure()
        for cat in risk_category_order:
            chart4.add_trace(go.Box(y=results.loc[results["Risk Level"] == cat, "Age"], name=cat))
        chart4.update_layout(title="Age Distribution by Risk Level", xaxis_title="Risk Level", yaxis_title="Age")
        st.plotly_chart(chart4, use_container_width=True)

        total = len(results)
        heart_detected = int((results["Prediction"] == "Heart Disease").sum())
        no_heart_detected = int((results["Prediction"] == "No Heart Disease").sum())
        avg_risk = float(results["Risk Score"].mean()) if total else 0.0
        max_idx = int(results["Risk Score"].idxmax()) if total else 0
        highest_row = results.loc[max_idx]
        most_common = results["Risk Level"].mode().iloc[0] if not results["Risk Level"].empty else "N/A"

        xgb_cm = metadata["metrics"]["XGBoost"]["confusion_matrix"]
        tn, fp = xgb_cm[0]
        fn, tp = xgb_cm[1]
        xgb_total = tn + fp + fn + tp
        xgb_accuracy = ((tn + tp) / xgb_total) * 100 if xgb_total else 0.0

        st.info(
            f"Total patients analyzed: {total}\n"
            f"Heart Disease detected: {heart_detected} ({(heart_detected/total*100):.1f}% )\n"
            f"No Heart Disease: {no_heart_detected} ({(no_heart_detected/total*100):.1f}% )\n"
            f"Average Risk Score: {avg_risk:.2f}\n"
            f"Highest Risk Patient: Patient#{int(highest_row['Patient#'])} (Risk: {float(highest_row['Risk Score']):.1f}%)\n"
            f"Most Common Risk Level: {most_common}\n"
            f"Model used: XGBoost ({xgb_accuracy:.1f}% accurate)"
        )

with tabs[3]:
    st.header("Patient History")
    conn = sqlite3.connect(DB_PATH)
    history_df = pd.read_sql_query(
        "SELECT id, patient_name, created_at, risk_score, risk_level, best_model, prediction FROM patient_predictions ORDER BY id DESC",
        conn,
    )
    conn.close()

    if history_df.empty:
        st.info("No patient history yet.")
    else:
        search = st.text_input("Search by patient name")
        filtered = history_df.copy()
        if search.strip():
            filtered = filtered[filtered["patient_name"].str.contains(search, case=False, na=False)]
        filtered["prediction"] = filtered["prediction"].map({0: "No Heart Disease", 1: "Heart Disease"})

        patient_groups = filtered.groupby("patient_name")["risk_score"].agg(["first", "last", "count"]).reset_index()
        trend_map = {}
        for _, row in patient_groups.iterrows():
            if row["count"] <= 1:
                trend_map[row["patient_name"]] = "→ stable"
            elif row["last"] > row["first"]:
                trend_map[row["patient_name"]] = "↑ increasing"
            elif row["last"] < row["first"]:
                trend_map[row["patient_name"]] = "↓ decreasing"
            else:
                trend_map[row["patient_name"]] = "→ stable"
        filtered["Risk Trend"] = filtered["patient_name"].map(trend_map)

        styled = filtered.style.applymap(color_risk, subset=["risk_level"])
        st.dataframe(styled, use_container_width=True)
        st.download_button("Download History as CSV", filtered.to_csv(index=False), "patient_history.csv", "text/csv")

        st.subheader("Delete Record")
        del_col1, del_col2 = st.columns([3, 1])
        with del_col1:
            selected_id = st.selectbox("Select record ID to delete", filtered["id"].tolist())
        with del_col2:
            if st.button("Delete Record"):
                conn = sqlite3.connect(DB_PATH)
                conn.execute("DELETE FROM patient_predictions WHERE id = ?", (int(selected_id),))
                conn.commit()
                conn.close()
                st.success("Record deleted.")
                st.rerun()

        trend = filtered.iloc[::-1]
        trend_fig = go.Figure()
        trend_fig.add_trace(go.Scatter(x=trend["created_at"], y=trend["risk_score"], mode="lines+markers", name="Risk Score"))
        trend_fig.update_layout(height=320, xaxis_title="Time", yaxis_title="Risk Score")
        st.plotly_chart(trend_fig, use_container_width=True)

with tabs[4]:
    st.header("💬 AI Assistant")

    top_badge_col, top_change_col = st.columns([6, 1])
    if st.session_state.gemini_connected:
        top_badge_col.success("🔑 Connected")
        if top_change_col.button("Change Key"):
            st.session_state.gemini_connected = False
            st.session_state.gemini_api_key = ""
            st.session_state.api_key = ""
            st.rerun()
    else:
        top_badge_col.info("🔑 Not connected")

    st.markdown(
        """
        <div style="border:1px solid #e0e0e0; border-radius:10px; padding:16px; margin:10px 0; background:#fcfcfc;">
            <div style="font-size:18px; font-weight:700;">🔑 Setup Required (One-time only)</div>
            <div style="margin-top:6px;">This AI Assistant uses Google Gemini AI (completely FREE - no credit card needed)</div>
            <div style="margin-top:6px; line-height:1.45;">
                <div>Step 1: Go to https://aistudio.google.com</div>
                <div>Step 2: Click "Get API Key" → "Create"</div>
                <div>Step 3: Copy and paste it below</div>
                <div style="color:#555;">✅ Your key is saved for this session only. We never store it permanently.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    api_key_input = st.text_input(
        "Paste your Gemini API key here",
        type="password",
        placeholder="AIza...",
        value=st.session_state.gemini_api_key,
    )
    if st.button("✅ Save & Connect"):
        if not api_key_input.strip():
            st.warning("Please enter an API key")
        else:
            ok, err = validate_gemini_key(api_key_input.strip())
            if ok:
                st.session_state.gemini_api_key = api_key_input.strip()
                st.session_state.api_key = api_key_input.strip()  # back-compat
                st.session_state.gemini_connected = True
                st.success("✅ Connected successfully!")
                st.rerun()
            else:
                st.error(f"❌ Connection failed: {err or 'Unknown Gemini API error'}")

    st.markdown(
        """
        <style>
        .chat-user {
            text-align: right;
            background: #d8ebff;
            color: #0f172a;
            padding: 8px;
            border-radius: 8px;
            margin: 5px 0;
        }
        .chat-ai {
            text-align: left;
            background: #f0f2f5;
            color: #111827;
            padding: 8px;
            border-radius: 8px;
            margin: 5px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.gemini_connected:
        quick_cols = st.columns(4)
        quick_questions = [
            "What does my risk score mean?",
            "How can I reduce my heart disease risk?",
            "Explain the SHAP chart",
            "What lifestyle changes should I make?",
        ]

        for msg in st.session_state.chat_history:
            css = "chat-user" if msg["role"] == "user" else "chat-ai"
            st.markdown(f'<div class="{css}">{msg["text"]}</div>', unsafe_allow_html=True)

        stream_placeholder = st.empty()

        for idx, question in enumerate(quick_questions):
            if quick_cols[idx].button(question):
                st.session_state.chat_history.append({"role": "user", "text": question})
                reply, err = stream_chat_with_gemini(
                    st.session_state.gemini_api_key,
                    question,
                    stream_placeholder,
                )
                st.session_state.chat_history.append({"role": "assistant", "text": reply if reply else err})
                st.rerun()

        user_query = st.text_input("Ask about your results or heart health...", key="chat_input_box")
        if st.button("Send"):
            if not user_query.strip():
                st.warning("Please enter a question.")
            else:
                st.session_state.chat_history.append({"role": "user", "text": user_query.strip()})
                reply, err = stream_chat_with_gemini(
                    st.session_state.gemini_api_key,
                    user_query.strip(),
                    stream_placeholder,
                )
                st.session_state.chat_history.append({"role": "assistant", "text": reply if reply else err})
                st.rerun()

with tabs[5]:
    st.header("Model Metrics")
    view_mode = st.radio(
        "View Mode",
        ["👥 Simple View", "🔬 Technical View (for developers)"],
        horizontal=True,
        index=0,
    )

    metrics_table = []
    for model_name, model_metrics in metadata["metrics"].items():
        cm = model_metrics["confusion_matrix"]
        tn, fp = cm[0]
        fn, tp = cm[1]
        total = tn + fp + fn + tp
        accuracy = ((tn + tp) / total) * 100 if total else 0
        metrics_table.append(
            {
                "Model": model_name,
                "F1 Score": round(model_metrics["f1"], 4),
                "ROC-AUC": round(model_metrics["roc_auc"], 4),
                "Confusion Matrix": cm,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
                "Total": total,
                "AccuracyPct": round(accuracy, 1),
            }
        )

    if view_mode == "🔬 Technical View (for developers)":
        tech_df = pd.DataFrame(metrics_table)[["Model", "F1 Score", "ROC-AUC", "Confusion Matrix"]]
        st.dataframe(tech_df, use_container_width=True)
        st.write(f"Best model: **{metadata['best_model']}**")
    else:
        friendly_name = {
            "Logistic Regression": "Classic Method - Fast & Reliable",
            "Decision Tree": "Decision Map - Easy to Explain",
            "Random Forest": "Team of Trees - Strong & Stable",
            "XGBoost": "Boosted Intelligence - Most Accurate ⭐",
            "SVM": "Pattern Finder - Good with Complex Data",
            "Neural Network": "Brain-Inspired - Learns Deep Patterns",
        }

        st.subheader("🏆 Which AI Model is Best?")
        for row in metrics_table:
            is_best = row["Model"] == metadata["best_model"]
            border_color = "#f1c40f" if is_best else "#3b82f6"
            crown = " 👑 Best Model" if is_best else ""
            st.markdown(
                f"""
                <div style="border:2px solid {border_color}; border-radius:10px; padding:10px; margin-bottom:10px;">
                    <div style="font-weight:700; font-size:18px;">{row["Model"]}{crown}</div>
                    <div style="color:#4b5563;">{friendly_name.get(row["Model"], row["Model"])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(max(row["F1 Score"], 0.0), 1.0))
            st.caption(f'F1 Accuracy Score: {row["F1 Score"] * 100:.1f}%')

        st.subheader("📊 How Accurate Are They? (Simple View)")
        for row in metrics_table:
            st.markdown(f"### {row['Model']}")
            cm_df = pd.DataFrame(
                [
                    [f"✅ Correctly said NO disease: {row['TN']}", f"❌ Missed disease: {row['FP']} patients"],
                    [f"❌ Missed healthy: {row['FN']} patient(s)", f"✅ Correctly said YES disease: {row['TP']}"],
                ],
                columns=["Negative Class", "Positive Class"],
            )
            st.table(cm_df)
            correct = row["TN"] + row["TP"]
            st.caption(f"Out of {row['Total']} patients, this model got {correct} right ({row['AccuracyPct']}%).")

        st.subheader("🎯 What Do These Numbers Mean?")
        best_acc = max(metrics_table, key=lambda x: x["AccuracyPct"])["AccuracyPct"]
        st.info(
            f"📖 Understanding the Results\n\n"
            f"✅ Accuracy: Out of 100 patients, how many did the model get right? Our best model: {best_acc:.1f}% correct.\n\n"
            f"🔴 False Negative (most important!): Patient HAS heart disease but model said NO. We minimize this to be safe.\n\n"
            f"🟡 False Positive: Patient is healthy but model said YES. Less dangerous but we track it."
        )

        st.subheader("📈 Model Comparison Chart")
        chart_df = pd.DataFrame(metrics_table)
        color_scale = []
        for val in chart_df["AccuracyPct"]:
            if val > 98:
                color_scale.append("#16a34a")
            elif val >= 95:
                color_scale.append("#eab308")
            else:
                color_scale.append("#f97316")
        fig = go.Figure(
            go.Bar(
                x=chart_df["AccuracyPct"],
                y=chart_df["Model"],
                orientation="h",
                marker_color=color_scale,
                hovertemplate="%{y}: %{x}% accurate on test data<extra></extra>",
            )
        )
        fig.add_vline(x=95, line_dash="dash", line_color="gray", annotation_text="Good threshold")
        fig.update_layout(
            title="Model Accuracy Comparison (higher is better)",
            xaxis_title="Accuracy %",
            yaxis_title="Model",
            xaxis=dict(range=[0, 100]),
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("💡 Why We Use Multiple Models")
        st.write(
            "We test 6 different AI models and pick the best one "
            f"(currently {metadata['best_model']} with {best_acc:.1f}% accuracy). "
            "Using multiple models is like getting a second opinion from different doctors - "
            "it makes the prediction more reliable. All our models score high, which means "
            "for every 100 patients, most predictions are correct."
        )
