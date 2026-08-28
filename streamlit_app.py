"""
Pearls AQI Predictor - Production AI Environmental Intelligence Platform v2.2.0

Fixes in this version:
- Auth left panel: replaced invisible gradient text with solid visible colors
- Selectbox/dropdown: forced dark background + white text on all dropdowns
- Dashboard: city selector moved from navbar to dashboard header (clean design)
- All text visibility: removed all -webkit-text-fill-color:transparent usages
- Navbar: clean 6-item navigation without city clutter
- History dropdown: fixed white-on-white option text
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from utils.db import (
    authenticate_user,
    create_session,
    get_prediction_history,
    get_user_preferences,
    log_prediction,
    logout_session,
    register_user,
    update_user_preferences,
    validate_session,
)
from utils.feature_store import feature_store

# ============================================================================
# BOOTSTRAP
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("pearls_streamlit")

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Pearls AQI Predictor | Environmental AI Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# PATHS & CONSTANTS
# ============================================================================

MODEL_DIR = PROJECT_ROOT / "models" / "3cities"
REPORT_DIR = PROJECT_ROOT / "reports"
EVALUATION_FILE = REPORT_DIR / "model_evaluation" / "3cities" / "training_report_3cities.json"
BEST_MODELS_FILE = REPORT_DIR / "model_evaluation" / "best_models.json"
SHAP_DIR = REPORT_DIR / "explainability"

CITIES: Dict[str, Dict[str, Any]] = {
    "Lahore": {
        "lat": 31.5204, "lon": 74.3587,
        "region": "Punjab", "tagline": "Provincial Capital & Cultural Hub",
    },
    "Islamabad": {
        "lat": 33.6844, "lon": 73.0479,
        "region": "Federal Territory", "tagline": "Federal Capital & Margalla Foothills",
    },
    "Faisalabad": {
        "lat": 31.4504, "lon": 73.1350,
        "region": "Punjab", "tagline": "Industrial Center & Textile Capital",
    },
}

AQI_BREAKPOINTS: List[Tuple] = [
    (0,   50,  "Good",                          "#10B981", "rgba(16,185,129,0.15)"),
    (51,  100, "Moderate",                       "#F59E0B", "rgba(245,158,11,0.15)"),
    (101, 150, "Unhealthy for Sensitive Groups", "#F97316", "rgba(249,115,22,0.15)"),
    (151, 200, "Unhealthy",                      "#EF4444", "rgba(239,68,68,0.15)"),
    (201, 300, "Very Unhealthy",                 "#A855F7", "rgba(168,85,247,0.15)"),
    (301, 500, "Hazardous",                      "#EC4899", "rgba(236,72,153,0.15)"),
]

HEALTH_ADVICE: Dict[str, str] = {
    "Good":                          "Air quality is satisfactory. Ideal conditions for all outdoor activities.",
    "Moderate":                      "Air quality is acceptable. Unusually sensitive individuals may notice minor effects.",
    "Unhealthy for Sensitive Groups": "Children, elderly, and those with respiratory conditions should limit prolonged outdoor exertion.",
    "Unhealthy":                     "Everyone may begin to experience health effects. Avoid prolonged outdoor exertion.",
    "Very Unhealthy":                "Health alert: everyone may experience serious effects. Stay indoors with air purification.",
    "Hazardous":                     "Health emergency. The entire population is likely to be affected. Stay indoors.",
}

# ============================================================================
# CSS  --  all text visible, dark dropdowns, no gradient transparency tricks
# ============================================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* ---- GLOBAL ---- */
html, body, [data-testid="stAppViewContainer"] {
    background: #070A11 !important;
    font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important;
}
[data-testid="stAppViewContainer"] > .main { padding-top: 0.5rem !important; }
.block-container {
    padding-top: 0.4rem !important;
    padding-bottom: 2rem !important;
    max-width: 1380px !important;
}
#MainMenu, footer, header { visibility: hidden; }

/* ---- ALL TEXT --- force white so nothing disappears ---- */
*, *::before, *::after {
    color: #F0F4F8 !important;
}
p, span, div, label, h1, h2, h3, h4, h5, h6,
.stMarkdown p, .stMarkdown span,
[data-testid="stText"], [data-testid="stMarkdown"] {
    color: #F0F4F8 !important;
}
.stCaption, figcaption, small {
    color: #94A3B8 !important;
}

/* ---- ULTRA AGGRESSIVE DROPDOWN / SELECT FIX ---- */
/* Force ALL dropdown elements to have visible text */
div[data-baseweb="select"] *,
div[data-baseweb="popover"] *,
ul[data-baseweb="menu"] *,
[role="listbox"] *,
li[role="option"] *,
div[data-baseweb="option"] * {
    color: #FFFFFF !important;
    background-color: transparent !important;
}
/* Dropdown container backgrounds */
div[data-baseweb="popover"],
ul[data-baseweb="menu"],
[role="listbox"] {
    background-color: #1E293B !important;
}
/* Each option background */
li[role="option"],
div[data-baseweb="option"] {
    background-color: #1E293B !important;
}
/* Hover state */
li[role="option"]:hover *,
div[data-baseweb="option"]:hover *,
li[aria-selected="true"] *,
div[aria-selected="true"] * {
    color: #38BDF8 !important;
}

/* ---- BUTTON TEXT VISIBILITY FIX ---- */
.stButton > button,
.stButton > button *,
button,
button * {
    color: #FFFFFF !important;
}

/* ---- INPUTS ---- */
.stTextInput input {
    background: #0F172A !important;
    border: 1.5px solid rgba(255,255,255,0.2) !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
    caret-color: #38BDF8 !important;
}
.stTextInput input::placeholder { color: #475569 !important; }
.stTextInput input:focus {
    border-color: #38BDF8 !important;
    box-shadow: 0 0 0 2px rgba(56,189,248,0.2) !important;
    outline: none !important;
}
.stTextInput > label {
    color: #CBD5E1 !important;
    font-weight: 700 !important;
    font-size: 0.875rem !important;
}

/* ---- SELECTBOX / DROPDOWN  --  the key fix for white dropdown issue ---- */
/* Container */
div[data-baseweb="select"] > div {
    background-color: #0F172A !important;
    border: 1.5px solid rgba(255,255,255,0.2) !important;
    border-radius: 10px !important;
}
/* Selected value text */
div[data-baseweb="select"] [data-testid="stSelectboxValue"],
div[data-baseweb="select"] span,
div[data-baseweb="select"] div {
    color: #FFFFFF !important;
}
/* Dropdown popup background */
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
ul[data-baseweb="menu"],
[role="listbox"] {
    background-color: #1E293B !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
}
/* Each dropdown option */
li[role="option"],
div[data-baseweb="option"] {
    background-color: #1E293B !important;
    color: #F0F4F8 !important;
}
li[role="option"]:hover,
div[data-baseweb="option"]:hover,
li[aria-selected="true"],
div[aria-selected="true"] {
    background-color: rgba(56,189,248,0.15) !important;
    color: #38BDF8 !important;
}
/* selectbox label */
.stSelectbox > label {
    color: #CBD5E1 !important;
    font-weight: 700 !important;
    font-size: 0.875rem !important;
}

/* ---- MULTISELECT ---- */
div[data-baseweb="select"][data-multi="true"] > div {
    background-color: #0F172A !important;
}
span[data-baseweb="tag"] {
    background-color: rgba(56,189,248,0.2) !important;
    color: #38BDF8 !important;
}

/* ---- AUTH panels: force button full width ---- */
[data-testid="column"] .stButton > button {
    width: 100% !important;
    padding: 12px 20px !important;
    font-size: 1rem !important;
    border-radius: 12px !important;
}

/* ---- Streamlit column padding reset for auth ---- */
[data-testid="stHorizontalBlock"] > [data-testid="column"] {
    padding: 0 8px !important;
}

/* ---- BUTTONS ---- */
.stButton > button {
    background: linear-gradient(135deg, #0EA5E9, #2563EB) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(14,165,233,0.3) !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(14,165,233,0.45) !important;
}

/* ---- TABS ---- */
[data-baseweb="tab-list"] {
    background: rgba(15,23,42,0.7) !important;
    border-radius: 10px !important;
    padding: 4px !important;
}
[data-baseweb="tab"] {
    color: #94A3B8 !important;
    font-weight: 600 !important;
    border-radius: 7px !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: #38BDF8 !important;
    background: rgba(56,189,248,0.12) !important;
}

/* ---- RADIO (navigation) ---- */
[data-testid="stHorizontalBlock"] {
    gap: 2px !important;
    flex-wrap: nowrap !important;
}
.stRadio label {
    color: #CBD5E1 !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    white-space: nowrap !important;
}

/* ---- COLUMN ALIGNMENT FOR DASHBOARD CARDS ---- */
[data-testid="column"] {
    display: flex !important;
    flex-direction: column !important;
}
[data-testid="column"] > div {
    flex: 1 !important;
    display: flex !important;
    flex-direction: column !important;
}

/* ---- DATAFRAME ---- */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] th {
    background: rgba(30,41,59,0.95) !important;
    color: #38BDF8 !important;
    font-weight: 700 !important;
}
[data-testid="stDataFrame"] td {
    color: #E2E8F0 !important;
    background: rgba(15,23,42,0.7) !important;
}

/* ---- METRIC WIDGET ---- */
[data-testid="stMetricLabel"] { color: #94A3B8 !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"] { color: #F0F4F8 !important; font-weight: 800 !important; }
[data-testid="stMetricDelta"] { color: #10B981 !important; }

/* ---- CHECKBOX ---- */
.stCheckbox label { color: #CBD5E1 !important; font-size: 0.875rem !important; }

/* ---- SLIDER ---- */
.stSlider label { color: #CBD5E1 !important; font-weight: 600 !important; }
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] { color: #64748B !important; }

/* ---- EXPANDER ---- */
.streamlit-expanderHeader { color: #94A3B8 !important; font-weight: 600 !important; }

/* ---- ALERTS ---- */
[data-testid="stInfo"]    { background: rgba(56,189,248,0.08)  !important; color: #E2E8F0 !important; border-color: rgba(56,189,248,0.3) !important; }
[data-testid="stWarning"] { background: rgba(245,158,11,0.08)  !important; color: #E2E8F0 !important; border-color: rgba(245,158,11,0.3) !important; }
[data-testid="stError"]   { background: rgba(239,68,68,0.08)   !important; color: #E2E8F0 !important; border-color: rgba(239,68,68,0.3)  !important; }
[data-testid="stSuccess"] { background: rgba(16,185,129,0.08)  !important; color: #E2E8F0 !important; border-color: rgba(16,185,129,0.3) !important; }

/* ---- FORM ---- */
[data-testid="stForm"] {
    background: rgba(15,23,42,0.6) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
    padding: 20px !important;
}

/* ---- CUSTOM CARD CLASSES ---- */
.glass-card {
    background: rgba(15,23,42,0.75);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-sizing: border-box !important;
    display: flex;
    flex-direction: column;
}
.hero-card {
    background: rgba(15,23,42,0.85);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 18px;
    padding: 26px;
    text-align: center;
    box-sizing: border-box !important;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.metric-card {
    background: rgba(30,41,59,0.6);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 13px;
    padding: 16px;
    text-align: center;
    box-sizing: border-box !important;
    min-height: 120px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}
.forecast-card {
    background: rgba(15,23,42,0.8);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    min-height: 280px;
    box-sizing: border-box !important;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.forecast-unavail {
    background: rgba(30,41,59,0.4);
    border: 1px dashed rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    min-height: 280px;
    box-sizing: border-box !important;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}
.alert-high     { background: rgba(239,68,68,0.07);  border-left: 4px solid #EF4444; border-radius: 10px; padding: 16px 18px; margin-bottom: 12px; }
.alert-moderate { background: rgba(245,158,11,0.07); border-left: 4px solid #F59E0B; border-radius: 10px; padding: 16px 18px; margin-bottom: 12px; }
.alert-critical { background: rgba(168,85,247,0.07); border-left: 4px solid #A855F7; border-radius: 10px; padding: 16px 18px; margin-bottom: 12px; }

.pill {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(30,41,59,0.8);
    border: 1px solid rgba(56,189,248,0.3);
    color: #38BDF8;
    padding: 4px 12px; border-radius: 9999px;
    font-size: 0.75rem; font-weight: 700; white-space: nowrap;
}
.pill-green { border-color: rgba(16,185,129,0.4); color: #10B981; }
.pill-amber { border-color: rgba(245,158,11,0.4);  color: #F59E0B; }
.pill-red   { border-color: rgba(239,68,68,0.4);   color: #EF4444; }
.divider { border: none; height: 1px; background: rgba(255,255,255,0.07); margin: 14px 0; }

/* ---- AUTH PANELS ---- */
.auth-left {
    background: linear-gradient(160deg, #0F1B2D 0%, #162032 60%, #0A1628 100%);
    border: 1px solid rgba(56,189,248,0.18);
    border-radius: 20px;
    padding: 36px 38px;
    min-height: 560px;
}
.auth-right {
    background: rgba(10,18,35,0.95);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 32px 34px;
}
.brand-badge {
    display: inline-flex; align-items: center; gap: 7px;
    background: rgba(56,189,248,0.12);
    border: 1px solid rgba(56,189,248,0.3);
    color: #38BDF8;
    padding: 5px 14px; border-radius: 9999px;
    font-size: 0.78rem; font-weight: 800;
    letter-spacing: 0.06em; text-transform: uppercase;
    margin-bottom: 20px; display: inline-block;
}
.capability-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px; padding: 12px;
    text-align: center;
}
.strength-bar-bg {
    background: rgba(255,255,255,0.08);
    height: 5px; border-radius: 3px; overflow: hidden;
    margin: 4px 0 8px 0;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================================
# HELPERS
# ============================================================================

def safe_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def fmt(val: Optional[float], dec: int = 1) -> str:
    return f"{val:.{dec}f}" if val is not None else "—"


def calculate_us_aqi(pm25) -> int:
    if pm25 is None:
        return 0
    try:
        pm25 = float(pm25)
    except (TypeError, ValueError):
        return 0
    if np.isnan(pm25) or np.isinf(pm25) or pm25 < 0:
        return 0
    if pm25 <= 12.0:    return int(round((50 / 12.0) * pm25))
    elif pm25 <= 35.4:  return int(round(50  + (50  / 23.4) * (pm25 - 12.1)))
    elif pm25 <= 55.4:  return int(round(100 + (50  / 20.0) * (pm25 - 35.5)))
    elif pm25 <= 150.4: return int(round(150 + (50  / 95.0) * (pm25 - 55.5)))
    elif pm25 <= 250.4: return int(round(200 + (100 / 100.) * (pm25 - 150.5)))
    elif pm25 <= 350.4: return int(round(300 + (100 / 100.) * (pm25 - 250.5)))
    elif pm25 <= 500.4: return int(round(400 + (100 / 150.) * (pm25 - 350.5)))
    return 500


def get_aqi_details(aqi: int) -> Tuple[str, str, str, str]:
    for lo, hi, cat, color, bg in AQI_BREAKPOINTS:
        if lo <= aqi <= hi:
            return cat, color, bg, HEALTH_ADVICE[cat]
    return "Hazardous", "#EC4899", "rgba(236,72,153,0.15)", HEALTH_ADVICE["Hazardous"]


def pwd_strength(pwd: str) -> Tuple[int, str, str]:
    if not pwd:
        return 0, "Empty", "#64748B"
    s = 0
    if len(pwd) >= 6:                   s += 15
    if len(pwd) >= 10:                  s += 25
    if re.search(r"[A-Z]", pwd):        s += 20
    if re.search(r"[0-9]", pwd):        s += 20
    if re.search(r"[^A-Za-z0-9]", pwd): s += 20
    if s < 40:   return s, "Weak",   "#EF4444"
    elif s < 75: return s, "Medium", "#F59E0B"
    else:        return s, "Strong", "#10B981"


# ============================================================================
# DATA LOADERS
# ============================================================================

@st.cache_data(ttl=60, show_spinner=False)
def load_features() -> pd.DataFrame:
    return feature_store.load_features()


@st.cache_data(ttl=300, show_spinner=False)
def load_eval_report() -> dict:
    if EVALUATION_FILE.exists():
        try:
            with open(EVALUATION_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


@st.cache_data(ttl=300, show_spinner=False)
def load_best_models() -> list:
    if BEST_MODELS_FILE.exists():
        try:
            with open(BEST_MODELS_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


@st.cache_data(ttl=300, show_spinner=False)
def load_shap(horizon: str) -> Optional[pd.DataFrame]:
    p = SHAP_DIR / f"shap_feature_importance_{horizon}.csv"
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            pass
    return None


@st.cache_resource
def load_model(horizon: int):
    p = MODEL_DIR / f"best_model_{horizon}h.joblib"
    if not p.exists():
        return None
    try:
        return joblib.load(p)
    except Exception as e:
        logger.error("Model load error [%dh]: %s", horizon, e)
        return None


def run_forecasts(city_df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    """
    Run 24h/48h/72h AQI forecasts using real trained models.
    
    Includes comprehensive sanity checks to prevent:
    - NaN predictions
    - Infinite predictions
    - Zero-on-failure predictions
    - Negative predictions
    """
    results: Dict[int, Dict[str, Any]] = {}
    
    # Early validation: check if data exists
    if city_df.empty:
        for h in (24, 48, 72):
            results[h] = {
                "status": "unavailable",
                "error": "No feature data available for this city."
            }
        return results
    
    # Get feature columns (exclude metadata and targets)
    excluded = {"city", "hour", "timestamp", "coverage_quality", 
                "target_24h", "target_48h", "target_72h",
                "target_pm2_5_24h", "target_pm2_5_48h", "target_pm2_5_72h"}
    fcols = [c for c in city_df.columns 
             if c not in excluded and np.issubdtype(city_df[c].dtype, np.number)]
    
    if not fcols:
        for h in (24, 48, 72):
            results[h] = {
                "status": "unavailable",
                "error": "No numeric feature columns found."
            }
        return results
    
    # Get latest observation and prepare input
    latest_row = city_df.iloc[-1]
    X = pd.DataFrame([latest_row[fcols]])
    
    # Pass raw NaN values to model - Ridge pipeline has SimpleImputer(strategy='median')
    # DO NOT use fillna with zero as it bypasses the trained imputer
    
    # Run prediction for each horizon
    for h in (24, 48, 72):
        model = load_model(h)
        
        if model is None:
            results[h] = {
                "status": "unavailable",
                "error": f"Model artifact not found: best_model_{h}h.joblib"
            }
            continue
        
        try:
            # Execute model prediction
            raw_prediction = model.predict(X)[0]
            
            # SANITY CHECK 1: Detect NaN
            if pd.isna(raw_prediction):
                logger.error(f"Model returned NaN prediction for +{h}h horizon")
                results[h] = {
                    "status": "unavailable",
                    "error": f"Model returned invalid NaN prediction for +{h}h"
                }
                continue
            
            # SANITY CHECK 2: Detect Infinity
            if np.isinf(raw_prediction):
                logger.error(f"Model returned Inf prediction for +{h}h horizon")
                results[h] = {
                    "status": "unavailable",
                    "error": f"Model returned invalid Inf prediction for +{h}h"
                }
                continue
            
            # SANITY CHECK 3: Convert to float and validate
            try:
                pm25 = float(raw_prediction)
            except (TypeError, ValueError) as e:
                logger.error(f"Cannot convert prediction to float for +{h}h: {e}")
                results[h] = {
                    "status": "unavailable",
                    "error": f"Invalid prediction type for +{h}h"
                }
                continue
            
            # SANITY CHECK 4: Ensure non-negative PM2.5
            pm25 = max(0.0, pm25)
            
            # SANITY CHECK 5: Validate reasonable range (0-1000 µg/m³)
            if pm25 > 1000:
                logger.warning(f"Unusually high PM2.5 prediction for +{h}h: {pm25:.2f}")
                # Don't reject, but log for investigation
            
            # Calculate AQI from validated PM2.5
            aqi = calculate_us_aqi(pm25)
            
            # SANITY CHECK 6: Validate AQI calculation
            if aqi == 0 and pm25 > 0:
                logger.error(f"AQI calculation failed for PM2.5={pm25:.2f}")
                results[h] = {
                    "status": "unavailable",
                    "error": f"AQI calculation error for +{h}h"
                }
                continue
            
            # Get AQI category and health advice
            cat, color, bg, health = get_aqi_details(aqi)
            
            # SUCCESS: Valid prediction
            results[h] = {
                "status": "success",
                "pm25": round(pm25, 2),
                "aqi": aqi,
                "category": cat,
                "color": color,
                "bg": bg,
                "health": health
            }
            
        except Exception as exc:
            logger.error(f"Prediction failed for +{h}h horizon: {exc}")
            results[h] = {
                "status": "unavailable",
                "error": f"Prediction error: {str(exc)}"
            }
    
    return results


def build_alerts(city_df, city, live_aqi, forecasts, threshold=150):
    alerts = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if live_aqi is not None and live_aqi > threshold:
        cat, color, _, health = get_aqi_details(live_aqi)
        sev = "CRITICAL" if live_aqi > 300 else ("HIGH" if live_aqi > 200 else "MODERATE")
        alerts.append({"id": f"LIVE-{city[:3].upper()}", "city": city, "horizon": "Now",
                        "severity": sev, "title": f"Live AQI Alert — {city}",
                        "message": f"Current AQI {live_aqi} ({cat}) exceeds your threshold of {threshold}.",
                        "aqi": live_aqi, "category": cat, "color": color, "recommendation": health, "ts": ts})
    for h, r in forecasts.items():
        if r.get("status") != "success":
            continue
        pa = r["aqi"]
        if pa > threshold:
            sev = "CRITICAL" if pa > 300 else ("HIGH" if pa > 200 else "MODERATE")
            alerts.append({"id": f"+{h}H-{city[:3].upper()}", "city": city, "horizon": f"+{h}h",
                            "severity": sev, "title": f"Forecast Alert — {city} (+{h}h)",
                            "message": f"Model forecast AQI {pa} ({r['category']}) in +{h}h exceeds threshold {threshold}.",
                            "aqi": pa, "category": r["category"], "color": r["color"],
                            "recommendation": r["health"], "ts": ts})
    return alerts


# ============================================================================
# SESSION STATE
# ============================================================================

for k, v in [("user", None), ("auth_token", None), ("selected_city", "Lahore"),
             ("current_nav", "Dashboard"), ("show_login_pwd", False), ("show_reg_pwd", False)]:
    if k not in st.session_state:
        st.session_state[k] = v

if not st.session_state["user"] and st.session_state["auth_token"]:
    restored = validate_session(st.session_state["auth_token"])
    st.session_state["user"] = restored
    if not restored:
        st.session_state["auth_token"] = None

# ============================================================================
# AUTH PORTAL
# ============================================================================

if not st.session_state["user"]:
    col_l, col_r = st.columns([1.15, 1])

    # ---- LEFT BRAND PANEL ------------------------------------------------
    with col_l:
        st.markdown("""
<div style="background:linear-gradient(160deg,#0F1B2D 0%,#162032 60%,#0A1628 100%);
            border:1px solid rgba(56,189,248,0.18); border-radius:20px;
            padding:36px 38px; min-height:580px; box-sizing:border-box;">

  <div style="display:inline-block; background:rgba(56,189,248,0.12);
              border:1px solid rgba(56,189,248,0.3); color:#38BDF8;
              padding:5px 14px; border-radius:9999px; font-size:0.78rem;
              font-weight:800; letter-spacing:0.06em; text-transform:uppercase;
              margin-bottom:22px;">
    ⚡ AI Environmental Intelligence
  </div>

  <div style="margin-bottom:12px;">
    <span style="font-size:2.6rem; font-weight:900; color:#38BDF8;
                 line-height:1.1; display:block; font-family:inherit;">
      Pearls AQI
    </span>
    <span style="font-size:2.6rem; font-weight:900; color:#FFFFFF;
                 line-height:1.1; display:block; font-family:inherit;">
      Predictor
    </span>
  </div>

  <p style="font-size:1.05rem; font-weight:700; color:#38BDF8; margin:0 0 12px 0;">
    Breathe smarter. Predict cleaner.
  </p>

  <p style="font-size:0.9rem; color:#94A3B8; line-height:1.65;
            max-width:460px; margin-bottom:26px;">
    End-to-end ML forecasting platform. Ingests real OpenAQ v3 sensor
    data and Open-Meteo atmospheric forecasts to predict 3-day AQI
    across Lahore, Islamabad and Faisalabad.
  </p>

  <svg width="100%" height="72" viewBox="0 0 420 72" fill="none"
       xmlns="http://www.w3.org/2000/svg" style="margin-bottom:22px; display:block;">
    <path d="M6 36 C80 6,170 66,414 36" stroke="#38BDF8" stroke-width="2.2"
          stroke-linecap="round" opacity="0.7"/>
    <path d="M6 52 C100 78,250 4,414 52" stroke="#F59E0B" stroke-width="1.6"
          stroke-dasharray="5 5" opacity="0.5"/>
    <circle cx="100" cy="22" r="4" fill="#38BDF8"/>
    <circle cx="100" cy="22" r="9" fill="#38BDF8" fill-opacity="0.2"/>
    <circle cx="252" cy="54" r="5" fill="#818CF8"/>
    <circle cx="252" cy="54" r="10" fill="#818CF8" fill-opacity="0.2"/>
    <circle cx="370" cy="30" r="4" fill="#10B981"/>
    <circle cx="370" cy="30" r="8" fill="#10B981" fill-opacity="0.2"/>
  </svg>

  <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-bottom:26px;">
    <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
                border-radius:10px; padding:14px; text-align:center;">
      <div style="font-size:1.2rem; margin-bottom:5px;">📡</div>
      <div style="font-size:0.75rem; font-weight:700; color:#E2E8F0;">Live Telemetry</div>
      <div style="font-size:0.67rem; color:#64748B; margin-top:2px;">OpenAQ v3</div>
    </div>
    <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
                border-radius:10px; padding:14px; text-align:center;">
      <div style="font-size:1.2rem; margin-bottom:5px;">🔮</div>
      <div style="font-size:0.75rem; font-weight:700; color:#E2E8F0;">3-Day Forecast</div>
      <div style="font-size:0.67rem; color:#64748B; margin-top:2px;">Ridge Models</div>
    </div>
    <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
                border-radius:10px; padding:14px; text-align:center;">
      <div style="font-size:1.2rem; margin-bottom:5px;">🧠</div>
      <div style="font-size:0.75rem; font-weight:700; color:#E2E8F0;">Explainability</div>
      <div style="font-size:0.67rem; color:#64748B; margin-top:2px;">Feature Importance</div>
    </div>
  </div>

  <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
              border-radius:12px; padding:14px 18px;">
    <div style="font-size:0.7rem; font-weight:800; color:#38BDF8; text-transform:uppercase;
                letter-spacing:0.07em; margin-bottom:8px;">Operational Coverage</div>
    <div style="display:flex; gap:18px; font-size:0.86rem; color:#E2E8F0;">
      <span>📍 <strong>Lahore</strong></span>
      <span>📍 <strong>Islamabad</strong></span>
      <span>📍 <strong>Faisalabad</strong></span>
    </div>
  </div>

</div>""", unsafe_allow_html=True)

    # ---- RIGHT AUTH CARD -------------------------------------------------
    with col_r:
        st.markdown("""
<div style="background:rgba(10,18,35,0.95); border:1px solid rgba(255,255,255,0.1);
            border-radius:20px; padding:32px 34px; box-sizing:border-box;">
""", unsafe_allow_html=True)
        tab_in, tab_reg = st.tabs(["🔐 Sign In", "📝 Create Account"])

        with tab_in:
            st.markdown("""
<div style="margin-bottom:18px;">
  <h3 style="font-size:1.45rem; font-weight:800; color:#F0F4F8; margin:0 0 5px 0;">Welcome Back</h3>
  <p style="font-size:0.85rem; color:#64748B; margin:0;">Sign in to your Pearls AQI intelligence dashboard.</p>
</div>""", unsafe_allow_html=True)

            login_email = st.text_input("Email Address", placeholder="name@company.com", key="li_email")
            pwd_t = "default" if st.session_state["show_login_pwd"] else "password"
            login_pwd = st.text_input("Password", type=pwd_t, placeholder="••••••••••", key="li_pwd")
            c1, c2 = st.columns(2)
            with c1:
                st.session_state["show_login_pwd"] = st.checkbox("Show password", value=st.session_state["show_login_pwd"], key="ck_lip")
            with c2:
                st.checkbox("Remember me", value=True, key="ck_rem")
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            if st.button("Sign In to Platform", key="btn_li"):
                ec = login_email.strip().lower()
                if not ec or not login_pwd:
                    st.error("Please enter both email and password.")
                elif "@" not in ec:
                    st.error("Enter a valid email address.")
                else:
                    with st.spinner("Verifying…"):
                        ok, msg, ud = authenticate_user(ec, login_pwd)
                        if ok and ud:
                            st.session_state["user"] = ud
                            st.session_state["auth_token"] = create_session(ud["id"])
                            st.success("Authentication successful — opening dashboard…")
                            st.rerun()
                        else:
                            st.error(msg)

            with st.expander("Trouble signing in?"):
                st.caption("Password reset is managed by the system administrator. Use the Create Account tab to register.")

        with tab_reg:
            st.markdown("""
<div style="margin-bottom:18px;">
  <h3 style="font-size:1.4rem; font-weight:800; color:#F0F4F8; margin:0 0 4px 0;">Create Account</h3>
  <p style="font-size:0.85rem; color:#64748B; margin:0;">Start monitoring air quality and receiving 3-day AQI forecasts.</p>
</div>""", unsafe_allow_html=True)

            reg_name  = st.text_input("Full Name", placeholder="Alex Morgan", key="rg_name")
            reg_email = st.text_input("Email Address", placeholder="alex@company.com", key="rg_email")
            reg_pt    = "default" if st.session_state["show_reg_pwd"] else "password"
            reg_pwd   = st.text_input("Password", type=reg_pt, placeholder="Create a strong password", key="rg_pwd")

            if reg_pwd:
                sc, lb, cl = pwd_strength(reg_pwd)
                ck = {
                    "6+ chars": len(reg_pwd) >= 6,
                    "Uppercase": bool(re.search(r"[A-Z]", reg_pwd)),
                    "Number": bool(re.search(r"[0-9]", reg_pwd)),
                    "Symbol": bool(re.search(r"[^A-Za-z0-9]", reg_pwd)),
                }
                ck_html = " &nbsp; ".join(
                    f'<span style="color:{"#10B981" if v else "#64748B"};font-size:0.72rem;">{"✓" if v else "○"} {k}</span>'
                    for k, v in ck.items()
                )
                st.markdown(f"""
<div style="margin:4px 0 10px 0;">
  <div style="display:flex; justify-content:space-between; font-size:0.78rem; font-weight:700; color:{cl};">
    <span>Strength: {lb}</span><span>{sc}%</span>
  </div>
  <div class="strength-bar-bg">
    <div style="width:{sc}%; background:{cl}; height:100%; border-radius:3px;"></div>
  </div>
  <div>{ck_html}</div>
</div>""", unsafe_allow_html=True)

            reg_conf = st.text_input("Confirm Password", type=reg_pt, placeholder="Re-enter password", key="rg_conf")
            st.session_state["show_reg_pwd"] = st.checkbox("Show passwords", value=st.session_state["show_reg_pwd"], key="ck_rsp")
            terms = st.checkbox("I agree to the terms & privacy policy.", key="ck_terms")
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            if st.button("Create Account & Access Platform", key="btn_reg"):
                ec = reg_email.strip().lower()
                if not ec or not reg_pwd:
                    st.error("Email and password are required.")
                elif "@" not in ec:
                    st.error("Enter a valid email address.")
                elif len(reg_pwd) < 6:
                    st.error("Password must be at least 6 characters.")
                elif reg_pwd != reg_conf:
                    st.error("Passwords do not match.")
                elif not terms:
                    st.warning("Please agree to the terms to continue.")
                else:
                    with st.spinner("Creating account…"):
                        ok, msg, _ = register_user(ec, reg_pwd, reg_name.strip())
                        if ok:
                            st.success("Account created! Switch to Sign In.")
                        else:
                            st.error(msg)

        st.markdown("</div>", unsafe_allow_html=True)  # closes auth-right inner div
    st.stop()


# ============================================================================
# AUTHENTICATED — LOAD SHARED DATA
# ============================================================================

user  = st.session_state["user"]
city  = st.session_state["selected_city"]
prefs = get_user_preferences(user["id"])
alert_threshold = int(prefs.get("alert_aqi_threshold", 150))

df_all = load_features()
fs_status = feature_store.get_status()

city_df: pd.DataFrame
if not df_all.empty and "city" in df_all.columns:
    city_df = df_all[df_all["city"].str.lower() == city.lower()].copy()
    if "hour" in city_df.columns:
        city_df["hour"] = pd.to_datetime(city_df["hour"], errors="coerce", utc=True)
        city_df = city_df.dropna(subset=["hour"]).sort_values("hour").reset_index(drop=True)
else:
    city_df = pd.DataFrame()

# Safe metric extraction
latest_pm25 = latest_temp = latest_hum = latest_wind = latest_pres = None
latest_ts: Optional[str] = None

if not city_df.empty:
    row = city_df.iloc[-1]
    for col in ("pm25_mean", "pm25_median", "pm2_5_24h_mean"):
        v = safe_float(row.get(col))
        if v is not None:
            latest_pm25 = v
            break
    latest_temp  = safe_float(row.get("temperature"))
    latest_hum   = safe_float(row.get("humidity"))
    latest_wind  = safe_float(row.get("wind_speed"))
    latest_pres  = safe_float(row.get("pressure"))
    ts = row.get("hour")
    if ts is not None and not pd.isna(ts):
        try:
            latest_ts = str(pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M UTC"))
        except Exception:
            latest_ts = str(ts)

live_aqi: Optional[int] = calculate_us_aqi(latest_pm25) if latest_pm25 is not None else None
live_cat, live_col, live_bg, live_health = get_aqi_details(live_aqi if live_aqi is not None else 0)
is_live = latest_pm25 is not None

report_data  = load_eval_report()
eval_results = report_data.get("results", {})
feat_count   = report_data.get("feature_count", "—")
dataset_rows = report_data.get("dataset_rows", "—")
now_utc = datetime.now(timezone.utc).strftime("%H:%M UTC")

# ============================================================================
# TOP NAVBAR  — clean 6-item nav + logout, NO city in navbar
# ============================================================================

nav_left, nav_mid, nav_right = st.columns([1.8, 3.8, 1.4])

with nav_left:
    st.markdown(f"""
<div style="display:flex; align-items:center; gap:10px; padding-top:4px;">
  <span style="font-size:1.35rem;">⚡</span>
  <div>
    <div style="font-weight:800; font-size:1.05rem; color:#38BDF8;">Pearls AQI Predictor</div>
    <div style="font-size:0.7rem; color:#475569;">Environmental Intelligence Platform</div>
  </div>
</div>""", unsafe_allow_html=True)

with nav_mid:
    NAV = ["Dashboard", "Forecast", "Analytics", "Model Insights", "History", "Alerts", "Profile"]
    sel = st.radio("nav", NAV,
                   index=NAV.index(st.session_state["current_nav"]),
                   horizontal=True, label_visibility="collapsed", key="main_nav")
    st.session_state["current_nav"] = sel

with nav_right:
    pill_cls = "pill-green" if is_live else "pill-amber"
    pill_lbl = "🟢 Live" if is_live else "🟡 Cached"
    nc1, nc2 = st.columns([1.5, 1])
    with nc1:
        st.markdown(f"""<div class="pill {pill_cls}" style="margin-top:6px;">{pill_lbl}</div>
<div style="font-size:0.68rem; color:#475569; margin-top:2px;">{now_utc}</div>""", unsafe_allow_html=True)
    with nc2:
        if st.button("Logout", key="nav_logout"):
            logout_session(st.session_state["auth_token"])
            st.session_state["user"] = None
            st.session_state["auth_token"] = None
            st.rerun()

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ============================================================================
# PAGE: DASHBOARD
# ============================================================================

if st.session_state["current_nav"] == "Dashboard":

    # ---- Dashboard header with city selector embedded here (not in navbar) ----
    dh_left, dh_right = st.columns([2.5, 1])
    with dh_left:
        name = user.get("full_name") or user.get("email", "").split("@")[0].title()
        st.markdown(f"""
<div style="margin-bottom:16px;">
  <h2 style="font-size:1.5rem; font-weight:800; color:#F0F4F8; margin:0 0 3px 0;">
    Good day, {name}
  </h2>
  <div style="font-size:0.85rem; color:#64748B;">
    Air Quality Intelligence &nbsp;·&nbsp;
    <strong style="color:#94A3B8;">📍 {city}</strong> — {CITIES[city]['tagline']}
    &nbsp;·&nbsp; <span style="color:#475569;">OpenAQ v3 + Open-Meteo</span>
  </div>
</div>""", unsafe_allow_html=True)

    with dh_right:
        # City selector lives here on Dashboard — clean placement
        new_city = st.selectbox(
            "Select City",
            options=list(CITIES.keys()),
            index=list(CITIES.keys()).index(city),
            key="dash_city",
        )
        if new_city != city:
            st.session_state["selected_city"] = new_city
            st.rerun()

    # ---- Hero row ----
    h1, h2 = st.columns([1.1, 2])

    with h1:
        aqi_disp = str(live_aqi) if live_aqi is not None else "—"
        src_note = latest_ts or "Feature store offline"
        st.markdown(f"""
<div class="hero-card">
  <div style="font-size:0.7rem; font-weight:700; color:#475569; text-transform:uppercase;
              letter-spacing:0.08em; margin-bottom:6px;">Current US EPA AQI</div>
  <div style="font-size:4rem; font-weight:900; color:{live_col}; line-height:1; margin:6px 0 10px 0;">
    {aqi_disp}
  </div>
  <div style="display:inline-block; background:{live_bg}; color:{live_col};
              padding:4px 16px; border-radius:9999px; font-weight:800; font-size:0.9rem;
              margin-bottom:10px;">
    {live_cat if live_aqi is not None else "Unavailable"}
  </div>
  <div style="font-size:0.7rem; color:#475569; margin-top:4px;">
    OpenAQ v3<br>{src_note}
  </div>
</div>""", unsafe_allow_html=True)

    with h2:
        health_text = live_health if live_aqi is not None else "Health guidance unavailable — feature store is offline."
        st.markdown(f"""
<div class="glass-card" style="height:100%; display:flex; flex-direction:column; justify-content:center;">
  <div style="font-size:0.95rem; font-weight:800; color:#38BDF8; margin-bottom:8px;">
    🛡️ Health Guidance & Advisory
  </div>
  <div style="font-size:0.93rem; color:#CBD5E1; line-height:1.65;">
    {health_text}
  </div>
  <div style="margin-top:12px; padding-top:10px; border-top:1px solid rgba(255,255,255,0.06);
              font-size:0.75rem; color:#475569;">
    Standard: US EPA PM2.5 AQI &nbsp;|&nbsp; Alert threshold: AQI &gt; {alert_threshold}
  </div>
</div>""", unsafe_allow_html=True)

    # ---- 3-day forecast ----
    st.markdown("""<div style="font-size:1.1rem; font-weight:800; color:#F0F4F8;
                     margin:20px 0 10px 0;">🔮 3-Day ML Multi-Horizon Forecast</div>""",
                unsafe_allow_html=True)

    with st.spinner("Running model inference…"):
        forecasts = run_forecasts(city_df)

    fc1, fc2, fc3 = st.columns(3)
    all_ok = True
    fc_aqis: Dict[int, int] = {}

    for col_obj, h in zip([fc1, fc2, fc3], [24, 48, 72]):
        r = forecasts[h]
        with col_obj:
            if r["status"] == "success":
                fc_aqis[h] = r["aqi"]
                st.markdown(f"""
<div class="forecast-card" style="border-top:4px solid {r['color']};">
  <div style="font-size:0.68rem; font-weight:700; color:#475569;
              text-transform:uppercase; letter-spacing:0.07em; margin-bottom:4px;">
    +{h} Hours
  </div>
  <div style="font-size:2.8rem; font-weight:900; color:{r['color']}; line-height:1; margin:4px 0 6px 0;">
    {r['aqi']}
  </div>
  <div style="display:inline-block; background:{r['bg']}; color:{r['color']};
              padding:3px 12px; border-radius:9999px; font-weight:800; font-size:0.78rem;
              margin-bottom:10px;">
    {r['category']}
  </div>
  <div style="font-size:0.8rem; color:#94A3B8; margin-top:8px;">
    PM2.5: <strong style="color:#38BDF8;">{r['pm25']:.1f} µg/m³</strong>
  </div>
  <div style="font-size:0.67rem; color:#475569; margin-top:3px;">
    Ridge · best_model_{h}h
  </div>
</div>""", unsafe_allow_html=True)
            else:
                all_ok = False
                st.markdown(f"""
<div class="forecast-unavail">
  <div style="font-size:0.68rem; font-weight:700; color:#475569;
              text-transform:uppercase; letter-spacing:0.07em; margin-bottom:6px;">+{h} Hours</div>
  <div style="font-size:0.95rem; font-weight:700; color:#EF4444; margin-bottom:5px;">
    Forecast Unavailable
  </div>
  <div style="font-size:0.75rem; color:#64748B; line-height:1.5;">
    {r.get('error', 'Model or data unavailable.')}
  </div>
</div>""", unsafe_allow_html=True)

    if all_ok and fc_aqis:
        log_prediction(city, fc_aqis.get(24, 0), fc_aqis.get(48, 0), fc_aqis.get(72, 0), user_id=user["id"])

    # ---- Environmental metrics ----
    st.markdown("""<div style="font-size:1.05rem; font-weight:800; color:#F0F4F8;
                     margin:20px 0 10px 0;">📊 Environmental Indicators</div>""",
                unsafe_allow_html=True)

    metrics_data = [
        ("PM2.5",    latest_pm25, 1, "µg/m³", "#38BDF8"),
        ("Temp",     latest_temp, 1, "°C",    "#F8FAFC"),
        ("Humidity", latest_hum,  0, "%",     "#818CF8"),
        ("Wind",     latest_wind, 1, "km/h",  "#10B981"),
        ("Pressure", latest_pres, 1, "hPa",   "#F59E0B"),
    ]
    m_cols = st.columns(5)
    for i, (label, val, dec, unit, color) in enumerate(metrics_data):
        with m_cols[i]:
            disp = fmt(val, dec) if val is not None else "—"
            unit_disp = unit if val is not None else "n/a"
            st.markdown(f"""
<div class="metric-card">
  <div style="font-size:0.7rem; font-weight:700; color:#475569;
              text-transform:uppercase; letter-spacing:0.07em; margin-bottom:5px;">{label}</div>
  <div style="font-size:1.7rem; font-weight:800; color:{color}; line-height:1.1;">{disp}</div>
  <div style="font-size:0.68rem; color:#475569; margin-top:3px;">{unit_disp}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("""<div style="font-size:0.72rem; color:#334155; margin-top:4px;">
  PM10, NO₂, SO₂, CO, O₃ are not available from the current OpenAQ sensor network
  for this location and are omitted rather than fabricated.
</div>""", unsafe_allow_html=True)

    # ---- Active alerts strip ----
    active = build_alerts(city_df, city, live_aqi, forecasts, alert_threshold)
    if active:
        st.markdown(f"""<div style="font-size:1.05rem; font-weight:800; color:#EF4444;
                         margin:20px 0 8px 0;">🚨 Active Alerts ({len(active)})</div>""",
                    unsafe_allow_html=True)
        for a in active[:3]:
            sev = a["severity"]
            sc = "alert-critical" if sev == "CRITICAL" else ("alert-high" if sev == "HIGH" else "alert-moderate")
            sc_col = "#A855F7" if sev == "CRITICAL" else ("#EF4444" if sev == "HIGH" else "#F59E0B")
            st.markdown(f"""
<div class="{sc}">
  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:5px; flex-wrap:wrap; gap:4px;">
    <span style="font-weight:800; color:{sc_col};">{a['title']}</span>
    <span class="pill" style="border-color:{sc_col}; color:{sc_col};">{sev} · {a['horizon']}</span>
  </div>
  <div style="font-size:0.88rem; color:#CBD5E1;">{a['message']}</div>
</div>""", unsafe_allow_html=True)


# ============================================================================
# PAGE: FORECAST
# ============================================================================

elif st.session_state["current_nav"] == "Forecast":
    # City selector at top of page
    fc_h1, fc_h2 = st.columns([3, 1])
    with fc_h1:
        st.markdown(f"""
<h2 style="font-size:1.5rem; font-weight:800; color:#F0F4F8; margin:0 0 4px 0;">
  🔮 3-Day Forecast Trajectory
</h2>
<div style="font-size:0.85rem; color:#64748B; margin-bottom:16px;">
  Multi-horizon predictions (+24h, +48h, +72h) from distinct Ridge Regression models.
</div>""", unsafe_allow_html=True)
    with fc_h2:
        fc_city = st.selectbox("City", list(CITIES.keys()),
                               index=list(CITIES.keys()).index(city), key="fc_city_sel")
        if fc_city != city:
            st.session_state["selected_city"] = fc_city
            st.rerun()

    with st.spinner("Running forecast models…"):
        forecasts = run_forecasts(city_df)

    c1, c2, c3 = st.columns(3)
    chart_pts = []
    if live_aqi is not None:
        chart_pts.append({"Horizon": "Current", "AQI": live_aqi})

    for col_obj, h in zip([c1, c2, c3], [24, 48, 72]):
        r = forecasts[h]
        with col_obj:
            if r["status"] == "success":
                chart_pts.append({"Horizon": f"+{h}h", "AQI": r["aqi"]})
                st.markdown(f"""
<div class="forecast-card" style="border-top:4px solid {r['color']};">
  <div style="font-size:0.68rem; font-weight:700; color:#475569;
              text-transform:uppercase; letter-spacing:0.07em; margin-bottom:4px;">+{h} Hours</div>
  <div style="font-size:3rem; font-weight:900; color:{r['color']}; line-height:1; margin:4px 0 6px 0;">
    {r['aqi']}
  </div>
  <div style="display:inline-block; background:{r['bg']}; color:{r['color']};
              padding:4px 14px; border-radius:9999px; font-weight:800; font-size:0.82rem; margin-bottom:10px;">
    {r['category']}
  </div>
  <div style="font-size:0.82rem; color:#94A3B8; margin-bottom:6px;">
    PM2.5: <strong style="color:#38BDF8;">{r['pm25']:.1f} µg/m³</strong>
  </div>
  <div style="font-size:0.78rem; color:#64748B; line-height:1.5;">{r['health']}</div>
</div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
<div class="forecast-unavail">
  <div style="font-size:0.68rem; font-weight:700; color:#475569;
              text-transform:uppercase; letter-spacing:0.07em; margin-bottom:6px;">+{h} Hours</div>
  <div style="font-size:1rem; font-weight:700; color:#EF4444; margin-bottom:5px;">Forecast Unavailable</div>
  <div style="font-size:0.75rem; color:#64748B;">{r.get('error','Model or data unavailable.')}</div>
</div>""", unsafe_allow_html=True)

    valid_pts = [p for p in chart_pts if p.get("AQI") is not None]
    if len(valid_pts) >= 2:
        st.markdown("<br>", unsafe_allow_html=True)
        cdf = pd.DataFrame(valid_pts)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cdf["Horizon"], y=cdf["AQI"],
            mode="lines+markers+text", text=cdf["AQI"].astype(str),
            textposition="top center", textfont=dict(color="#F0F4F8", size=13),
            line=dict(color="#38BDF8", width=3),
            marker=dict(size=11, color="#818CF8", line=dict(color="#38BDF8", width=2)),
            fill="tozeroy", fillcolor="rgba(56,189,248,0.07)",
        ))
        for ref_v, ref_l, ref_c in [(150, "USG", "#F97316"), (200, "Unhealthy", "#EF4444")]:
            fig.add_hline(y=ref_v, line_dash="dot", line_color=ref_c, opacity=0.5,
                          annotation_text=ref_l, annotation_font_color=ref_c)
        fig.update_layout(
            title=f"AQI Trajectory — {city}",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.5)",
            font_color="#F0F4F8", height=360,
            margin=dict(l=40, r=40, t=50, b=30),
            yaxis=dict(title="AQI", gridcolor="rgba(255,255,255,0.05)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig, width="stretch")

    bm = load_best_models()
    if bm:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.9rem; font-weight:700; color:#94A3B8; margin-bottom:6px;'>Best model evaluation (best_models.json)</div>", unsafe_allow_html=True)
        bdf = pd.DataFrame(bm)[["horizon_hours", "model", "mae", "rmse", "r2", "training_samples", "testing_samples"]]
        bdf.columns = ["Horizon (h)", "Model", "MAE", "RMSE", "R²", "Train", "Test"]
        bdf["MAE"] = bdf["MAE"].round(2)
        bdf["RMSE"] = bdf["RMSE"].round(2)
        bdf["R²"] = bdf["R²"].round(4)
        st.dataframe(bdf, width="stretch", hide_index=True)


# ============================================================================
# PAGE: ANALYTICS
# ============================================================================

elif st.session_state["current_nav"] == "Analytics":
    al_h1, al_h2 = st.columns([3, 1])
    with al_h1:
        st.markdown(f"""
<h2 style="font-size:1.5rem; font-weight:800; color:#F0F4F8; margin:0 0 4px 0;">
  📊 Climate Analytics & Trends
</h2>
<div style="font-size:0.85rem; color:#64748B; margin-bottom:14px;">
  All charts generated from the live feature store. Showing data for <strong style="color:#94A3B8">{city}</strong>.
</div>""", unsafe_allow_html=True)
    with al_h2:
        al_city = st.selectbox("City", list(CITIES.keys()),
                               index=list(CITIES.keys()).index(city), key="al_city_sel")
        if al_city != city:
            st.session_state["selected_city"] = al_city
            st.rerun()

    if city_df.empty:
        st.warning("No feature data available. Run `python feature_pipeline/daily_live_refresh.py` to refresh.")
    else:
        city_df["aqi"] = city_df.get("pm25_mean", pd.Series(dtype=float)).apply(
            lambda x: calculate_us_aqi(x) if pd.notna(x) else None
        )

        ac1, ac2 = st.columns(2)
        with ac1:
            st.markdown("<div style='font-weight:700;color:#94A3B8;margin-bottom:6px;'>📈 72-Hour AQI Trend</div>", unsafe_allow_html=True)
            trend = city_df.dropna(subset=["hour", "aqi"]).tail(72)
            if len(trend) >= 2:
                fig1 = px.line(trend, x="hour", y="aqi",
                               title=f"72h AQI Trend — {city}",
                               color_discrete_sequence=["#38BDF8"],
                               labels={"aqi": "AQI", "hour": "Time"})
                fig1.add_hline(y=150, line_dash="dot", line_color="#F97316", opacity=0.5,
                               annotation_text="USG", annotation_font_color="#F97316")
                fig1.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                   plot_bgcolor="rgba(15,23,42,0.5)", height=320,
                                   font_color="#F0F4F8", margin=dict(l=30,r=20,t=45,b=30))
                st.plotly_chart(fig1, width="stretch")
            else:
                st.info("Need at least 2 data points for the trend chart.")

        with ac2:
            st.markdown("<div style='font-weight:700;color:#94A3B8;margin-bottom:6px;'>🌡️ Temperature vs AQI</div>", unsafe_allow_html=True)
            scat = city_df.dropna(subset=["temperature", "aqi", "humidity"]).tail(200)
            if len(scat) >= 5:
                fig2 = px.scatter(scat, x="temperature", y="aqi", color="humidity",
                                  title=f"Temp & Humidity vs AQI — {city}",
                                  color_continuous_scale="Viridis", opacity=0.75,
                                  labels={"aqi": "AQI", "temperature": "Temp (°C)", "humidity": "Humidity (%)"})
                fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                   plot_bgcolor="rgba(15,23,42,0.5)", height=320,
                                   font_color="#F0F4F8", margin=dict(l=30,r=20,t=45,b=30))
                st.plotly_chart(fig2, width="stretch")
            else:
                st.info("Need at least 5 rows with temperature and humidity data.")

        pm_col = next((c for c in ("pm25_mean", "pm25_median") if c in city_df.columns), None)
        if pm_col:
            pm_s = city_df[pm_col].dropna()
            if len(pm_s) >= 10:
                st.markdown("<div style='font-weight:700;color:#94A3B8;margin:14px 0 6px 0;'>📉 PM2.5 Distribution</div>", unsafe_allow_html=True)
                fig3 = px.histogram(pm_s, nbins=40, color_discrete_sequence=["#818CF8"],
                                    labels={"value": "PM2.5 (µg/m³)"})
                fig3.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                   plot_bgcolor="rgba(15,23,42,0.5)", height=260,
                                   font_color="#F0F4F8", margin=dict(l=30,r=20,t=30,b=30))
                st.plotly_chart(fig3, width="stretch")

        if "hour" in city_df.columns:
            hod = city_df.dropna(subset=["aqi"]).copy()
            hod["hod"] = pd.to_datetime(hod["hour"], errors="coerce").dt.hour
            hod = hod.dropna(subset=["hod"])
            if len(hod) >= 24:
                st.markdown("<div style='font-weight:700;color:#94A3B8;margin:14px 0 6px 0;'>🕐 Average AQI by Hour of Day</div>", unsafe_allow_html=True)
                ha = hod.groupby("hod")["aqi"].mean().reset_index()
                ha.columns = ["Hour", "Avg AQI"]
                fig4 = px.bar(ha, x="Hour", y="Avg AQI", color="Avg AQI", color_continuous_scale="RdYlGn_r")
                fig4.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                   plot_bgcolor="rgba(15,23,42,0.5)", height=260,
                                   font_color="#F0F4F8", margin=dict(l=30,r=20,t=30,b=30))
                st.plotly_chart(fig4, width="stretch")


# ============================================================================
# PAGE: MODEL INSIGHTS
# ============================================================================

elif st.session_state["current_nav"] == "Model Insights":
    st.markdown("""
<h2 style="font-size:1.5rem; font-weight:800; color:#F0F4F8; margin:0 0 14px 0;">
  ⚙️ Model Intelligence & Feature Importance
</h2>""", unsafe_allow_html=True)

    mi1, mi2, mi3, mi4 = st.columns(4)
    with mi1: st.metric("Active Estimator", "Ridge Regression")
    with mi2: st.metric("Feature Count", f"{feat_count} features")
    with mi3: st.metric("Training Rows", f"{dataset_rows:,}" if isinstance(dataset_rows, int) else str(dataset_rows))
    with mi4: st.metric("Horizons", "24h / 48h / 72h")

    st.markdown("""
<div class="glass-card" style="margin-top:14px;">
  <div style="font-weight:800; color:#38BDF8; margin-bottom:6px;">Model Selection</div>
  <div style="font-size:0.9rem; color:#CBD5E1; line-height:1.65;">
    Ridge Regression is deployed for all three horizons based on chronological 80/20
    train/test evaluation. Ridge achieved the lowest MAE at all horizons vs Random Forest
    and MLPRegressor, with &lt;3 KB artifact size vs 80–87 MB for Random Forest.
  </div>
</div>""", unsafe_allow_html=True)

    if eval_results:
        st.markdown("<div style='font-size:1rem; font-weight:800; color:#94A3B8; margin:16px 0 8px 0;'>Per-Horizon Model Comparison</div>", unsafe_allow_html=True)
        for hk in ("24h", "48h", "72h"):
            hd = eval_results.get(hk, {})
            if not hd:
                continue
            st.markdown(f"<div style='font-weight:700; color:#F0F4F8; margin:10px 0 4px 0;'>Horizon +{hk} — Best: <span style='color:#10B981;'>{hd.get('best_model','—')}</span></div>", unsafe_allow_html=True)
            rows = []
            for mk, mn in [("ridge","Ridge"),("random_forest","Random Forest"),("deep_learning_mlp","MLP")]:
                m = hd.get(mk, {})
                if m:
                    rows.append({"Model": mn, "MAE": round(m.get("MAE",0),2),
                                 "RMSE": round(m.get("RMSE",0),2), "R²": round(m.get("R2",0),4)})
            if rows:
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    else:
        st.info("Run `python training_pipeline/train_3cities.py` to generate the evaluation report.")

    st.markdown("""
<div style="font-size:1rem; font-weight:800; color:#94A3B8; margin:16px 0 8px 0;">
  Feature Importance (Ridge |Coefficient| Values)
</div>
<div style="background:rgba(245,158,11,0.07); border:1px solid rgba(245,158,11,0.3);
            border-radius:10px; padding:12px 16px; margin-bottom:14px;">
  <strong style="color:#F59E0B;">Method Note:</strong>
  <span style="font-size:0.86rem; color:#CBD5E1;">
    These are Ridge regression |coefficient| values, not true SHAP.
    Install the <code>shap</code> package and re-run <code>explainability/shap_analysis.py</code> for exact SHAP.
  </span>
</div>""", unsafe_allow_html=True)

    hz = st.selectbox("Forecast Horizon", ["24h", "48h", "72h"], key="fi_hz")
    sdf = load_shap(hz)
    if sdf is not None and not sdf.empty:
        top = sdf.nlargest(min(15, len(sdf)), "mean_absolute_shap").copy()
        top["feature"] = top["feature"].str.replace("_", " ").str.title()
        fig_fi = px.bar(top, x="mean_absolute_shap", y="feature", orientation="h",
                        title=f"Top Feature Importances — +{hz}",
                        color="mean_absolute_shap", color_continuous_scale="Plasma",
                        labels={"mean_absolute_shap": "|Ridge Coef|", "feature": "Feature"})
        fig_fi.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="rgba(15,23,42,0.5)", height=400, font_color="#F0F4F8",
                             yaxis=dict(autorange="reversed"),
                             margin=dict(l=10,r=20,t=50,b=30), coloraxis_showscale=False)
        st.plotly_chart(fig_fi, width="stretch")
        top5 = sdf.nlargest(5, "mean_absolute_shap")["feature"].tolist()
        st.markdown(f"""
<div class="glass-card">
  <div style="font-weight:800; color:#38BDF8; margin-bottom:6px;">Why is the model predicting this?</div>
  <div style="font-size:0.9rem; color:#CBD5E1; line-height:1.65;">
    Top 5 driving features for +{hz}:
    <strong style="color:#F0F4F8;">{', '.join(top5)}</strong>.
    PM2.5 lag and rolling-mean features dominate at short horizons;
    calendar and pressure features gain weight at longer horizons.
  </div>
</div>""", unsafe_allow_html=True)
    else:
        st.info(f"Run `python explainability/shap_analysis.py` to generate +{hz} importances.")


# ============================================================================
# PAGE: HISTORY
# ============================================================================

elif st.session_state["current_nav"] == "History":
    st.markdown("""
<h2 style="font-size:1.5rem; font-weight:800; color:#F0F4F8; margin:0 0 4px 0;">
  📜 Prediction Audit Log
</h2>
<div style="font-size:0.85rem; color:#64748B; margin-bottom:14px;">
  Dashboard and Forecast page predictions are logged automatically for your account.
</div>""", unsafe_allow_html=True)

    history = get_prediction_history(user_id=user["id"], limit=100)

    if not history:
        st.markdown("""
<div class="glass-card" style="text-align:center; padding:40px;">
  <div style="font-size:2rem; margin-bottom:10px;">📭</div>
  <div style="font-weight:700; color:#94A3B8; font-size:1rem;">No prediction history yet</div>
  <div style="font-size:0.85rem; color:#475569; margin-top:6px;">
    Go to Dashboard or Forecast to generate predictions.
  </div>
</div>""", unsafe_allow_html=True)
    else:
        hdf = pd.DataFrame(history)
        hs1, hs2 = st.columns([2.5, 1])
        with hs1:
            search = st.text_input("Search by city or date", "", key="hist_q",
                                   placeholder="e.g. Lahore or 2026")
        with hs2:
            cf = st.selectbox("Filter by city", ["All"] + list(CITIES.keys()), key="hist_cf")
        if search:
            hdf = hdf[hdf["city"].str.contains(search, case=False, na=False) |
                      hdf["timestamp"].astype(str).str.contains(search, na=False)]
        if cf != "All":
            hdf = hdf[hdf["city"].str.lower() == cf.lower()]

        st.markdown(f"<div style='font-size:0.85rem; color:#64748B; margin-bottom:6px;'>{len(hdf)} records</div>", unsafe_allow_html=True)
        dcols = [c for c in ["id","city","predicted_aqi_24h","predicted_aqi_48h","predicted_aqi_72h","model_version","timestamp"] if c in hdf.columns]
        rename = {"id":"ID","city":"City","predicted_aqi_24h":"+24h","predicted_aqi_48h":"+48h",
                  "predicted_aqi_72h":"+72h","model_version":"Model","timestamp":"Time"}
        st.dataframe(hdf[dcols].rename(columns=rename), width="stretch", hide_index=True)


# ============================================================================
# PAGE: ALERTS
# ============================================================================

elif st.session_state["current_nav"] == "Alerts":
    al_ah1, al_ah2 = st.columns([3, 1])
    with al_ah1:
        st.markdown(f"""
<h2 style="font-size:1.5rem; font-weight:800; color:#F0F4F8; margin:0 0 4px 0;">
  🚨 AQI Hazard Alerts
</h2>
<div style="font-size:0.85rem; color:#64748B; margin-bottom:14px;">
  Alerts fire when AQI exceeds your threshold (AQI &gt; {alert_threshold}).
  Configure in Profile.
</div>""", unsafe_allow_html=True)
    with al_ah2:
        al_city2 = st.selectbox("City", list(CITIES.keys()),
                                index=list(CITIES.keys()).index(city), key="al_city2")
        if al_city2 != city:
            st.session_state["selected_city"] = al_city2
            st.rerun()

    with st.spinner("Evaluating alerts…"):
        forecasts = run_forecasts(city_df)
        active = build_alerts(city_df, city, live_aqi, forecasts, alert_threshold)

    if not active:
        st.markdown(f"""
<div class="glass-card" style="text-align:center; padding:40px; border-color:rgba(16,185,129,0.3);">
  <div style="font-size:2rem; margin-bottom:10px;">✅</div>
  <div style="font-weight:700; color:#10B981; font-size:1.1rem;">No Active Alerts</div>
  <div style="font-size:0.88rem; color:#475569; margin-top:8px;">
    Current AQI ({live_aqi if live_aqi is not None else "—"}) and all forecasts for {city}
    are within your threshold of {alert_threshold}.
  </div>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='font-weight:700; color:#EF4444; margin-bottom:10px;'>{len(active)} active alert(s) for {city}</div>", unsafe_allow_html=True)
        for a in active:
            sev = a["severity"]
            sc = "alert-critical" if sev == "CRITICAL" else ("alert-high" if sev == "HIGH" else "alert-moderate")
            sc_col = "#A855F7" if sev == "CRITICAL" else ("#EF4444" if sev == "HIGH" else "#F59E0B")
            st.markdown(f"""
<div class="{sc}">
  <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:6px; margin-bottom:7px;">
    <span style="font-weight:800; font-size:1rem; color:{sc_col};">{a['title']}</span>
    <div style="display:flex; gap:6px; flex-wrap:wrap;">
      <span class="pill" style="border-color:{sc_col}; color:{sc_col};">{sev}</span>
      <span class="pill">{a['horizon']}</span>
      <span class="pill">AQI {a['aqi']}</span>
    </div>
  </div>
  <div style="font-size:0.88rem; color:#CBD5E1; margin-bottom:6px;">{a['message']}</div>
  <div style="font-size:0.8rem; color:#94A3B8;"><strong>Action:</strong> {a['recommendation']}</div>
  <div style="font-size:0.68rem; color:#334155; margin-top:5px;">{a['ts']}</div>
</div>""", unsafe_allow_html=True)

    with st.expander("US EPA AQI Reference Scale"):
        ref = pd.DataFrame({
            "AQI Range": ["0-50","51-100","101-150","151-200","201-300","301-500"],
            "Category": ["Good","Moderate","Unhealthy for Sensitive Groups","Unhealthy","Very Unhealthy","Hazardous"],
            "Risk": ["None","Sensitive only","Sensitive groups","Everyone","Everyone — serious","Entire population"],
        })
        st.dataframe(ref, width="stretch", hide_index=True)


# ============================================================================
# PAGE: PROFILE
# ============================================================================

elif st.session_state["current_nav"] == "Profile":
    st.markdown("""
<h2 style="font-size:1.5rem; font-weight:800; color:#F0F4F8; margin:0 0 14px 0;">
  👤 Profile & Preferences
</h2>""", unsafe_allow_html=True)

    pr1, pr2 = st.columns(2)
    with pr1:
        st.markdown(f"""
<div class="glass-card">
  <div style="font-size:0.7rem; font-weight:800; color:#475569; text-transform:uppercase;
              letter-spacing:0.07em; margin-bottom:10px;">Account Details</div>
  <div style="margin-bottom:10px;">
    <div style="font-size:0.75rem; color:#64748B;">Full Name</div>
    <div style="font-weight:700; color:#F0F4F8;">{user.get('full_name') or '—'}</div>
  </div>
  <div style="margin-bottom:10px;">
    <div style="font-size:0.75rem; color:#64748B;">Email</div>
    <div style="font-weight:700; color:#F0F4F8;">{user.get('email','—')}</div>
  </div>
  <div>
    <div style="font-size:0.75rem; color:#64748B;">Member Since</div>
    <div style="font-weight:700; color:#F0F4F8;">{str(user.get('created_at','—'))[:10]}</div>
  </div>
</div>""", unsafe_allow_html=True)

    with pr2:
        models_ok = (MODEL_DIR / "best_model_24h.joblib").exists()
        fs_ok = fs_status.get("local_store_available", False)
        st.markdown(f"""
<div class="glass-card">
  <div style="font-size:0.7rem; font-weight:800; color:#475569; text-transform:uppercase;
              letter-spacing:0.07em; margin-bottom:10px;">Platform Status</div>
  <div style="margin-bottom:10px;">
    <div style="font-size:0.75rem; color:#64748B;">Feature Store</div>
    <div style="font-weight:700; color:{'#10B981' if fs_ok else '#EF4444'};">
      {'Connected — ' + str(fs_status.get('record_count',0)) + ' records' if fs_ok else 'Offline'}
    </div>
  </div>
  <div style="margin-bottom:10px;">
    <div style="font-size:0.75rem; color:#64748B;">Models (24h/48h/72h)</div>
    <div style="font-weight:700; color:{'#10B981' if models_ok else '#EF4444'};">
      {'All loaded' if models_ok else 'Missing — run train_3cities.py'}
    </div>
  </div>
  <div>
    <div style="font-size:0.75rem; color:#64748B;">Last Feature Update</div>
    <div style="font-weight:700; color:#F0F4F8;">{str(fs_status.get('last_updated','—'))[:19]}</div>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='font-size:1rem; font-weight:800; color:#94A3B8; margin:16px 0 10px 0;'>Alert & Notification Settings</div>", unsafe_allow_html=True)

    with st.form("pref_form"):
        fav = st.multiselect("Favourite Cities", list(CITIES.keys()),
                             default=prefs.get("favorite_cities", ["Lahore"]))
        thr = st.slider("Alert AQI Threshold", 50, 300,
                        value=int(prefs.get("alert_aqi_threshold", 150)), step=10,
                        help="Alerts fire when AQI exceeds this value.")
        ae  = st.text_input("Alert Email (optional)",
                            value=prefs.get("alert_email", user.get("email", "")),
                            placeholder="alerts@company.com")
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        save = st.form_submit_button("Save Preferences")

    if save:
        if not fav:
            st.error("Select at least one city.")
        else:
            ok = update_user_preferences(user["id"], fav, thr, ae.strip())
            if ok:
                st.success("Preferences saved.")
                st.rerun()
            else:
                st.error("Failed to save preferences.")

    st.markdown("<div style='font-size:1rem; font-weight:800; color:#94A3B8; margin:16px 0 10px 0;'>Session Management</div>", unsafe_allow_html=True)
    st.markdown("""
<div class="glass-card">
  <div style="font-size:0.88rem; color:#94A3B8; margin-bottom:12px;">
    Session secured with UUID4 bearer token. Tokens expire after 7 days.
  </div>
</div>""", unsafe_allow_html=True)
    if st.button("Sign Out", key="prof_logout"):
        logout_session(st.session_state["auth_token"])
        st.session_state["user"] = None
        st.session_state["auth_token"] = None
        st.rerun()
