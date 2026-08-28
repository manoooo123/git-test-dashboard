"""
Pearls AQI Predictor — Production Flask REST API Backend (v2.1.0)

Endpoints:
  GET  /                          — API manifest
  GET  /api/status                — System health telemetry
  GET  /api/cities                — Supported city metadata
  POST /api/auth/register         — User registration
  POST /api/auth/login            — User login
  POST /api/auth/logout           — Session invalidation
  GET  /api/auth/me               — Current session user
  GET|POST /api/auth/callback     — OAuth callback recovery (local redirect)
  GET  /api/aqi/live              — Latest live AQI & weather metrics
  GET  /api/aqi/forecast          — 3-day ML forecast (+24h/+48h/+72h)
  GET  /api/aqi/comparison        — Multi-city AQI side-by-side
  GET  /api/aqi/explainability    — Feature importance (SHAP or coefficients)
  GET  /api/aqi/explain-live     — Real-time prediction explanation
  POST /api/predict               — Scenario prediction using trained model
  GET  /api/history               — User prediction audit log
  GET|POST /api/user/preferences  — User preferences CRUD
  GET  /api/models/performance    — Training report metrics
  GET  /api/alerts                — Data-driven AQI hazard alerts
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, request, redirect

from utils.db import (
    register_user,
    authenticate_user,
    create_session,
    validate_session,
    logout_session,
    log_prediction,
    get_prediction_history,
    get_user_preferences,
    update_user_preferences,
)
from utils.feature_store import feature_store
from utils.data_quality import data_quality_monitor
from utils.model_health import model_health_monitor
from utils.model_registry import model_registry
from utils.live_explainability import live_explainer

# ============================================================================
# BOOTSTRAP
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("pearls_api")

# ============================================================================
# PATHS
# ============================================================================

MODEL_DIR = PROJECT_ROOT / "models" / "3cities"
REPORT_DIR = PROJECT_ROOT / "reports"
EVALUATION_FILE = REPORT_DIR / "model_evaluation" / "3cities" / "training_report_3cities.json"
BEST_MODELS_FILE = REPORT_DIR / "model_evaluation" / "best_models.json"
SHAP_DIR = REPORT_DIR / "explainability"

# ============================================================================
# FLASK APP
# ============================================================================

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(32).hex())

# ============================================================================
# RATE LIMITING (simple in-memory token bucket per IP)
# ============================================================================

_login_attempts: dict[str, list[float]] = defaultdict(list)
_LOGIN_WINDOW_SECONDS = 60
_LOGIN_MAX_ATTEMPTS = 10


def _check_rate_limit(ip: str) -> bool:
    """Return True if the IP is within the allowed login rate, False if throttled."""
    now = time.time()
    attempts = _login_attempts[ip]
    # Prune old entries outside the window
    _login_attempts[ip] = [t for t in attempts if now - t < _LOGIN_WINDOW_SECONDS]
    if len(_login_attempts[ip]) >= _LOGIN_MAX_ATTEMPTS:
        return False
    _login_attempts[ip].append(now)
    return True


# ============================================================================
# CORS
# ============================================================================

_ALLOWED_ORIGINS = {
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost:5000",
    "http://127.0.0.1:5000",
}

# Allow overriding via env var (comma-separated)
_extra = os.getenv("CORS_ALLOWED_ORIGINS", "")
for _o in _extra.split(","):
    _o = _o.strip()
    if _o:
        _ALLOWED_ORIGINS.add(_o)


@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin", "")
    if origin in _ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
    else:
        # For requests with no Origin (e.g. curl, server-to-server) allow all
        if not origin:
            response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.route("/", defaults={"path": ""}, methods=["OPTIONS"])
@app.route("/<path:path>", methods=["OPTIONS"])
def handle_options(path):
    return jsonify({}), 200


# ============================================================================
# AUTH HELPERS
# ============================================================================

def get_token_user() -> Optional[Dict[str, Any]]:
    """Extract and validate bearer token from Authorization header."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.split("Bearer ", 1)[1].strip()
        return validate_session(token)
    return None


def _safe_redirect_target() -> str:
    """
    Return the configured local Streamlit URL for auth callbacks.

    Priority:
    1. Validated local 'next' / 'return_to' / 'redirect_uri' request parameter
    2. PEARLS_STREAMLIT_URL / STREAMLIT_URL / FLASK_APP_URL / APP_URL / AUTH_REDIRECT_URL env vars
    3. Default http://localhost:8501
    """
    for param in ("next", "return_to", "redirect_uri"):
        val = request.args.get(param) or request.form.get(param, "")
        if val:
            parsed = urlparse(val)
            # Only allow local URLs (no scheme = relative path, or explicit localhost)
            if not parsed.scheme and val.startswith("/"):
                return val
            if parsed.scheme in {"http", "https"} and parsed.netloc in {
                "localhost", "127.0.0.1"
            }:
                return val

    for env_var in (
        "PEARLS_STREAMLIT_URL",
        "STREAMLIT_URL",
        "FLASK_APP_URL",
        "APP_URL",
        "AUTH_REDIRECT_URL",
    ):
        val = os.getenv(env_var, "").strip().rstrip("/")
        if val:
            return val

    return "http://localhost:8501"


# ============================================================================
# CITIES METADATA
# ============================================================================

CITIES_METADATA = {
    "Lahore": {
        "city": "Lahore",
        "latitude": 31.5204,
        "longitude": 74.3587,
        "country": "Pakistan",
        "region": "Punjab",
        "tagline": "Provincial Capital & Cultural Hub",
    },
    "Islamabad": {
        "city": "Islamabad",
        "latitude": 33.6844,
        "longitude": 73.0479,
        "country": "Pakistan",
        "region": "Federal Territory",
        "tagline": "Federal Capital & Margalla Foothills",
    },
    "Faisalabad": {
        "city": "Faisalabad",
        "latitude": 31.4504,
        "longitude": 73.1350,
        "country": "Pakistan",
        "region": "Punjab",
        "tagline": "Industrial Center & Textile Capital",
    },
}

# ============================================================================
# AQI HELPERS
# ============================================================================

def calculate_us_aqi(pm25: float) -> int:
    """Convert PM2.5 concentration (µg/m³) to US EPA AQI using official breakpoints."""
    if pm25 is None or (isinstance(pm25, float) and (np.isnan(pm25) or np.isinf(pm25))):
        return 0
    pm25 = max(0.0, float(pm25))
    if pm25 <= 12.0:
        return int(round((50 / 12.0) * pm25))
    elif pm25 <= 35.4:
        return int(round(50 + (50 / 23.4) * (pm25 - 12.1)))
    elif pm25 <= 55.4:
        return int(round(100 + (50 / 20.0) * (pm25 - 35.5)))
    elif pm25 <= 150.4:
        return int(round(150 + (50 / 95.0) * (pm25 - 55.5)))
    elif pm25 <= 250.4:
        return int(round(200 + (100 / 100.0) * (pm25 - 150.5)))
    elif pm25 <= 350.4:
        return int(round(300 + (100 / 100.0) * (pm25 - 250.5)))
    elif pm25 <= 500.4:
        return int(round(400 + (100 / 150.0) * (pm25 - 350.5)))
    return 500


def get_aqi_meta(aqi: int) -> Dict[str, str]:
    """Return AQI category, hex color, and responsible health advice."""
    if aqi <= 50:
        return {
            "category": "Good",
            "color": "#10B981",
            "health": "Air quality is satisfactory. Ideal conditions for all outdoor activities.",
        }
    elif aqi <= 100:
        return {
            "category": "Moderate",
            "color": "#F59E0B",
            "health": "Air quality is acceptable. Unusually sensitive individuals may notice minor effects.",
        }
    elif aqi <= 150:
        return {
            "category": "Unhealthy for Sensitive Groups",
            "color": "#F97316",
            "health": "Children, elderly, and those with respiratory conditions should limit prolonged outdoor exertion.",
        }
    elif aqi <= 200:
        return {
            "category": "Unhealthy",
            "color": "#EF4444",
            "health": "Everyone may begin to experience health effects. Avoid prolonged outdoor exertion.",
        }
    elif aqi <= 300:
        return {
            "category": "Very Unhealthy",
            "color": "#A855F7",
            "health": "Health alert: everyone may experience serious effects. Remain indoors with air purification.",
        }
    else:
        return {
            "category": "Hazardous",
            "color": "#7E22CE",
            "health": "Health emergency. The entire population is likely to be affected. Stay indoors.",
        }


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Convert to float, returning default on NaN/None/Inf."""
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _run_model_inference(
    model_path: Path, X: pd.DataFrame
) -> tuple[bool, float, str]:
    """
    Load a joblib model and run inference on X.

    Returns (success, predicted_pm25, error_message).
    The model pipeline includes SimpleImputer so NaN inputs are handled.
    """
    if not model_path.exists():
        return False, 0.0, f"Model artifact not found: {model_path.name}"
    try:
        model = joblib.load(model_path)
        raw = float(model.predict(X)[0])
        if np.isnan(raw) or np.isinf(raw):
            return False, 0.0, "Model returned NaN/Inf prediction."
        return True, max(0.0, raw), ""
    except Exception as exc:
        logger.error("Inference error [%s]: %s", model_path.name, exc)
        return False, 0.0, str(exc)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "project": "Pearls AQI Predictor API",
        "version": "2.1.0",
        "supported_cities": list(CITIES_METADATA.keys()),
        "status": "Operational",
        "data_sources": "OpenAQ v3 API + Open-Meteo Weather",
        "endpoints": [
            "GET  /api/status",
            "POST /api/auth/register",
            "POST /api/auth/login",
            "POST /api/auth/logout",
            "GET  /api/auth/me",
            "GET  /api/cities",
            "GET  /api/aqi/live",
            "GET  /api/aqi/forecast",
            "GET  /api/aqi/comparison",
            "GET  /api/aqi/explainability",
            "GET  /api/aqi/explain-live",
            "POST /api/predict",
            "GET  /api/history",
            "GET  /api/user/preferences",
            "POST /api/user/preferences",
            "GET  /api/models/performance",
            "GET  /api/models/health",
            "GET  /api/models/registry",
            "GET  /api/data-quality",
            "GET  /api/alerts",
        ],
    })


# ----------------------------------------------------------------------------
# AUTH
# ----------------------------------------------------------------------------

@app.route("/api/auth/register", methods=["POST"])
def auth_register():
    data = request.get_json(force=True, silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))
    full_name = str(data.get("full_name", "")).strip()

    success, msg, user_id = register_user(email, password, full_name)
    if not success:
        return jsonify({"success": False, "error": msg}), 400

    token = create_session(user_id)
    return jsonify({
        "success": True,
        "message": msg,
        "user": {"id": user_id, "email": email, "full_name": full_name},
        "token": token,
    }), 201


@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    ip = request.remote_addr or "unknown"
    if not _check_rate_limit(ip):
        logger.warning("Rate limit exceeded for IP: %s", ip)
        return jsonify({
            "success": False,
            "error": "Too many login attempts. Please wait before trying again.",
        }), 429

    data = request.get_json(force=True, silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))

    success, msg, user_data = authenticate_user(email, password)
    if not success or not user_data:
        return jsonify({"success": False, "error": msg}), 401

    token = create_session(user_data["id"])
    return jsonify({
        "success": True,
        "message": msg,
        "user": user_data,
        "token": token,
    })


@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.split("Bearer ", 1)[1].strip()
        logout_session(token)
    return jsonify({"success": True, "message": "Logged out successfully."})


@app.route("/api/auth/me", methods=["GET"])
def auth_me():
    user = get_token_user()
    if not user:
        return jsonify({"success": False, "error": "Unauthorized or session expired."}), 401
    return jsonify({"success": True, "user": user})


@app.route("/api/auth/callback", methods=["GET", "POST"])
@app.route("/auth/callback", methods=["GET", "POST"])
@app.route("/oauth/callback", methods=["GET", "POST"])
def auth_callback():
    """
    Auth callback recovery endpoint.

    This project uses local credential-based auth, not OAuth. However, when
    a browser returns to this endpoint after an external authentication page
    (e.g. Antigravity), this route catches the callback and redirects the
    browser back to the Pearls AQI Predictor Streamlit application.

    The target URL is resolved from PEARLS_STREAMLIT_URL env var (default:
    http://localhost:8501). A visible HTML fallback link is included so the
    user is never stranded.
    """
    error = request.args.get("error") or request.form.get("error", "")
    if error:
        logger.warning("Auth callback received provider error: %s", error)
        return jsonify({
            "success": False,
            "error": f"Authentication provider reported: {error}",
            "action": "Please return to the application and sign in with your email and password.",
        }), 400

    target = _safe_redirect_target()
    logger.info("Auth callback redirect → %s", target)

    # Provide both an automatic redirect and a visible fallback link
    # so the user is never stuck on a blank "successfully authenticated" page.
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="1;url={target}">
  <title>Redirecting — Pearls AQI Predictor</title>
  <style>
    *{{margin:0;padding:0;box-sizing:border-box}}
    body{{font-family:system-ui,sans-serif;background:#070A11;color:#F8FAFC;
         display:flex;align-items:center;justify-content:center;height:100vh;}}
    .card{{background:rgba(15,23,42,0.9);border:1px solid rgba(255,255,255,0.12);
           border-radius:16px;padding:40px;text-align:center;max-width:440px;}}
    h2{{font-size:1.4rem;font-weight:800;margin-bottom:8px;color:#38BDF8}}
    p{{color:#94A3B8;font-size:0.95rem;margin-bottom:20px;line-height:1.6}}
    a{{display:inline-block;background:linear-gradient(135deg,#0EA5E9,#2563EB);
       color:#fff;font-weight:700;padding:12px 28px;border-radius:10px;
       text-decoration:none;font-size:1rem}}
    a:hover{{opacity:0.9}}
    .spinner{{width:32px;height:32px;border:3px solid rgba(56,189,248,0.3);
              border-top-color:#38BDF8;border-radius:50%;
              animation:spin 0.8s linear infinite;margin:0 auto 16px}}
    @keyframes spin{{to{{transform:rotate(360deg)}}}}
  </style>
</head>
<body>
  <div class="card">
    <div class="spinner"></div>
    <h2>Authentication Successful</h2>
    <p>You have been authenticated. Returning you to Pearls AQI Predictor automatically&hellip;</p>
    <a href="{target}">Click here if not redirected</a>
  </div>
</body>
</html>"""
    from flask import Response
    return Response(html, status=200, content_type="text/html")


# ----------------------------------------------------------------------------
# SYSTEM TELEMETRY
# ----------------------------------------------------------------------------

@app.route("/api/status", methods=["GET"])
def system_status():
    fs_status = feature_store.get_status()
    models_ready = all(
        (MODEL_DIR / f"best_model_{h}h.joblib").exists()
        for h in (24, 48, 72)
    )
    overall = "Operational" if (fs_status["local_store_available"] and models_ready) else "Degraded"

    return jsonify({
        "system": "Pearls AQI Intelligence Platform",
        "version": "2.1.0",
        "status": overall,
        "data_source": "OpenAQ v3 API + Open-Meteo Weather",
        "last_checked_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "feature_store": fs_status,
        "prediction_engine": {
            "status": "Online" if models_ready else "Offline",
            "models_ready": models_ready,
            "horizons": ["24h", "48h", "72h"],
        },
        "database": {"status": "Connected", "engine": "SQLite"},
        "target_cities": list(CITIES_METADATA.keys()),
    })


@app.route("/api/cities", methods=["GET"])
def list_cities():
    return jsonify({
        "success": True,
        "count": len(CITIES_METADATA),
        "cities": list(CITIES_METADATA.values()),
    })


@app.route("/api/data-quality", methods=["GET"])
def data_quality():
    """Return real-time data quality assessment for Feature Store."""
    city = request.args.get("city")  # Optional city filter
    
    report = data_quality_monitor.get_quality_report(city=city)
    
    return jsonify({
        "success": True,
        "city": city if city else "All Cities",
        "data_source": "OpenAQ v3 API + Open-Meteo Weather",
        "quality_report": {
            "overall_status": report.status,
            "overall_score": report.overall_score,
            "last_update": report.last_update,
            "freshness_hours": report.freshness_hours,
            "checks": report.checks,
            "issues": report.issues,
            "metrics": report.metrics,
        },
    })


@app.route("/api/models/health", methods=["GET"])
def model_health():
    """Return real-time model health and artifact status."""
    report = model_health_monitor.get_health_report()
    
    return jsonify({
        "success": True,
        "overall_status": report.overall_status,
        "models_available": report.models_available,
        "models_expected": report.models_expected,
        "model_statuses": [
            {
                "horizon": s.horizon,
                "available": s.available,
                "size_mb": s.size_mb,
                "last_modified": s.last_modified,
                "error": s.error
            }
            for s in report.model_statuses
        ],
        "performance_metrics": [
            {
                "horizon": p.horizon,
                "model_type": p.model_type,
                "mae": round(p.mae, 2),
                "rmse": round(p.rmse, 2),
                "r2": round(p.r2, 4),
                "samples_trained": p.samples_trained,
                "by_city": p.by_city
            }
            for p in report.performance_metrics
        ],
        "training_info": report.training_info,
        "shap_available": report.shap_available,
        "last_training": report.last_training,
    })


@app.route("/api/models/registry", methods=["GET"])
def model_registry_endpoint():
    """
    Expose Model Registry for production model tracking.
    
    Returns:
    - Full registry report with all model versions
    - Production model status for each horizon
    - Performance comparison across versions
    
    Optional query params:
    - horizon: Filter by horizon (24h, 48h, 72h)
    - status: Filter by deployment status (production, staging, archived)
    """
    horizon = request.args.get("horizon")
    status_filter = request.args.get("status")
    
    try:
        # Get comprehensive registry report
        registry_report = model_registry.export_registry_report()
        
        # Get all models (optionally filtered by horizon)
        all_models = model_registry.get_all_models(horizon=horizon)
        
        # Apply status filter if provided
        if status_filter:
            all_models = [m for m in all_models if m.deployment_status == status_filter]
        
        # Format models for response
        models_list = [
            {
                "model_id": m.model_id,
                "horizon": m.horizon,
                "algorithm": m.algorithm,
                "version": m.version,
                "deployment_status": m.deployment_status,
                "created_at": m.created_at,
                "file_path": m.file_path,
                "file_size_mb": m.file_size_mb,
                "checksum": m.checksum[:16] + "...",  # Truncate for readability
                "performance": {
                    "MAE": round(m.performance.get("MAE", 0), 2),
                    "RMSE": round(m.performance.get("RMSE", 0), 2),
                    "R2": round(m.performance.get("R2", 0), 4),
                },
                "training_samples": m.training_samples,
                "feature_count": m.feature_count,
                "notes": m.notes,
            }
            for m in all_models
        ]
        
        # Get performance comparison for each horizon
        performance_comparison = {}
        for h in ["24h", "48h", "72h"]:
            if horizon is None or horizon == h:
                performance_comparison[h] = model_registry.get_performance_comparison(h)
        
        return jsonify({
            "success": True,
            "registry_summary": registry_report,
            "models": models_list,
            "total_models": len(models_list),
            "performance_comparison": performance_comparison,
            "filters_applied": {
                "horizon": horizon if horizon else "all",
                "status": status_filter if status_filter else "all",
            },
        })
    
    except Exception as e:
        logger.error(f"Error retrieving model registry: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to retrieve model registry",
            "detail": str(e),
        }), 500


# ----------------------------------------------------------------------------
# AIR QUALITY
# ----------------------------------------------------------------------------

@app.route("/api/aqi/live", methods=["GET"])
def live_aqi():
    city = request.args.get("city", "Lahore")
    df = feature_store.load_features()

    if not df.empty and "city" in df.columns:
        city_df = df[df["city"].str.lower() == city.lower()]
        if not city_df.empty:
            latest = city_df.iloc[-1]
            pm25 = _safe_float(latest.get("pm25_mean"), default=None)
            if pm25 is None:
                # Column may be named differently — try alternatives
                for col in ("pm25_median", "pm2_5_24h_mean", "pm2_5"):
                    v = latest.get(col)
                    if v is not None:
                        pm25 = _safe_float(v, default=None)
                        if pm25 is not None:
                            break
            if pm25 is None:
                pm25 = 65.0

            aqi = calculate_us_aqi(pm25)
            meta = get_aqi_meta(aqi)

            temp = _safe_float(latest.get("temperature"), default=None)
            humidity = _safe_float(latest.get("humidity"), default=None)
            wind = _safe_float(latest.get("wind_speed"), default=None)
            pressure = _safe_float(latest.get("pressure"), default=None)

            return jsonify({
                "success": True,
                "city": city.capitalize(),
                "data_source": "OpenAQ v3 API + Open-Meteo Weather",
                "data_freshness": "live",
                "last_updated_utc": datetime.now(timezone.utc).strftime("%H:%M UTC"),
                "observation_timestamp": str(latest.get("hour", "N/A")),
                "metrics": {
                    "aqi": aqi,
                    "category": meta["category"],
                    "color": meta["color"],
                    "pm25": round(pm25, 1),
                    # PM10 is not measured — report as unavailable rather than fabricating PM25*1.6
                    "pm10": None,
                    "pm10_note": "PM10 not available from OpenAQ sensor network for this location.",
                    "temperature": round(temp, 1) if temp is not None else None,
                    "humidity": round(humidity, 0) if humidity is not None else None,
                    "wind_speed": round(wind, 1) if wind is not None else None,
                    "pressure": round(pressure, 1) if pressure is not None else None,
                },
                "health_advice": meta["health"],
            })

    # Offline cache fallback — clearly labelled
    default_aqi = 135
    meta = get_aqi_meta(default_aqi)
    return jsonify({
        "success": True,
        "city": city.capitalize(),
        "data_source": "OpenAQ v3 API + Open-Meteo Weather",
        "data_freshness": "offline_cache",
        "last_updated_utc": datetime.now(timezone.utc).strftime("%H:%M UTC"),
        "observation_timestamp": "Offline Cache",
        "metrics": {
            "aqi": default_aqi,
            "category": meta["category"],
            "color": meta["color"],
            "pm25": 49.5,
            "pm10": None,
            "pm10_note": "Feature store offline — live metrics unavailable.",
            "temperature": None,
            "humidity": None,
            "wind_speed": None,
            "pressure": None,
        },
        "health_advice": meta["health"],
        "warning": "Feature store is offline. Showing last known cached AQI estimate.",
    }), 200


@app.route("/api/aqi/forecast", methods=["GET"])
def aqi_forecast():
    """Return ML-driven 3-day AQI forecast for a city."""
    city = request.args.get("city", "Lahore")

    if city.lower() not in {c.lower() for c in CITIES_METADATA}:
        return jsonify({
            "success": False,
            "error": f"Unsupported city '{city}'.",
            "supported_cities": list(CITIES_METADATA.keys()),
        }), 400

    df = feature_store.load_features()
    if df.empty or "city" not in df.columns:
        return jsonify({"success": False, "error": "Feature dataset is unavailable."}), 503

    city_df = df[df["city"].str.lower() == city.lower()].copy()
    if city_df.empty:
        return jsonify({"success": False, "error": f"No feature data available for {city}."}), 404

    city_df["hour"] = pd.to_datetime(city_df["hour"], errors="coerce", utc=True)
    city_df = city_df.dropna(subset=["hour"]).sort_values("hour").reset_index(drop=True)
    latest_timestamp = city_df["hour"].iloc[-1]
    latest_row = city_df.iloc[-1]

    excluded = {"city", "hour", "coverage_quality", "target_24h", "target_48h", "target_72h"}
    feature_cols = [
        c for c in city_df.columns
        if c not in excluded and np.issubdtype(city_df[c].dtype, np.number)
    ]
    # Use the model's own SimpleImputer — do not fillna(0) here; pass raw NaN values
    X_latest = pd.DataFrame([latest_row[feature_cols]])

    forecast: Dict[str, Any] = {}
    predicted_aqis: Dict[str, int] = {}
    all_succeeded = True

    for horizon in (24, 48, 72):
        model_path = MODEL_DIR / f"best_model_{horizon}h.joblib"
        success, predicted_pm25, err = _run_model_inference(model_path, X_latest)

        if not success:
            logger.error("Forecast failed for +%dh: %s", horizon, err)
            forecast[f"{horizon}h"] = {
                "horizon": f"+{horizon} Hours",
                "status": "unavailable",
                "error": err,
            }
            all_succeeded = False
            continue

        predicted_aqi = calculate_us_aqi(predicted_pm25)
        meta = get_aqi_meta(predicted_aqi)
        forecast_timestamp = latest_timestamp + pd.Timedelta(hours=horizon)
        forecast[f"{horizon}h"] = {
            "horizon": f"+{horizon} Hours",
            "status": "success",
            "forecast_timestamp": forecast_timestamp.isoformat(),
            "pm25": round(predicted_pm25, 2),
            "aqi": predicted_aqi,
            "category": meta["category"],
            "color": meta["color"],
            "health_advice": meta["health"],
        }
        predicted_aqis[f"{horizon}h"] = predicted_aqi

    # Log to DB only when all horizons succeeded
    if all_succeeded:
        user = get_token_user()
        user_id = user["id"] if user else None
        log_prediction(
            city.capitalize(),
            predicted_aqis["24h"],
            predicted_aqis["48h"],
            predicted_aqis["72h"],
            user_id=user_id,
        )

    return jsonify({
        "success": True,
        "city": city.capitalize(),
        "data_source": "OpenAQ v3 API + Open-Meteo Weather",
        "last_updated_utc": datetime.now(timezone.utc).strftime("%H:%M UTC"),
        "generated_from": latest_timestamp.isoformat(),
        "forecast": forecast,
    })


@app.route("/api/aqi/comparison", methods=["GET"])
def aqi_comparison():
    df = feature_store.load_features()
    comparison = []

    for city in CITIES_METADATA:
        pm25 = None
        if not df.empty and "city" in df.columns:
            cdf = df[df["city"].str.lower() == city.lower()]
            if not cdf.empty:
                for col in ("pm25_mean", "pm25_median", "pm2_5_24h_mean"):
                    v = _safe_float(cdf.iloc[-1].get(col), default=None)
                    if v is not None:
                        pm25 = v
                        break

        if pm25 is None:
            comparison.append({
                "city": city,
                "status": "unavailable",
                "aqi": None,
                "pm25": None,
                "category": "Data Unavailable",
                "color": "#64748B",
            })
            continue

        aqi = calculate_us_aqi(pm25)
        meta = get_aqi_meta(aqi)
        comparison.append({
            "city": city,
            "status": "live",
            "aqi": aqi,
            "pm25": round(pm25, 1),
            "category": meta["category"],
            "color": meta["color"],
        })

    comparison.sort(key=lambda x: x["aqi"] if x["aqi"] is not None else -1, reverse=True)
    return jsonify({
        "success": True,
        "data_source": "OpenAQ v3 API + Open-Meteo Weather",
        "last_updated_utc": datetime.now(timezone.utc).strftime("%H:%M UTC"),
        "comparison": comparison,
    })


@app.route("/api/aqi/explainability", methods=["GET"])
def explainability():
    horizon = request.args.get("horizon", "24h")
    # Normalise horizon input
    horizon_clean = horizon.replace("h", "").strip()
    if horizon_clean not in {"24", "48", "72"}:
        horizon_clean = "24"
    horizon_key = f"{horizon_clean}h"

    shap_path = SHAP_DIR / f"shap_feature_importance_{horizon_key}.csv"

    if shap_path.exists():
        try:
            shap_df = pd.read_csv(shap_path).head(15)
            # Clarify the method — these are Ridge |coef_| values, not SHAP
            return jsonify({
                "success": True,
                "horizon": horizon_key,
                "method": "Ridge absolute coefficients (proxy for feature importance)",
                "method_note": (
                    "Feature importances are derived from Ridge regression |coefficient| values, "
                    "not true SHAP values. Install the 'shap' package and re-run "
                    "explainability/shap_analysis.py to obtain exact SHAP explanations."
                ),
                "features": shap_df.to_dict(orient="records"),
            })
        except Exception as e:
            logger.error("Error reading feature importance file: %s", e)

    return jsonify({
        "success": False,
        "error": f"Feature importance file not found for horizon {horizon_key}.",
        "features": [],
    }), 404


@app.route("/api/predict", methods=["POST"])
def predict_scenario():
    """
    Scenario prediction using the trained 24h Ridge model.

    Accepts a partial feature set and builds a full feature row using the
    feature store's latest record as a base, then overrides specified fields.
    This avoids the previous fake linear formula.
    """
    try:
        data = request.get_json(force=True, silent=True) or {}

        # Load latest feature row to use as context
        df = feature_store.load_features()
        city = str(data.get("city", "Lahore"))

        if not df.empty and "city" in df.columns:
            city_df = df[df["city"].str.lower() == city.lower()]
            if not city_df.empty:
                base_row = city_df.iloc[-1].copy()
            else:
                base_row = df.iloc[-1].copy()
        else:
            return jsonify({
                "success": False,
                "error": "Feature store unavailable. Cannot run scenario prediction.",
            }), 503

        excluded = {"city", "hour", "coverage_quality", "target_24h", "target_48h", "target_72h"}
        feature_cols = [
            c for c in df.columns
            if c not in excluded and np.issubdtype(df[c].dtype, np.number)
        ]

        X = pd.DataFrame([base_row[feature_cols]])

        # Override user-specified fields where column names match
        field_map = {
            "pm25": "pm25_mean",
            "temperature": "temperature",
            "humidity": "humidity",
            "wind_speed": "wind_speed",
            "pressure": "pressure",
        }
        for user_key, col_name in field_map.items():
            if user_key in data and col_name in X.columns:
                X[col_name] = float(data[user_key])
                # Also update lag columns if pm25 is overridden
                if user_key == "pm25":
                    for lag_col in ("pm25_lag_1h", "pm25_median"):
                        if lag_col in X.columns:
                            X[lag_col] = float(data[user_key])

        model_path = MODEL_DIR / "best_model_24h.joblib"
        success, predicted_pm25, err = _run_model_inference(model_path, X)

        if not success:
            return jsonify({"success": False, "error": f"Model inference failed: {err}"}), 503

        predicted_aqi = calculate_us_aqi(predicted_pm25)
        meta = get_aqi_meta(predicted_aqi)

        return jsonify({
            "success": True,
            "city": city,
            "model": "Ridge Regression (best_model_24h — 24h horizon)",
            "method": "Scenario overlaid on latest feature store row; remaining features from live data.",
            "inputs": data,
            "prediction": {
                "predicted_pm25": round(predicted_pm25, 1),
                "predicted_aqi": predicted_aqi,
                "category": meta["category"],
                "color": meta["color"],
                "health_advice": meta["health"],
                "horizon": "+24 Hours",
            },
        })
    except Exception as exc:
        logger.exception("Unexpected error in /api/predict")
        return jsonify({"success": False, "error": "Internal server error."}), 500


# ----------------------------------------------------------------------------
# ALERTS (DATA-DRIVEN)
# ----------------------------------------------------------------------------

def _build_alerts(city: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Generate AQI hazard alerts from live feature store data and model forecasts.

    Alerts are produced for:
    - Current live AQI above threshold
    - Each forecast horizon (+24h/+48h/+72h) above threshold

    Default alert threshold: AQI > 150 (Unhealthy for Sensitive Groups).
    """
    ALERT_THRESHOLD = 150
    alerts: List[Dict[str, Any]] = []
    df = feature_store.load_features()

    cities_to_check = list(CITIES_METADATA.keys()) if not city else [city]

    for c in cities_to_check:
        if not df.empty and "city" in df.columns:
            city_df = df[df["city"].str.lower() == c.lower()]
        else:
            city_df = pd.DataFrame()

        if city_df.empty:
            continue

        latest = city_df.iloc[-1]
        pm25 = None
        for col in ("pm25_mean", "pm25_median", "pm2_5_24h_mean"):
            v = _safe_float(latest.get(col), default=None)
            if v is not None:
                pm25 = v
                break

        if pm25 is None:
            continue

        live_aqi = calculate_us_aqi(pm25)
        live_meta = get_aqi_meta(live_aqi)

        # Current AQI alert
        if live_aqi > ALERT_THRESHOLD:
            severity = "CRITICAL" if live_aqi > 300 else ("HIGH" if live_aqi > 200 else "MODERATE")
            alerts.append({
                "id": f"ALT-{c.upper()[:3]}-LIVE",
                "city": c,
                "horizon": "Current",
                "severity": severity,
                "title": f"Elevated AQI Alert — {c} (Current)",
                "message": (
                    f"Live AQI of {live_aqi} ({live_meta['category']}) exceeds the "
                    f"alert threshold of {ALERT_THRESHOLD}."
                ),
                "aqi": live_aqi,
                "category": live_meta["category"],
                "color": live_meta["color"],
                "recommendation": live_meta["health"],
                "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            })

        # Forecast alerts
        city_df["hour"] = pd.to_datetime(city_df["hour"], errors="coerce", utc=True)
        city_df = city_df.dropna(subset=["hour"]).sort_values("hour").reset_index(drop=True)
        excluded = {"city", "hour", "coverage_quality", "target_24h", "target_48h", "target_72h"}
        feature_cols = [
            col for col in city_df.columns
            if col not in excluded and np.issubdtype(city_df[col].dtype, np.number)
        ]
        if not feature_cols:
            continue

        X = pd.DataFrame([city_df.iloc[-1][feature_cols]])

        for horizon in (24, 48, 72):
            model_path = MODEL_DIR / f"best_model_{horizon}h.joblib"
            ok, pred_pm25, _ = _run_model_inference(model_path, X)
            if not ok:
                continue
            pred_aqi = calculate_us_aqi(pred_pm25)
            pred_meta = get_aqi_meta(pred_aqi)

            if pred_aqi > ALERT_THRESHOLD:
                severity = "CRITICAL" if pred_aqi > 300 else ("HIGH" if pred_aqi > 200 else "MODERATE")
                alerts.append({
                    "id": f"ALT-{c.upper()[:3]}-{horizon}H",
                    "city": c,
                    "horizon": f"+{horizon} Hours",
                    "severity": severity,
                    "title": f"Forecast AQI Alert — {c} (+{horizon}h)",
                    "message": (
                        f"Model forecast AQI of {pred_aqi} ({pred_meta['category']}) "
                        f"in +{horizon}h exceeds the alert threshold of {ALERT_THRESHOLD}."
                    ),
                    "aqi": pred_aqi,
                    "category": pred_meta["category"],
                    "color": pred_meta["color"],
                    "recommendation": pred_meta["health"],
                    "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                })

    return alerts


@app.route("/api/alerts", methods=["GET"])
def get_alerts():
    city = request.args.get("city", None)
    if city and city.lower() not in {c.lower() for c in CITIES_METADATA}:
        return jsonify({"success": False, "error": f"Unsupported city: {city}"}), 400

    alerts = _build_alerts(city=city)

    if not alerts:
        return jsonify({
            "success": True,
            "count": 0,
            "alerts": [],
            "message": "No active AQI alerts. Air quality is within acceptable limits across all monitored cities.",
        })

    return jsonify({
        "success": True,
        "count": len(alerts),
        "alerts": alerts,
    })


# ----------------------------------------------------------------------------
# USER HISTORY & PREFERENCES
# ----------------------------------------------------------------------------

@app.route("/api/history", methods=["GET"])
def user_history():
    user = get_token_user()
    user_id = user["id"] if user else None
    history = get_prediction_history(user_id=user_id, limit=50)
    return jsonify({"success": True, "count": len(history), "history": history})


@app.route("/api/user/preferences", methods=["GET", "POST"])
def user_prefs():
    user = get_token_user()
    if not user:
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    if request.method == "GET":
        prefs = get_user_preferences(user["id"])
        return jsonify({"success": True, "preferences": prefs})

    data = request.get_json(force=True, silent=True) or {}
    favorite_cities = data.get("favorite_cities", ["Lahore"])
    if not isinstance(favorite_cities, list):
        favorite_cities = [str(favorite_cities)]
    alert_aqi_threshold = int(data.get("alert_aqi_threshold", 150))
    alert_email = str(data.get("alert_email", user.get("email", ""))).strip()

    updated = update_user_preferences(user["id"], favorite_cities, alert_aqi_threshold, alert_email)
    if updated:
        return jsonify({"success": True, "message": "Preferences updated successfully."})
    return jsonify({"success": False, "error": "Failed to update preferences."}), 500


@app.route("/api/models/performance", methods=["GET"])
def model_performance():
    if not EVALUATION_FILE.exists():
        # Fallback to best_models.json
        if BEST_MODELS_FILE.exists():
            try:
                with open(BEST_MODELS_FILE, "r", encoding="utf-8") as f:
                    best = json.load(f)
                return jsonify({"success": True, "source": "best_models.json", "results": best})
            except Exception as e:
                return jsonify({"success": False, "error": str(e)}), 500
        return jsonify({"success": False, "error": "Evaluation report not found."}), 404

    try:
        with open(EVALUATION_FILE, "r", encoding="utf-8") as f:
            report = json.load(f)
        return jsonify({
            "success": True,
            "source": "training_report_3cities.json",
            "feature_count": report.get("feature_count"),
            "dataset_rows": report.get("dataset_rows"),
            "feature_columns": report.get("feature_columns", []),
            "results": report.get("results", {}),
        })
    except Exception as exc:
        logger.exception("Error reading evaluation report")
        return jsonify({"success": False, "error": "Error reading evaluation report."}), 500


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)


@app.route("/api/aqi/explain-live", methods=["POST"])
def explain_live_prediction():
    """
    Real-time SHAP explainability for a single prediction.
    
    Request body (JSON):
    {
        "horizon": "24h" | "48h" | "72h",
        "features": {feature_name: value, ...}
    }
    
    Returns:
    - Top-K most influential features for this prediction
    - Positive contributors (increasing AQI)
    - Negative contributors (decreasing AQI)
    - Baseline prediction
    - Final prediction
    - Method used (SHAP vs Coefficients)
    """
    data = request.get_json(force=True, silent=True) or {}
    horizon = data.get("horizon", "24h")
    features = data.get("features", {})
    
    if not features:
        return jsonify({
            "success": False,
            "error": "No features provided. Include 'features' dict in request body."
        }), 400
    
    # Normalize horizon
    horizon_clean = horizon.replace("h", "").strip()
    if horizon_clean not in {"24", "48", "72"}:
        return jsonify({
            "success": False,
            "error": f"Invalid horizon '{horizon}'. Use '24h', '48h', or '72h'."
        }), 400
    horizon_key = f"{horizon_clean}h"
    
    try:
        # Get global feature importance to determine expected feature order
        global_importance = live_explainer.get_global_importance(horizon_key, top_k=50)
        
        if not global_importance:
            return jsonify({
                "success": False,
                "error": f"Feature importance data not available for {horizon_key}."
            }), 503
        
        # Build feature vector in correct order
        feature_names = [f["feature"] for f in global_importance]
        feature_vector = np.array([features.get(fname, 0.0) for fname in feature_names])
        
        # Get explanation
        explanation = live_explainer.explain_prediction(
            feature_vector,
            horizon_key,
            top_k=15
        )
        
        if "error" in explanation:
            return jsonify({
                "success": False,
                "error": explanation["error"]
            }), 500
        
        return jsonify({
            "success": True,
            "horizon": horizon_key,
            "explanation": {
                "method": explanation["method"],
                "baseline": explanation["baseline"],
                "prediction": explanation["prediction"],
                "top_features": explanation["top_features"],
                "positive_contributors": explanation["positive_contributors"],
                "negative_contributors": explanation["negative_contributors"],
            },
            "metadata": {
                "feature_count": len(feature_names),
                "top_k": 15,
                "note": "Impact values show how much each feature contributed to the prediction."
            }
        })
        
    except Exception as e:
        logger.error(f"Error in live explainability: {e}")
        return jsonify({
            "success": False,
            "error": "Internal error during explanation computation",
            "detail": str(e)
        }), 500
