"""
Application Configuration Module.

This file manages:
- Project paths
- Environment variables
- API configuration
- AQI system constants
"""

import os
from pathlib import Path
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


# ==============================
# Project Directory Configuration
# ==============================

BASE_DIR = Path(__file__).resolve().parent.parent


DATA_DIR = BASE_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"


MODEL_DIR = BASE_DIR / "models"


LOG_DIR = BASE_DIR / "logs"


# ==============================
# API Configuration
# ==============================

OPENWEATHER_API_KEY = os.getenv(
    "OPENWEATHER_API_KEY"
)


OPENWEATHER_BASE_URL = (
    "https://api.openweathermap.org/data/2.5/air_pollution"
)


# ==============================
# AQI Forecast Configuration
# ==============================

FORECAST_DAYS = 3


# ==============================
# Supported Cities Configuration
# ==============================

CITIES = {
    "Lahore": {
        "lat": 31.5204,
        "lon": 74.3587,
        "country": "Pakistan",
        "tagline": "Provincial Capital & Cultural Hub"
    },
    "Islamabad": {
        "lat": 33.6844,
        "lon": 73.0479,
        "country": "Pakistan",
        "tagline": "Federal Capital & Margalla Foothills"
    },
    "Faisalabad": {
        "lat": 31.4504,
        "lon": 73.1350,
        "country": "Pakistan",
        "tagline": "Industrial Center & Textile Capital"
    }
}

WEATHER_BASE_URL = (
    "https://api.openweathermap.org/data/2.5/weather"
)

# ==============================
# AQI Thresholds & Categories
# ==============================

AQI_CATEGORIES = [
    {"min": 0, "max": 50, "name": "Good", "color": "#10B981", "bg": "rgba(16, 185, 129, 0.12)", "health": "Air quality is satisfactory and poses little or no risk."},
    {"min": 51, "max": 100, "name": "Moderate", "color": "#F59E0B", "bg": "rgba(245, 158, 11, 0.12)", "health": "Air quality is acceptable; sensitive individuals may experience mild effects."},
    {"min": 101, "max": 150, "name": "Unhealthy for Sensitive Groups", "color": "#F97316", "bg": "rgba(249, 115, 22, 0.12)", "health": "Members of sensitive groups may experience health effects. General public less likely affected."},
    {"min": 151, "max": 200, "name": "Unhealthy", "color": "#EF4444", "bg": "rgba(239, 68, 68, 0.12)", "health": "Everyone may begin to experience health effects; sensitive groups may experience serious effects."},
    {"min": 201, "max": 300, "name": "Very Unhealthy", "color": "#A855F7", "bg": "rgba(168, 85, 247, 0.12)", "health": "Health alert: everyone may experience more serious health effects."},
    {"min": 301, "max": 999, "name": "Hazardous", "color": "#7E22CE", "bg": "rgba(126, 34, 206, 0.12)", "health": "Health emergency conditions. The entire population is likely to be affected."}
]