"""
Pearls AQI Predictor — Flask API launcher.

Run from the project root:
    python run_api.py

This ensures sys.path includes the project root so `utils`, `app`, etc. all resolve.
"""
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from app.flask_api import app

if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
