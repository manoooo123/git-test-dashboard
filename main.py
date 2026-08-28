"""
Pearls AQI Predictor - Unified Command Line Interface & Orchestrator.

Usage:
    python main.py --mode app         # Launch Streamlit Intelligence Dashboard
    python main.py --mode api         # Launch Flask REST API Backend
    python main.py --mode refresh     # Run Daily Feature Store Data Refresh
    python main.py --mode train       # Train Multi-City Machine Learning Models
    python main.py --mode evaluate    # Evaluate Model Metrics & Generate Reports
    python main.py --mode test        # Execute Automated Test Suite (Pytest)
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import logger


def run_app():
    """Launch Streamlit Dashboard."""
    logger.info("Launching Streamlit Air Quality Intelligence Dashboard...")
    cmd = [sys.executable, "-m", "streamlit", "run", str(PROJECT_ROOT / "streamlit_app.py")]
    subprocess.run(cmd, check=True)


def run_api(port: int = 5000, host: str = "0.0.0.0"):
    """Launch Flask REST API."""
    logger.info(f"Starting Flask REST API on {host}:{port}...")
    from app.flask_api import app
    app.run(host=host, port=port, debug=False)


def run_refresh():
    """Run feature store refresh."""
    logger.info("Executing daily feature store refresh pipeline...")
    cmd = [sys.executable, str(PROJECT_ROOT / "feature_pipeline" / "daily_live_refresh.py")]
    res = subprocess.run(cmd)
    if res.returncode == 0:
        logger.info("Feature store refresh completed successfully.")
    else:
        logger.error(f"Feature store refresh failed with return code {res.returncode}")


def run_train():
    """Run model training pipeline."""
    logger.info("Executing multi-city model training pipeline...")
    cmd = [sys.executable, str(PROJECT_ROOT / "training_pipeline" / "train_3cities.py")]
    res = subprocess.run(cmd)
    if res.returncode == 0:
        logger.info("Model training completed successfully.")
    else:
        logger.error(f"Model training failed with return code {res.returncode}")


def run_evaluate():
    """Run model evaluation pipeline."""
    logger.info("Executing model evaluation pipeline...")
    cmd = [sys.executable, str(PROJECT_ROOT / "training_pipeline" / "evaluate_3cities.py")]
    res = subprocess.run(cmd)
    if res.returncode == 0:
        logger.info("Model evaluation completed successfully.")
    else:
        logger.error(f"Model evaluation failed with return code {res.returncode}")


def run_test():
    """Run automated pytest suite."""
    logger.info("Executing automated test suite via Pytest...")
    cmd = [sys.executable, "-m", "pytest", str(PROJECT_ROOT / "tests"), "-v"]
    res = subprocess.run(cmd)
    if res.returncode == 0:
        logger.info("All automated tests passed successfully.")
    else:
        logger.error("Some automated tests failed.")


def main():
    parser = argparse.ArgumentParser(
        description="Pearls AQI Predictor - Enterprise Climate Intelligence CLI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["app", "api", "refresh", "train", "evaluate", "test"],
        default="app",
        help="""Execution mode:
  app       : Launch Streamlit Dashboard
  api       : Launch Flask REST API
  refresh   : Run Feature Pipeline Live Refresh
  train     : Train ML Forecasting Models
  evaluate  : Evaluate Trained Models & Reports
  test      : Run Automated Pytest Suite
""",
    )
    parser.add_argument("--port", type=int, default=5000, help="Port for API backend (default: 5000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for API backend (default: 0.0.0.0)")

    args = parser.parse_args()

    if args.mode == "app":
        run_app()
    elif args.mode == "api":
        run_api(port=args.port, host=args.host)
    elif args.mode == "refresh":
        run_refresh()
    elif args.mode == "train":
        run_train()
    elif args.mode == "evaluate":
        run_evaluate()
    elif args.mode == "test":
        run_test()


if __name__ == "__main__":
    main()
