"""
Pearls AQI Predictor — Streamlit App Entry Point (app/ subdirectory launcher).

This module exists only to let Streamlit be invoked from the app/ subdirectory:
    streamlit run app/streamlit_app.py

It adds the project root to sys.path and then imports the root streamlit_app
module properly — avoiding the fragile exec() pattern.

Usage:
    # From project root (preferred)
    streamlit run streamlit_app.py

    # From app/ subdirectory
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so all imports resolve correctly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Change working directory to project root so all relative paths resolve correctly
import os
os.chdir(PROJECT_ROOT)

# Import and run the root application module
# We use runpy to execute it as __main__ so Streamlit picks up set_page_config correctly
import runpy
runpy.run_path(str(PROJECT_ROOT / "streamlit_app.py"), run_name="__main__")
