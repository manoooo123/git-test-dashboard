"""
WSGI Entry Point for Pearls AQI Predictor Flask API

Production deployment using Gunicorn or Waitress:
    gunicorn --bind 0.0.0.0:5000 --workers 4 wsgi:app
    waitress-serve --host=0.0.0.0 --port=5000 wsgi:app
"""

from app.flask_api import app

if __name__ == "__main__":
    app.run()
