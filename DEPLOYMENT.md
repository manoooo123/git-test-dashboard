# Pearls AQI Predictor - Deployment Guide

## Prerequisites

- Python 3.13+
- 4GB+ RAM recommended
- Trained models in `models/` directory
- Feature store data in `data/processed/`

## Environment Configuration

Create `.env` file (use `.env.example` as template):

```bash
# Database
DATABASE_PATH=data/pearls_aqi.db

# Flask API
FLASK_SECRET_KEY=your-production-secret-key-here
FLASK_ENV=production

# Model Configuration
MODEL_DIR=models/3cities

# Feature Store
FEATURE_STORE_PATH=data/processed/model_features_3cities.csv

# API Configuration
API_BASE_URL=http://localhost:5000
```

## Installation

```bash
# Clone repository
git clone https://github.com/your-username/Pearls-AQI-Predictor.git
cd Pearls-AQI-Predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install SHAP for explainability
pip install shap==0.46.0
```

## Running in Production

### Option 1: Gunicorn (Linux/Mac)

```bash
# Start Flask API
gunicorn --bind 0.0.0.0:5000 --workers 4 --timeout 120 wsgi:app

# Start Streamlit Dashboard (separate terminal)
streamlit run streamlit_app.py --server.port 8502
```

### Option 2: Waitress (Windows)

```bash
# Start Flask API
waitress-serve --host=0.0.0.0 --port=5000 --threads=4 wsgi:app

# Start Streamlit Dashboard (separate terminal)
streamlit run streamlit_app.py --server.port 8502
```

### Option 3: Docker (Recommended for Production)

```bash
# Build image
docker build -t pearls-aqi-predictor .

# Run container
docker run -d -p 5000:5000 -p 8502:8502 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  --env-file .env \
  pearls-aqi-predictor
```

## Health Checks

### Flask API

```bash
curl http://localhost:5000/api/status
```

Expected response:
```json
{
  "status": "operational",
  "database": "connected",
  "feature_store": "available",
  "models_loaded": 3
}
```

### Streamlit Dashboard

Open browser: `http://localhost:8502`

## Production Checklist

- [ ] `.env` file configured with production secrets
- [ ] `FLASK_SECRET_KEY` is strong and unique
- [ ] Database file has correct permissions
- [ ] Models directory is accessible
- [ ] Feature store CSV is present
- [ ] Firewall allows ports 5000 (API) and 8502 (Dashboard)
- [ ] HTTPS/TLS configured (use nginx/Apache reverse proxy)
- [ ] Monitoring and logging configured
- [ ] Backup strategy for database and predictions
- [ ] Resource limits configured (memory, CPU)

## Security Recommendations

1. **Use HTTPS**: Deploy behind nginx/Apache with SSL certificates
2. **Secure Secrets**: Use environment variables, never commit `.env`
3. **Database Backups**: Regular backups of `data/pearls_aqi.db`
4. **Rate Limiting**: Configure API rate limits in production
5. **Authentication**: Ensure strong password policies
6. **CORS**: Configure appropriate CORS settings for your domain

## Monitoring

### Logs

```bash
# Flask API logs
tail -f logs/flask_api.log

# Streamlit logs
tail -f logs/streamlit.log
```

### Resource Usage

```bash
# Check memory/CPU
htop

# Check disk usage
df -h
```

## Troubleshooting

### Issue: Models not loading

**Solution**: Verify `MODEL_DIR` path in `.env` and ensure `.joblib` files exist

### Issue: Feature store unavailable

**Solution**: Check `FEATURE_STORE_PATH` and verify CSV file exists with correct columns

### Issue: Database locked

**Solution**: Ensure only one process accesses SQLite database, or migrate to PostgreSQL

### Issue: Forecast returns NaN

**Solution**: Verify feature store has recent data and models are trained properly

## Scaling Considerations

### Horizontal Scaling

- Use PostgreSQL instead of SQLite for multi-process access
- Deploy API behind load balancer (nginx, HAProxy)
- Use Redis for session management

### Performance Optimization

- Enable model caching
- Use connection pooling for database
- Implement CDN for static assets
- Configure Streamlit for production mode

## Maintenance

### Update Feature Store

```bash
python feature_pipeline/feature_engineering_3cities.py
```

### Retrain Models

```bash
python training_pipeline/train_3cities.py
```

### Run Tests

```bash
pytest tests/ -v
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/your-username/Pearls-AQI-Predictor/issues
- Documentation: https://github.com/your-username/Pearls-AQI-Predictor/wiki
