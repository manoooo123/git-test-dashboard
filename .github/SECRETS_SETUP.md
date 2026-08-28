# GitHub Actions Secrets Configuration

This document explains the required secrets for the Pearls AQI Predictor CI/CD pipeline.

## Required Secrets

The GitHub Actions workflow (`.github/workflows/aqi_pipeline.yml`) requires the following secrets to be configured in your repository settings:

### 1. `OPENAQ_API_KEY` (Required)

**Purpose**: Access OpenAQ v3 API for real-time air quality measurements

**How to obtain**:
1. Visit [OpenAQ](https://openaq.org/)
2. Create a free account
3. Navigate to your profile settings
4. Generate an API key

**Used in**:
- Hourly feature pipeline (data collection from OpenAQ sensors)
- Daily model retraining (historical data retrieval)

---

### 2. `HOPSWORKS_API_KEY` (Optional but Recommended)

**Purpose**: Connect to Hopsworks Feature Store for cloud-based feature management

**How to obtain**:
1. Sign up at [Hopsworks](https://www.hopsworks.ai/)
2. Create a new project named `pearls_aqi_predictor` (or customize in code)
3. Go to Settings → API Keys → Generate new API key
4. Copy the API key value

**Fallback behavior**:
- If not configured, the system uses local CSV-based feature store (`data/processed/model_features_3cities.csv`)
- All functionality remains operational

**Used in**:
- Feature pipeline (saving features to cloud)
- Model training (loading features from cloud)
- Dashboard (real-time feature store status)

---

### 3. `FLASK_SECRET_KEY` (Required)

**Purpose**: Secure Flask session management and JWT token signing

**How to generate**:
```python
import secrets
print(secrets.token_urlsafe(32))
```

Or use this one-liner in terminal:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

**Security notes**:
- Must be at least 32 characters
- Should be cryptographically random
- Never commit to version control
- Rotate periodically for production systems

**Used in**:
- Flask API authentication
- Session token generation
- User password hashing verification

---

### 4. `OPENWEATHER_API_KEY` (Legacy - Not Currently Used)

**Status**: Optional (kept for backward compatibility)

**Purpose**: Originally used for weather data, now replaced by Open-Meteo API (no key required)

**Note**: The data collection pipeline now uses Open-Meteo's free API, which doesn't require authentication. You can safely leave this secret blank.

---

## How to Configure Secrets

### GitHub Repository Secrets

1. Navigate to your repository on GitHub
2. Go to **Settings** → **Secrets and variables** → **Actions**
3. Click **"New repository secret"**
4. Add each secret with the exact name shown above
5. Save

### Local Development

For local development, create a `.env` file in the project root:

```env
OPENAQ_API_KEY=your_openaq_key_here
HOPSWORKS_API_KEY=your_hopsworks_key_here
FLASK_SECRET_KEY=your_random_secret_key_here
OPENWEATHER_API_KEY=optional_legacy_key
```

**Important**: The `.env` file is gitignored by default - never commit it!

---

## Verification

### Test if secrets are properly configured:

1. **Trigger GitHub Actions Workflow**:
   - Go to **Actions** tab in GitHub
   - Select "Pearls AQI Predictor — CI/CD Pipeline"
   - Click "Run workflow"

2. **Check workflow logs**:
   - Green checkmarks = all secrets configured correctly
   - Red X = missing or invalid secrets (check error messages)

3. **Local verification**:
   ```bash
   python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('OpenAQ:', 'OK' if os.getenv('OPENAQ_API_KEY') else 'MISSING'); print('Flask:', 'OK' if os.getenv('FLASK_SECRET_KEY') else 'MISSING')"
   ```

---

## Workflow Schedule

The CI/CD pipeline runs automatically:

- **Hourly** (`0 * * * *`): Feature refresh (fetch live data → engineer features → store)
- **Daily** (`0 1 * * *`): Model retraining + SHAP feature importance regeneration
- **On Push**: CI validation (linting + test suite) on `main`/`master` branches
- **Manual**: Can be triggered via GitHub Actions UI ("workflow_dispatch")

---

## Troubleshooting

### "OpenAQ API authentication failed"
- Verify your API key is correct
- Check if you've exceeded rate limits (free tier: 2,000 requests/day)
- Ensure the key hasn't expired

### "Hopsworks connection failed"
- Check API key validity
- Verify project name matches (`pearls_aqi_predictor` by default)
- System will automatically fall back to local feature store

### "Flask secret key too short"
- Generate a new key with at least 32 characters
- Use the Python snippet provided above

### Workflow fails with "permission denied"
- Ensure workflow has `contents: write` permissions (already configured)
- Check if branch protection rules allow Actions to push

---

## Security Best Practices

✅ **DO**:
- Rotate secrets periodically (every 90 days recommended)
- Use GitHub's secret masking (secrets are automatically hidden in logs)
- Generate cryptographically random keys
- Limit API key permissions to minimum required

❌ **DON'T**:
- Commit secrets to git
- Share secrets via email/Slack
- Reuse secrets across projects
- Log secret values in application code

---

## Need Help?

If you encounter issues with secrets configuration:
1. Check the workflow run logs in GitHub Actions
2. Verify `.env` file exists locally with correct values
3. Test API keys manually using the verification commands above
4. Ensure all secrets are spelled exactly as shown (case-sensitive)

---

**Last Updated**: 2025-01-XX  
**Maintained by**: Pearls AQI Predictor Team
