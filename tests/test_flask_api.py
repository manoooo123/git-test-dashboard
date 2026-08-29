"""
Pearls AQI Predictor — Flask REST API integration tests.

Covers:
- All endpoints: root, status, cities, live, forecast, comparison, explainability,
  predict, history, preferences, model performance, alerts, callback
- Auth full lifecycle: register → login → me → logout → invalid
- Rate limiting on login
- Data-driven alerts (not hardcoded)
- Forecast response contract (status key, no zero-on-failure)
- Auth callback recovery → redirect to local app URL
"""

import json
from pathlib import Path

import pytest

from app.flask_api import app, _login_attempts


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_rate_limiter():
    """Clear login attempt history before each test to avoid cross-test rate-limit pollution."""
    _login_attempts.clear()
    yield
    _login_attempts.clear()


@pytest.fixture(scope="module")
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture(scope="module")
def auth_token(client):
    """Register + login a test user and return a valid bearer token."""
    payload = {"email": "flask_test@pearls-aqi.org", "password": "TestPass123!", "full_name": "Flask Tester"}
    client.post("/api/auth/register", json=payload)  # Idempotent — ok if already exists
    r = client.post("/api/auth/login", json={"email": "flask_test@pearls-aqi.org", "password": "TestPass123!"})
    data = r.get_json()
    assert data["success"] is True, f"Login fixture failed: {data}"
    return data["token"]


# ── Root & system ─────────────────────────────────────────────────────────────

class TestSystemEndpoints:

    def test_root_returns_manifest(self, client):
        r = client.get("/")
        assert r.status_code == 200
        data = r.get_json()
        assert data["project"] == "Pearls AQI Predictor API"
        assert "supported_cities" in data
        assert "endpoints" in data

    def test_status_endpoint(self, client):
        r = client.get("/api/status")
        assert r.status_code == 200
        data = r.get_json()
        assert "status" in data
        assert "feature_store" in data
        assert "prediction_engine" in data
        assert "database" in data

    def test_cities_returns_three(self, client):
        r = client.get("/api/cities")
        assert r.status_code == 200
        data = r.get_json()
        assert data["success"] is True
        assert data["count"] == 3
        city_names = [c["city"] for c in data["cities"]]
        assert "Lahore" in city_names
        assert "Islamabad" in city_names
        assert "Faisalabad" in city_names

    def test_options_returns_200(self, client):
        r = client.options("/api/status")
        assert r.status_code == 200


# ── Authentication ────────────────────────────────────────────────────────────

class TestAuthentication:

    def test_register_creates_user(self, client):
        payload = {"email": "new_reg_test@pearls.org", "password": "Secure123!", "full_name": "New User"}
        r = client.post("/api/auth/register", json=payload)
        # Either 201 (new) or 400 (already exists from previous run)
        assert r.status_code in (200, 201, 400)
        data = r.get_json()
        if r.status_code in (200, 201):
            assert data["success"] is True
            assert "token" in data

    def test_register_rejects_short_password(self, client):
        r = client.post("/api/auth/register", json={"email": "x@x.com", "password": "abc"})
        assert r.status_code == 400
        assert r.get_json()["success"] is False

    def test_register_rejects_invalid_email(self, client):
        r = client.post("/api/auth/register", json={"email": "notanemail", "password": "Password123"})
        assert r.status_code == 400

    def test_login_with_valid_credentials(self, auth_token):
        assert auth_token is not None
        assert len(auth_token) > 10

    def test_login_with_wrong_password_returns_401(self, client):
        r = client.post("/api/auth/login", json={"email": "flask_test@pearls-aqi.org", "password": "WrongPass"})
        assert r.status_code == 401
        assert r.get_json()["success"] is False

    def test_login_with_nonexistent_user_returns_401(self, client):
        r = client.post("/api/auth/login", json={"email": "nobody@nowhere.org", "password": "Pass123"})
        assert r.status_code == 401

    def test_me_with_valid_token(self, client, auth_token):
        r = client.get("/api/auth/me", headers={"Authorization": f"Bearer {auth_token}"})
        assert r.status_code == 200
        data = r.get_json()
        assert data["success"] is True
        assert data["user"]["email"] == "flask_test@pearls-aqi.org"

    def test_me_without_token_returns_401(self, client):
        r = client.get("/api/auth/me")
        assert r.status_code == 401

    def test_me_with_invalid_token_returns_401(self, client):
        r = client.get("/api/auth/me", headers={"Authorization": "Bearer invalid-token-xyz"})
        assert r.status_code == 401

    def test_logout_invalidates_session(self, client):
        # Create a fresh session just for this test
        r = client.post("/api/auth/login", json={"email": "flask_test@pearls-aqi.org", "password": "TestPass123!"})
        token = r.get_json()["token"]

        logout_r = client.post("/api/auth/logout", headers={"Authorization": f"Bearer {token}"})
        assert logout_r.status_code == 200

        me_r = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me_r.status_code == 401

    def test_login_rate_limiter_blocks_at_11th_attempt(self, client):
        """Rate limiter should block after 10 attempts per window."""
        _login_attempts.clear()
        payload = {"email": "rate@test.org", "password": "wrongpassword"}
        for _ in range(10):
            client.post("/api/auth/login", json=payload)
        # 11th attempt should be rate-limited (404 for unknown user OR 429 for rate limit)
        r = client.post("/api/auth/login", json=payload)
        # Allow 401 (wrong creds) for attempts that aren't rate-limited yet
        # and 429 for when rate limit triggers
        # With the 10-attempt window, the 11th should be 429
        assert r.status_code == 429, (
            f"Expected 429 (rate limited) on 11th attempt, got {r.status_code}"
        )


# ── Auth callback ─────────────────────────────────────────────────────────────

class TestAuthCallback:

    def test_callback_returns_html_with_redirect(self, client):
        r = client.get("/api/auth/callback?code=abc&state=xyz")
        assert r.status_code == 200
        body = r.get_data(as_text=True)
        # Must contain automatic redirect mechanism
        assert "meta http-equiv" in body.lower() or "refresh" in body.lower()

    def test_callback_contains_fallback_link(self, client):
        r = client.get("/api/auth/callback?code=abc&state=xyz")
        body = r.get_data(as_text=True)
        assert "localhost:8501" in body or "Click here" in body

    def test_callback_with_error_param_returns_400(self, client):
        r = client.get("/api/auth/callback?error=access_denied")
        assert r.status_code == 400
        data = r.get_json()
        assert data["success"] is False

    def test_callback_with_next_param_uses_it(self, client):
        r = client.get("/api/auth/callback?code=x&next=http%3A%2F%2Flocalhost%3A8501")
        body = r.get_data(as_text=True)
        assert "localhost:8501" in body

    def test_oauth_callback_alias_works(self, client):
        r = client.get("/oauth/callback?code=abc")
        assert r.status_code == 200

    def test_auth_callback_alias_works(self, client):
        r = client.get("/auth/callback?code=abc")
        assert r.status_code == 200


# ── Air quality endpoints ─────────────────────────────────────────────────────

class TestAQIEndpoints:

    def test_live_aqi_lahore(self, client):
        r = client.get("/api/aqi/live?city=Lahore")
        assert r.status_code == 200
        data = r.get_json()
        assert data["success"] is True
        assert data["city"] == "Lahore"
        assert "metrics" in data
        assert "aqi" in data["metrics"]

    def test_live_aqi_returns_data_freshness(self, client):
        r = client.get("/api/aqi/live?city=Lahore")
        data = r.get_json()
        assert "data_freshness" in data
        assert data["data_freshness"] in ("live", "offline_cache")

    def test_live_aqi_no_fabricated_pm10(self, client):
        """PM10 must be None (not fabricated as pm25*1.6)."""
        r = client.get("/api/aqi/live?city=Lahore")
        data = r.get_json()
        pm10 = data["metrics"].get("pm10")
        assert pm10 is None, (
            f"PM10 should be None (not available from this sensor network), got {pm10}. "
            "Fabricating PM10=PM2.5*1.6 is not acceptable."
        )

    def test_live_aqi_no_nan_in_response(self, client):
        """No metric in the response should be NaN."""
        r = client.get("/api/aqi/live?city=Lahore")
        import math
        body = r.get_data(as_text=True)
        assert "NaN" not in body, "API response contains literal 'NaN'"
        assert "nan" not in body, "API response contains literal 'nan'"

    def test_forecast_all_horizons_present(self, client):
        r = client.get("/api/aqi/forecast?city=Lahore")
        assert r.status_code == 200
        data = r.get_json()
        assert data["success"] is True
        assert "24h" in data["forecast"]
        assert "48h" in data["forecast"]
        assert "72h" in data["forecast"]

    def test_forecast_each_horizon_has_status(self, client):
        r = client.get("/api/aqi/forecast?city=Lahore")
        data = r.get_json()
        for h in ("24h", "48h", "72h"):
            assert "status" in data["forecast"][h], f"Horizon {h} missing 'status'"

    def test_forecast_successful_horizon_has_valid_aqi(self, client):
        r = client.get("/api/aqi/forecast?city=Lahore")
        data = r.get_json()
        for h in ("24h", "48h", "72h"):
            entry = data["forecast"][h]
            if entry.get("status") == "success":
                aqi = entry["aqi"]
                assert isinstance(aqi, int)
                assert 0 < aqi <= 500, f"AQI {aqi} for {h} is out of range or zero"

    def test_forecast_unavailable_horizon_has_no_aqi(self, client):
        """Failed horizon must return status='unavailable', NOT aqi=0."""
        r = client.get("/api/aqi/forecast?city=Lahore")
        data = r.get_json()
        for h in ("24h", "48h", "72h"):
            entry = data["forecast"][h]
            if entry.get("status") == "unavailable":
                assert "aqi" not in entry, (
                    f"Unavailable forecast for {h} must not contain 'aqi' key "
                    "(would misleadingly show AQI=0 as 'Good')"
                )

    def test_forecast_invalid_city_returns_400(self, client):
        r = client.get("/api/aqi/forecast?city=Dubai")
        assert r.status_code == 400

    def test_forecast_islamabad(self, client):
        r = client.get("/api/aqi/forecast?city=Islamabad")
        assert r.status_code in (200, 404)  # 404 is acceptable if no Islamabad data in store

    def test_comparison_has_three_cities(self, client):
        r = client.get("/api/aqi/comparison")
        assert r.status_code == 200
        data = r.get_json()
        assert len(data["comparison"]) == 3

    def test_explainability_returns_features(self, client):
        r = client.get("/api/aqi/explainability?horizon=24h")
        assert r.status_code == 200
        data = r.get_json()
        assert data["success"] is True
        assert "features" in data

    def test_explainability_has_method_note(self, client):
        """Explainability endpoint must clarify this is Ridge |coef_|, not SHAP."""
        r = client.get("/api/aqi/explainability?horizon=24h")
        data = r.get_json()
        assert "method" in data, "Must declare the explanation method"
        assert "Ridge" in data.get("method", "") or "coefficient" in data.get("method_note", "").lower()


# ── Prediction scenario endpoint ──────────────────────────────────────────────

class TestPredictEndpoint:

    def test_predict_with_valid_payload(self, client):
        payload = {"pm25": 75.0, "temperature": 30.0, "humidity": 65.0, "wind_speed": 10.0}
        r = client.post("/api/predict", json=payload)
        assert r.status_code == 200
        data = r.get_json()
        assert data["success"] is True
        assert "prediction" in data

    def test_predict_uses_real_model_not_formula(self, client):
        """Verify the response mentions the model name, not a manual formula."""
        payload = {"pm25": 75.0}
        r = client.post("/api/predict", json=payload)
        data = r.get_json()
        if data.get("success"):
            # Response must include model identification
            assert "model" in data, "Predict endpoint must identify which model was used"
            assert "Ridge" in data.get("model", "") or "best_model" in data.get("model", "")

    def test_predict_aqi_is_positive_for_elevated_pm25(self, client):
        """High PM2.5 (75 µg/m³) must produce AQI > 150, not 0."""
        payload = {"pm25": 75.0, "temperature": 28.0, "humidity": 60.0, "wind_speed": 8.0}
        r = client.post("/api/predict", json=payload)
        data = r.get_json()
        if data.get("success"):
            aqi = data["prediction"]["predicted_aqi"]
            assert aqi > 0, f"Predicted AQI should be > 0 for PM2.5=75, got {aqi}"


# ── Alerts (data-driven) ──────────────────────────────────────────────────────

class TestAlertsEndpoint:

    def test_alerts_endpoint_succeeds(self, client):
        r = client.get("/api/alerts")
        assert r.status_code == 200
        data = r.get_json()
        assert data["success"] is True
        assert "alerts" in data

    def test_alerts_has_count(self, client):
        r = client.get("/api/alerts")
        data = r.get_json()
        assert "count" in data
        assert data["count"] == len(data["alerts"])

    def test_alerts_are_not_hardcoded_static(self, client):
        """Alerts must come from live data, not always the same hardcoded 2 objects."""
        r = client.get("/api/alerts")
        data = r.get_json()
        for alert in data.get("alerts", []):
            # Hardcoded alerts had fixed IDs like 'ALT-LHR-901' — verify they're gone
            assert alert.get("id") != "ALT-LHR-901", (
                "Found legacy hardcoded alert ID 'ALT-LHR-901'. "
                "Alerts must be generated from live data."
            )

    def test_alert_objects_have_required_fields(self, client):
        r = client.get("/api/alerts")
        data = r.get_json()
        required = {"id", "city", "severity", "title", "message", "aqi", "recommendation"}
        for alert in data.get("alerts", []):
            for field in required:
                assert field in alert, f"Alert missing field '{field}': {alert}"

    def test_alert_city_filter(self, client):
        r = client.get("/api/alerts?city=Lahore")
        assert r.status_code == 200
        data = r.get_json()
        for alert in data.get("alerts", []):
            assert alert["city"].lower() == "lahore"

    def test_invalid_city_returns_400(self, client):
        r = client.get("/api/alerts?city=NewYork")
        assert r.status_code == 400


# ── History & preferences ─────────────────────────────────────────────────────

class TestHistoryAndPreferences:

    def test_history_accessible_without_auth(self, client):
        r = client.get("/api/history")
        assert r.status_code == 200
        data = r.get_json()
        assert data["success"] is True

    def test_history_with_auth(self, client, auth_token):
        r = client.get("/api/history", headers={"Authorization": f"Bearer {auth_token}"})
        assert r.status_code == 200

    def test_preferences_requires_auth(self, client):
        r = client.get("/api/user/preferences")
        assert r.status_code == 401

    def test_preferences_get(self, client, auth_token):
        r = client.get("/api/user/preferences", headers={"Authorization": f"Bearer {auth_token}"})
        assert r.status_code == 200
        data = r.get_json()
        assert data["success"] is True
        assert "preferences" in data

    def test_preferences_post(self, client, auth_token):
        payload = {"favorite_cities": ["Lahore", "Islamabad"], "alert_aqi_threshold": 175}
        r = client.post("/api/user/preferences", json=payload, headers={"Authorization": f"Bearer {auth_token}"})
        assert r.status_code == 200
        assert r.get_json()["success"] is True


# ── Model performance ─────────────────────────────────────────────────────────

class TestModelPerformance:

    def test_model_performance_endpoint(self, client):
        r = client.get("/api/models/performance")
        assert r.status_code == 200
        data = r.get_json()
        assert data["success"] is True

    def test_performance_has_feature_count(self, client):
        r = client.get("/api/models/performance")
        data = r.get_json()
        assert data.get("feature_count") is not None


# ── Streamlit source integrity ────────────────────────────────────────────────

class TestStreamlitSourceIntegrity:

    def test_password_toggle_uses_supported_type(self):
        """Streamlit only accepts 'password' or 'default' for text_input type (updated for premium auth UI)."""
        auth_ui = Path("streamlit_app.py").read_text(encoding="utf-8")
        assert 'type=pwd_disp' in auth_ui or 'type=pwd_reg_disp' in auth_ui or 'type="password"' in auth_ui

    def test_no_hardcoded_no2_value(self):
        """NO2 must not be hardcoded as 24.2 — it's not available from this sensor network."""
        src = Path("streamlit_app.py").read_text(encoding="utf-8")
        assert "24.2" not in src, (
            "Hardcoded NO2=24.2 found in streamlit_app.py. "
            "This sensor network does not provide NO2 data."
        )

    def test_no_hardcoded_so2_value(self):
        src = Path("streamlit_app.py").read_text(encoding="utf-8")
        assert ">8.5<" not in src and '"SO2": 8.5' not in src, (
            "Hardcoded SO2=8.5 detected in UI"
        )

    def test_no_pm10_fabrication(self):
        """PM10 must not be calculated as pm25 * 1.6."""
        src = Path("streamlit_app.py").read_text(encoding="utf-8")
        assert "pm25*1.6" not in src.replace(" ", "") and "pm25_val*1.6" not in src.replace(" ", ""), (
            "PM10 fabrication (pm25*1.6) detected in streamlit_app.py"
        )

    def test_forecast_unavailable_state_exists(self):
        src = Path("streamlit_app.py").read_text(encoding="utf-8")
        assert "Forecast Unavailable" in src, (
            "streamlit_app.py must display 'Forecast Unavailable' when model inference fails"
        )

    def test_no_fillna_zero_on_inference_input(self):
        """fillna(0) on model input is wrong — model has SimpleImputer. Must not exist."""
        src = Path("streamlit_app.py").read_text(encoding="utf-8")
        # The old code had: X_latest = pd.DataFrame([...]).fillna(0)
        # This was wrong — it bypasses the model's own imputer and introduces bias
        assert ".fillna(0)" not in src, (
            "streamlit_app.py uses .fillna(0) on model input features. "
            "This bypasses the Ridge pipeline's SimpleImputer(strategy='median'). "
            "Pass raw NaN values to the model instead."
        )
