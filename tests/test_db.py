"""
Unit tests for SQLite Database & Authentication Infrastructure (utils/db.py).
"""

import pytest
import os
from utils.db import (
    init_db,
    register_user,
    authenticate_user,
    create_session,
    validate_session,
    logout_session,
    log_prediction,
    get_prediction_history,
    get_user_preferences,
    update_user_preferences
)


def test_init_db():
    init_db()
    # Verify tables created without exception


def test_user_registration_and_authentication():
    test_email = "test_user_qa@pearls-aqi.org"
    test_pwd = "SecretPassword123!"  # Added special character for security validation

    # Register user
    success, msg, user_id = register_user(test_email, test_pwd, "QA Tester")
    assert success is True or "already exists" in msg

    # Authenticate valid user
    auth_success, auth_msg, user_data = authenticate_user(test_email, test_pwd)
    assert auth_success is True
    assert user_data["email"] == test_email

    # Authenticate invalid password
    bad_auth, bad_msg, _ = authenticate_user(test_email, "WrongPassword")
    assert bad_auth is False


def test_session_lifecycle():
    test_email = "session_user@pearls-aqi.org"
    _, _, user_id = register_user(test_email, "Password123!", "Session User")  # Added special character
    if not user_id:
        auth_success, _, user_data = authenticate_user(test_email, "Password123!")
        user_id = user_data["id"]

    token = create_session(user_id)
    assert token is not None

    sess = validate_session(token)
    assert sess is not None
    assert sess["email"] == test_email

    logout_success = logout_session(token)
    assert logout_success is True

    expired_sess = validate_session(token)
    assert expired_sess is None


def test_prediction_logging():
    logged = log_prediction("Lahore", 120, 115, 110)
    assert logged is True

    history = get_prediction_history(limit=5)
    assert len(history) > 0
    assert history[0]["city"] in ["Lahore", "Islamabad", "Faisalabad"]


def test_user_preferences():
    test_email = "pref_user@pearls-aqi.org"
    _, _, user_id = register_user(test_email, "Password123", "Pref User")
    if not user_id:
        _, _, user_data = authenticate_user(test_email, "Password123")
        user_id = user_data["id"]

    updated = update_user_preferences(user_id, ["Lahore", "Islamabad"], 180, test_email)
    assert updated is True

    prefs = get_user_preferences(user_id)
    assert "Islamabad" in prefs["favorite_cities"]
    assert prefs["alert_aqi_threshold"] == 180
