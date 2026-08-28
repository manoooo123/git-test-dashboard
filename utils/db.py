"""
Pearls AQI Predictor - SQLite Database & Authentication Infrastructure.

Provides thread-safe operations for:
- Database schema initialization (users, sessions, prediction_history, user_preferences)
- Secure password hashing (PBKDF2-HMAC-SHA256 with per-user salt)
- User registration, authentication, and session token validation
- Prediction history logging and retrieval
- User preferences management (favorite cities, alert thresholds)

Security Features:
- Strong password hashing with salt
- Session expiration and cleanup
- Input validation and sanitization
- Audit logging for security events
"""

import os
import json
import sqlite3
import hashlib
import uuid
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from utils.security import (
    PasswordValidator,
    InputSanitizer,
    SessionCleaner,
    AuditLogger
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "pearls_aqi.db"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("pearls_db")


def get_db_connection() -> sqlite3.Connection:
    """Create a thread-safe connection to the SQLite database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize database tables and schema."""
    conn = get_db_connection()
    try:
        with conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    full_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    city TEXT NOT NULL,
                    predicted_aqi_24h INTEGER,
                    predicted_aqi_48h INTEGER,
                    predicted_aqi_72h INTEGER,
                    model_version TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id INTEGER PRIMARY KEY,
                    favorite_cities TEXT DEFAULT '["Lahore"]',
                    alert_aqi_threshold INTEGER DEFAULT 150,
                    alert_email TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );
            """)
        logger.info(f"Database schema initialized successfully at {DB_PATH}")
    finally:
        conn.close()


def _hash_password(password: str, salt: bytes = None) -> Tuple[str, str]:
    """Generate PBKDF2 hash and salt for a password."""
    if salt is None:
        salt = os.urandom(16)
    hash_bytes = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return hash_bytes.hex(), salt.hex()


def register_user(email: str, password: str, full_name: str = "") -> Tuple[bool, str, Optional[int]]:
    """
    Register a new user account with enhanced security validation.
    Returns: (success, message, user_id)
    """
    # Validate and sanitize email
    valid_email, sanitized_email, email_error = InputSanitizer.sanitize_email(email)
    if not valid_email:
        AuditLogger.log_security_event("REGISTRATION_FAILED", f"Invalid email: {email}", "WARNING")
        return False, email_error, None
    
    # Validate password strength
    valid_password, password_error = PasswordValidator.validate(password, min_length=8)
    if not valid_password:
        AuditLogger.log_security_event("REGISTRATION_FAILED", f"Weak password for {sanitized_email}", "WARNING")
        return False, password_error, None
    
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE email = ?", (sanitized_email,))
        if cur.fetchone():
            AuditLogger.log_security_event("REGISTRATION_FAILED", f"Duplicate email: {sanitized_email}", "INFO")
            return False, "An account with this email address already exists.", None

        pwd_hash, salt_hex = _hash_password(password)
        cur.execute(
            "INSERT INTO users (email, password_hash, salt, full_name) VALUES (?, ?, ?, ?)",
            (sanitized_email, pwd_hash, salt_hex, full_name.strip())
        )
        user_id = cur.lastrowid
        cur.execute(
            "INSERT INTO user_preferences (user_id, favorite_cities, alert_email) VALUES (?, ?, ?)",
            (user_id, json.dumps(["Lahore"]), sanitized_email)
        )
        conn.commit()
        logger.info(f"Registered user: {sanitized_email} (ID: {user_id})")
        AuditLogger.log_security_event("REGISTRATION_SUCCESS", f"User {sanitized_email} registered", "INFO")
        return True, "User registered successfully.", user_id
    except Exception as e:
        conn.rollback()
        logger.error(f"Error registering user {sanitized_email}: {e}")
        AuditLogger.log_security_event("REGISTRATION_ERROR", f"Database error for {sanitized_email}: {e}", "WARNING")
        return False, f"Database error during registration: {str(e)}", None
    finally:
        conn.close()


def authenticate_user(email: str, password: str, ip_address: str = "unknown") -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Authenticate user credentials with enhanced security logging.
    Returns: (success, message, user_dict)
    """
    # Validate email format
    valid_email, sanitized_email, email_error = InputSanitizer.sanitize_email(email)
    if not valid_email:
        AuditLogger.log_auth_attempt(email, False, ip_address, "Invalid email format")
        return False, "Invalid email or password.", None
    
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, email, password_hash, salt, full_name, created_at FROM users WHERE email = ?", (sanitized_email,))
        row = cur.fetchone()
        if not row:
            AuditLogger.log_auth_attempt(sanitized_email, False, ip_address, "User not found")
            return False, "Invalid email or password.", None

        salt_bytes = bytes.fromhex(row["salt"])
        computed_hash, _ = _hash_password(password, salt_bytes)
        if computed_hash != row["password_hash"]:
            AuditLogger.log_auth_attempt(sanitized_email, False, ip_address, "Incorrect password")
            return False, "Invalid email or password.", None

        user_data = {
            "id": row["id"],
            "email": row["email"],
            "full_name": row["full_name"],
            "created_at": row["created_at"]
        }
        AuditLogger.log_auth_attempt(sanitized_email, True, ip_address, "Success")
        return True, "Authentication successful.", user_data
    except Exception as e:
        logger.error(f"Error authenticating user {sanitized_email}: {e}")
        AuditLogger.log_auth_attempt(sanitized_email, False, ip_address, f"Database error: {e}")
        return False, "An error occurred during authentication.", None
    finally:
        conn.close()


def create_session(user_id: int, days_valid: int = 7, ip_address: str = "unknown") -> str:
    """Create a new session token for a user with audit logging."""
    token = str(uuid.uuid4())
    expires_at = datetime.now(timezone.utc) + timedelta(days=days_valid)
    conn = get_db_connection()
    try:
        with conn:
            conn.execute(
                "INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
                (token, user_id, expires_at.isoformat())
            )
        AuditLogger.log_session_created(user_id, token, ip_address)
        return token
    finally:
        conn.close()


def validate_session(token: str) -> Optional[Dict[str, Any]]:
    """Validate session token and return active user object if valid."""
    if not token:
        return None
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT s.token, s.expires_at, u.id, u.email, u.full_name
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.token = ?
        """, (token,))
        row = cur.fetchone()
        if not row:
            return None

        expires_at = datetime.fromisoformat(row["expires_at"])
        if datetime.now(timezone.utc) > expires_at:
            cur.execute("DELETE FROM sessions WHERE token = ?", (token,))
            conn.commit()
            AuditLogger.log_session_invalidated(token, "expired")
            return None

        return {
            "id": row["id"],
            "email": row["email"],
            "full_name": row["full_name"],
            "token": row["token"]
        }
    except Exception as e:
        logger.error(f"Session validation error: {e}")
        return None
    finally:
        conn.close()


def logout_session(token: str) -> bool:
    """Delete a session token upon logout with audit logging."""
    conn = get_db_connection()
    try:
        with conn:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        AuditLogger.log_session_invalidated(token, "logout")
        return True
    finally:
        conn.close()


def log_prediction(city: str, aqi_24h: int, aqi_48h: int, aqi_72h: int, user_id: Optional[int] = None, model_version: str = "v1.0.0-3cities") -> bool:
    """Log an executed prediction into the database."""
    conn = get_db_connection()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO prediction_history
                (user_id, city, predicted_aqi_24h, predicted_aqi_48h, predicted_aqi_72h, model_version)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, city, aqi_24h, aqi_48h, aqi_72h, model_version)
            )
        return True
    except Exception as e:
        logger.error(f"Error logging prediction: {e}")
        return False
    finally:
        conn.close()


def get_prediction_history(user_id: Optional[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve prediction history logs."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        if user_id is not None:
            cur.execute(
                "SELECT * FROM prediction_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
                (user_id, limit)
            )
        else:
            cur.execute(
                "SELECT * FROM prediction_history ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_user_preferences(user_id: int) -> Dict[str, Any]:
    """Get preferences for a given user."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT favorite_cities, alert_aqi_threshold, alert_email FROM user_preferences WHERE user_id = ?", (user_id,))
        row = cur.fetchone()
        if not row:
            return {"favorite_cities": ["Lahore"], "alert_aqi_threshold": 150, "alert_email": ""}
        return {
            "favorite_cities": json.loads(row["favorite_cities"] or '["Lahore"]'),
            "alert_aqi_threshold": row["alert_aqi_threshold"],
            "alert_email": row["alert_email"] or ""
        }
    finally:
        conn.close()


def update_user_preferences(user_id: int, favorite_cities: List[str], alert_aqi_threshold: int, alert_email: str) -> bool:
    """Update preferences for a user."""
    conn = get_db_connection()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO user_preferences (user_id, favorite_cities, alert_aqi_threshold, alert_email)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    favorite_cities=excluded.favorite_cities,
                    alert_aqi_threshold=excluded.alert_aqi_threshold,
                    alert_email=excluded.alert_email
                """,
                (user_id, json.dumps(favorite_cities), alert_aqi_threshold, alert_email)
            )
        return True
    except Exception as e:
        logger.error(f"Error updating preferences for user {user_id}: {e}")
        return False
    finally:
        conn.close()


# Ensure DB schema initialized on module import
init_db()


def cleanup_expired_sessions() -> int:
    """
    Clean up expired sessions from database.
    Returns number of sessions deleted.
    """
    return SessionCleaner.cleanup_expired_sessions()


def get_password_strength(password: str) -> Tuple[int, str]:
    """
    Check password strength score and category.
    Returns: (score 0-100, category)
    """
    return PasswordValidator.get_strength_score(password)
