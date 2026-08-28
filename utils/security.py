"""
Security Utilities & Enhanced Authentication Features.

Provides production-grade security enhancements:
- Password strength validation
- Session expiration cleanup
- Rate limiting helpers
- Input sanitization
- Security headers configuration
- Audit logging
"""

import re
import logging
from datetime import datetime, timezone
from typing import Tuple, Dict, Any, Optional
from pathlib import Path
import sqlite3

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "pearls_aqi.db"

logger = logging.getLogger(__name__)


class PasswordValidator:
    """
    Production-grade password strength validator.
    
    Requirements:
    - Minimum 8 characters (recommended 12+)
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    - At least one special character
    - No common passwords
    """
    
    COMMON_PASSWORDS = {
        "password", "123456", "12345678", "password123", "admin", "letmein",
        "welcome", "monkey", "qwerty", "abc123", "password1", "admin123",
        "root", "toor", "pass", "test", "guest", "user", "1234", "12345"
    }
    
    @staticmethod
    def validate(password: str, min_length: int = 8) -> Tuple[bool, str]:
        """
        Validate password strength.
        
        Args:
            password: Password to validate
            min_length: Minimum required length (default: 8)
            
        Returns:
            (is_valid, error_message)
        """
        if not password:
            return False, "Password cannot be empty"
        
        if len(password) < min_length:
            return False, f"Password must be at least {min_length} characters long"
        
        if len(password) > 128:
            return False, "Password too long (max 128 characters)"
        
        # Check for common passwords
        if password.lower() in PasswordValidator.COMMON_PASSWORDS:
            return False, "Password is too common - please choose a more secure password"
        
        # Character class requirements
        has_upper = bool(re.search(r'[A-Z]', password))
        has_lower = bool(re.search(r'[a-z]', password))
        has_digit = bool(re.search(r'\d', password))
        has_special = bool(re.search(r'[!@#$%^&*(),.?":{}|<>_\-=+\[\]\\\/;\'`~]', password))
        
        missing_requirements = []
        if not has_upper:
            missing_requirements.append("one uppercase letter")
        if not has_lower:
            missing_requirements.append("one lowercase letter")
        if not has_digit:
            missing_requirements.append("one digit")
        if not has_special:
            missing_requirements.append("one special character (!@#$%^&* etc.)")
        
        if missing_requirements:
            return False, f"Password must contain at least {', '.join(missing_requirements)}"
        
        return True, "Password meets security requirements"
    
    @staticmethod
    def get_strength_score(password: str) -> Tuple[int, str]:
        """
        Calculate password strength score (0-100) and category.
        
        Returns:
            (score, category) where category is "Weak", "Fair", "Good", or "Strong"
        """
        if not password:
            return 0, "Weak"
        
        score = 0
        
        # Length scoring (max 40 points)
        if len(password) >= 8:
            score += 10
        if len(password) >= 12:
            score += 10
        if len(password) >= 16:
            score += 10
        if len(password) >= 20:
            score += 10
        
        # Character diversity (max 40 points)
        if re.search(r'[a-z]', password):
            score += 10
        if re.search(r'[A-Z]', password):
            score += 10
        if re.search(r'\d', password):
            score += 10
        if re.search(r'[!@#$%^&*(),.?":{}|<>_\-=+\[\]\\\/;\'`~]', password):
            score += 10
        
        # Complexity bonus (max 20 points)
        unique_chars = len(set(password))
        if unique_chars >= 8:
            score += 5
        if unique_chars >= 12:
            score += 5
        
        # No sequential characters
        if not re.search(r'(012|123|234|345|456|567|678|789|890|abc|bcd|cde)', password.lower()):
            score += 5
        
        # No repeating characters
        if not re.search(r'(.)\1{2,}', password):
            score += 5
        
        # Penalize common passwords
        if password.lower() in PasswordValidator.COMMON_PASSWORDS:
            score = max(0, score - 50)
        
        # Categorize
        if score >= 80:
            category = "Strong"
        elif score >= 60:
            category = "Good"
        elif score >= 40:
            category = "Fair"
        else:
            category = "Weak"
        
        return min(100, score), category


class SessionCleaner:
    """Automatic cleanup of expired sessions."""
    
    @staticmethod
    def cleanup_expired_sessions() -> int:
        """
        Remove all expired sessions from database.
        
        Returns:
            Number of sessions deleted
        """
        try:
            conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
            cur = conn.cursor()
            
            now = datetime.now(timezone.utc).isoformat()
            cur.execute("SELECT COUNT(*) FROM sessions WHERE expires_at < ?", (now,))
            count = cur.fetchone()[0]
            
            if count > 0:
                cur.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
                conn.commit()
                logger.info(f"Cleaned up {count} expired sessions")
            
            conn.close()
            return count
            
        except Exception as e:
            logger.error(f"Error cleaning expired sessions: {e}")
            return 0
    
    @staticmethod
    def get_session_stats() -> Dict[str, Any]:
        """
        Get statistics about active sessions.
        
        Returns:
            Dict with total_sessions, expired_sessions, active_sessions
        """
        try:
            conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
            cur = conn.cursor()
            
            now = datetime.now(timezone.utc).isoformat()
            
            cur.execute("SELECT COUNT(*) FROM sessions")
            total = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM sessions WHERE expires_at < ?", (now,))
            expired = cur.fetchone()[0]
            
            active = total - expired
            
            conn.close()
            
            return {
                "total_sessions": total,
                "expired_sessions": expired,
                "active_sessions": active
            }
            
        except Exception as e:
            logger.error(f"Error getting session stats: {e}")
            return {
                "total_sessions": 0,
                "expired_sessions": 0,
                "active_sessions": 0
            }


class InputSanitizer:
    """Input validation and sanitization for user inputs."""
    
    @staticmethod
    def sanitize_email(email: str) -> Tuple[bool, str, Optional[str]]:
        """
        Validate and sanitize email address.
        
        Returns:
            (is_valid, sanitized_email, error_message)
        """
        if not email or not isinstance(email, str):
            return False, "", "Email is required"
        
        email = email.strip().lower()
        
        if len(email) > 254:
            return False, "", "Email too long (max 254 characters)"
        
        # RFC 5322 compliant email regex (simplified)
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if not re.match(email_regex, email):
            return False, "", "Invalid email format"
        
        # Block disposable/temporary email domains
        disposable_domains = {
            "tempmail.com", "throwaway.email", "guerrillamail.com",
            "10minutemail.com", "mailinator.com", "trashmail.com"
        }
        
        domain = email.split('@')[1] if '@' in email else ""
        if domain in disposable_domains:
            return False, "", "Disposable email addresses are not allowed"
        
        return True, email, None
    
    @staticmethod
    def sanitize_city_name(city: str) -> Tuple[bool, str, Optional[str]]:
        """
        Validate city name input.
        
        Returns:
            (is_valid, sanitized_city, error_message)
        """
        if not city or not isinstance(city, str):
            return False, "", "City name is required"
        
        city = city.strip().title()
        
        if len(city) > 50:
            return False, "", "City name too long"
        
        # Only allow letters, spaces, hyphens
        if not re.match(r'^[A-Za-z\s\-]+$', city):
            return False, "", "City name contains invalid characters"
        
        # Whitelist of supported cities
        supported_cities = {"Lahore", "Islamabad", "Faisalabad"}
        
        if city not in supported_cities:
            return False, "", f"Unsupported city. Supported: {', '.join(supported_cities)}"
        
        return True, city, None


def get_security_headers() -> Dict[str, str]:
    """
    Get recommended security headers for HTTP responses.
    
    Returns:
        Dict of header_name: header_value
    """
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; connect-src 'self' https://api.openaq.org https://api.open-meteo.com;",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
    }


class AuditLogger:
    """Security audit logging for critical operations."""
    
    @staticmethod
    def log_auth_attempt(email: str, success: bool, ip_address: str = "unknown", reason: str = ""):
        """Log authentication attempt."""
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"AUTH {status} | Email: {email} | IP: {ip_address} | Reason: {reason}")
    
    @staticmethod
    def log_session_created(user_id: int, token: str, ip_address: str = "unknown"):
        """Log session creation."""
        logger.info(f"SESSION CREATED | UserID: {user_id} | Token: {token[:8]}... | IP: {ip_address}")
    
    @staticmethod
    def log_session_invalidated(token: str, reason: str = "logout"):
        """Log session invalidation."""
        logger.info(f"SESSION INVALIDATED | Token: {token[:8]}... | Reason: {reason}")
    
    @staticmethod
    def log_security_event(event_type: str, details: str, severity: str = "INFO"):
        """Log security-related events."""
        logger.log(
            logging.WARNING if severity == "WARNING" else logging.INFO,
            f"SECURITY EVENT | Type: {event_type} | Details: {details} | Severity: {severity}"
        )


# Initialize session cleanup on module import
try:
    cleaned = SessionCleaner.cleanup_expired_sessions()
    if cleaned > 0:
        logger.info(f"Initial session cleanup removed {cleaned} expired sessions")
except Exception as e:
    logger.error(f"Error during initial session cleanup: {e}")
