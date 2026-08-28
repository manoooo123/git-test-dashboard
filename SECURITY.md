# Security Policy & Best Practices

## Overview

Pearls AQI Predictor implements production-grade security measures to protect user data, prevent unauthorized access, and ensure system integrity.

---

## Authentication & Authorization

### Password Security

**Hashing Algorithm**: PBKDF2-HMAC-SHA256
- **Iterations**: 100,000 (OWASP recommended minimum)
- **Salt**: 16 bytes cryptographically random per user
- **Storage**: Hash + salt stored separately in database

**Password Requirements**:
- ✅ Minimum 8 characters (12+ recommended)
- ✅ At least one uppercase letter
- ✅ At least one lowercase letter
- ✅ At least one digit
- ✅ At least one special character
- ✅ No common passwords (blacklist check)

**Password Strength Scoring**:
- **Weak** (0-39): Rejected at registration
- **Fair** (40-59): Accepted with warning
- **Good** (60-79): Recommended strength
- **Strong** (80-100): Excellent security

### Session Management

**Session Tokens**:
- UUID v4 format (128-bit random)
- 7-day expiration by default
- Stored in database with foreign key to users table
- Automatic cleanup of expired sessions

**Session Security**:
- ✅ HTTP-only cookies (if using cookies)
- ✅ Secure flag in production (HTTPS only)
- ✅ Session invalidation on logout
- ✅ Automatic expiration enforcement
- ✅ No session fixation vulnerabilities

### Authentication Flow

1. **Registration**:
   ```
   User Input → Email Validation → Password Strength Check → 
   Hash Password → Store User → Create Preferences → Audit Log
   ```

2. **Login**:
   ```
   User Input → Email Validation → Database Lookup → 
   Password Verification → Create Session → Audit Log
   ```

3. **Session Validation**:
   ```
   Token → Database Lookup → Expiration Check → 
   Return User Object or Invalidate
   ```

---

## Input Validation & Sanitization

### Email Validation

- RFC 5322 compliant regex
- Maximum 254 characters
- Lowercase normalization
- Disposable email domain blocking
- Duplicate email prevention

**Blocked Domains**:
- tempmail.com
- throwaway.email
- guerrillamail.com
- 10minutemail.com
- mailinator.com
- trashmail.com

### City Name Validation

- Whitelist: Lahore, Islamabad, Faisalabad
- Character restriction: letters, spaces, hyphens only
- Maximum 50 characters
- Case normalization (Title Case)

---

## API Security

### Rate Limiting

**Login Endpoint** (`/api/auth/login`):
- 10 attempts per IP per 60 seconds
- 429 Too Many Requests response on limit
- In-memory token bucket implementation

**Future Enhancement**: Redis-based distributed rate limiting for multi-instance deployments

### CORS Configuration

**Allowed Origins**:
```python
{
    "http://localhost:8501",  # Streamlit dashboard
    "http://127.0.0.1:8501",
    "http://localhost:5000",  # Flask API
    "http://127.0.0.1:5000"
}
```

**Headers**:
- `Access-Control-Allow-Origin`: Dynamic (matched origin or *)
- `Access-Control-Allow-Methods`: GET, POST, OPTIONS
- `Access-Control-Allow-Headers`: Content-Type, Authorization

### HTTP Security Headers

**Recommended Headers** (via `utils/security.py`):
```python
{
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'self'; ...",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
}
```

**CSP Policy**:
- `default-src 'self'`: Only load resources from same origin
- `script-src 'self' 'unsafe-inline' 'unsafe-eval'`: Allow inline scripts (Streamlit requirement)
- `style-src 'self' 'unsafe-inline'`: Allow inline styles
- `connect-src`: Whitelist OpenAQ and Open-Meteo APIs

---

## Data Protection

### Sensitive Data Handling

**Never Logged**:
- ❌ Raw passwords
- ❌ Session tokens (only first 8 chars in logs)
- ❌ API keys

**Encrypted/Hashed**:
- ✅ User passwords (PBKDF2-HMAC-SHA256)
- ✅ Session tokens (UUID, not predictable)

**Environment Variables**:
```env
OPENAQ_API_KEY=<key>
HOPSWORKS_API_KEY=<key>
FLASK_SECRET_KEY=<key>
```

**⚠️ Never commit `.env` file to git!**

### Database Security

**SQLite Configuration**:
- Thread-safe connections (`check_same_thread=False`)
- Row factory for dict-like access
- Foreign key constraints enabled
- Automatic schema initialization

**Backup Strategy**:
- Regular backups of `data/pearls_aqi.db`
- Exclude from `.gitignore` in production
- Encrypted backups for sensitive data

---

## Audit Logging

### Security Events Logged

1. **Authentication**:
   - Registration attempts (success/failure)
   - Login attempts (success/failure + IP)
   - Logout events
   - Session creation/invalidation

2. **Security Violations**:
   - Rate limit exceeded
   - Invalid input attempts
   - Expired session access

3. **Critical Operations**:
   - User registration
   - Password changes (future)
   - Preference updates

### Log Format

```
YYYY-MM-DD HH:MM:SS | LEVEL | MODULE | EVENT_TYPE | Details
```

**Example**:
```
2025-01-15 10:23:45 | INFO | pearls_db | AUTH SUCCESS | Email: user@example.com | IP: 192.168.1.1
2025-01-15 10:25:12 | WARNING | pearls_db | AUTH FAILED | Email: attacker@evil.com | IP: 203.0.113.45 | Reason: Incorrect password
```

---

## Vulnerability Prevention

### SQL Injection

**Protection**: Parameterized queries everywhere
```python
# ✅ SAFE
cur.execute("SELECT * FROM users WHERE email = ?", (email,))

# ❌ NEVER DO THIS
cur.execute(f"SELECT * FROM users WHERE email = '{email}'")
```

### XSS (Cross-Site Scripting)

**Protection**:
- Flask auto-escapes HTML in templates
- Streamlit sanitizes inputs by default
- CSP header restricts script sources

### CSRF (Cross-Site Request Forgery)

**Protection**:
- Token-based authentication (Bearer tokens)
- CORS restrictions
- SameSite cookie attribute (if using cookies)

### Session Fixation

**Protection**:
- New session token on each login
- Session invalidation on logout
- No predictable session IDs (UUID v4)

### Brute Force Attacks

**Protection**:
- Rate limiting on login endpoint
- Account lockout after N failed attempts (future)
- Strong password requirements

---

## Security Checklist

### Production Deployment

- [ ] Set `FLASK_SECRET_KEY` to cryptographically random 32+ char string
- [ ] Enable HTTPS/TLS (Let's Encrypt recommended)
- [ ] Configure firewall rules (allow 80/443 only)
- [ ] Set up database backups (daily minimum)
- [ ] Enable security headers in web server config
- [ ] Review and update `.gitignore` (ensure secrets excluded)
- [ ] Set appropriate file permissions (640 for .env, 600 for DB)
- [ ] Configure log rotation (logrotate)
- [ ] Set up monitoring/alerts (failed login spikes, errors)
- [ ] Test backup restore procedure

### Code Review

- [ ] No hardcoded credentials
- [ ] All inputs validated/sanitized
- [ ] Parameterized SQL queries only
- [ ] Sensitive data not logged
- [ ] Error messages don't leak information
- [ ] Dependencies up to date (pip-audit)

### Regular Maintenance

- [ ] Update dependencies monthly (security patches)
- [ ] Review audit logs weekly
- [ ] Clean expired sessions daily (automatic)
- [ ] Rotate API keys quarterly
- [ ] Review user accounts monthly (remove inactive)
- [ ] Test backup restoration quarterly

---

## Reporting Security Issues

**DO NOT** create public GitHub issues for security vulnerabilities.

**Contact**:
- Email: [security@yourdomain.com] (set this up)
- PGP Key: [link to public key] (optional but recommended)

**Response Time**:
- Acknowledgment: 48 hours
- Initial assessment: 5 business days
- Fix deployment: Based on severity

**Severity Levels**:
- **Critical**: RCE, authentication bypass, data breach
- **High**: Privilege escalation, SQL injection, XSS
- **Medium**: CSRF, information disclosure, DoS
- **Low**: Non-exploitable bugs, best practice violations

---

## Security Updates

### Version History

**v2.1.0** (Current)
- ✅ Enhanced password validation (8+ chars, complexity requirements)
- ✅ Input sanitization for email and city names
- ✅ Session expiration and automatic cleanup
- ✅ Audit logging for authentication events
- ✅ Rate limiting on login endpoint
- ✅ Security headers configuration
- ✅ Disposable email blocking

**v2.0.0**
- ✅ PBKDF2-HMAC-SHA256 password hashing (100k iterations)
- ✅ Per-user salt generation
- ✅ Session token management
- ✅ CORS configuration

---

## Dependencies Security

### Known Vulnerabilities

Check dependencies with:
```bash
pip install pip-audit
pip-audit
```

### Update Strategy

- **Security patches**: Apply immediately
- **Minor versions**: Monthly review
- **Major versions**: Quarterly review + testing

### Pinned Versions

See `requirements.txt` for pinned dependency versions. All versions are tested for compatibility and security.

---

## Compliance

### GDPR Considerations

If deploying in EU:
- ✅ User consent for data collection
- ✅ Right to access (user can view prediction history)
- ✅ Right to deletion (implement user account deletion)
- ✅ Data portability (export prediction history as JSON)
- ⚠️ Privacy policy required
- ⚠️ Cookie consent banner required (if using cookies)

### OWASP Top 10 Compliance

- [x] A01: Broken Access Control → Token-based auth, session validation
- [x] A02: Cryptographic Failures → PBKDF2 hashing, TLS in production
- [x] A03: Injection → Parameterized queries
- [x] A04: Insecure Design → Secure by design, threat modeling
- [x] A05: Security Misconfiguration → Security headers, CSP
- [x] A06: Vulnerable Components → Regular dependency updates
- [x] A07: Authentication Failures → Strong passwords, rate limiting
- [x] A08: Data Integrity Failures → Input validation, hash verification
- [x] A09: Logging Failures → Comprehensive audit logging
- [x] A10: SSRF → No user-controlled URLs for external requests

---

## Additional Resources

- [OWASP Cheat Sheets](https://cheatsheetseries.owasp.org/)
- [NIST Password Guidelines](https://pages.nist.gov/800-63-3/)
- [Flask Security Best Practices](https://flask.palletsprojects.com/en/2.3.x/security/)
- [SQLite Security](https://www.sqlite.org/security.html)

---

**Last Updated**: 2025-01-XX  
**Maintained by**: Pearls AQI Predictor Security Team
