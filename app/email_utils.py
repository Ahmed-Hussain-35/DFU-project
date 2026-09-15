"""
SMTP email helper. Configure via env vars: SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS.
If unconfigured, send_email() silently skips (logs to console) instead of crashing —
so the app works fine without email set up, and you can add it later.

Example (Gmail): SMTP_HOST=smtp.gmail.com  SMTP_PORT=587
                  SMTP_USER=you@gmail.com   SMTP_PASS=<16-char App Password, not your real password>
"""
import os, smtplib
from email.mime.text import MIMEText

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
FROM_EMAIL = os.getenv("FROM_EMAIL", SMTP_USER or "no-reply@dfuplatform.local")


def send_email(to_email: str, subject: str, body: str) -> bool:
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS):
        print(f"[email skipped — SMTP not configured] to={to_email} subject={subject!r}")
        return False
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = FROM_EMAIL
        msg["To"] = to_email
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(FROM_EMAIL, [to_email], msg.as_string())
        return True
    except Exception as e:
        print(f"[email failed] {e}")
        return False
