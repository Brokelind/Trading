import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Optional: load secrets locally
try:
    import env  # local env.py file with credentials
except ImportError:
    env = None

# === Config ===
STRONG_SIGNAL_THRESHOLD = 0.65
RESULTS_DIR = "results"

EMAIL_SENDER = os.getenv("EMAIL_SENDER") or getattr(env, "EMAIL_SENDER", None)
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER") or getattr(env, "EMAIL_RECEIVER", None)
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD") or getattr(env, "GMAIL_APP_PASSWORD", None)


def load_results():
    summaries = []

    for file in os.listdir(RESULTS_DIR):
        if file.endswith("_summary.json"):
            with open(os.path.join(RESULTS_DIR, file), "r") as f:
                data = json.load(f)
                if data.get("confidence", 0) >= STRONG_SIGNAL_THRESHOLD:
                    summaries.append(data)
    return summaries


def compose_email_body(results):
    if not results:
        return None

    body = "<strong>🚨 Strong Trading Signals Detected</strong><br><br>"
    for res in results:
        body += (
            f"<b>Ticker:</b> {res['ticker']}<br>"
            f"<b>Signal:</b> {res['signal']}<br>"
            f"<b>Confidence:</b> {res['confidence']:.2f}<br>"
            f"<b>Strategy:</b> {res.get('strategy')}<br>"
            f"<b>Model performance vs Buy & Hold:</b> {res.get('model performance vs Buy & Hold')}%<br>"
            f"<b>Sentiment Score:</b> {res.get('sentiment_score')}<br>"
            f"<b>Sentiment Confidence:</b> {res.get('sentiment_confidence')}<br>"
            f"<b>Predicted difference:</b> {res.get('predicted_diff')}<br>"
            f"<b>Accuracy:</b> {res.get('accuracy')}%<br><br>"
        )
    return body


def send_email(subject, body):
    if not (EMAIL_SENDER and EMAIL_RECEIVER and GMAIL_APP_PASSWORD):
        print("Missing email credentials.")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER

    html_part = MIMEText(body, "html")
    msg.attach(html_part)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, GMAIL_APP_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        print("✅ Email sent successfully.")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")


if __name__ == "__main__":
    print("📈 Compiling signal summary...")
    summaries = load_results()

    email_body = compose_email_body(summaries)
    if email_body:
        send_email("Daily Trading Summary - Strong Signals", email_body)
    else:
        print("📭 No strong signals to report today.")
