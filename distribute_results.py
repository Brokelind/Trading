import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
# Removed MIMEImage import

# Optional: load secrets locally
try:
    import env  # local env.py file with credentials
except ImportError:
    env = None

# === Config ===
STRONG_SIGNAL_THRESHOLD = 0.65
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

EMAIL_SENDER = os.environ.get("EMAIL_SENDER") or getattr(env, "EMAIL_SENDER", None)
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER") or getattr(env, "EMAIL_RECEIVER", None)
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD") or getattr(env, "GMAIL_APP_PASSWORD", None)

def save_result_json(ticker, payload):
    path = os.path.join(RESULTS_DIR, f"{ticker}_summary.json")
    with open(path, "w") as f:
        json.dump(payload, f, default=str, indent=2)
    return path

def load_results():
    summaries = []

    for file in os.listdir(RESULTS_DIR):
        if file.endswith("_summary.json"):
            with open(os.path.join(RESULTS_DIR, file), "r") as f:
                data = json.load(f)

                # Safely get pct_diff from chosen model
                chosen_model = data.get("chosen_model")
                preds = data.get("predictions", {})
                pct_diff = 0
                if chosen_model and chosen_model in preds:
                    if chosen_model == "Ensemble":
                        pct_diff = preds[chosen_model].get("pct_diff_weighted", 0)
                    else:
                        pct_diff = preds[chosen_model].get("pct_diff", 0)

                sentiment_conf = data.get("sentiment", {}).get("confidence", 0)

                if abs(pct_diff) >= 0.01 and sentiment_conf >= 0.4:
                    summaries.append(data)

    return summaries

def compose_html_email(results):
    if not results:
        return None

    table_rows = "".join(f"""
        <tr>
            <td>{res['ticker']}</td>
            <td>{res['signal']}</td>
            <td>{res.get('confidence', 0):.2f}</td>
            <td>{res.get('strategy', '')}</td>
            <td>{res.get('model performance vs Buy & Hold', '')}%</td>
            <td>{res.get('sentiment_score', '')}</td>
            <td>{res.get('sentiment_confidence', '')}</td>
            <td>{res.get('predicted_diff', '')}</td>
            <td>{res.get('accuracy', '')}%</td>
        </tr>
    """ for res in results)

    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; background-color: #111; color: #ddd; padding: 20px; }}
            table {{ border-collapse: collapse; width: 100%; background-color: #222; color: #ddd; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #444; }}
            th {{ background-color: #333; color: #00ff99; }}
            tr:hover {{ background-color: #2a2a2a; }}
            .signal-buy {{ color: #00ff99; font-weight: bold; }}
            .signal-sell {{ color: #ff4444; font-weight: bold; }}
            .header {{ color: #00ff99; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>📊 Daily Trading Report</h2>
            <p>Generated on {os.path.basename(RESULTS_DIR)}</p>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Ticker</th><th>Signal</th><th>Confidence</th><th>Strategy</th>
                    <th>Perf vs B&H</th><th>Sentiment</th><th>Sent. Conf.</th><th>Pred. Diff</th><th>Accuracy</th>
                </tr>
            </thead>
            <tbody>{table_rows}</tbody>
        </table>
        <br>
        <div style="color: #888; font-size: 12px;">
            <p><em>Note: Charts and detailed analysis available in the results directory.</em></p>
        </div>
    </body>
    </html>
    """
    return html

def compose_text_email(results):
    """Fallback plain text version"""
    if not results:
        return None
        
    text = "🚨 Strong Trading Signals Detected\n\n"
    for res in results:
        text += (
            f"Ticker: {res['ticker']}\n"
            f"Signal: {res['signal']}\n"
            f"Confidence: {res.get('confidence', 0):.2f}\n"
            f"Strategy: {res.get('strategy', '')}\n"
            f"Model performance vs Buy & Hold: {res.get('model performance vs Buy & Hold', '')}%\n"
            f"Sentiment Score: {res.get('sentiment_score', '')}\n"
            f"Sentiment Confidence: {res.get('sentiment_confidence', '')}\n"
            f"Predicted difference: {res.get('predicted_diff', '')}\n"
            f"Accuracy: {res.get('accuracy', '')}%\n"
            f"{'-' * 50}\n"
        )
    return text

def send_email(subject, html_body):
    if not (EMAIL_SENDER and EMAIL_RECEIVER and GMAIL_APP_PASSWORD):
        print("Missing email credentials.")
        return False

    # Create message container
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER

    # Create text fallback
    text_body = "Please view this email in an HTML-compatible email client."
    
    # Attach both text and HTML versions
    part1 = MIMEText(text_body, 'plain')
    part2 = MIMEText(html_body, 'html')
    
    msg.attach(part1)
    msg.attach(part2)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, GMAIL_APP_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        print("✅ Email sent successfully.")
        return True
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False

def send_daily_summary():
    """Main function to compile and send daily summary"""
    print("📈 Compiling signal summary...")
    summaries = load_results()

    if summaries:
        print(f"📧 Found {len(summaries)} strong signals. Sending email...")
        
        # Compose HTML email (no images)
        email_body = compose_html_email(summaries)
        
        if email_body:
            success = send_email(
                f"Daily Trading Summary - {len(summaries)} Strong Signals", 
                email_body
            )
            return success
        else:
            print("❌ Failed to compose email body")
            return False
    else:
        print("📭 No strong signals to report today.")
        return True  # No error, just nothing to send

if __name__ == "__main__":
    send_daily_summary()