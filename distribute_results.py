import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
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

def compose_html_email(results, image_cids=None):
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

    imgs_html = ""
    if image_cids:
        for cid in image_cids:
            imgs_html += f'<br><img src="cid:{cid}" style="max-width:100%;">'

    html = f"""
    <html>
    <body style="font-family:Arial, sans-serif; background-color:#111; color:#ddd; padding:20px;">
        <h2 style="color:#00ff99;">📊 Daily Trading Report</h2>
        <table style="border-collapse:collapse; width:100%; background-color:#222; color:#ddd;">
            <thead>
                <tr style="background-color:#333; color:#00ff99;">
                    <th>Ticker</th><th>Signal</th><th>Confidence</th><th>Strategy</th>
                    <th>Perf vs B&H</th><th>Sentiment</th><th>Sent. Conf.</th><th>Pred. Diff</th><th>Accuracy</th>
                </tr>
            </thead>
            <tbody>{table_rows}</tbody>
        </table>
        {imgs_html}
    </body>
    </html>
    """
    return html

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


def send_email(subject, body, inline_images=None):
    if not (EMAIL_SENDER and EMAIL_RECEIVER and GMAIL_APP_PASSWORD):
        print("Missing email credentials.")
        return

    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(body, "html"))
    msg.attach(alt)

    if inline_images:
        for i, img_path in enumerate(inline_images):
            with open(img_path, "rb") as f:
                img = MIMEImage(f.read())
                img.add_header("Content-ID", f"<chart{i}>")
                img.add_header("Content-Disposition", "inline", filename=os.path.basename(img_path))
                msg.attach(img)

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

    if summaries:
        # example: you generate these paths somewhere else, replace with actual paths
        backtest_images = ["path/to/backtest1.png", "path/to/backtest2.png"]  
        prediction_images = ["path/to/prediction1.png", "path/to/prediction2.png"]
        all_images = backtest_images + prediction_images

        image_cids = [f"chart{i}" for i in range(len(all_images))]

        email_body = compose_html_email(summaries, image_cids=image_cids)

        send_email("Daily Trading Summary - Strong Signals", email_body, inline_images=all_images)
    else:
        print("📭 No strong signals to report today.")
