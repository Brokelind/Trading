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

# Table style for email - defined before use
_TABLE_STYLE = (
    "border-collapse:collapse;width:100%;background:rgba(255,255,255,0.03);"
    "border-radius:10px;overflow:hidden;font-size:0.88em"
)

EMAIL_SENDER = os.environ.get("EMAIL_SENDER") or getattr(env, "EMAIL_SENDER", None)
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER") or getattr(env, "EMAIL_RECEIVER", None)
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD") or getattr(env, "GMAIL_APP_PASSWORD", None)

def save_result_json(ticker, payload):
    path = os.path.join(RESULTS_DIR, f"{ticker}_summary.json")
    with open(path, "w") as f:
        json.dump(payload, f, default=str, indent=2)
    return path

def load_results():
    """Load and filter trading results with improved criteria"""
    summaries = []

    for file in os.listdir(RESULTS_DIR):
        if file.endswith("_summary.json"):
            try:
                with open(os.path.join(RESULTS_DIR, file), "r") as f:
                    data = json.load(f)

                ticker = data.get("ticker")
                chosen_model = data.get("chosen_model")
                preds = data.get("predictions", {})
                signal = data.get("signal", "HOLD")
                
                # Calculate percentage difference
                pct_diff = 0
                if chosen_model and chosen_model in preds:
                    if chosen_model == "Ensemble":
                        last_price = data.get("last_price", 0)
                        ensemble_price = preds[chosen_model].get("predicted_price", 0)
                        pct_diff = (ensemble_price - last_price) / last_price * 100 if last_price else 0
                    else:
                        pct_diff = preds[chosen_model].get("pct_diff", 0)

                # Get confidence from ensemble or use default
                confidence = preds.get("Ensemble", {}).get("confidence", 0.5)
                
                # Get sentiment confidence
                sentiment_conf = data.get("sentiment", {}).get("confidence", 0)
                
                # Enhanced filtering criteria
                meets_criteria = (
                    signal != "HOLD" and  # Only show BUY/SELL signals
                    abs(pct_diff) >= 0.5 and  # Minimum 0.5% predicted change
                    confidence >= 0.4 and  # Minimum 40% model confidence
                    sentiment_conf >= 0.3  # Minimum 30% sentiment confidence
                )

                # Add confidence and pct_diff to data for sorting
                data['confidence'] = confidence
                data['pct_diff'] = pct_diff
                
                if meets_criteria:
                    summaries.append(data)
                else:
                    print(f"Filtered out {ticker}: signal={signal}, pct_diff={pct_diff:.2f}%, conf={confidence:.2f}")
                    
            except Exception as e:
                print(f"Error loading {file}: {e}")

    print(f"Loaded {len(summaries)} strong signals after filtering")
    return summaries


def compose_html_email(results, crypto_signals=None):
    """Compose a premium consolidated HTML email with equity + crypto sections."""
    from datetime import datetime as _dt
    if not results and not crypto_signals:
        return None

    run_date = _dt.now().strftime('%A, %B %d %Y  %H:%M UTC')

    # ── Equity rows ──────────────────────────────────────────────────────────
    equity_rows = ""
    buy_count = sell_count = hold_count = 0
    for res in (results or []):
        sig = res.get('signal', 'HOLD')
        if sig == 'BUY':   buy_count  += 1; sig_color = '#00e676'
        elif sig == 'SELL': sell_count += 1; sig_color = '#ff5252'
        else:               hold_count += 1; sig_color = '#ffa726'

        conf   = res.get('confidence', 0)
        pct    = res.get('pct_diff', 0)
        sent   = res.get('sentiment', {})
        s_sig  = sent.get('signal', 'N/A')
        s_scr  = sent.get('score', 0)
        model  = res.get('chosen_model', 'N/A')
        pct_color = '#00e676' if pct > 0 else '#ff5252'
        s_color   = '#00e676' if s_scr > 0 else '#ff5252'
        equity_rows += f"""
        <tr>
            <td style="font-weight:700;color:#e5e7eb">{res.get('ticker','')}</td>
            <td><span style="color:{sig_color};font-weight:700;background:rgba(255,255,255,0.05);
                padding:3px 10px;border-radius:20px;font-size:0.82em">{sig}</span></td>
            <td style="color:{pct_color}">{pct:+.2f}%</td>
            <td style="color:#9ca3af">{conf:.0%}</td>
            <td style="color:{s_color}">{s_sig} ({s_scr:+.2f})</td>
            <td style="color:#a78bfa;font-size:0.85em">{model}</td>
        </tr>"""

    # ── Crypto rows ───────────────────────────────────────────────────────────
    crypto_rows = ""
    for sig in (crypto_signals or []):
        sym = sig.get('ticker', '').replace('-USD', '')
        csig = sig.get('signal', 'HOLD')
        conf = sig.get('confidence', 0)
        ret  = sig.get('sub_return', 0)
        sig_color = '#00e676' if csig == 'BUY' else '#ff5252' if csig == 'SELL' else '#ffa726'
        ret_color = '#00e676' if ret > 0 else '#ff5252'
        crypto_rows += f"""
        <tr>
            <td style="font-weight:700;color:#c4b5fd">{sym}</td>
            <td><span style="color:{sig_color};font-weight:700;background:rgba(255,255,255,0.05);
                padding:3px 10px;border-radius:20px;font-size:0.82em">{csig}</span></td>
            <td style="color:{ret_color}">{ret:+.2f}%</td>
            <td style="color:#9ca3af">{conf:.0%}</td>
        </tr>"""

    equity_section = f"""
    <h3 style="color:#00e5ff;font-size:1em;letter-spacing:0.1em;text-transform:uppercase;margin:28px 0 12px">
        Equity Signals ({len(results or [])} tickers)
    </h3>
    <table width="100%" cellpadding="0" cellspacing="0" style="{_TABLE_STYLE}">
        <thead>
            <tr>
                <th>Ticker</th><th>Signal</th><th>Pred. Change</th>
                <th>Confidence</th><th>Sentiment (FinBERT)</th><th>Model</th>
            </tr>
        </thead>
        <tbody>{equity_rows}</tbody>
    </table>""" if results else ""

    crypto_section = f"""
    <h3 style="color:#a78bfa;font-size:1em;letter-spacing:0.1em;text-transform:uppercase;margin:28px 0 12px">
        Crypto Lead-Lag Signals ({len(crypto_signals)} alerts)
    </h3>
    <table width="100%" cellpadding="0" cellspacing="0" style="{_TABLE_STYLE}">
        <thead>
            <tr><th>Symbol</th><th>Signal</th><th>3h Return</th><th>Confidence</th></tr>
        </thead>
        <tbody>{crypto_rows}</tbody>
    </table>""" if crypto_signals else ""

    stats_bar = (f"<span style='margin-right:20px'>🟢 {buy_count} BUY</span>"
                 f"<span style='margin-right:20px'>🔴 {sell_count} SELL</span>"
                 f"<span>🟡 {hold_count} HOLD</span>") if results else ""

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background-color:#0b0f19;font-family:'Segoe UI',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#0b0f19;padding:20px">
<tr><td align="center">
<table width="640" cellpadding="0" cellspacing="0" style="background:#111827;border-radius:16px;
    border:1px solid rgba(255,255,255,0.08);overflow:hidden">

    <!-- Header -->
    <tr><td style="background:linear-gradient(135deg,#0b0f19,#1a1040);padding:32px 36px;text-align:center;
                   border-bottom:1px solid rgba(255,255,255,0.07)">
        <div style="font-size:1.8em;font-weight:700;background:linear-gradient(90deg,#00e5ff,#7b2fff);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:6px">
            AI Trading Report
        </div>
        <div style="color:#6b7280;font-size:0.85em">{run_date}</div>
    </td></tr>

    <!-- Stats bar -->
    <tr><td style="padding:16px 36px;background:rgba(255,255,255,0.03);
                   border-bottom:1px solid rgba(255,255,255,0.07);color:#9ca3af;font-size:0.88em">
        {stats_bar}
    </td></tr>

    <!-- Content -->
    <tr><td style="padding:20px 36px 36px">
        {equity_section}
        {crypto_section}

        <p style="color:#4b5563;font-size:0.78em;margin-top:32px;padding-top:20px;
                  border-top:1px solid rgba(255,255,255,0.06)">
            This report is auto-generated by the AI Trading System.<br>
            Sentiment powered by FinBERT. Not financial advice.
        </p>
    </td></tr>

</table>
</td></tr>
</table>
</body>
</html>"""
    return html


def compose_text_email(results, crypto_signals=None):
    """Fallback plain text version"""
    from datetime import datetime as _dt
    if not results and not crypto_signals:
        return None

    text = f"AI Trading Report — {_dt.now().strftime('%Y-%m-%d %H:%M UTC')}\n"
    text += "=" * 50 + "\n\n"

    if results:
        text += "EQUITY SIGNALS\n" + "-" * 30 + "\n"
        for res in results:
            sig = res.get('signal', 'HOLD')
            pct = res.get('pct_diff', 0)
            conf = res.get('confidence', 0)
            sent = res.get('sentiment', {}).get('signal', 'N/A')
            text += (
                f"  {res['ticker']:6s}  {sig:4s}  Change: {pct:+.2f}%  "
                f"Conf: {conf:.0%}  Sentiment: {sent}\n"
            )

    if crypto_signals:
        text += "\nCRYPTO LEAD-LAG SIGNALS\n" + "-" * 30 + "\n"
        for sig in crypto_signals:
            sym = sig.get('ticker', '').replace('-USD', '')
            csig = sig.get('signal', 'HOLD')
            ret = sig.get('sub_return', 0)
            conf = sig.get('confidence', 0)
            text += f"  {sym:6s}  {csig:4s}  3h Return: {ret:+.2f}%  Conf: {conf:.0%}\n"

    text += "\nNot financial advice."
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

    # Attach both text and HTML versions
    part1 = MIMEText("Please view this email in an HTML-compatible email client.", 'plain')
    part2 = MIMEText(html_body, 'html')
    msg.attach(part1)
    msg.attach(part2)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, GMAIL_APP_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        print("Email sent successfully.")
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False


def send_daily_summary():
    """Main function to compile and send a consolidated daily summary (equities + crypto)."""
    print("Compiling signal summary...")
    summaries = load_results()

    # Load crypto signals
    crypto_signals = []
    crypto_path = os.path.join(RESULTS_DIR, "crypto_signals.json")
    if os.path.exists(crypto_path):
        try:
            with open(crypto_path) as f:
                crypto_signals = json.load(f)
            print(f"Loaded {len(crypto_signals)} crypto signal(s).")
        except Exception as e:
            print(f"Could not load crypto signals: {e}")

    total = len(summaries) + len(crypto_signals)
    if total == 0:
        print("No signals to report today.")
        return True

    print(f"Sending consolidated email: {len(summaries)} equity + {len(crypto_signals)} crypto signals...")
    email_body = compose_html_email(summaries, crypto_signals)
    if email_body:
        subject = (
            f"Trading Report: {len(summaries)} Equity + "
            f"{len(crypto_signals)} Crypto Signal(s)"
        )
        return send_email(subject, email_body)
    else:
        print("Failed to compose email body")
        return False