from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

SENDGRID_API_KEY = "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"
EMAIL_SENDER = "appbison406@gmail.com"
EMAIL_RECEIVER = "hugobrokelind@gmail.com"

message = Mail(
    from_email=EMAIL_SENDER,
    to_emails=EMAIL_RECEIVER,
    subject="Test Email from SendGrid",
    html_content="<strong>This is a test email sent via SendGrid API</strong>"
)

try:
    sg = SendGridAPIClient(SENDGRID_API_KEY)
    response = sg.send(message)
    print(f"Email sent. Status: {response.status_code}")
except Exception as e:
    print(f"Send failed: {e}")
