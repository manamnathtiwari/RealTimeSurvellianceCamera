import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Your email credentials
gmail_user = 'hackathonmanam@gmail.com'
gmail_password = '8217034929q'

# Email setup
def send_email(to_email, subject, body):
    try:
        # Set up the MIME
        msg = MIMEMultipart()
        msg['From'] = gmail_user
        msg['To'] = to_email
        msg['Subject'] = subject

        # Attach the body to the message
        msg.attach(MIMEText(body, 'plain'))

        # Connect to Gmail's SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(gmail_user, gmail_password)

        # Send the email
        server.sendmail(gmail_user, to_email, msg.as_string())
        server.quit()

        print("Email successfully sent to", to_email)
    except Exception as e:
        print(f"Failed to send email. Error: {str(e)}")

# Example usage
if __name__ == '__main__':
    to_email = "recipient@example.com"
    subject = "Test Email"
    body = "This is a test email sent from Python using Gmail SMTP."

    send_email(to_email, subject, body)
