import smtplib

# Replace these with your actual credentials
sender_email = "manamnathtiwari@gmail.com"
receiver_email = "manamtiwari786@gmail.com" 
password = ""

try:
    # Create an SMTP client session object
    s = smtplib.SMTP("smtp.gmail.com", 587)  # Correct SMTP server for Gmail
    s.starttls()  # Upgrade the connection to secure
    s.login(sender_email, password)  # Log in to your email account

    # Send your email here
    message = "Subject: Test Email\n\nThis is a test email."
    s.sendmail(sender_email, receiver_email, message)
    print("Email sent successfully!")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    s.quit()  # Close the connection
