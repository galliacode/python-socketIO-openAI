from urllib.parse import urlencode
from fastapi import BackgroundTasks
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from pydantic import EmailStr
import os

conf = ConnectionConfig(
    MAIL_USERNAME = os.getenv("MAIL_USERNAME"),
    MAIL_PASSWORD = os.getenv("MAIL_PASSWORD"),
    MAIL_FROM = os.getenv("MAIL_FROM"),
    MAIL_PORT = os.getenv("MAIL_PORT"),
    MAIL_SERVER = os.getenv("MAIL_SERVER"),
    MAIL_STARTTLS = os.getenv("MAIL_STARTTLS"),  # Replace MAIL_TLS with MAIL_STARTTLS if required
    MAIL_SSL_TLS = os.getenv("MAIL_SSL_TLS")   # Replace MAIL_SSL with MAIL_SSL_TLS if required
)

def generate_password_reset_url(email_to: EmailStr, token: str) -> str:
    query_params = urlencode({"token": token, "email": email_to})
    return f"{os.getenv('HOST')}/reset-password?{query_params}"

def send_password_reset_email(background_tasks: BackgroundTasks, email_to: EmailStr, token: str):
    reset_link = generate_password_reset_url(email_to=email_to, token=token)
    message = MessageSchema(
        subject="Analitiq: Request Password Reset",
        recipients=[email_to], 
        body=f"Please click on the link to reset your password: <a href='{reset_link}'>{reset_link}</a>",
        subtype="html"
    )

    fm = FastMail(conf)
    background_tasks.add_task(fm.send_message, message)
