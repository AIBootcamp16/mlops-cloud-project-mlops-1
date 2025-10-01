# data-pipeline/src/utils/send_notification.py
"""
알림 전송 유틸리티: Slack, Email
"""
import os
import sys
import json
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def send_slack_notification(message: str, status: str = "success"):
    """
    Slack 웹훅으로 알림 전송

    Args:
        message: 전송할 메시지
        status: success, warning, error
    """
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    channel = os.getenv("SLACK_CHANNEL", "#mlops-alerts")

    if not webhook_url:
        logger.warning("SLACK_WEBHOOK_URL not set, skipping Slack notification")
        return False

    # 상태에 따른 이모지
    emoji_map = {
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
        "info": "ℹ️"
    }
    emoji = emoji_map.get(status, "📊")

    # 상태에 따른 색상
    color_map = {
        "success": "#36a64f",  # 녹색
        "warning": "#ff9800",  # 주황색
        "error": "#d32f2f",  # 빨간색
        "info": "#2196f3"  # 파란색
    }
    color = color_map.get(status, "#2196f3")

    # Slack 메시지 페이로드
    payload = {
        "channel": channel,
        "username": "MLOps Bot",
        "icon_emoji": ":robot_face:",
        "attachments": [
            {
                "color": color,
                "title": f"{emoji} COVID-19 배치 예측 알림",
                "text": message,
                "fields": [
                    {
                        "title": "Status",
                        "value": status.upper(),
                        "short": True
                    },
                    {
                        "title": "Timestamp",
                        "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "short": True
                    }
                ],
                "footer": "COVID-19 MLOps Pipeline",
                "footer_icon": "https://platform.slack-edge.com/img/default_application_icon.png"
            }
        ]
    }

    try:
        response = requests.post(
            webhook_url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=10
        )

        if response.status_code == 200:
            logger.info("✅ Slack notification sent successfully")
            return True
        else:
            logger.error(f"❌ Slack notification failed: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        logger.error(f"❌ Error sending Slack notification: {e}")
        return False


def send_email_notification(subject: str, message: str, status: str = "success"):
    """
    이메일 알림 전송

    Args:
        subject: 이메일 제목
        message: 이메일 본문
        status: success, warning, error
    """
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    email_to = os.getenv("EMAIL_TO")

    if not all([smtp_user, smtp_password, email_to]):
        logger.warning("Email credentials not set, skipping email notification")
        return False

    # 상태에 따른 이모지
    emoji_map = {
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
        "info": "ℹ️"
    }
    emoji = emoji_map.get(status, "📊")

    # HTML 이메일 본문
    html_body = f"""
    <html>
      <head>
        <style>
          body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
          .header {{ background-color: #f4f4f4; padding: 20px; text-align: center; }}
          .content {{ padding: 20px; }}
          .status-{status} {{ 
            padding: 10px; 
            border-left: 4px solid {'#36a64f' if status == 'success' else '#d32f2f' if status == 'error' else '#ff9800'};
            background-color: #f9f9f9;
            margin: 20px 0;
          }}
          .footer {{ background-color: #f4f4f4; padding: 10px; text-align: center; font-size: 12px; }}
        </style>
      </head>
      <body>
        <div class="header">
          <h2>{emoji} COVID-19 MLOps Pipeline Alert</h2>
        </div>
        <div class="content">
          <div class="status-{status}">
            <strong>Status:</strong> {status.upper()}<br>
            <strong>Time:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
          </div>
          <p>{message}</p>
        </div>
        <div class="footer">
          <p>This is an automated message from COVID-19 MLOps Pipeline</p>
        </div>
      </body>
    </html>
    """

    try:
        # 이메일 메시지 생성
        msg = MIMEMultipart('alternative')
        msg['From'] = smtp_user
        msg['To'] = email_to
        msg['Subject'] = f"{emoji} {subject}"

        # HTML 파트 추가
        html_part = MIMEText(html_body, 'html')
        msg.attach(html_part)

        # SMTP 서버 연결 및 전송
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)

        logger.info("✅ Email notification sent successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Error sending email notification: {e}")
        return False


def send_notification(message: str, subject: str = None, status: str = "success"):
    """
    모든 알림 채널로 전송 (Slack + Email)

    Args:
        message: 알림 메시지
        subject: 이메일 제목 (없으면 message 사용)
        status: success, warning, error, info
    """
    if subject is None:
        subject = f"COVID-19 Pipeline - {status.upper()}"

    results = {
        "slack": send_slack_notification(message, status),
        "email": send_email_notification(subject, message, status)
    }

    return results


def main():
    """CLI 실행용"""
    import argparse

    parser = argparse.ArgumentParser(description="Send notification")
    parser.add_argument("--message", type=str, required=True, help="Notification message")
    parser.add_argument("--subject", type=str, default=None, help="Email subject")
    parser.add_argument("--status", type=str, default="success",
                        choices=["success", "warning", "error", "info"],
                        help="Notification status")
    parser.add_argument("--channel", type=str, default="all",
                        choices=["all", "slack", "email"],
                        help="Notification channel")

    args = parser.parse_args()

    if args.channel == "all":
        results = send_notification(args.message, args.subject, args.status)
        logger.info(f"Notification results: {results}")
    elif args.channel == "slack":
        send_slack_notification(args.message, args.status)
    elif args.channel == "email":
        subject = args.subject or f"COVID-19 Pipeline - {args.status.upper()}"
        send_email_notification(subject, args.message, args.status)


if __name__ == "__main__":
    main()