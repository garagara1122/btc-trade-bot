# telegram_alert.py

import requests

# 텔레그램 봇 토큰과 챗 ID 입력
TELEGRAM_TOKEN = "7540304059:AAHkMfDkE3-xH8whXfwiPbNbT3BUHA7KSU0"
CHAT_ID = "7587985949"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, data=payload)
        return response.json()
    except Exception as e:
        print(f"❌ 텔레그램 전송 실패: {e}")
