import requests
from typing import Dict, Any, Optional

import certifi

BASE_URL = "https://api.telegram.org/bot"

class TelegramBot:
    def __init__(self, token: str, chat_id: str, base_url: Optional[str] = None):
        self.token = token
        self.chat_id = chat_id
        self.base_url = base_url if base_url else BASE_URL
        self.api_url = f"{self.base_url.rstrip('/')}{self.token}"

        # Create session with SSL context
        self.session = requests.Session()
        self.session.verify = certifi.where()
        self.session.timeout = 10

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def close(self):
        """close requests session"""
        if self.session:
            self.session.close()

    def send_text(self, content: str, parse_mode: str = "HTML") -> Dict[str, Any]:
        """Send a text message to Telegram"""
        payload = {
            "chat_id": self.chat_id,
            "text": content,
            "parse_mode": parse_mode
        }
        return self._send_message("sendMessage", payload)

    def _send_message(self, method: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Internal method to send messages to Telegram API"""
        url = f"{self.api_url}/{method}"
        
        try:
            response = self.session.post(url, json=payload)
            response_data = response.json()
            if not response_data.get("ok", False):
                print(f"Telegram send message failed: {response_data}")
            return response_data
        except Exception as e:
            print(f"Telegram send message failed: {e}")
            return {"ok": False, "error": str(e)}
