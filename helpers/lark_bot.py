import os
import ssl
import aiohttp
from typing import Dict, Any, Optional

import certifi

BASE_URL = "https://www.feishu.cn/flow/api/trigger-webhook/"

class LarkBot:
    def __init__(self, token: str, base_url: Optional[str]=None):
        self.token = token
        self.base_url = base_url if base_url else BASE_URL
        self.webhook_url = f"{self.base_url.rstrip('/')}/{self.token}"

        self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        self.connector = aiohttp.TCPConnector(limit=5, ssl=self.ssl_context)
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=aiohttp.ClientTimeout(total=5),
            trust_env=True
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def close(self):
        """close ClientSession"""
        if self.session:
            await self.session.close()

    async def send_text(self, content: str) -> Dict[str, Any]:
        payload = {
            "msg_type": "text",
            "content": {
                "text": content
            }
        }
        return await self._send_message(payload)

    async def _send_message(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.session:
            self.session = aiohttp.ClientSession()

        try:
            async with self.session.post(self.webhook_url, json=payload) as response:
                response_data = await response.json()
                if response.status != 200 or response_data.get("code", 0) != 0:
                    print(f"Lark send message failed: {response_data}")
                return response_data
        except Exception as e:
            print(f"Lark send message failed: {e}")
            return {"code": -1, "error": str(e)}

# example
async def main():
    lark_token = os.getenv("LARK_TOKEN")
    if not lark_token:
        print("LARK_TOKEN is not set")
        return
    async with LarkBot(lark_token) as bot:
        text_response = await bot.send_text("This is a test message!")
        print("Text response:", text_response)


if __name__ == "__main__":
    import asyncio
    import dotenv
    dotenv.load_dotenv()
    asyncio.run(main())
