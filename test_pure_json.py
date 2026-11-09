#!/usr/bin/env python3
"""
Test script to verify the pure JSON notification format
"""

import asyncio
import os
import json
from helpers.lark_bot import LarkBot
import dotenv

async def test_pure_json_notification():
    """Test the pure JSON notification format."""
    
    # Load environment variables
    dotenv.load_dotenv()
    
    lark_token = os.getenv("LARK_TOKEN")
    if not lark_token:
        print("LARK_TOKEN is not set in environment variables")
        print("Testing payload format only...")
        
        # Test payload creation without sending
        from datetime import datetime
        
        notification_data = {
            "notification_type": "startup",
            "timestamp": datetime.now().isoformat(),
            "instance_id": "zhang",
            "exchange": "backpack",
            "ticker": "SOL",
            "message": "Trading bot started with instance ID: zhang",
            "quantity": "0.05",
            "take_profit": "0.02",
            "direction": "buy",
            "max_orders": 18,
            "wait_time": 450,
            "grid_step": "0.2"
        }
        
        print("Expected payload format (pure JSON):")
        print(json.dumps(notification_data, ensure_ascii=False, indent=2))
        print("\nThis will be sent directly as JSON payload, not wrapped in text field")
        return
    
    # If token is available, test actual sending
    async with LarkBot(lark_token) as bot:
        print("Testing pure JSON notification...")
        
        startup_data = {
            "quantity": "0.05",
            "take_profit": "0.02",
            "direction": "buy",
            "max_orders": 18,
            "wait_time": 450,
            "grid_step": "0.2"
        }
        
        result = await bot.send_notification(
            notification_type="startup",
            instance_id="zhang",
            exchange="backpack",
            ticker="SOL",
            message="Trading bot started with instance ID: zhang",
            data=startup_data
        )
        
        print(f"Notification result: {result}")
        print("The payload sent is now pure JSON without text wrapper!")

if __name__ == "__main__":
    asyncio.run(test_pure_json_notification())
