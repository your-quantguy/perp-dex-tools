#!/usr/bin/env python3
"""
Test script for Lark notification functionality
"""

import asyncio
import os
from decimal import Decimal
from helpers.lark_bot import LarkBot
import dotenv

async def test_lark_notifications():
    """Test the new JSON notification system."""
    
    # Load environment variables
    dotenv.load_dotenv()
    
    lark_token = os.getenv("LARK_TOKEN")
    if not lark_token:
        print("LARK_TOKEN is not set in environment variables")
        return
    
    async with LarkBot(lark_token) as bot:
        # Test startup notification
        print("Testing startup notification...")
        startup_data = {
            "quantity": "0.1",
            "take_profit": "0.02",
            "direction": "buy",
            "max_orders": 40,
            "wait_time": 450,
            "grid_step": "-100"
        }
        
        result = await bot.send_notification(
            notification_type="startup",
            instance_id="test_instance",
            exchange="edgex",
            ticker="ETH",
            message="Trading bot started with instance ID: test_instance",
            data=startup_data
        )
        print(f"Startup notification result: {result}")
        
        # Test hourly stats notification
        print("\nTesting hourly stats notification...")
        stats_data = {
            "position_operations": 5,
            "closing_operations": 3,
            "successful_fills": 4,
            "canceled_orders": 1,
            "estimated_fees": "0.005",
            "current_position": "0.5",
            "active_closing_amount": "0.5",
            "hour_period": "2025-09-20 10:00 - 2025-09-20 11:00"
        }
        
        result = await bot.send_notification(
            notification_type="hourly_stats",
            instance_id="test_instance",
            exchange="edgex",
            ticker="ETH",
            message="Hourly trading statistics for instance test_instance",
            data=stats_data
        )
        print(f"Hourly stats notification result: {result}")
        
        # Test error notification
        print("\nTesting error notification...")
        error_data = {
            "current_position": "1.0",
            "active_closing_amount": "0.7",
            "position_difference": "0.3",
            "max_allowed_difference": "0.2"
        }
        
        result = await bot.send_notification(
            notification_type="error",
            instance_id="test_instance",
            exchange="edgex",
            ticker="ETH",
            message="Position mismatch detected! Manual intervention required.",
            data=error_data
        )
        print(f"Error notification result: {result}")

if __name__ == "__main__":
    asyncio.run(test_lark_notifications())
