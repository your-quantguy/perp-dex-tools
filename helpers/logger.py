"""
Trading logger with structured output and error handling.
"""

import os
import csv
import logging
import time
import asyncio
from datetime import datetime
import pytz
from decimal import Decimal


class TradingLogger:
    """Enhanced logging with structured output and error handling."""

    def __init__(self, exchange: str, ticker: str, instance_id: str = "default", log_to_console: bool = False):
        self.exchange = exchange
        self.ticker = ticker
        self.instance_id = instance_id
        # Ensure logs directory exists at the project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        logs_dir = os.path.join(project_root, 'logs')
        os.makedirs(logs_dir, exist_ok=True)

        # Determine file naming based on instance_id and ACCOUNT_NAME
        # Priority: instance_id (if not "default") > ACCOUNT_NAME > default naming
        account_name = os.getenv('ACCOUNT_NAME')
        if instance_id != "default":
            # Use instance_id for naming if it's not the default value
            order_file_name = f"{exchange}_{ticker}_{instance_id}_orders.csv"
            debug_log_file_name = f"{exchange}_{ticker}_{instance_id}_activity.log"
        elif account_name:
            # Fall back to ACCOUNT_NAME if instance_id is default
            order_file_name = f"{exchange}_{ticker}_{account_name}_orders.csv"
            debug_log_file_name = f"{exchange}_{ticker}_{account_name}_activity.log"
        else:
            # Default naming
            order_file_name = f"{exchange}_{ticker}_orders.csv"
            debug_log_file_name = f"{exchange}_{ticker}_activity.log"

        # Log file paths inside logs directory
        self.order_file = os.path.join(logs_dir, order_file_name)
        self.debug_log_file = os.path.join(logs_dir, debug_log_file_name)
        self.log_file = self.order_file  # Alias for backward compatibility
        self.timezone = pytz.timezone(os.getenv('TIMEZONE', 'Asia/Shanghai'))
        self.logger = self._setup_logger(log_to_console)
        self.last_lark_notification_time = 0
        self.lark_notification_interval = int(os.getenv('LARK_NOTIFICATION_INTERVAL', 60))  # Default to 60 minutes

    def _setup_logger(self, log_to_console: bool) -> logging.Logger:
        """Setup the logger with proper configuration."""
        logger = logging.getLogger(f"trading_bot_{self.exchange}_{self.ticker}")
        logger.setLevel(logging.INFO)

        # Prevent propagation to root logger to avoid duplicate messages
        logger.propagate = False

        # Prevent duplicate handlers
        if logger.handlers:
            return logger

        class TimeZoneFormatter(logging.Formatter):
            def __init__(self, fmt=None, datefmt=None, tz=None):
                super().__init__(fmt=fmt, datefmt=datefmt)
                self.tz = tz

            def formatTime(self, record, datefmt=None):
                dt = datetime.fromtimestamp(record.created, tz=self.tz)
                if datefmt:
                    return dt.strftime(datefmt)
                return dt.isoformat()

        formatter = TimeZoneFormatter(
            "%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            tz=self.timezone
        )

        # File handler
        file_handler = logging.FileHandler(self.debug_log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler if requested
        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        return logger

    def log(self, message: str, level: str = "INFO"):
        """Log a message with the specified level."""
        formatted_message = f"[{self.exchange.upper()}_{self.ticker.upper()}] {message}"
        log_methods = {
            "DEBUG": self.logger.debug,
            "INFO": self.logger.info,
            "WARNING": self.logger.warning,
            "ERROR": self.logger.error
        }
        log_method = log_methods.get(level.upper(), self.logger.info)
        log_method(formatted_message)

        # Send Lark notification for ERROR level logs
        if level.upper() == "ERROR":
            self._send_lark_notification(formatted_message)

    def _send_lark_notification(self, message: str):
        """Send a Lark notification if the interval has passed."""
        current_time = time.time()
        if current_time - self.last_lark_notification_time > self.lark_notification_interval * 60:
            lark_token = os.getenv("LARK_TOKEN")
            if lark_token:
                # Instead of using asyncio.run(), create a task if we're in an event loop
                try:
                    # Try to get the running event loop
                    loop = asyncio.get_running_loop()
                    # If we're in an event loop, create a task instead of using asyncio.run()
                    task = loop.create_task(self._send_lark_message(lark_token, message))
                    # Add a callback to handle any exceptions in the task
                    task.add_done_callback(self._handle_task_exception)
                except RuntimeError:
                    # If there's no running event loop, use asyncio.run()
                    asyncio.run(self._send_lark_message(lark_token, message))
            self.last_lark_notification_time = current_time

    def _handle_task_exception(self, task):
        """Handle exceptions in async tasks."""
        try:
            task.result()
        except Exception as e:
            self.log(f"Error in Lark notification task: {e}", "ERROR")

    async def _send_lark_message(self, lark_token: str, message: str):
        """Send a formatted message to Lark."""
        from helpers.lark_bot import LarkBot
        # Create a rich text message for Lark
        rich_text_message = {
            "msg_type": "post",
            "content": {
                "post": {
                    "zh_cn": {
                        "title": f"🚨 Trading Bot Error Alert [{self.exchange.upper()}_{self.ticker.upper()}_{self.instance_id}]",
                        "content": [
                            [
                                {
                                    "tag": "text",
                                    "text": message
                                }
                            ],
                            [
                                {
                                    "tag": "at",
                                    "user_id": "all"
                                }
                            ]
                        ]
                    }
                }
            }
        }
        async with LarkBot(lark_token) as bot:
            await bot.send(rich_text_message)

    def log_transaction(self, order_id: str, side: str, quantity: Decimal, price: Decimal, status: str):
        """Log a transaction to CSV file."""
        try:
            timestamp = datetime.now(self.timezone).strftime("%Y-%m-%d %H:%M:%S")
            row = [timestamp, order_id, side, quantity, price, status]

            # Check if file exists to write headers
            file_exists = os.path.isfile(self.log_file)

            with open(self.log_file, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow(['Timestamp', 'OrderID', 'Side', 'Quantity', 'Price', 'Status'])
                writer.writerow(row)

        except Exception as e:
            self.log(f"Failed to log transaction: {e}", "ERROR")
