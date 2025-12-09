import asyncio
import json
import signal
import logging
import os
import sys
import time
import argparse
import traceback
import csv
from decimal import Decimal
from typing import Tuple

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exchanges.grvt import GrvtClient
from datetime import datetime
import pytz
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Simple config class to wrap dictionary for GRVT client."""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


class HedgeBot:
    """Trading bot that places post-only orders on GRVT and hedges with market orders on another GRVT account."""

    def __init__(self, ticker: str, order_quantity: Decimal, fill_timeout: int = 5, iterations: int = 20, sleep_time: int = 0, initial_direction: str = 'buy', trade_type: str = 'single'):
        self.ticker = ticker
        self.order_quantity = order_quantity
        self.fill_timeout = fill_timeout
        self.grvt2_order_filled = False
        self.iterations = iterations
        self.sleep_time = sleep_time
        self.initial_direction = initial_direction
        self.trade_type = trade_type
        self.grvt1_position = Decimal('0')
        self.grvt2_position = Decimal('0')
        self.current_order = {}

        # Initialize logging to file
        os.makedirs("logs", exist_ok=True)
        self.log_filename = f"logs/grvt_grvt_{ticker}_hedge_mode_log.txt"
        self.csv_filename = f"logs/grvt_grvt_{ticker}_hedge_mode_trades.csv"
        self.original_stdout = sys.stdout

        # Initialize CSV file with headers if it doesn't exist
        self._initialize_csv_file()

        # Setup logger
        self.logger = logging.getLogger(f"hedge_bot_{ticker}")
        self.logger.setLevel(logging.INFO)

        # Clear any existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Disable verbose logging from external libraries
        logging.getLogger('urllib3').setLevel(logging.CRITICAL)
        logging.getLogger('requests').setLevel(logging.CRITICAL)
        logging.getLogger('websockets').setLevel(logging.CRITICAL)
        logging.getLogger('pysdk').setLevel(logging.CRITICAL)
        logging.getLogger('pysdk.grvt_ccxt').setLevel(logging.CRITICAL)
        logging.getLogger('pysdk.grvt_ccxt_ws').setLevel(logging.CRITICAL)
        logging.getLogger('pysdk.grvt_ccxt_logging_selector').setLevel(logging.CRITICAL)
        logging.getLogger('pysdk.grvt_ccxt_env').setLevel(logging.CRITICAL)
        
        # Disable root logger propagation to prevent external logs
        logging.getLogger().setLevel(logging.CRITICAL)

        # Create file handler
        file_handler = logging.FileHandler(self.log_filename)
        file_handler.setLevel(logging.INFO)

        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Create different formatters for file and console
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Prevent propagation to root logger to avoid duplicate messages and external logs
        self.logger.propagate = False
        
        # Ensure our logger only shows our messages
        self.logger.setLevel(logging.INFO)

        # State management
        self.stop_flag = False
        self.order_counter = 0

        # GRVT1 state (主账户 - 下限价单)
        self.grvt1_client = None
        self.grvt1_contract_id = None
        self.grvt1_tick_size = None
        self.grvt1_order_status = None

        # GRVT2 state (对冲账户 - 下市价单)
        self.grvt2_client = None
        self.grvt2_contract_id = None
        self.grvt2_tick_size = None
        self.grvt2_order_status = None

        # Order execution tracking
        self.order_execution_complete = False

        # Current order details for immediate execution
        self.current_grvt2_side = None
        self.current_grvt2_quantity = None
        self.current_grvt2_price = None
        self.grvt2_order_info = None

        # GRVT1 configuration (主账户)
        self.grvt1_trading_account_id = os.getenv('GRVT_TRADING_ACCOUNT_ID')
        self.grvt1_private_key = os.getenv('GRVT_PRIVATE_KEY')
        self.grvt1_api_key = os.getenv('GRVT_API_KEY')
        self.grvt1_environment = os.getenv('GRVT_ENVIRONMENT', 'prod')

        # GRVT2 configuration (对冲账户)
        self.grvt2_trading_account_id = os.getenv('GRVT2_TRADING_ACCOUNT_ID')
        self.grvt2_private_key = os.getenv('GRVT2_PRIVATE_KEY')
        self.grvt2_api_key = os.getenv('GRVT2_API_KEY')
        self.grvt2_environment = os.getenv('GRVT2_ENVIRONMENT', 'prod')

        # Strategy state
        self.waiting_for_grvt2_fill = False
        self.wait_start_time = None

    def shutdown(self, signum=None, frame=None):
        """Graceful shutdown handler."""
        self.stop_flag = True
        self.logger.info("\n🛑 Stopping...")

        # Close WebSocket connections
        if self.grvt1_client:
            try:
                self.logger.info("🔌 GRVT1 WebSocket will be disconnected")
            except Exception as e:
                self.logger.error(f"Error disconnecting GRVT1 WebSocket: {e}")

        if self.grvt2_client:
            try:
                self.logger.info("🔌 GRVT2 WebSocket will be disconnected")
            except Exception as e:
                self.logger.error(f"Error disconnecting GRVT2 WebSocket: {e}")

        # Close logging handlers properly
        for handler in self.logger.handlers[:]:
            try:
                handler.close()
                self.logger.removeHandler(handler)
            except Exception:
                pass

    def _initialize_csv_file(self):
        """Initialize CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.csv_filename):
            with open(self.csv_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['exchange', 'timestamp', 'side', 'price', 'quantity'])

    def log_trade_to_csv(self, exchange: str, side: str, price: str, quantity: str):
        """Log trade details to CSV file."""
        timestamp = datetime.now(pytz.UTC).isoformat()

        with open(self.csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                exchange,
                timestamp,
                side,
                price,
                quantity
            ])

        self.logger.info(f"📊 Trade logged to CSV: {exchange} {side} {quantity} @ {price}")

    def handle_grvt2_order_result(self, order_data):
        """Handle GRVT2 order result."""
        try:
            side = order_data.get('side', '')
            filled_size = Decimal(order_data.get('filled_size', '0'))
            price = order_data.get('price', '0')
            
            if side == 'sell':
                order_type = "OPEN"
                self.grvt2_position -= filled_size
            else:
                order_type = "CLOSE"
                self.grvt2_position += filled_size
            
            order_id = order_data.get('order_id', '')

            self.logger.info(f"[{order_id}] [{order_type}] [GRVT2] [FILLED]: "
                             f"{filled_size} @ {price}")

            # Log GRVT2 trade to CSV
            self.log_trade_to_csv(
                exchange='GRVT2',
                side=side.upper(),
                price=str(price),
                quantity=str(filled_size)
            )

            # Mark execution as complete
            self.grvt2_order_filled = True
            self.order_execution_complete = True

        except Exception as e:
            self.logger.error(f"Error handling GRVT2 order result: {e}")

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    def initialize_grvt1_client(self):
        """Initialize the GRVT1 client (主账户)."""
        if not all([self.grvt1_trading_account_id, self.grvt1_private_key, self.grvt1_api_key]):
            raise ValueError("GRVT_TRADING_ACCOUNT_ID, GRVT_PRIVATE_KEY, and GRVT_API_KEY must be set in environment variables")

        # Temporarily set environment variables for GRVT1
        original_env = {}
        for key in ['GRVT_TRADING_ACCOUNT_ID', 'GRVT_PRIVATE_KEY', 'GRVT_API_KEY', 'GRVT_ENVIRONMENT']:
            original_env[key] = os.getenv(key)
        
        os.environ['GRVT_TRADING_ACCOUNT_ID'] = self.grvt1_trading_account_id
        os.environ['GRVT_PRIVATE_KEY'] = self.grvt1_private_key
        os.environ['GRVT_API_KEY'] = self.grvt1_api_key
        os.environ['GRVT_ENVIRONMENT'] = self.grvt1_environment

        # Create config for GRVT1 client
        config_dict = {
            'ticker': self.ticker,
            'contract_id': '',
            'quantity': self.order_quantity,
            'tick_size': Decimal('0.01'),
            'close_order_side': 'sell'
        }

        config = Config(config_dict)
        self.grvt1_client = GrvtClient(config)

        # Restore original environment
        for key, value in original_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]

        self.logger.info("✅ GRVT1 client initialized successfully")
        return self.grvt1_client

    def initialize_grvt2_client(self):
        """Initialize the GRVT2 client (对冲账户)."""
        if not all([self.grvt2_trading_account_id, self.grvt2_private_key, self.grvt2_api_key]):
            raise ValueError("GRVT2_TRADING_ACCOUNT_ID, GRVT2_PRIVATE_KEY, and GRVT2_API_KEY must be set in environment variables")

        # Temporarily set environment variables for GRVT2
        original_env = {}
        for key in ['GRVT_TRADING_ACCOUNT_ID', 'GRVT_PRIVATE_KEY', 'GRVT_API_KEY', 'GRVT_ENVIRONMENT']:
            original_env[key] = os.getenv(key)
        
        os.environ['GRVT_TRADING_ACCOUNT_ID'] = self.grvt2_trading_account_id
        os.environ['GRVT_PRIVATE_KEY'] = self.grvt2_private_key
        os.environ['GRVT_API_KEY'] = self.grvt2_api_key
        os.environ['GRVT_ENVIRONMENT'] = self.grvt2_environment

        # Create config for GRVT2 client
        config_dict = {
            'ticker': self.ticker,
            'contract_id': '',
            'quantity': self.order_quantity,
            'tick_size': Decimal('0.01'),
            'close_order_side': 'buy'  # 对冲方向相反
        }

        config = Config(config_dict)
        self.grvt2_client = GrvtClient(config)

        # Restore original environment
        for key, value in original_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]

        self.logger.info("✅ GRVT2 client initialized successfully")
        return self.grvt2_client

    async def get_grvt1_contract_info(self) -> Tuple[str, Decimal]:
        """Get GRVT1 contract ID and tick size."""
        if not self.grvt1_client:
            raise Exception("GRVT1 client not initialized")

        contract_id, tick_size = await self.grvt1_client.get_contract_attributes()

        if self.order_quantity < self.grvt1_client.config.quantity:
            raise ValueError(
                f"Order quantity is less than min quantity: {self.order_quantity} < {self.grvt1_client.config.quantity}")

        return contract_id, tick_size

    async def get_grvt2_contract_info(self) -> Tuple[str, Decimal]:
        """Get GRVT2 contract ID and tick size."""
        if not self.grvt2_client:
            raise Exception("GRVT2 client not initialized")

        contract_id, tick_size = await self.grvt2_client.get_contract_attributes()

        if self.order_quantity < self.grvt2_client.config.quantity:
            raise ValueError(
                f"Order quantity is less than min quantity: {self.order_quantity} < {self.grvt2_client.config.quantity}")

        return contract_id, tick_size

    async def fetch_grvt1_bbo_prices(self) -> Tuple[Decimal, Decimal]:
        """Fetch best bid/ask prices from GRVT1 using REST API."""
        if not self.grvt1_client:
            raise Exception("GRVT1 client not initialized")

        best_bid, best_ask = await self.grvt1_client.fetch_bbo_prices(self.grvt1_contract_id)

        return best_bid, best_ask

    def round_to_tick(self, price: Decimal, tick_size: Decimal) -> Decimal:
        """Round price to tick size."""
        if tick_size is None:
            return price
        return (price / tick_size).quantize(Decimal('1')) * tick_size

    async def place_bbo_order(self, side: str, quantity: Decimal):
        """Place BBO order on GRVT1."""
        order_result = await self.grvt1_client.place_open_order(
            contract_id=self.grvt1_contract_id,
            quantity=quantity,
            direction=side.lower()
        )

        if order_result.success:
            return order_result.order_id, order_result.price
        else:
            raise Exception(f"Failed to place order: {order_result.error_message}")

    async def place_grvt1_post_only_order(self, side: str, quantity: Decimal):
        """Place a post-only order on GRVT1."""
        if not self.grvt1_client:
            raise Exception("GRVT1 client not initialized")

        self.grvt1_order_status = None
        self.logger.info(f"[OPEN] [GRVT1] [{side}] Placing GRVT1 POST-ONLY order")
        order_id, order_price = await self.place_bbo_order(side, quantity)

        start_time = time.time()
        while not self.stop_flag:
            if self.grvt1_order_status == 'CANCELED':
                self.grvt1_order_status = 'NEW'
                order_id, order_price = await self.place_bbo_order(side, quantity)
                start_time = time.time()
                await asyncio.sleep(0.5)
            elif self.grvt1_order_status in ['NEW', 'OPEN', 'PENDING', 'CANCELING', 'PARTIALLY_FILLED']:
                await asyncio.sleep(0.5)
                # Check if we need to cancel and replace the order
                should_cancel = False
                best_bid, best_ask = await self.fetch_grvt1_bbo_prices()
                if side == 'buy':
                    if order_price < best_bid:
                        should_cancel = True
                else:
                    if order_price > best_ask:
                        should_cancel = True
                if time.time() - start_time > 10:
                    if should_cancel:
                        try:
                            cancel_result = await self.grvt1_client.cancel_order(order_id)
                            if not cancel_result.success:
                                self.logger.error(f"❌ Error canceling GRVT1 order: {cancel_result.error_message}")
                        except Exception as e:
                            self.logger.error(f"❌ Error canceling GRVT1 order: {e}")
                    else:
                        self.logger.info(f"Order {order_id} is at best bid/ask, waiting for fill")
                        start_time = time.time()
            elif self.grvt1_order_status == 'FILLED':
                break
            else:
                if self.grvt1_order_status is not None:
                    self.logger.error(f"❌ Unknown GRVT1 order status: {self.grvt1_order_status}")
                    break
                else:
                    await asyncio.sleep(0.5)

    def handle_grvt1_order_update(self, order_data):
        """Handle GRVT1 order updates from WebSocket."""
        side = order_data.get('side', '').lower()
        filled_size = Decimal(order_data.get('filled_size', '0'))
        price = Decimal(order_data.get('price', '0'))

        # 更新 GRVT1 仓位
        if side == 'buy':
            self.grvt1_position += filled_size
            grvt2_side = 'sell'
        else:
            self.grvt1_position -= filled_size
            grvt2_side = 'buy'

        # Store order details for immediate execution
        self.current_grvt2_side = grvt2_side
        self.current_grvt2_quantity = filled_size
        self.current_grvt2_price = price

        self.grvt2_order_info = {
            'grvt2_side': grvt2_side,
            'quantity': filled_size,
            'price': price
        }

        self.waiting_for_grvt2_fill = True

    async def place_grvt2_market_order(self, grvt2_side: str, quantity: Decimal, price: Decimal):
        """Place market order on GRVT2 for hedging."""
        if not self.grvt2_client:
            raise Exception("GRVT2 client not initialized")

        # 根据方向确定订单类型
        if grvt2_side.lower() == 'buy':
            order_type = "CLOSE"
        else:
            order_type = "OPEN"

        # Reset order state
        self.grvt2_order_filled = False

        try:
            self.logger.info(f"[{order_type}] [GRVT2] [OPEN]: Placing market order {grvt2_side} {quantity}")

            # 使用 GRVT SDK 下市价单
            # TODO: 需要实现市价单方法，目前使用限价单模拟
            # 获取当前最优价格
            best_bid, best_ask = await self.grvt2_client.fetch_bbo_prices(self.grvt2_contract_id)
            
            # 根据方向设置价格，确保能够成交
            if grvt2_side.lower() == 'buy':
                # 买单设置为高于当前卖一价
                market_price = best_ask * Decimal('1.01')
            else:
                # 卖单设置为低于当前买一价
                market_price = best_bid * Decimal('0.99')
            
            market_price = self.round_to_tick(market_price, self.grvt2_tick_size)
            
            # 下限价单，但价格设置为能立即成交
            order_result = self.grvt2_client.rest_client.create_limit_order(
                symbol=self.grvt2_contract_id,
                side=grvt2_side.lower(),
                amount=quantity,
                price=market_price,
                params={
                    'post_only': False,  # 允许吃单
                    'order_duration_secs': 30 * 86400 - 1,
                }
            )

            if not order_result:
                raise Exception(f"[{order_type}] Error placing GRVT2 market order")

            order_id = order_result.get('order_id', '')
            self.logger.info(f"[{order_id}] [{order_type}] [GRVT2] Market order placed: {quantity} @ {market_price}")

            # 监控订单状态
            await self.monitor_grvt2_order(order_id)

            return order_id

        except Exception as e:
            self.logger.error(f"❌ Error placing GRVT2 market order: {e}")
            self.logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            return None

    async def monitor_grvt2_order(self, order_id: str):
        """Monitor GRVT2 order and wait for fill."""
        start_time = time.time()
        while not self.grvt2_order_filled and not self.stop_flag:
            # Check for timeout (30 seconds total)
            if time.time() - start_time > 30:
                self.logger.error(f"❌ Timeout waiting for GRVT2 order fill after {time.time() - start_time:.1f}s")
                self.logger.error(f"❌ Order state - Filled: {self.grvt2_order_filled}")

                # Fallback: 手动更新仓位并标记为已成交
                self.logger.warning("⚠️ Using fallback - manually updating position and marking as filled")
                
                # 从 current_grvt2_side 和 current_grvt2_quantity 更新仓位
                if self.current_grvt2_side and self.current_grvt2_quantity:
                    if self.current_grvt2_side.lower() == 'buy':
                        self.grvt2_position += self.current_grvt2_quantity
                    else:
                        self.grvt2_position -= self.current_grvt2_quantity
                    self.logger.warning(f"⚠️ Fallback position update: GRVT2 {self.current_grvt2_side} {self.current_grvt2_quantity}, new position: {self.grvt2_position}")
                
                self.grvt2_order_filled = True
                self.waiting_for_grvt2_fill = False
                self.order_execution_complete = True
                break

            await asyncio.sleep(0.1)

    async def setup_grvt1_websocket(self):
        """Setup GRVT1 websocket for order updates."""
        if not self.grvt1_client:
            raise Exception("GRVT1 client not initialized")

        def order_update_handler(order_data):
            """Handle order updates from GRVT1 WebSocket."""
            if order_data.get('contract_id') != self.grvt1_contract_id:
                return
            try:
                order_id = order_data.get('order_id')
                status = order_data.get('status')
                side = order_data.get('side', '').lower()
                filled_size = Decimal(order_data.get('filled_size', '0'))
                size = Decimal(order_data.get('size', '0'))
                price = order_data.get('price', '0')

                if side == 'buy':
                    order_type = "OPEN"
                else:
                    order_type = "CLOSE"
                
                if status == 'CANCELED' and filled_size > 0:
                    status = 'FILLED'

                # Handle the order update
                if status == 'FILLED' and self.grvt1_order_status != 'FILLED':
                    # 仓位更新在 handle_grvt1_order_update 中进行，避免重复
                    self.logger.info(f"[{order_id}] [{order_type}] [GRVT1] [{status}]: {filled_size} @ {price}")
                    self.grvt1_order_status = status

                    # Log GRVT1 trade to CSV
                    self.log_trade_to_csv(
                        exchange='GRVT1',
                        side=side,
                        price=str(price),
                        quantity=str(filled_size)
                    )

                    self.handle_grvt1_order_update({
                        'order_id': order_id,
                        'side': side,
                        'status': status,
                        'size': size,
                        'price': price,
                        'contract_id': self.grvt1_contract_id,
                        'filled_size': filled_size
                    })
                elif self.grvt1_order_status != 'FILLED':
                    if status == 'OPEN':
                        self.logger.info(f"[{order_id}] [{order_type}] [GRVT1] [{status}]: {size} @ {price}")
                    else:
                        self.logger.info(f"[{order_id}] [{order_type}] [GRVT1] [{status}]: {filled_size} @ {price}")
                    self.grvt1_order_status = status

            except Exception as e:
                self.logger.error(f"Error handling GRVT1 order update: {e}")

        try:
            self.grvt1_client.setup_order_update_handler(order_update_handler)
            self.logger.info("✅ GRVT1 WebSocket order update handler set up")

            await self.grvt1_client.connect()
            self.logger.info("✅ GRVT1 WebSocket connection established")

        except Exception as e:
            self.logger.error(f"Could not setup GRVT1 WebSocket handlers: {e}")

    async def setup_grvt2_websocket(self):
        """Setup GRVT2 websocket for order updates."""
        if not self.grvt2_client:
            raise Exception("GRVT2 client not initialized")

        def order_update_handler(order_data):
            """Handle order updates from GRVT2 WebSocket."""
            if order_data.get('contract_id') != self.grvt2_contract_id:
                return
            try:
                order_id = order_data.get('order_id')
                status = order_data.get('status')
                side = order_data.get('side', '').lower()
                filled_size = Decimal(order_data.get('filled_size', '0'))
                size = Decimal(order_data.get('size', '0'))
                price = order_data.get('price', '0')

                if side == 'buy':
                    order_type = "CLOSE"
                else:
                    order_type = "OPEN"
                
                if status == 'CANCELED' and filled_size > 0:
                    status = 'FILLED'

                # Handle the order update
                if status == 'FILLED' and self.grvt2_order_status != 'FILLED':
                    # 仓位更新在 handle_grvt2_order_result 中进行，避免重复
                    self.logger.info(f"[{order_id}] [{order_type}] [GRVT2] [{status}]: {filled_size} @ {price}")
                    self.grvt2_order_status = status

                    # Log GRVT2 trade to CSV
                    self.log_trade_to_csv(
                        exchange='GRVT2',
                        side=side,
                        price=str(price),
                        quantity=str(filled_size)
                    )

                    self.handle_grvt2_order_result({
                        'order_id': order_id,
                        'side': side,
                        'status': status,
                        'size': size,
                        'price': price,
                        'contract_id': self.grvt2_contract_id,
                        'filled_size': filled_size
                    })
                elif self.grvt2_order_status != 'FILLED':
                    if status == 'OPEN':
                        self.logger.info(f"[{order_id}] [{order_type}] [GRVT2] [{status}]: {size} @ {price}")
                    else:
                        self.logger.info(f"[{order_id}] [{order_type}] [GRVT2] [{status}]: {filled_size} @ {price}")
                    self.grvt2_order_status = status

            except Exception as e:
                self.logger.error(f"Error handling GRVT2 order update: {e}")

        try:
            self.grvt2_client.setup_order_update_handler(order_update_handler)
            self.logger.info("✅ GRVT2 WebSocket order update handler set up")

            await self.grvt2_client.connect()
            self.logger.info("✅ GRVT2 WebSocket connection established")

        except Exception as e:
            self.logger.error(f"Could not setup GRVT2 WebSocket handlers: {e}")

    async def trading_loop(self):
        """Main trading loop implementing the hedge strategy."""
        self.logger.info(f"🚀 Starting hedge bot for {self.ticker}")

        # Initialize clients
        try:
            self.initialize_grvt1_client()
            self.initialize_grvt2_client()

            # Get contract info
            self.grvt1_contract_id, self.grvt1_tick_size = await self.get_grvt1_contract_info()
            self.grvt2_contract_id, self.grvt2_tick_size = await self.get_grvt2_contract_info()

            self.logger.info(f"Contract info loaded - GRVT1: {self.grvt1_contract_id}, "
                             f"GRVT2: {self.grvt2_contract_id}")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize: {e}")
            return

        # Setup GRVT1 websocket
        try:
            await self.setup_grvt1_websocket()
            self.logger.info("✅ GRVT1 WebSocket connection established")

        except Exception as e:
            self.logger.error(f"❌ Failed to setup GRVT1 websocket: {e}")
            return

        # Setup GRVT2 websocket
        try:
            await self.setup_grvt2_websocket()
            self.logger.info("✅ GRVT2 WebSocket connection established")

        except Exception as e:
            self.logger.error(f"❌ Failed to setup GRVT2 websocket: {e}")
            return

        await asyncio.sleep(5)

        # 获取初始持仓并更新本地缓存
        try:
            self.logger.info("📊 Fetching initial positions...")
            self.grvt1_position = await self.grvt1_client.get_real_position()
            self.grvt2_position = await self.grvt2_client.get_real_position()
            self.logger.info(f"✅ Initial positions - GRVT1: {self.grvt1_position}, GRVT2: {self.grvt2_position}")
        except Exception as e:
            self.logger.error(f"❌ Failed to get initial positions: {e}")
            self.logger.warning(f"⚠️ Continuing with default positions (0, 0)")

        iterations = 0
        while iterations < self.iterations and not self.stop_flag:
            iterations += 1
            self.logger.info("-----------------------------------------------")
            self.logger.info(f"🔄 Trading loop iteration {iterations}")
            self.logger.info("-----------------------------------------------")

            self.logger.info(f"[STEP 1] GRVT1 position: {self.grvt1_position} | GRVT2 position: {self.grvt2_position}")

            if abs(self.grvt1_position + self.grvt2_position) > self.order_quantity * 2:
                self.logger.error(f"❌ Position diff is too large: {self.grvt1_position + self.grvt2_position}")
                break

            self.order_execution_complete = False
            self.waiting_for_grvt2_fill = False
            try:
                # Open position
                side = self.initial_direction
                await self.place_grvt1_post_only_order(side, self.order_quantity)
            except Exception as e:
                self.logger.error(f"⚠️ Error in trading loop: {e}")
                self.logger.error(f"⚠️ Full traceback: {traceback.format_exc()}")
                break

            start_time = time.time()
            while not self.order_execution_complete and not self.stop_flag:
                # Check if GRVT1 order filled and we need to place GRVT2 order
                if self.waiting_for_grvt2_fill:
                    await self.place_grvt2_market_order(
                        self.current_grvt2_side,
                        self.current_grvt2_quantity,
                        self.current_grvt2_price
                    )
                    break

                await asyncio.sleep(0.01)
                if time.time() - start_time > 180:
                    self.logger.error("❌ Timeout waiting for trade completion")
                    break

            if self.stop_flag:
                break

            # Sleep after step 1
            if self.sleep_time > 0:
                self.logger.info(f"💤 Sleeping {self.sleep_time} seconds after STEP 1...")
                await asyncio.sleep(self.sleep_time)

            # 如果是 single 模式，跳过 STEP 2 和 STEP 3
            if self.trade_type == 'single':
                self.logger.info(f"[SINGLE MODE] Skipping close position steps")
                continue

            # Close position (仅在 twice 模式执行)
            self.logger.info(f"[STEP 2] GRVT1 position: {self.grvt1_position} | GRVT2 position: {self.grvt2_position}")
            self.order_execution_complete = False
            self.waiting_for_grvt2_fill = False
            try:
                side = 'buy' if self.initial_direction == 'sell' else 'sell'
                await self.place_grvt1_post_only_order(side, self.order_quantity)
            except Exception as e:
                self.logger.error(f"⚠️ Error in trading loop: {e}")
                self.logger.error(f"⚠️ Full traceback: {traceback.format_exc()}")
                break

            start_time = time.time()
            while not self.order_execution_complete and not self.stop_flag:
                # Check if GRVT1 order filled and we need to place GRVT2 order
                if self.waiting_for_grvt2_fill:
                    await self.place_grvt2_market_order(
                        self.current_grvt2_side,
                        self.current_grvt2_quantity,
                        self.current_grvt2_price
                    )
                    break

                await asyncio.sleep(0.01)
                if time.time() - start_time > 180:
                    self.logger.error("❌ Timeout waiting for trade completion")
                    break

            # Close remaining position
            self.logger.info(f"[STEP 3] GRVT1 position: {self.grvt1_position} | GRVT2 position: {self.grvt2_position}")
            self.order_execution_complete = False
            self.waiting_for_grvt2_fill = False
            if self.grvt1_position == 0:
                continue
            elif self.grvt1_position > 0:
                side = 'sell'
            else:
                side = 'buy'

            try:
                await self.place_grvt1_post_only_order(side, abs(self.grvt1_position))
            except Exception as e:
                self.logger.error(f"⚠️ Error in trading loop: {e}")
                self.logger.error(f"⚠️ Full traceback: {traceback.format_exc()}")
                break

            start_time = time.time()
            while not self.order_execution_complete and not self.stop_flag:
                # Check if GRVT1 order filled and we need to place GRVT2 order
                if self.waiting_for_grvt2_fill:
                    await self.place_grvt2_market_order(
                        self.current_grvt2_side,
                        self.current_grvt2_quantity,
                        self.current_grvt2_price
                    )
                    break

                await asyncio.sleep(0.01)
                if time.time() - start_time > 180:
                    self.logger.error("❌ Timeout waiting for trade completion")
                    break

    async def run(self):
        """Run the hedge bot."""
        self.setup_signal_handlers()

        try:
            await self.trading_loop()
        except KeyboardInterrupt:
            self.logger.info("\n🛑 Received interrupt signal...")
        finally:
            self.logger.info("🔄 Cleaning up...")
            self.shutdown()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Trading bot for GRVT to GRVT hedge')
    parser.add_argument('--ticker', type=str, default='BTC',
                        help='Ticker symbol (default: BTC)')
    parser.add_argument('--size', type=str,
                        help='Number of tokens to buy/sell per order')
    parser.add_argument('--iter', type=int,
                        help='Number of iterations to run')
    parser.add_argument('--fill-timeout', type=int, default=5,
                        help='Timeout in seconds for maker order fills (default: 5)')
    parser.add_argument('--sleep', type=int, default=0,
                        help='Sleep time in seconds after each step (default: 0)')
    parser.add_argument('--direction', type=str, default='buy', choices=['buy', 'sell'],
                        help='Initial direction for STEP 1: buy or sell (default: buy)')
    parser.add_argument('--type', type=str, default='single', choices=['single', 'twice'],
                        help='Trade type: single (open only) or twice (open then close) (default: single)')

    return parser.parse_args()


async def main():
    """Main function."""
    args = parse_arguments()

    # Validate required arguments
    if not args.size:
        print("Error: --size is required")
        return

    if not args.iter:
        print("Error: --iter is required")
        return

    # Create and run the bot
    bot = HedgeBot(
        ticker=args.ticker,
        order_quantity=Decimal(args.size),
        fill_timeout=args.fill_timeout,
        iterations=args.iter,
        sleep_time=args.sleep,
        initial_direction=args.direction,
        trade_type=args.type
    )

    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
