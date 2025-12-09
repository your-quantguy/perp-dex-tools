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
from exchanges.aster import AsterClient
from datetime import datetime
import pytz
from dotenv import load_dotenv

load_dotenv('.grvt_aster_env')

class Config:
    """Simple config class to wrap dictionary for exchange clients."""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


class HedgeBot:
    """Trading bot that places post-only orders on GRVT and hedges with market orders on Aster."""

    def __init__(self, ticker: str, order_quantity: Decimal, fill_timeout: int = 5, iterations: int = 20, sleep_time: int = 0, initial_direction: str = 'buy', trade_type: str = 'single', open_rate: Decimal = Decimal('0.001'), close_rate: Decimal = Decimal('-0.001'), max_size: Decimal = Decimal('0')):
        self.ticker = ticker
        self.order_quantity = order_quantity
        self.fill_timeout = fill_timeout
        self.aster_order_filled = False
        self.iterations = iterations
        self.sleep_time = sleep_time
        self.initial_direction = initial_direction
        self.trade_type = trade_type
        self.open_rate = open_rate
        self.close_rate = close_rate
        self.max_size = max_size
        self.grvt_position = Decimal('0')
        self.aster_position = Decimal('0')
        self.current_order = {}

        # Initialize logging to file
        os.makedirs("logs", exist_ok=True)
        self.log_filename = f"logs/grvt_aster_{ticker}_hedge_mode_log.txt"
        self.csv_filename = f"logs/grvt_aster_{ticker}_hedge_mode_trades.csv"
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

        # GRVT state (主账户 - 下限价单)
        self.grvt_client = None
        self.grvt_contract_id = None
        self.grvt_tick_size = None
        self.grvt_order_status = None

        # Aster state (对冲账户 - 下市价单)
        self.aster_client = None
        self.aster_contract_id = None
        self.aster_tick_size = None
        self.aster_order_status = None

        # Order execution tracking
        self.order_execution_complete = False

        # Current order details for immediate execution
        self.current_aster_side = None
        self.current_aster_quantity = None
        self.current_aster_price = None
        self.aster_order_info = None

        # GRVT configuration
        self.grvt_trading_account_id = os.getenv('GRVT_TRADING_ACCOUNT_ID')
        self.grvt_private_key = os.getenv('GRVT_PRIVATE_KEY')
        self.grvt_api_key = os.getenv('GRVT_API_KEY')
        self.grvt_environment = os.getenv('GRVT_ENVIRONMENT', 'prod')

        # Aster configuration
        self.aster_api_key = os.getenv('ASTER_API_KEY')
        self.aster_secret_key = os.getenv('ASTER_SECRET_KEY')

        # Strategy state
        self.waiting_for_aster_fill = False
        self.wait_start_time = None

    def shutdown(self, signum=None, frame=None):
        """Graceful shutdown handler."""
        self.stop_flag = True
        self.logger.info("\n🛑 Stopping...")

        # Close WebSocket connections
        if self.grvt_client:
            try:
                self.logger.info("🔌 GRVT WebSocket will be disconnected")
            except Exception as e:
                self.logger.error(f"Error disconnecting GRVT WebSocket: {e}")

        if self.aster_client:
            try:
                self.logger.info("🔌 Aster WebSocket will be disconnected")
            except Exception as e:
                self.logger.error(f"Error disconnecting Aster WebSocket: {e}")

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

    def handle_aster_order_result(self, order_data):
        """Handle Aster order result."""
        try:
            side = order_data.get('side', '')
            filled_size = Decimal(order_data.get('filled_size', '0'))
            price = order_data.get('price', '0')
            
            if side == 'sell':
                order_type = "OPEN"
                self.aster_position -= filled_size
            else:
                order_type = "CLOSE"
                self.aster_position += filled_size
            
            order_id = order_data.get('order_id', '')

            self.logger.info(f"[{order_id}] [{order_type}] [ASTER] [FILLED]: "
                             f"{filled_size} @ {price}")

            # Log Aster trade to CSV
            self.log_trade_to_csv(
                exchange='ASTER',
                side=side.upper(),
                price=str(price),
                quantity=str(filled_size)
            )

            # Mark execution as complete
            self.aster_order_filled = True
            self.order_execution_complete = True

        except Exception as e:
            self.logger.error(f"Error handling Aster order result: {e}")

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    def initialize_grvt_client(self):
        """Initialize the GRVT client."""
        if not all([self.grvt_trading_account_id, self.grvt_private_key, self.grvt_api_key]):
            raise ValueError("GRVT_TRADING_ACCOUNT_ID, GRVT_PRIVATE_KEY, and GRVT_API_KEY must be set in environment variables")

        # Temporarily set environment variables for GRVT
        original_env = {}
        for key in ['GRVT_TRADING_ACCOUNT_ID', 'GRVT_PRIVATE_KEY', 'GRVT_API_KEY', 'GRVT_ENVIRONMENT']:
            original_env[key] = os.getenv(key)
        
        os.environ['GRVT_TRADING_ACCOUNT_ID'] = self.grvt_trading_account_id
        os.environ['GRVT_PRIVATE_KEY'] = self.grvt_private_key
        os.environ['GRVT_API_KEY'] = self.grvt_api_key
        os.environ['GRVT_ENVIRONMENT'] = self.grvt_environment

        # Create config for GRVT client
        config_dict = {
            'ticker': self.ticker,
            'contract_id': '',
            'quantity': self.order_quantity,
            'tick_size': Decimal('0.01'),
            'close_order_side': 'sell'
        }

        config = Config(config_dict)
        self.grvt_client = GrvtClient(config)

        # Restore original environment
        for key, value in original_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]

        self.logger.info("✅ GRVT client initialized successfully")
        return self.grvt_client

    def initialize_aster_client(self):
        """Initialize the Aster client."""
        if not all([self.aster_api_key, self.aster_secret_key]):
            raise ValueError("ASTER_API_KEY and ASTER_SECRET_KEY must be set in environment variables")

        # Create config for Aster client
        # Aster 使用 ticker + USDT 格式，如 ETHUSDT
        aster_ticker = self.ticker  # Aster 使用基础币种名称
        
        config_dict = {
            'ticker': aster_ticker,
            'contract_id': '',
            'quantity': self.order_quantity,
            'tick_size': Decimal('0.01'),
            'close_order_side': 'buy',  # 对冲方向相反
            'direction': 'sell'  # Aster 需要 direction 参数
        }

        config = Config(config_dict)
        self.aster_client = AsterClient(config)

        self.logger.info("✅ Aster client initialized successfully")
        return self.aster_client

    async def get_grvt_contract_info(self) -> Tuple[str, Decimal]:
        """Get GRVT contract ID and tick size."""
        if not self.grvt_client:
            raise Exception("GRVT client not initialized")

        contract_id, tick_size = await self.grvt_client.get_contract_attributes()

        if self.order_quantity < self.grvt_client.config.quantity:
            raise ValueError(
                f"Order quantity is less than min quantity: {self.order_quantity} < {self.grvt_client.config.quantity}")

        return contract_id, tick_size

    async def get_aster_contract_info(self) -> Tuple[str, Decimal]:
        """Get Aster contract ID and tick size."""
        if not self.aster_client:
            raise Exception("Aster client not initialized")

        contract_id, tick_size = await self.aster_client.get_contract_attributes()

        if self.order_quantity < self.aster_client.config.quantity:
            raise ValueError(
                f"Order quantity is less than min quantity: {self.order_quantity} < {self.aster_client.config.quantity}")

        return contract_id, tick_size

    async def fetch_grvt_bbo_prices(self) -> Tuple[Decimal, Decimal]:
        """Fetch best bid/ask prices from GRVT using REST API."""
        if not self.grvt_client:
            raise Exception("GRVT client not initialized")

        best_bid, best_ask = await self.grvt_client.fetch_bbo_prices(self.grvt_contract_id)

        return best_bid, best_ask

    async def fetch_aster_bbo_prices(self) -> Tuple[Decimal, Decimal]:
        """Fetch best bid/ask prices from Aster using REST API."""
        if not self.aster_client:
            raise Exception("Aster client not initialized")

        best_bid, best_ask = await self.aster_client.fetch_bbo_prices(self.aster_contract_id)

        return best_bid, best_ask

    async def calculate_spreads(self) -> Tuple[Decimal, Decimal]:
        """Calculate open_spread and close_spread.
        
        Returns:
            Tuple[Decimal, Decimal]: (open_spread, close_spread)
            open_spread = (grvt_ask - aster_ask) / grvt_ask
            close_spread = (grvt_bid - aster_bid) / grvt_bid
        """
        grvt_bid, grvt_ask = await self.fetch_grvt_bbo_prices()
        aster_bid, aster_ask = await self.fetch_aster_bbo_prices()

        if grvt_ask == 0 or grvt_bid == 0:
            raise Exception("Invalid GRVT prices")

        open_spread = (grvt_ask - aster_ask) / grvt_ask
        close_spread = (grvt_bid - aster_bid) / grvt_bid

        return open_spread, close_spread

    def round_to_tick(self, price: Decimal, tick_size: Decimal) -> Decimal:
        """Round price to tick size."""
        if tick_size is None:
            return price
        return (price / tick_size).quantize(Decimal('1')) * tick_size

    async def place_bbo_order(self, side: str, quantity: Decimal):
        """Place BBO order on GRVT."""
        order_result = await self.grvt_client.place_open_order(
            contract_id=self.grvt_contract_id,
            quantity=quantity,
            direction=side.lower()
        )

        if order_result.success:
            return order_result.order_id, order_result.price
        else:
            raise Exception(f"Failed to place order: {order_result.error_message}")

    async def place_grvt_post_only_order(self, side: str, quantity: Decimal):
        """Place a post-only order on GRVT."""
        if not self.grvt_client:
            raise Exception("GRVT client not initialized")

        self.grvt_order_status = None
        self.logger.info(f"[OPEN] [GRVT] [{side}] Placing GRVT POST-ONLY order")
        order_id, order_price = await self.place_bbo_order(side, quantity)

        start_time = time.time()
        while not self.stop_flag:
            if self.grvt_order_status == 'CANCELED':
                self.grvt_order_status = 'NEW'
                order_id, order_price = await self.place_bbo_order(side, quantity)
                start_time = time.time()
                await asyncio.sleep(0.5)
            elif self.grvt_order_status in ['NEW', 'OPEN', 'PENDING', 'CANCELING', 'PARTIALLY_FILLED']:
                await asyncio.sleep(0.5)
                # Check if we need to cancel and replace the order
                should_cancel = False
                best_bid, best_ask = await self.fetch_grvt_bbo_prices()
                if side == 'buy':
                    if order_price < best_bid:
                        should_cancel = True
                else:
                    if order_price > best_ask:
                        should_cancel = True
                if time.time() - start_time > 10:
                    if should_cancel:
                        try:
                            cancel_result = await self.grvt_client.cancel_order(order_id)
                            if not cancel_result.success:
                                self.logger.error(f"❌ Error canceling GRVT order: {cancel_result.error_message}")
                        except Exception as e:
                            self.logger.error(f"❌ Error canceling GRVT order: {e}")
                    else:
                        self.logger.info(f"Order {order_id} is at best bid/ask, waiting for fill")
                        start_time = time.time()
            elif self.grvt_order_status == 'FILLED':
                break
            else:
                if self.grvt_order_status is not None:
                    self.logger.error(f"❌ Unknown GRVT order status: {self.grvt_order_status}")
                    break
                else:
                    await asyncio.sleep(0.5)

    def handle_grvt_order_update(self, order_data):
        """Handle GRVT order updates from WebSocket."""
        side = order_data.get('side', '').lower()
        filled_size = Decimal(order_data.get('filled_size', '0'))
        price = Decimal(order_data.get('price', '0'))

        # 更新 GRVT 仓位
        if side == 'buy':
            self.grvt_position += filled_size
            aster_side = 'sell'
        else:
            self.grvt_position -= filled_size
            aster_side = 'buy'

        # Store order details for immediate execution
        self.current_aster_side = aster_side
        self.current_aster_quantity = filled_size
        self.current_aster_price = price

        self.aster_order_info = {
            'aster_side': aster_side,
            'quantity': filled_size,
            'price': price
        }

        self.waiting_for_aster_fill = True

    async def place_aster_market_order(self, aster_side: str, quantity: Decimal, price: Decimal):
        """Place market order on Aster for hedging."""
        if not self.aster_client:
            raise Exception("Aster client not initialized")

        # 根据方向确定订单类型
        if aster_side.lower() == 'buy':
            order_type = "CLOSE"
        else:
            order_type = "OPEN"

        # Reset order state
        self.aster_order_filled = False

        try:
            self.logger.info(f"[{order_type}] [ASTER] [OPEN]: Placing market order {aster_side} {quantity}")

            # 使用 Aster REST API 下市价单
            # 获取当前最优价格
            best_bid, best_ask = await self.aster_client.fetch_bbo_prices(self.aster_contract_id)
            
            # 根据方向设置价格，确保能够成交
            if aster_side.lower() == 'buy':
                # 买单设置为高于当前卖一价
                market_price = best_ask * Decimal('1.001')
            else:
                # 卖单设置为低于当前买一价
                market_price = best_bid * Decimal('0.999')
            
            market_price = self.round_to_tick(market_price, self.aster_tick_size)
            
            # Aster 使用 _make_request 下单
            order_data = {
                'symbol': self.aster_contract_id,
                'side': aster_side.upper(),
                'type': 'LIMIT',
                'quantity': str(quantity),
                'price': str(market_price),
                'timeInForce': 'GTC'  # Good Till Cancel，允许吃单
            }

            order_result = await self.aster_client._make_request('POST', '/fapi/v1/order', data=order_data)

            if not order_result:
                raise Exception(f"[{order_type}] Error placing Aster market order")

            order_id = str(order_result.get('orderId', ''))
            self.logger.info(f"[{order_id}] [{order_type}] [ASTER] Market order placed: {quantity} @ {market_price}")

            # 监控订单状态
            await self.monitor_aster_order(order_id)

            return order_id

        except Exception as e:
            self.logger.error(f"❌ Error placing Aster market order: {e}")
            self.logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            return None

    async def monitor_aster_order(self, order_id: str):
        """Monitor Aster order and wait for fill."""
        start_time = time.time()
        while not self.aster_order_filled and not self.stop_flag:
            # Check for timeout (30 seconds total)
            if time.time() - start_time > 30:
                self.logger.error(f"❌ Timeout waiting for Aster order fill after {time.time() - start_time:.1f}s")
                self.logger.error(f"❌ Order state - Filled: {self.aster_order_filled}")

                # Fallback: 手动更新仓位并标记为已成交
                self.logger.warning("⚠️ Using fallback - manually updating position and marking as filled")
                
                # 从 current_aster_side 和 current_aster_quantity 更新仓位
                if self.current_aster_side and self.current_aster_quantity:
                    if self.current_aster_side.lower() == 'buy':
                        self.aster_position += self.current_aster_quantity
                    else:
                        self.aster_position -= self.current_aster_quantity
                    self.logger.warning(f"⚠️ Fallback position update: Aster {self.current_aster_side} {self.current_aster_quantity}, new position: {self.aster_position}")
                
                self.aster_order_filled = True
                self.waiting_for_aster_fill = False
                self.order_execution_complete = True
                break

            await asyncio.sleep(0.1)

    async def setup_grvt_websocket(self):
        """Setup GRVT websocket for order updates."""
        if not self.grvt_client:
            raise Exception("GRVT client not initialized")

        def order_update_handler(order_data):
            """Handle order updates from GRVT WebSocket."""
            if order_data.get('contract_id') != self.grvt_contract_id:
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
                if status == 'FILLED' and self.grvt_order_status != 'FILLED':
                    self.logger.info(f"[{order_id}] [{order_type}] [GRVT] [{status}]: {filled_size} @ {price}")
                    self.grvt_order_status = status

                    # Log GRVT trade to CSV
                    self.log_trade_to_csv(
                        exchange='GRVT',
                        side=side,
                        price=str(price),
                        quantity=str(filled_size)
                    )

                    self.handle_grvt_order_update({
                        'order_id': order_id,
                        'side': side,
                        'status': status,
                        'size': size,
                        'price': price,
                        'contract_id': self.grvt_contract_id,
                        'filled_size': filled_size
                    })
                elif self.grvt_order_status != 'FILLED':
                    if status == 'OPEN':
                        self.logger.info(f"[{order_id}] [{order_type}] [GRVT] [{status}]: {size} @ {price}")
                    else:
                        self.logger.info(f"[{order_id}] [{order_type}] [GRVT] [{status}]: {filled_size} @ {price}")
                    self.grvt_order_status = status

            except Exception as e:
                self.logger.error(f"Error handling GRVT order update: {e}")

        try:
            self.grvt_client.setup_order_update_handler(order_update_handler)
            self.logger.info("✅ GRVT WebSocket order update handler set up")

            await self.grvt_client.connect()
            self.logger.info("✅ GRVT WebSocket connection established")

        except Exception as e:
            self.logger.error(f"Could not setup GRVT WebSocket handlers: {e}")

    async def setup_aster_websocket(self):
        """Setup Aster websocket for order updates."""
        if not self.aster_client:
            raise Exception("Aster client not initialized")

        def order_update_handler(order_data):
            """Handle order updates from Aster WebSocket."""
            try:
                # Aster WebSocket 传递的是已经处理好的订单数据（来自 aster.py 的 order_update_callback）
                # 不是原始 WebSocket 消息，所以直接使用标准字段名
                order_id = str(order_data.get('order_id', ''))
                status = order_data.get('status', '')
                side = order_data.get('side', '').lower()
                filled_size = Decimal(order_data.get('filled_size', '0'))
                size = Decimal(order_data.get('size', '0'))
                price = Decimal(order_data.get('price', '0'))

                # 使用 order_data 中的 order_type（aster.py 已经计算好了）
                order_type = order_data.get('order_type', 'UNKNOWN')
                
                # status 已经在 aster.py 中映射过了（NEW -> OPEN 等）
                mapped_status = status

                # Handle the order update
                if mapped_status == 'FILLED' and self.aster_order_status != 'FILLED':
                    self.logger.info(f"[{order_id}] [{order_type}] [ASTER] [{mapped_status}]: {filled_size} @ {price}")
                    self.aster_order_status = mapped_status

                    # CSV 记录在 handle_aster_order_result 中进行，避免重复
                    self.handle_aster_order_result({
                        'order_id': order_id,
                        'side': side,
                        'status': mapped_status,
                        'size': size,
                        'price': price,
                        'filled_size': filled_size
                    })
                elif self.aster_order_status != 'FILLED':
                    if mapped_status == 'NEW':
                        self.logger.info(f"[{order_id}] [{order_type}] [ASTER] [{mapped_status}]: {size} @ {price}")
                    else:
                        self.logger.info(f"[{order_id}] [{order_type}] [ASTER] [{mapped_status}]: {filled_size} @ {price}")
                    self.aster_order_status = mapped_status

            except Exception as e:
                self.logger.error(f"Error handling Aster order update: {e}")

        try:
            # 设置 Aster WebSocket 的 logger
            if hasattr(self.aster_client, 'ws_manager') and self.aster_client.ws_manager:
                self.aster_client.ws_manager.set_logger(self.logger)
            
            self.aster_client.setup_order_update_handler(order_update_handler)
            self.logger.info("✅ Aster WebSocket order update handler set up")

            await self.aster_client.connect()
            self.logger.info("✅ Aster WebSocket connection established")

        except Exception as e:
            self.logger.error(f"Could not setup Aster WebSocket handlers: {e}")

    async def trading_loop(self):
        """Main trading loop implementing the hedge strategy."""
        self.logger.info(f"🚀 Starting hedge bot for {self.ticker}")

        # Initialize clients
        try:
            self.initialize_grvt_client()
            self.initialize_aster_client()

            # Get contract info
            self.grvt_contract_id, self.grvt_tick_size = await self.get_grvt_contract_info()
            self.aster_contract_id, self.aster_tick_size = await self.get_aster_contract_info()

            self.logger.info(f"Contract info loaded - GRVT: {self.grvt_contract_id}, "
                             f"Aster: {self.aster_contract_id}")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize: {e}")
            return

        # Setup GRVT websocket
        try:
            await self.setup_grvt_websocket()
            self.logger.info("✅ GRVT WebSocket connection established")

        except Exception as e:
            self.logger.error(f"❌ Failed to setup GRVT websocket: {e}")
            return

        # Setup Aster websocket
        try:
            await self.setup_aster_websocket()
            self.logger.info("✅ Aster WebSocket connection established")

        except Exception as e:
            self.logger.error(f"❌ Failed to setup Aster websocket: {e}")
            return

        await asyncio.sleep(5)

        # 获取初始持仓并更新本地缓存
        try:
            self.logger.info("📊 Fetching initial positions...")
            self.grvt_position = await self.grvt_client.get_real_position()
            self.aster_position = await self.aster_client.get_real_position()
            self.logger.info(f"✅ Initial positions - GRVT: {self.grvt_position}, Aster: {self.aster_position}")
        except Exception as e:
            self.logger.error(f"❌ Failed to get initial positions: {e}")
            self.logger.warning(f"⚠️ Continuing with default positions (0, 0)")

        iterations = 0
        while iterations < self.iterations and not self.stop_flag:
            # Auto 模式下先检查条件，满足才增加 iterations
            if self.trade_type != 'auto':
                iterations += 1
            
            self.logger.info("-----------------------------------------------")
            self.logger.info(f"🔄 Trading loop iteration {iterations + 1 if self.trade_type == 'auto' else iterations}")
            self.logger.info("-----------------------------------------------")

            self.logger.info(f"[STEP 1] GRVT position: {self.grvt_position} | Aster position: {self.aster_position}")

            if abs(self.grvt_position + self.aster_position) > self.order_quantity * 2:
                self.logger.error(f"❌ Position diff is too large: {self.grvt_position + self.aster_position}")
                break

            # Auto 模式：根据价差决定交易方向
            if self.trade_type == 'auto':
                try:
                    open_spread, close_spread = await self.calculate_spreads()
                    self.logger.info(f"📊 Spreads - Open: {open_spread:.6f} (threshold: {self.open_rate}), Close: {close_spread:.6f} (threshold: {self.close_rate})")

                    # 判断是否满足开仓条件
                    if open_spread > self.open_rate:
                        side = 'sell'
                        # 检查 max_size 限制（仅在 max_size > 0 时生效）
                        if self.max_size > 0 and self.grvt_position <= -self.max_size:
                            self.logger.info(f"⚠️ Max size limit reached: GRVT position {self.grvt_position} <= -{self.max_size}, skipping SELL")
                            await asyncio.sleep(3)
                            continue
                        self.logger.info(f"✅ Open spread condition met: {open_spread:.6f} > {self.open_rate}, direction: SELL")
                        iterations += 1  # 满足条件才增加计数
                    elif close_spread < self.close_rate:
                        side = 'buy'
                        # 检查 max_size 限制（仅在 max_size > 0 时生效）
                        if self.max_size > 0 and self.grvt_position >= self.max_size:
                            self.logger.info(f"⚠️ Max size limit reached: GRVT position {self.grvt_position} >= {self.max_size}, skipping BUY")
                            await asyncio.sleep(3)
                            continue
                        self.logger.info(f"✅ Close spread condition met: {close_spread:.6f} < {self.close_rate}, direction: BUY")
                        iterations += 1  # 满足条件才增加计数
                    else:
                        self.logger.info(f"⏭️ No condition met, waiting 3 seconds...")
                        await asyncio.sleep(3)
                        continue
                except Exception as e:
                    self.logger.error(f"❌ Error calculating spreads: {e}")
                    await asyncio.sleep(3)
                    continue
            else:
                # Single/Twice 模式：使用 initial_direction
                side = self.initial_direction

            self.order_execution_complete = False
            self.waiting_for_aster_fill = False
            try:
                # Open position
                await self.place_grvt_post_only_order(side, self.order_quantity)
            except Exception as e:
                self.logger.error(f"⚠️ Error in trading loop: {e}")
                self.logger.error(f"⚠️ Full traceback: {traceback.format_exc()}")
                break

            start_time = time.time()
            while not self.order_execution_complete and not self.stop_flag:
                # Check if GRVT order filled and we need to place Aster order
                if self.waiting_for_aster_fill:
                    await self.place_aster_market_order(
                        self.current_aster_side,
                        self.current_aster_quantity,
                        self.current_aster_price
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

            # 如果是 single 或 auto 模式，跳过 STEP 2 和 STEP 3
            if self.trade_type in ['single', 'auto']:
                self.logger.info(f"[{self.trade_type.upper()} MODE] Skipping close position steps")
                continue

            # Close position (仅在 twice 模式执行)
            self.logger.info(f"[STEP 2] GRVT position: {self.grvt_position} | Aster position: {self.aster_position}")
            self.order_execution_complete = False
            self.waiting_for_aster_fill = False
            try:
                side = 'buy' if self.initial_direction == 'sell' else 'sell'
                await self.place_grvt_post_only_order(side, self.order_quantity)
            except Exception as e:
                self.logger.error(f"⚠️ Error in trading loop: {e}")
                self.logger.error(f"⚠️ Full traceback: {traceback.format_exc()}")
                break

            start_time = time.time()
            while not self.order_execution_complete and not self.stop_flag:
                # Check if GRVT order filled and we need to place Aster order
                if self.waiting_for_aster_fill:
                    await self.place_aster_market_order(
                        self.current_aster_side,
                        self.current_aster_quantity,
                        self.current_aster_price
                    )
                    break

                await asyncio.sleep(0.01)
                if time.time() - start_time > 180:
                    self.logger.error("❌ Timeout waiting for trade completion")
                    break

            # Close remaining position
            self.logger.info(f"[STEP 3] GRVT position: {self.grvt_position} | Aster position: {self.aster_position}")
            self.order_execution_complete = False
            self.waiting_for_aster_fill = False
            if self.grvt_position == 0:
                continue
            elif self.grvt_position > 0:
                side = 'sell'
            else:
                side = 'buy'

            try:
                await self.place_grvt_post_only_order(side, abs(self.grvt_position))
            except Exception as e:
                self.logger.error(f"⚠️ Error in trading loop: {e}")
                self.logger.error(f"⚠️ Full traceback: {traceback.format_exc()}")
                break

            start_time = time.time()
            while not self.order_execution_complete and not self.stop_flag:
                # Check if GRVT order filled and we need to place Aster order
                if self.waiting_for_aster_fill:
                    await self.place_aster_market_order(
                        self.current_aster_side,
                        self.current_aster_quantity,
                        self.current_aster_price
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
    parser = argparse.ArgumentParser(description='Trading bot for GRVT to Aster hedge')
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
    parser.add_argument('--type', type=str, default='single', choices=['single', 'twice', 'auto'],
                        help='Trade type: single (open only), twice (open then close), or auto (based on spread) (default: single)')
    parser.add_argument('--open-rate', type=str, default='0.001',
                        help='Open rate threshold for auto mode (default: 0.001)')
    parser.add_argument('--close-rate', type=str, default='-0.001',
                        help='Close rate threshold for auto mode (default: -0.001)')
    parser.add_argument('--max-size', type=str, default='0',
                        help='Max position size limit for auto mode (default: 0, no limit). Position must be <= max_size and >= -max_size')

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
        trade_type=args.type,
        open_rate=Decimal(args.open_rate),
        close_rate=Decimal(args.close_rate),
        max_size=Decimal(args.max_size)
    )

    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
