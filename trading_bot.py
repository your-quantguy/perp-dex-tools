"""
Modular Trading Bot - Supports multiple exchanges
"""

import os
import time
import asyncio
import traceback
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional
from datetime import datetime

from exchanges import ExchangeFactory
from helpers import TradingLogger
from helpers.lark_webhook import get_lark_webhook_bot, CardColor
from helpers.telegram_bot import TelegramBot


@dataclass
class TradingConfig:
    """Configuration class for trading parameters."""
    ticker: str
    contract_id: str
    quantity: Decimal
    take_profit: Decimal
    tick_size: Decimal
    direction: str
    max_orders: int
    wait_time: int
    exchange: str
    grid_step: Decimal
    instance_id: str = "default"  # Unique identifier for the bot instance
    stop_price: Decimal = Decimal(-1)
    pause_price: Decimal = Decimal(-1)
    stop_loss: Decimal = Decimal(-1)  # Stop loss percentage (e.g., 0.4 = 0.4% loss)
    boost_mode: bool = False

    @property
    def close_order_side(self) -> str:
        """Get the close order side based on bot direction."""
        return 'buy' if self.direction == "sell" else 'sell'


@dataclass
class OrderMonitor:
    """Thread-safe order monitoring state."""
    order_id: Optional[str] = None
    filled: bool = False
    filled_price: Optional[Decimal] = None
    filled_qty: Decimal = 0.0

    def reset(self):
        """Reset the monitor state."""
        self.order_id = None
        self.filled = False
        self.filled_price = None
        self.filled_qty = 0.0


class TradingBot:
    """Modular Trading Bot - Main trading logic supporting multiple exchanges."""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = TradingLogger(config.exchange, config.ticker, config.instance_id, log_to_console=True)

        # Create exchange client
        try:
            self.exchange_client = ExchangeFactory.create_exchange(
                config.exchange,
                config
            )
        except ValueError as e:
            raise ValueError(f"Failed to create exchange client: {e}")

        # Trading state
        self.active_close_orders = []
        self.last_close_orders = 0
        self.last_open_order_time = 0
        self.last_log_time = 0
        self.current_order_status = None
        self.order_filled_event = asyncio.Event()
        self.order_canceled_event = asyncio.Event()
        self.shutdown_requested = False
        self.loop = None

        # Enhanced hourly statistics
        self.hourly_operations = 0
        self.hourly_fees = Decimal(0)
        self.last_hour_reset = time.time()
        
        # Separate statistics for position operations and closing operations
        self.hourly_position_operations = 0  # Current Position related operations
        self.hourly_closing_operations = 0   # Active closing amount related operations
        self.hourly_successful_fills = 0     # Successfully filled orders
        self.hourly_canceled_orders = 0      # Canceled orders

        # Register order callback
        self._setup_websocket_handlers()

    async def send_startup_notification(self):
        """Send startup notification to Lark."""
        bot = get_lark_webhook_bot()
        if bot:
            try:
                # 构建策略参数字符串
                strategy_params = (
                    f"Direction={self.config.direction} | "
                    f"Qty={self.config.quantity} | "
                    f"TP={self.config.take_profit}% | "
                    f"MaxOrders={self.config.max_orders} | "
                    f"Wait={self.config.wait_time}s | "
                    f"Grid={self.config.grid_step}%"
                )
                
                metrics = {
                    "数量": str(self.config.quantity),
                    "止盈": f"{self.config.take_profit}%",
                    "方向": self.config.direction.upper(),
                    "最大订单": str(self.config.max_orders),
                    "等待时间": f"{self.config.wait_time}秒",
                    "网格步长": f"{self.config.grid_step}%"
                }
                
                await bot.send_system_status(
                    status_type='startup',
                    message=f"交易机器人已启动",
                    metrics=metrics,
                    instance_id=self.config.instance_id,
                    exchange=self.config.exchange,
                    ticker=self.config.ticker,
                    strategy=strategy_params,
                    color=CardColor.GREEN
                )
                self.logger.log(f"Startup notification sent for instance: {self.config.instance_id}", "INFO")
            except Exception as e:
                self.logger.log(f"Failed to send startup notification: {e}", "ERROR")

    async def send_error_notification(self, error_type: str, details: str):
        """Send error notification to Lark."""
        bot = get_lark_webhook_bot()
        if bot:
            try:
                await bot.send_error_card(
                    error_type=error_type,
                    error_details=details,
                    instance_id=self.config.instance_id,
                    exchange=self.config.exchange,
                    ticker=self.config.ticker
                )
                self.logger.log(f"Error notification sent: {error_type}", "INFO")
            except Exception as e:
                self.logger.log(f"Failed to send error notification: {e}", "ERROR")

    async def graceful_shutdown(self, reason: str = "Unknown"):
        """Perform graceful shutdown of the trading bot."""
        self.logger.log(f"Starting graceful shutdown: {reason}", "INFO")
        self.shutdown_requested = True

        try:
            # Disconnect from exchange
            await self.exchange_client.disconnect()
            self.logger.log("Graceful shutdown completed", "INFO")

        except Exception as e:
            self.logger.log(f"Error during graceful shutdown: {e}", "ERROR")

    def _setup_websocket_handlers(self):
        """Setup WebSocket handlers for order updates."""
        def order_update_handler(message):
            """Handle order updates from WebSocket."""
            try:
                # Check if this is for our contract
                if message.get('contract_id') != self.config.contract_id:
                    return

                order_id = message.get('order_id')
                status = message.get('status')
                side = message.get('side', '')
                order_type = message.get('order_type', '')
                filled_size = Decimal(message.get('filled_size'))
                if order_type == "OPEN":
                    self.current_order_status = status

                if status == 'FILLED':
                    if order_type == "OPEN":
                        self.order_filled_amount = filled_size
                        # Ensure thread-safe interaction with asyncio event loop
                        if self.loop is not None:
                            self.loop.call_soon_threadsafe(self.order_filled_event.set)
                        else:
                            # Fallback (should not happen after run() starts)
                            self.order_filled_event.set()
                        
                        # Update position operation statistics
                        self.hourly_position_operations += 1
                        self.hourly_successful_fills += 1
                    elif order_type == "CLOSE":
                        # Update closing operation statistics
                        self.hourly_closing_operations += 1

                    self.logger.log(f"[{order_type}] [{order_id}] {status} "
                                    f"{message.get('size')} @ {message.get('price')}", "INFO")
                    self.logger.log_transaction(order_id, side, message.get('size'), message.get('price'), status)
                elif status == "CANCELED":
                    if order_type == "OPEN":
                        self.order_filled_amount = filled_size
                        if self.loop is not None:
                            self.loop.call_soon_threadsafe(self.order_canceled_event.set)
                        else:
                            self.order_canceled_event.set()

                        # Only count as canceled operation if there was no fill
                        if filled_size == 0:
                            self.hourly_canceled_orders += 1
                        else:
                            # Partial fill, count as position operation
                            self.hourly_position_operations += 1

                        if self.order_filled_amount > 0:
                            self.logger.log_transaction(order_id, side, self.order_filled_amount, message.get('price'), status)

                    # PATCH
                    if self.config.exchange == "extended":
                        self.logger.log(f"[{order_type}] [{order_id}] {status} "
                                        f"{Decimal(message.get('size')) - filled_size} @ {message.get('price')}", "INFO")
                    else:
                        self.logger.log(f"[{order_type}] [{order_id}] {status} "
                                        f"{message.get('size')} @ {message.get('price')}", "INFO")
                elif status == "PARTIALLY_FILLED":
                    self.logger.log(f"[{order_type}] [{order_id}] {status} "
                                    f"{filled_size} @ {message.get('price')}", "INFO")
                else:
                    self.logger.log(f"[{order_type}] [{order_id}] {status} "
                                    f"{message.get('size')} @ {message.get('price')}", "INFO")

            except Exception as e:
                self.logger.log(f"Error handling order update: {e}", "ERROR")
                self.logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")

        # Setup order update handler
        self.exchange_client.setup_order_update_handler(order_update_handler)

    def _calculate_wait_time(self) -> Decimal:
        """Calculate wait time between orders."""
        cool_down_time = self.config.wait_time

        if len(self.active_close_orders) < self.last_close_orders:
            self.last_close_orders = len(self.active_close_orders)
            return 0

        self.last_close_orders = len(self.active_close_orders)
        if len(self.active_close_orders) >= self.config.max_orders:
            return 1

        if len(self.active_close_orders) / self.config.max_orders >= 2/3:
            cool_down_time = 2 * self.config.wait_time
        elif len(self.active_close_orders) / self.config.max_orders >= 1/3:
            cool_down_time = self.config.wait_time
        elif len(self.active_close_orders) / self.config.max_orders >= 1/6:
            cool_down_time = self.config.wait_time / 2
        else:
            cool_down_time = self.config.wait_time / 4

        # if the program detects active_close_orders during startup, it is necessary to consider cooldown_time
        if self.last_open_order_time == 0 and len(self.active_close_orders) > 0:
            self.last_open_order_time = time.time()

        if time.time() - self.last_open_order_time > cool_down_time:
            return 0
        else:
            return 1

    async def _place_and_monitor_open_order(self) -> bool:
        """Place an order and monitor its execution."""
        try:
            # Reset state before placing order
            self.order_filled_event.clear()
            self.current_order_status = 'OPEN'
            self.order_filled_amount = 0.0

            # Place the order
            order_result = await self.exchange_client.place_open_order(
                self.config.contract_id,
                self.config.quantity,
                self.config.direction
            )

            if not order_result.success:
                # Check if it's an insufficient margin error
                if order_result.error_message and 'Insufficient margin' in order_result.error_message:
                    self.logger.log(f"⚠️ 资金不足，无法开仓，将休眠10分钟后重试", "WARNING")
                    await self.send_error_notification("资金不足 - 暂停交易", 
                        f"账户保证金不足，无法开仓\n交易对: {self.config.ticker}\n数量: {self.config.quantity}\n\n⏰ 将休眠10分钟后自动重试\n请及时充值或减少交易数量")
                    # Sleep for 10 minutes before retrying
                    await asyncio.sleep(600)  # 600 seconds = 10 minutes
                    self.logger.log(f"💤 休眠结束，继续尝试交易", "INFO")
                return False

            if order_result.status == 'FILLED':
                return await self._handle_order_result(order_result)
            elif not self.order_filled_event.is_set():
                timeout_seconds = int(os.getenv('ORDER_TIMEOUT_SECONDS', 10))  # Default to 10 seconds
                try:
                    await asyncio.wait_for(self.order_filled_event.wait(), timeout=timeout_seconds)
                except asyncio.TimeoutError:
                    self.logger.log(f"[OPEN] Order {order_result.order_id} timed out after {timeout_seconds} seconds, trying to cancel order", "INFO")
                    pass

            # Handle order result
            return await self._handle_order_result(order_result)

        except Exception as e:
            self.logger.log(f"Error placing order: {e}", "ERROR")
            self.logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")
            return False

    async def _handle_order_result(self, order_result) -> bool:
        """Handle the result of an order placement."""
        order_id = order_result.order_id
        filled_price = order_result.price

        if self.order_filled_event.is_set() or order_result.status == 'FILLED':
            if self.config.boost_mode:
                close_order_result = await self.exchange_client.place_market_order(
                    self.config.contract_id,
                    self.config.quantity,
                    self.config.close_order_side
                )
            else:
                self.last_open_order_time = time.time()
                # Place close order
                close_side = self.config.close_order_side
                close_price = self._calculate_close_price(filled_price)
                
                # Also place stop-loss order if configured
                if self.config.stop_loss > 0:
                    stop_loss_price = self._calculate_stop_loss_price(filled_price)
                    self.logger.log(f"[STOP-LOSS] Setting stop-loss order at {stop_loss_price} ({self.config.stop_loss}%)", "INFO")
                    
                    # Place stop-loss order first
                    stop_loss_order_result = await self.exchange_client.place_close_order(
                        self.config.contract_id,
                        self.config.quantity,
                        stop_loss_price,
                        close_side
                    )
                    
                    if stop_loss_order_result.success:
                        self.logger.log(f"[STOP-LOSS] Order placed: {stop_loss_order_result.order_id}", "INFO")

                close_order_result = await self.exchange_client.place_close_order(
                    self.config.contract_id,
                    self.config.quantity,
                    close_price,
                    close_side
                )
                if self.config.exchange == "lighter":
                    await asyncio.sleep(1)

                if not close_order_result.success:
                    self.logger.log(f"[CLOSE] Failed to place close order: {close_order_result.error_message}", "ERROR")
                    raise Exception(f"[CLOSE] Failed to place close order: {close_order_result.error_message}")

                return True

        else:
            new_order_price = await self.exchange_client.get_order_price(self.config.direction)

            def should_wait(direction: str, new_order_price: Decimal, order_result_price: Decimal) -> bool:
                if direction == "buy":
                    return new_order_price <= order_result_price
                elif direction == "sell":
                    return new_order_price >= order_result_price
                return False

            if self.config.exchange == "lighter":
                current_order_status = self.exchange_client.current_order.status
            else:
                order_info = await self.exchange_client.get_order_info(order_id)
                current_order_status = order_info.status

            while (
                should_wait(self.config.direction, new_order_price, order_result.price)
                and current_order_status == "OPEN"
            ):
                self.logger.log(f"[OPEN] [{order_id}] Waiting for order to be filled @ {order_result.price}", "INFO")
                await asyncio.sleep(5)
                if self.config.exchange == "lighter":
                    current_order_status = self.exchange_client.current_order.status
                else:
                    order_info = await self.exchange_client.get_order_info(order_id)
                    if order_info is not None:
                        current_order_status = order_info.status
                new_order_price = await self.exchange_client.get_order_price(self.config.direction)

            self.order_canceled_event.clear()
            # Cancel the order if it's still open
            self.logger.log(f"[OPEN] [{order_id}] Cancelling order and placing a new order", "INFO")
            if self.config.exchange == "lighter":
                cancel_result = await self.exchange_client.cancel_order(order_id)
                start_time = time.time()
                while (time.time() - start_time < 10 and self.exchange_client.current_order.status != 'CANCELED' and
                        self.exchange_client.current_order.status != 'FILLED'):
                    await asyncio.sleep(0.1)

                if self.exchange_client.current_order.status not in ['CANCELED', 'FILLED']:
                    raise Exception(f"[OPEN] Error cancelling order: {self.exchange_client.current_order.status}")
                else:
                    self.order_filled_amount = self.exchange_client.current_order.filled_size
            else:
                try:
                    cancel_result = await self.exchange_client.cancel_order(order_id)
                    if not cancel_result.success:
                        self.order_canceled_event.set()
                        self.logger.log(f"[CLOSE] Failed to cancel order {order_id}: {cancel_result.error_message}", "WARNING")
                    else:
                        self.current_order_status = "CANCELED"

                except Exception as e:
                    self.order_canceled_event.set()
                    self.logger.log(f"[CLOSE] Error canceling order {order_id}: {e}", "ERROR")

                if self.config.exchange == "backpack" or self.config.exchange == "extended":
                    self.order_filled_amount = cancel_result.filled_size
                else:
                    # Wait for cancel event or timeout
                    if not self.order_canceled_event.is_set():
                        try:
                            await asyncio.wait_for(self.order_canceled_event.wait(), timeout=5)
                        except asyncio.TimeoutError:
                            order_info = await self.exchange_client.get_order_info(order_id)
                            self.order_filled_amount = order_info.filled_size

            if self.order_filled_amount > 0:
                close_side = self.config.close_order_side
                if self.config.boost_mode:
                    close_order_result = await self.exchange_client.place_close_order(
                        self.config.contract_id,
                        self.order_filled_amount,
                        filled_price,
                        close_side
                    )
                else:
                    if close_side == 'sell':
                        close_price = filled_price * (1 + self.config.take_profit/100)
                    else:
                        close_price = filled_price * (1 - self.config.take_profit/100)

                    close_order_result = await self.exchange_client.place_close_order(
                        self.config.contract_id,
                        self.order_filled_amount,
                        close_price,
                        close_side
                    )
                    if self.config.exchange == "lighter":
                        await asyncio.sleep(1)

                self.last_open_order_time = time.time()
                if not close_order_result.success:
                    self.logger.log(f"[CLOSE] Failed to place close order: {close_order_result.error_message}", "ERROR")

            return True

        return False

    async def _log_status_periodically(self):
        """Log status information periodically, including positions and hourly statistics."""
        if time.time() - self.last_log_time > 60 or self.last_log_time == 0:
            print("--------------------------------")
            try:
                # Get active orders
                active_orders = await self.exchange_client.get_active_orders(self.config.contract_id)

                # Filter close orders
                self.active_close_orders = []
                for order in active_orders:
                    if order.side == self.config.close_order_side:
                        self.active_close_orders.append({
                            'id': order.order_id,
                            'price': order.price,
                            'size': order.size
                        })

                # Get positions
                position_amt = await self.exchange_client.get_account_positions()
                position_amt = abs(position_amt)

                # Calculate active closing amount
                active_close_amount = sum(
                    Decimal(order.get('size', 0))
                    for order in self.active_close_orders
                    if isinstance(order, dict)
                )

                self.logger.log(f"Current Position: {position_amt} | Active closing amount: {active_close_amount} | "
                                f"Order quantity: {len(self.active_close_orders)}")
                self.last_log_time = time.time()
                # Check for position mismatch
                if abs(position_amt - active_close_amount) > (2 * self.config.quantity):
                    error_message = f"Position mismatch detected - Current position: {position_amt}, Active closing amount: {active_close_amount}"
                    
                    self.logger.log(f"\n\nERROR: [{self.config.exchange.upper()}_{self.config.ticker.upper()}] {error_message}", "ERROR")
                    self.logger.log("###### ERROR ###### ERROR ###### ERROR ###### ERROR #####", "ERROR")
                    self.logger.log("Please manually rebalance your position and take-profit orders", "ERROR")
                    self.logger.log("请手动平衡当前仓位和正在关闭的仓位", "ERROR")
                    self.logger.log("###### ERROR ###### ERROR ###### ERROR ###### ERROR #####", "ERROR")

                    bot = get_lark_webhook_bot()
                    if bot:
                        try:
                            strategy_params = (
                                f"Direction={self.config.direction} | "
                                f"Qty={self.config.quantity} | "
                                f"TP={self.config.take_profit}% | "
                                f"MaxOrders={self.config.max_orders}"
                            )
                            
                            await bot.send_error_alert(
                                error_type="仓位不匹配",
                                error_message="Position mismatch detected! Manual intervention required.",
                                instance=self.config.instance_id,
                                exchange=self.config.exchange,
                                ticker=self.config.ticker,
                                strategy=strategy_params,
                                traceback_preview=f"当前仓位: {position_amt}\n活跃平仓数量: {active_close_amount}\n差异: {abs(position_amt - active_close_amount)}",
                                error_count=1
                            )
                        except Exception as e:
                            self.logger.log(f"Failed to send error notification: {e}", "ERROR")
                    
                    # Also send simple notification for telegram support
                    simple_error_msg = f"\n\nERROR: [{self.config.exchange.upper()}_{self.config.ticker.upper()}] "
                    simple_error_msg += "Position mismatch detected\n"
                    simple_error_msg += "###### ERROR ###### ERROR ###### ERROR ###### ERROR #####\n"
                    simple_error_msg += "Please manually rebalance your position and take-profit orders\n"
                    simple_error_msg += "请手动平衡当前仓位和正在关闭的仓位\n"
                    simple_error_msg += f"current position: {position_amt} | active closing amount: {active_close_amount} | "f"Order quantity: {len(self.active_close_orders)}\n"
                    simple_error_msg += "###### ERROR ###### ERROR ###### ERROR ###### ERROR #####\n"
                    await self.send_notification(simple_error_msg.lstrip())

                    if not self.shutdown_requested:
                        self.shutdown_requested = True

                    mismatch_detected = True
                else:
                    mismatch_detected = False

                # Send hourly statistics to Lark
                current_time = time.time()
                if current_time - self.last_hour_reset > 3600:  # 1 hour
                    # Calculate estimated fees
                    estimated_fees = self.hourly_position_operations * self.config.quantity * Decimal(0.001)  # Assuming 0.1% fee
                    
                    hourly_stats_message = (f"Hourly Statistics for instance {self.config.instance_id} - "
                                          f"Position operations: {self.hourly_position_operations}, "
                                          f"Closing operations: {self.hourly_closing_operations}, "
                                          f"Successful fills: {self.hourly_successful_fills}, "
                                          f"Canceled orders: {self.hourly_canceled_orders}")
                    
                    self.logger.log(hourly_stats_message, "INFO")
                    
                    bot = get_lark_webhook_bot()
                    if bot:
                        try:
                            strategy_params = (
                                f"Direction={self.config.direction} | "
                                f"Qty={self.config.quantity} | "
                                f"TP={self.config.take_profit}%"
                            )
                            
                            metrics = {
                                "仓位操作数": str(self.hourly_position_operations),
                                "平仓操作数": str(self.hourly_closing_operations),
                                "成功成交数": str(self.hourly_successful_fills),
                                "取消订单数": str(self.hourly_canceled_orders),
                                "预估手续费": f"{estimated_fees:.4f}",
                                "当前仓位": str(position_amt),
                                "活跃平仓": str(active_close_amount),
                                "统计周期": f"{datetime.fromtimestamp(self.last_hour_reset).strftime('%H:%M')} - {datetime.fromtimestamp(current_time).strftime('%H:%M')}"
                            }
                            
                            await bot.send_system_status(
                                status_type='resource',
                                message="每小时交易统计",
                                metrics=metrics,
                                instance_id=self.config.instance_id,
                                exchange=self.config.exchange,
                                ticker=self.config.ticker,
                                strategy=strategy_params,
                                color=CardColor.BLUE
                            )
                        except Exception as e:
                            self.logger.log(f"Failed to send hourly stats notification: {e}", "ERROR")

                    # Reset statistics
                    self.hourly_position_operations = 0
                    self.hourly_closing_operations = 0
                    self.hourly_successful_fills = 0
                    self.hourly_canceled_orders = 0
                    self.last_hour_reset = current_time

                return mismatch_detected

            except Exception as e:
                self.logger.log(f"Error in periodic status check: {e}", "ERROR")
                self.logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")

            print("--------------------------------")

    async def _meet_grid_step_condition(self) -> bool:
        if self.active_close_orders:
            picker = min if self.config.direction == "buy" else max
            next_close_order = picker(self.active_close_orders, key=lambda o: o["price"])
            next_close_price = next_close_order["price"]

            best_bid, best_ask = await self.exchange_client.fetch_bbo_prices(self.config.contract_id)
            if best_bid <= 0 or best_ask <= 0 or best_bid >= best_ask:
                raise ValueError("No bid/ask data available")

            if self.config.direction == "buy":
                new_order_close_price = best_ask * (1 + self.config.take_profit/100)
                if next_close_price / new_order_close_price > 1 + self.config.grid_step/100:
                    return True
                else:
                    return False
            elif self.config.direction == "sell":
                new_order_close_price = best_bid * (1 - self.config.take_profit/100)
                if new_order_close_price / next_close_price > 1 + self.config.grid_step/100:
                    return True
                else:
                    return False
            else:
                raise ValueError(f"Invalid direction: {self.config.direction}")
        else:
            return True

    async def _check_price_condition(self) -> bool:
        stop_trading = False
        pause_trading = False

        if self.config.pause_price == self.config.stop_price == -1:
            return stop_trading, pause_trading

        best_bid, best_ask = await self.exchange_client.fetch_bbo_prices(self.config.contract_id)
        if best_bid <= 0 or best_ask <= 0 or best_bid >= best_ask:
            raise ValueError("No bid/ask data available")

        if self.config.stop_price != -1:
            if self.config.direction == "buy":
                if best_ask >= self.config.stop_price:
                    stop_trading = True
            elif self.config.direction == "sell":
                if best_bid <= self.config.stop_price:
                    stop_trading = True

        if self.config.pause_price != -1:
            if self.config.direction == "buy":
                if best_ask >= self.config.pause_price:
                    pause_trading = True
            elif self.config.direction == "sell":
                if best_bid <= self.config.pause_price:
                    pause_trading = True

        return stop_trading, pause_trading

    async def send_notification(self, message: str):
        """Send notification to Lark or Telegram based on configuration."""
        lark_bot = get_lark_webhook_bot()
        if lark_bot:
            await lark_bot.send_text(message)

        telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if telegram_token and telegram_chat_id:
            with TelegramBot(telegram_token, telegram_chat_id) as tg_bot:
                tg_bot.send_text(message)

    async def run(self):
        """Main trading loop."""
        try:
            self.config.contract_id, self.config.tick_size = await self.exchange_client.get_contract_attributes()

            # Log current TradingConfig
            self.logger.log("=== Trading Configuration ===", "INFO")
            self.logger.log(f"Ticker: {self.config.ticker}", "INFO")
            self.logger.log(f"Contract ID: {self.config.contract_id}", "INFO")
            self.logger.log(f"Quantity: {self.config.quantity}", "INFO")
            self.logger.log(f"Take Profit: {self.config.take_profit}%", "INFO")
            self.logger.log(f"Direction: {self.config.direction}", "INFO")
            self.logger.log(f"Max Orders: {self.config.max_orders}", "INFO")
            self.logger.log(f"Wait Time: {self.config.wait_time}s", "INFO")
            self.logger.log(f"Exchange: {self.config.exchange}", "INFO")
            self.logger.log(f"Grid Step: {self.config.grid_step}%", "INFO")
            self.logger.log(f"Stop Price: {self.config.stop_price}", "INFO")
            self.logger.log(f"Pause Price: {self.config.pause_price}", "INFO")
            self.logger.log(f"Boost Mode: {self.config.boost_mode}", "INFO")
            self.logger.log("=============================", "INFO")

            # Capture the running event loop for thread-safe callbacks
            self.loop = asyncio.get_running_loop()
            # Connect to exchange
            await self.exchange_client.connect()

            # Send startup notification
            await self.send_startup_notification()
            
            # wait for connection to establish
            await asyncio.sleep(5)

            # Main trading loop
            while not self.shutdown_requested:
                # Update active orders
                active_orders = await self.exchange_client.get_active_orders(self.config.contract_id)

                # Filter close orders
                self.active_close_orders = []
                for order in active_orders:
                    if order.side == self.config.close_order_side:
                        self.active_close_orders.append({
                            'id': order.order_id,
                            'price': order.price,
                            'size': order.size
                        })

                # Periodic logging
                mismatch_detected = await self._log_status_periodically()

                stop_trading, pause_trading = await self._check_price_condition()
                if stop_trading:
                    msg = f"\n\nWARNING: [{self.config.exchange.upper()}_{self.config.ticker.upper()}] \n"
                    msg += "Stopped trading due to stop price triggered\n"
                    msg += "价格已经达到停止交易价格，脚本将停止交易\n"
                    await self.send_notification(msg.lstrip())
                    await self.graceful_shutdown(msg)
                    continue

                if pause_trading:
                    await asyncio.sleep(5)
                    continue

                if not mismatch_detected:
                    wait_time = self._calculate_wait_time()

                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        meet_grid_step_condition = await self._meet_grid_step_condition()
                        if not meet_grid_step_condition:
                            await asyncio.sleep(1)
                            continue

                        await self._place_and_monitor_open_order()
                        self.last_close_orders += 1

        except KeyboardInterrupt:
            self.logger.log("Bot stopped by user")
            await self.graceful_shutdown("User interruption (Ctrl+C)")
        except Exception as e:
            self.logger.log(f"Critical error: {e}", "ERROR")
            self.logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")
            await self.graceful_shutdown(f"Critical error: {e}")
            raise
        finally:
            # Ensure all connections are closed even if graceful shutdown fails
            try:
                await self.exchange_client.disconnect()
            except Exception as e:
                self.logger.log(f"Error disconnecting from exchange: {e}", "ERROR")

    def _calculate_close_price(self, filled_price: Decimal) -> Decimal:
        """Calculate the close price based on the filled price and take profit."""
        if self.config.close_order_side == 'sell':
            return filled_price * (Decimal('1') + self.config.take_profit / Decimal('100'))
        else:
            return filled_price * (Decimal('1') - self.config.take_profit / Decimal('100'))

    def _calculate_stop_loss_price(self, filled_price: Decimal) -> Decimal:
        """Calculate the stop-loss price based on the filled price and stop loss percentage."""
        if self.config.close_order_side == 'sell':
            # For sell close (buy open), stop-loss is higher (loss when price goes up)
            return filled_price * (Decimal('1') + self.config.stop_loss / Decimal('100'))
        else:
            # For buy close (sell open), stop-loss is lower (loss when price goes down)
            return filled_price * (Decimal('1') - self.config.stop_loss / Decimal('100'))
