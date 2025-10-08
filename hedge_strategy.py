"""
Hedging strategy between Lighter and Paradex exchanges.
This strategy places limit orders on Lighter and monitors until filled,
then places opposite orders on Paradex for hedging.
Uses BBO prices from the order book for limit orders.
"""

import asyncio
import os
import time
from decimal import Decimal
from typing import Dict, Any, Optional, Tuple

from exchanges.lighter import LighterClient
from exchanges.paradex import ParadexClient
from helpers.logger import TradingLogger


class HedgeStrategy:
    """
    Hedging strategy between Lighter and Paradex exchanges.
    Places limit orders on Lighter, monitors until filled,
    then places opposite orders on Paradex for hedging.
    Uses BBO prices from the order book for limit orders.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the hedging strategy with configuration."""
        self.config = config
        self.logger = TradingLogger(
            exchange="hedge_strategy",
            ticker=self.config.get("ticker", "ETH-PERP"),
            log_to_console=True
        )
        
        # Initialize exchange clients
        self.lighter_client = LighterClient(config.get("lighter", {}))
        self.paradex_client = ParadexClient(config.get("paradex", {}))
        
        # Strategy parameters
        self.ticker = config.get("ticker", "ETH-PERP")
        self.quantity = Decimal(config.get("quantity", "0.1"))
        self.side = config.get("side", "buy")  # Initial side on Lighter
        
        # Order parameters
        self.price_offset_ticks = config.get("price_offset_ticks", 0)  # Ticks away from BBO
        self.order_timeout_seconds = config.get("order_timeout_seconds", 5)  # Time before canceling order (changed to 5 seconds)
        self.max_retries = config.get("max_retries", 10)  # Maximum number of order placement retries (increased for more attempts)
        
        # Tracking variables
        self.lighter_order_id = None
        self.paradex_order_id = None
        self.is_running = False
        self.lighter_order_filled = False
        self.order_placement_time = None
        self.current_retry_count = 0
        
        # Auto cancel and replace configuration
        self.auto_cancel_enabled = config.get('auto_cancel_enabled', True)
        self.price_check_interval = config.get('price_check_interval', 5)  # seconds
        self.price_tolerance = config.get('price_tolerance', 0.001)  # 0.1% tolerance

    async def initialize(self):
        """Initialize connections to exchanges."""
        self.logger.log("Initializing hedge strategy", "INFO")
        
        try:
            # Ensure contract attributes (contract_id, tick_size) are resolved first
            await self.lighter_client.get_contract_attributes()
            await self.paradex_client.get_contract_attributes()
            
            # Connect to exchanges
            await self.lighter_client.connect()
            await self.paradex_client.connect()
            self.logger.log("Connected to exchanges", "INFO")
            return True
        except Exception as e:
            self.logger.log(f"Failed to initialize strategy: {e}", "ERROR")
            return False

    async def shutdown(self):
        """Shutdown connections to exchanges."""
        self.logger.log("Shutting down hedge strategy", "INFO")
        self.is_running = False
        
        # Disconnect from exchanges with individual error handling
        try:
            await self.lighter_client.disconnect()
        except Exception as e:
            self.logger.log(f"Error disconnecting from Lighter: {e}", "ERROR")
            
        try:
            await self.paradex_client.disconnect()
        except Exception as e:
            self.logger.log(f"Error disconnecting from Paradex: {e}", "ERROR")
            
        self.logger.log("Hedge strategy shutdown completed", "INFO")

    async def get_bbo_price(self) -> Tuple[Decimal, Decimal]:
        """Get the best bid and offer prices using Lighter client helper."""
        try:
            contract_id = self.lighter_client.config.contract_id
            best_bid, best_ask = await self.lighter_client.fetch_bbo_prices(contract_id)
            self.logger.log(f"BBO prices - Bid: {best_bid}, Ask: {best_ask}", "INFO")
            return best_bid, best_ask
        except Exception as e:
            self.logger.log(f"Error getting BBO prices: {e}", "ERROR")
            raise
    
    async def calculate_limit_price(self) -> Decimal:
        """Calculate limit price based on BBO and side."""
        best_bid, best_ask = await self.get_bbo_price()
        tick_size = self.lighter_client.config.tick_size
        
        if self.side == "buy":
            # For buy orders, use bid price minus offset
            price = best_bid - (tick_size * self.price_offset_ticks)
        else:
            # For sell orders, use ask price plus offset
            price = best_ask + (tick_size * self.price_offset_ticks)
            
        # Ensure price is rounded to tick size
        price = self.lighter_client.round_to_tick(price)
        self.logger.log(f"Calculated limit price for {self.side}: {price}", "INFO")
        return price
    
    async def should_cancel_and_replace_order(self):
        """Check if the current order should be canceled and replaced based on price movement."""
        if not self.auto_cancel_enabled or not self.lighter_order_id:
            return False
            
        try:
            # Get current BBO price
            bid_price, ask_price = await self.get_bbo_price()
            
            # Use the appropriate BBO price based on order side
            if self.side == "buy":
                current_bbo_price = bid_price
            else:
                current_bbo_price = ask_price
                
            if not current_bbo_price:
                self.logger.log("Failed to get current BBO price for order replacement check", "WARNING")
                return False
            
            # Get the current order details
            orders = await self.lighter_client.get_active_orders(self.lighter_client.config.contract_id)
            current_order = None
            for order in orders:
                if order.order_id == self.lighter_order_id:
                    current_order = order
                    break
            
            if not current_order:
                self.logger.log(f"Current order {self.lighter_order_id} not found in active orders", "DEBUG")
                return False
            
            # Calculate price difference
            order_price = current_order.price
            price_diff = abs(current_bbo_price - order_price) / order_price
            
            self.logger.log(f"Price check: Order price={order_price}, Current BBO={current_bbo_price}, "
                          f"Diff={price_diff:.4f}, Tolerance={self.price_tolerance}", "DEBUG")
            
            # Check if price has moved beyond tolerance
            if price_diff > self.price_tolerance:
                self.logger.log(f"Price moved beyond tolerance ({price_diff:.4f} > {self.price_tolerance}). "
                              f"Will cancel and replace order.", "INFO")
                return True
                
            return False
            
        except Exception as e:
            self.logger.log(f"Error checking if order should be replaced: {e}", "ERROR")
            return False

    async def cancel_and_replace_lighter_order(self):
        """Cancel the current Lighter order and place a new one with updated BBO price."""
        if not self.lighter_order_id:
            self.logger.log("No active order to cancel and replace", "WARNING")
            return False
            
        try:
            # Cancel current order
            self.logger.log(f"Canceling order {self.lighter_order_id} for replacement", "INFO")
            cancel_result = await self.cancel_lighter_order()
            
            if not cancel_result:
                self.logger.log("Failed to cancel order for replacement", "ERROR")
                return False
            
            # Wait a moment for cancellation to process
            await asyncio.sleep(0.5)
            
            # Place new order with current BBO price
            self.logger.log("Placing replacement order with updated BBO price", "INFO")
            placement_result = await self.place_lighter_limit_order()
            
            if placement_result:
                self.logger.log("Successfully placed replacement order", "INFO")
                return True
            else:
                self.logger.log("Failed to place replacement order", "ERROR")
                return False
                
        except Exception as e:
            self.logger.log(f"Error during cancel and replace: {e}", "ERROR")
            return False

    async def cancel_lighter_order(self):
        """Cancel the current Lighter order if it exists."""
        if not self.lighter_order_id:
            return True
            
        try:
            self.logger.log(f"Canceling Lighter order {self.lighter_order_id}", "INFO")
            result = await self.lighter_client.cancel_order(self.lighter_order_id)
            
            if result:
                self.logger.log(f"Successfully canceled order {self.lighter_order_id}", "INFO")
                self.lighter_order_id = None
                return True
            else:
                self.logger.log(f"Failed to cancel order {self.lighter_order_id}", "WARNING")
                return False
        except Exception as e:
            self.logger.log(f"Error canceling order: {e}", "ERROR")
            return False
    
    async def place_lighter_limit_order(self):
        """Place a limit order on Lighter exchange using BBO price."""
        try:
            # Calculate limit price based on BBO
            price = await self.calculate_limit_price()
            
            self.logger.log(f"Placing {self.side} limit order on Lighter for {self.quantity} {self.ticker} at price {price}", "INFO")
            
            # Place limit order on Lighter using place_close_order for limit orders
            direction = "long" if self.side == "buy" else "short"
            order_result = await self.lighter_client.place_close_order(
                contract_id=self.ticker,
                quantity=self.quantity,
                price=price,
                side=self.side
            )
            
            if order_result.success:
                self.lighter_order_id = order_result.order_id
                self.order_placement_time = time.time()
                # Reset filled status for new order
                self.lighter_order_filled = False
                self.logger.log(f"Lighter order placed successfully: {self.lighter_order_id} (attempt {self.current_retry_count + 1})", "INFO")
                return True
            else:
                self.logger.log(f"Failed to place Lighter order: {order_result.error_message}", "ERROR")
                return False
        except Exception as e:
            self.logger.log(f"Error placing Lighter order: {e}", "ERROR")
            return False

    async def monitor_lighter_order(self):
        """Monitor the Lighter order until it's filled or timeout occurs."""
        self.logger.log(f"Monitoring Lighter order {self.lighter_order_id}", "INFO")
        
        # Set up order update handler for WebSocket if available
        try:
            self.lighter_client.setup_order_update_handler(self.on_lighter_order_update)
            self.logger.log("Successfully set up WebSocket order update handler", "INFO")
        except Exception as e:
            self.logger.log(f"Failed to set up WebSocket updates, falling back to polling: {e}", "WARNING")
        
        # Polling loop
        poll_count = 0
        last_price_check_time = time.time()
        
        while self.is_running and not self.lighter_order_filled:
            try:
                poll_count += 1
                self.logger.log(f"Polling Lighter order status (attempt {poll_count})", "DEBUG")
                
                current_time = time.time()
                
                # Check if order timeout has occurred
                if self.order_placement_time and (current_time - self.order_placement_time > self.order_timeout_seconds):
                    self.logger.log(f"Order timeout reached ({self.order_timeout_seconds}s). Canceling and replacing order.", "INFO")
                    await self.cancel_lighter_order()
                    
                    # Check if we should retry
                    if self.current_retry_count < self.max_retries:
                        self.current_retry_count += 1
                        self.logger.log(f"Auto-replacing order due to timeout (attempt {self.current_retry_count}/{self.max_retries})", "INFO")
                        if await self.place_lighter_limit_order():
                            # Reset timeout for new order
                            last_price_check_time = time.time()  # Reset price check timer
                            continue
                        else:
                            self.logger.log("Failed to place replacement order", "ERROR")
                            return False
                    else:
                        self.logger.log(f"Maximum retry attempts ({self.max_retries}) reached. Giving up.", "WARNING")
                        return False
                
                # Check if we should cancel and replace order based on price movement
                if (current_time - last_price_check_time >= self.price_check_interval):
                    self.logger.log("Checking if order should be canceled and replaced due to price movement", "DEBUG")
                    if await self.should_cancel_and_replace_order():
                        replace_result = await self.cancel_and_replace_lighter_order()
                        if replace_result:
                            # Reset timers for new order
                            last_price_check_time = time.time()
                            self.order_placement_time = time.time()
                            continue
                        else:
                            self.logger.log("Failed to replace order, continuing with current order", "WARNING")
                    last_price_check_time = time.time()
                
                # First check current_order from WebSocket if available
                if hasattr(self.lighter_client, 'current_order') and self.lighter_client.current_order:
                    current_order = self.lighter_client.current_order
                    if current_order.order_id == self.lighter_order_id:
                        status_upper = str(current_order.status).upper()
                        self.logger.log(f"Current order status from WebSocket: {status_upper}, filled: {current_order.filled_size}/{current_order.size}", "DEBUG")
                        
                        if status_upper == "FILLED":
                            self.lighter_order_filled = True
                            self.logger.log(f"Lighter order {self.lighter_order_id} filled (WebSocket)", "INFO")
                            return True
                        elif status_upper in ("CANCELED", "EXPIRED"):
                            self.logger.log(f"Lighter order {self.lighter_order_id} {status_upper} (WebSocket)", "WARNING")
                            return False
                        elif current_order.filled_size > 0:
                            self.logger.log(f"Order {self.lighter_order_id} partially filled: {current_order.filled_size}/{current_order.size} (WebSocket)", "INFO")
                
                # Fallback: Get active orders from Lighter API
                self.logger.log(f"Fetching active orders for contract: {self.lighter_client.config.contract_id}", "DEBUG")
                orders = await self.lighter_client.get_active_orders(self.lighter_client.config.contract_id)
                self.logger.log(f"Found {len(orders)} active orders", "DEBUG")

                found = False
                for order in orders:
                    if order.order_id == self.lighter_order_id:
                        found = True
                        status_upper = str(order.status).upper()
                        self.logger.log(f"Order {order.order_id} status: {status_upper}, filled: {order.filled_size}/{order.size}", "DEBUG")
                        
                        if status_upper == "FILLED":
                            self.lighter_order_filled = True
                            self.logger.log(f"Lighter order {self.lighter_order_id} filled", "INFO")
                            return True
                        elif status_upper in ("CANCELED", "EXPIRED"):
                            self.logger.log(f"Lighter order {self.lighter_order_id} {status_upper}", "WARNING")
                            return False
                        elif order.filled_size > 0:
                            self.logger.log(f"Order {self.lighter_order_id} partially filled: {order.filled_size}/{order.size}", "INFO")
                        break

                # If not found in active orders, check order info as final fallback
                if not found:
                    self.logger.log(f"Order {self.lighter_order_id} not found in active orders, checking order info", "DEBUG")
                    info = await self.lighter_client.get_order_info(self.lighter_order_id)
                    if info:
                        self.logger.log(f"Order info status: {info.status}", "DEBUG")
                        if info.status == "FILLED":
                            self.lighter_order_filled = True
                            self.logger.log(f"Lighter order {self.lighter_order_id} filled (by order info)", "INFO")
                            return True
                    else:
                        self.logger.log(f"Could not get order info for {self.lighter_order_id}", "WARNING")
                
                # Wait before checking again
                self.logger.log("Waiting 2 seconds before next status check", "DEBUG")
                await asyncio.sleep(2)
            except Exception as e:
                self.logger.log(f"Error monitoring Lighter order: {e}", "ERROR")
                self.logger.log("Waiting 5 seconds before retrying after error", "DEBUG")
                await asyncio.sleep(5)  # Longer wait on error
        
        self.logger.log(f"Exiting monitor loop, lighter_order_filled: {self.lighter_order_filled}", "DEBUG")
        return self.lighter_order_filled
    
    def on_lighter_order_update(self, order_update):
        """Callback for Lighter order updates via WebSocket."""
        if not order_update or not hasattr(order_update, 'order_id'):
            return
            
        if order_update.order_id == self.lighter_order_id:
            if order_update.status == "filled":
                self.lighter_order_filled = True
                self.logger.log(f"WebSocket: Lighter order {self.lighter_order_id} filled", "INFO")
            elif order_update.status == "canceled" or order_update.status == "expired":
                self.logger.log(f"WebSocket: Lighter order {self.lighter_order_id} {order_update.status}", "WARNING")

    async def place_paradex_hedge_order(self):
        """Place a hedge order on Paradex exchange (market order)."""
        # Determine opposite side for hedging
        hedge_side = "sell" if self.side == "buy" else "buy"
        self.logger.log(f"Placing {hedge_side} hedge order on Paradex for {self.quantity} {self.ticker}", "INFO")
        
        try:
            # Place market order on Paradex for immediate execution
            order_result = await self.paradex_client.place_market_order(
                contract_id=self.paradex_client.config.contract_id,
                quantity=self.quantity,
                direction=hedge_side
            )
            
            if order_result.success:
                self.paradex_order_id = order_result.order_id
                self.logger.log(f"Paradex hedge order placed successfully: {self.paradex_order_id}", "INFO")
                return True
            else:
                self.logger.log(f"Failed to place Paradex hedge order: {order_result.error_message}", "ERROR")
                # Fallback to aggressive limit that crosses the spread
                self.logger.log("Falling back to aggressive limit hedge on Paradex", "WARNING")
                order_result = await self.paradex_client.place_aggressive_limit_order(
                    contract_id=self.paradex_client.config.contract_id,
                    quantity=self.quantity,
                    direction=hedge_side
                )
                if order_result and order_result.success:
                    self.paradex_order_id = order_result.order_id
                    self.logger.log(f"Paradex hedge order placed via limit: {self.paradex_order_id}", "INFO")
                    return True
                return False
        except Exception as e:
            self.logger.log(f"Error placing Paradex hedge order: {e}", "ERROR")
            return False

    async def run(self):
        """Run the hedging strategy with BBO pricing and order cancellation."""
        self.is_running = True
        self.logger.log("Starting hedge strategy with BBO pricing", "INFO")
        
        try:
            # Initialize connections
            self.logger.log("Initializing exchange connections", "DEBUG")
            if not await self.initialize():
                self.logger.log("Failed to initialize. Exiting.", "ERROR")
                return False
            
            # Reset retry counter
            self.current_retry_count = 0
            self.logger.log(f"Reset retry counter to {self.current_retry_count}", "DEBUG")
            
            # Place limit order on Lighter using BBO price
            self.logger.log("Placing initial Lighter limit order", "DEBUG")
            if not await self.place_lighter_limit_order():
                self.logger.log("Failed to place initial Lighter order. Exiting.", "ERROR")
                await self.shutdown()
                return False
            
            # Monitor Lighter order until filled or timeout/cancel
            self.logger.log("Starting to monitor Lighter order", "DEBUG")
            monitor_result = await self.monitor_lighter_order()
            self.logger.log(f"Lighter order monitoring completed with result: {monitor_result}", "DEBUG")
            
            if not monitor_result:
                self.logger.log("Lighter order monitoring failed after retries. Exiting.", "ERROR")
                # Make sure any pending orders are canceled
                await self.cancel_lighter_order()
                await self.shutdown()
                return False
            
            # If we got here, the order was filled successfully
            self.logger.log("Lighter order filled successfully. Placing hedge order on Paradex.", "INFO")
            
            # Place hedge order on Paradex (market for immediacy, fallback to aggressive limit)
            self.logger.log("Placing hedge order on Paradex", "DEBUG")
            hedge_result = await self.place_paradex_hedge_order()
            self.logger.log(f"Paradex hedge order placement result: {hedge_result}", "DEBUG")
            
            if not hedge_result:
                self.logger.log("Failed to place Paradex hedge order. Exiting.", "ERROR")
                await self.shutdown()
                return False
            
            self.logger.log("Hedge strategy completed successfully", "INFO")
            await self.shutdown()
            return True
            
        except Exception as e:
            self.logger.log(f"Error in hedge strategy: {e}", "ERROR")
            self.logger.log("Attempting to cancel any pending orders during error handling", "DEBUG")
            # Attempt to cancel any pending orders
            await self.cancel_lighter_order()
            await self.shutdown()
            return False


async def main():
    """Main function to run the hedge strategy."""
    strategy = None
    try:
        # Load configuration
        config = {
            "ticker": "ETH-PERP",
            "quantity": "0.1",
            "price": "3000",  # Set to 0 for market price
            "side": "buy",    # Initial side on Lighter
            "lighter": {
                "ticker": "ETH-PERP",
                "tick_size": Decimal("0.01")
            },
            "paradex": {
                "ticker": "ETH-PERP",
                "tick_size": Decimal("0.01")
            }
        }
        
        # Create and run strategy
        strategy = HedgeStrategy(config)
        await strategy.run()
    except KeyboardInterrupt:
        print("\nReceived interrupt signal, shutting down...")
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        # Ensure proper cleanup
        if strategy:
            try:
                await strategy.shutdown()
            except Exception as e:
                print(f"Error during shutdown: {e}")


if __name__ == "__main__":
    asyncio.run(main())