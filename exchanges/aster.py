"""
Aster exchange client implementation.
"""

import os
import asyncio
import json
import time
import hmac
import hashlib
from decimal import Decimal
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlencode
import aiohttp
import websockets
import sys

from .base import BaseExchangeClient, OrderResult, OrderInfo, query_retry
from helpers.logger import TradingLogger


class AsterWebSocketManager:
    """WebSocket manager for Aster order updates."""

    def __init__(self, config: Dict[str, Any], api_key: str, secret_key: str, order_update_callback):
        self.api_key = api_key
        self.secret_key = secret_key
        self.order_update_callback = order_update_callback
        self.websocket = None
        self.running = False
        self.base_url = "https://fapi.asterdex.com"
        self.ws_url = "wss://fstream.asterdex.com"
        self.listen_key = None
        self.logger = None
        self._keepalive_task = None
        self._last_ping_time = None
        self.config = config

    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """Generate HMAC SHA256 signature for Aster API authentication."""
        # Use urlencode to properly format the query string
        query_string = urlencode(params)

        # Generate HMAC SHA256 signature
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        return signature

    async def _get_listen_key(self) -> str:
        """Get listen key for user data stream."""
        params = {
            'timestamp': int(time.time() * 1000)
        }
        signature = self._generate_signature(params)
        params['signature'] = signature

        headers = {
            'X-MBX-APIKEY': self.api_key,
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                'https://fapi.asterdex.com/fapi/v1/listenKey',
                headers=headers,
                data=params
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get('listenKey')
                else:
                    raise Exception(f"Failed to get listen key: {response.status}")

    async def _keepalive_listen_key(self) -> bool:
        """Keep alive the listen key to prevent timeout."""
        try:
            if not self.listen_key:
                return False

            params = {
                'timestamp': int(time.time() * 1000)
            }
            signature = self._generate_signature(params)
            params['signature'] = signature

            headers = {
                'X-MBX-APIKEY': self.api_key,
                'Content-Type': 'application/x-www-form-urlencoded'
            }

            async with aiohttp.ClientSession() as session:
                async with session.put(
                    f"{self.base_url}/fapi/v1/listenKey",
                    headers=headers,
                    data=params
                ) as response:
                    if response.status == 200:
                        if self.logger:
                            self.logger.log("Listen key keepalive successful", "DEBUG")
                        return True
                    else:
                        if self.logger:
                            self.logger.log(f"Failed to keepalive listen key: {response.status}", "WARNING")
                        return False
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error keeping alive listen key: {e}", "ERROR")
            return False

    async def _check_connection_health(self) -> bool:
        """Check if the WebSocket connection is healthy based on ping timing."""
        if not self._last_ping_time:
            return True  # No pings received yet, assume healthy

        # Check if we haven't received a ping in the last 10 minutes
        # (server sends pings every 5 minutes, so 10 minutes indicates a problem)
        time_since_last_ping = time.time() - self._last_ping_time
        if time_since_last_ping > 10 * 60:  # 10 minutes
            if self.logger:
                self.logger.log(
                    f"No ping received for {time_since_last_ping/60:.1f} minutes, "
                    "connection may be unhealthy", "WARNING"
                )
            return False

        return True

    async def _start_keepalive_task(self):
        """Start the keepalive task to extend listen key validity and monitor connection health."""
        while self.running:
            try:
                # Check connection health every 5 minutes
                await asyncio.sleep(5 * 60)

                if not self.running:
                    break

                # Check if connection is healthy
                if not await self._check_connection_health():
                    if self.logger:
                        self.logger.log("Connection health check failed, reconnecting...", "WARNING")
                    # Try to reconnect
                    try:
                        await self.connect()
                    except Exception as e:
                        if self.logger:
                            self.logger.log(f"Reconnection failed: {e}", "ERROR")
                        # Wait before retrying
                        await asyncio.sleep(30)
                    continue

                # Check if we need to keepalive the listen key (every 50 minutes)
                if self.listen_key and time.time() % (50 * 60) < 5 * 60:  # Within 5 minutes of 50-minute mark
                    success = await self._keepalive_listen_key()
                    if not success:
                        if self.logger:
                            self.logger.log("Listen key keepalive failed, reconnecting...", "WARNING")
                        # Try to reconnect
                        try:
                            await self.connect()
                        except Exception as e:
                            if self.logger:
                                self.logger.log(f"Reconnection failed: {e}", "ERROR")
                            # Wait before retrying
                            await asyncio.sleep(30)

            except Exception as e:
                if self.logger:
                    self.logger.log(f"Error in keepalive task: {e}", "ERROR")
                # Wait a bit before retrying
                await asyncio.sleep(60)

    async def connect(self):
        """Connect to Aster WebSocket."""
        try:
            # Get listen key
            self.listen_key = await self._get_listen_key()
            if not self.listen_key:
                raise Exception("Failed to get listen key")

            # Connect to WebSocket
            ws_url = f"{self.ws_url}/ws/{self.listen_key}"
            self.websocket = await websockets.connect(ws_url)
            self.running = True

            if self.logger:
                self.logger.log("Connected to Aster WebSocket with listen key", "INFO")

            # Start keepalive task
            self._keepalive_task = asyncio.create_task(self._start_keepalive_task())

            # Start listening for messages
            await self._listen()

        except Exception as e:
            if self.logger:
                self.logger.log(f"WebSocket connection error: {e}", "ERROR")
            raise

    async def _listen(self):
        """Listen for WebSocket messages."""
        try:
            async for message in self.websocket:
                if not self.running:
                    break

                # Check if this is a ping frame (websockets library handles pong automatically)
                if isinstance(message, bytes) and message == b'\x89\x00':  # Ping frame
                    self._last_ping_time = time.time()
                    if self.logger:
                        self.logger.log("Received ping frame, sending pong", "DEBUG")
                    continue

                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError as e:
                    if self.logger:
                        self.logger.log(f"Failed to parse WebSocket message: {e}", "ERROR")
                except Exception as e:
                    if self.logger:
                        self.logger.log(f"Error handling WebSocket message: {e}", "ERROR")

        except websockets.exceptions.ConnectionClosed:
            if self.logger:
                self.logger.log("WebSocket connection closed", "WARNING")
        except Exception as e:
            if self.logger:
                self.logger.log(f"WebSocket listen error: {e}", "ERROR")

    async def _handle_message(self, data: Dict[str, Any]):
        """Handle incoming WebSocket messages."""
        try:
            event_type = data.get('e', '')

            if event_type == 'ORDER_TRADE_UPDATE':
                await self._handle_order_update(data)
            elif event_type == 'listenKeyExpired':
                if self.logger:
                    self.logger.log("Listen key expired, reconnecting...", "WARNING")
                # Reconnect with new listen key
                await self.connect()
            else:
                if self.logger:
                    self.logger.log(f"Unknown WebSocket message: {data}", "DEBUG")

        except Exception as e:
            if self.logger:
                self.logger.log(f"Error handling WebSocket message: {e}", "ERROR")

    async def _handle_order_update(self, order_data: Dict[str, Any]):
        """Handle order update messages."""
        try:
            order_info = order_data.get('o', {})

            order_id = order_info.get('i', '')
            symbol = order_info.get('s', '')
            side = order_info.get('S', '')
            quantity = order_info.get('q', '0')
            price = order_info.get('p', '0')
            executed_qty = order_info.get('z', '0')
            status = order_info.get('X', '')

            # Map status
            status_map = {
                'NEW': 'OPEN',
                'PARTIALLY_FILLED': 'PARTIALLY_FILLED',
                'FILLED': 'FILLED',
                'CANCELED': 'CANCELED',
                'REJECTED': 'REJECTED',
                'EXPIRED': 'EXPIRED'
            }
            mapped_status = status_map.get(status, status)

            # Call the order update callback if it exists
            if hasattr(self, 'order_update_callback') and self.order_update_callback:
                if side.lower() == self.config.close_order_side:
                    order_type = "CLOSE"
                else:
                    order_type = "OPEN"

                await self.order_update_callback({
                    'order_id': order_id,
                    'side': side.lower(),
                    'order_type': order_type,
                    'status': mapped_status,
                    'size': quantity,
                    'price': price,
                    'contract_id': symbol,
                    'filled_size': executed_qty
                })

        except Exception as e:
            if self.logger:
                self.logger.log(f"Error handling order update: {e}", "ERROR")

    async def disconnect(self):
        """Disconnect from WebSocket."""
        self.running = False

        # Cancel keepalive task
        if self._keepalive_task and not self._keepalive_task.done():
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass

        if self.websocket:
            await self.websocket.close()
            if self.logger:
                self.logger.log("WebSocket disconnected", "INFO")

    def set_logger(self, logger):
        """Set the logger instance."""
        self.logger = logger


class AsterClient(BaseExchangeClient):
    """Aster exchange client implementation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Aster client."""
        super().__init__(config)

        # Aster credentials from environment
        self.api_key = os.getenv('ASTER_API_KEY')
        self.secret_key = os.getenv('ASTER_SECRET_KEY')
        self.base_url = 'https://fapi.asterdex.com'

        if not self.api_key or not self.secret_key:
            raise ValueError(
                "ASTER_API_KEY and ASTER_SECRET_KEY must be set in environment variables"
            )

        # Initialize logger early
        self.logger = TradingLogger(exchange="aster", ticker=self.config.ticker, log_to_console=False)
        self._order_update_handler = None

    def _validate_config(self) -> None:
        """Validate Aster configuration."""
        required_env_vars = ['ASTER_API_KEY', 'ASTER_SECRET_KEY']
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")

    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """Generate HMAC SHA256 signature for Aster API authentication."""
        # Use urlencode to properly format the query string
        query_string = urlencode(params)

        # Generate HMAC SHA256 signature
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        return signature

    async def _make_request(
        self, method: str, endpoint: str, params: Dict[str, Any] = None, data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Make authenticated request to Aster API."""
        if params is None:
            params = {}
        if data is None:
            data = {}

        # Add timestamp and recvWindow
        timestamp = int(time.time() * 1000)
        params['timestamp'] = timestamp
        params['recvWindow'] = 5000

        url = f"{self.base_url}{endpoint}"
        headers = {
            'X-MBX-APIKEY': self.api_key,
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        async with aiohttp.ClientSession() as session:
            if method.upper() == 'GET':
                # For GET requests, signature is based on query parameters only
                signature = self._generate_signature(params)
                params['signature'] = signature

                async with session.get(url, params=params, headers=headers) as response:
                    result = await response.json()
                    if response.status != 200:
                        raise Exception(f"API request failed: {result}")
                    return result
            elif method.upper() == 'POST':
                # For POST requests, signature must include both query string and request body
                # According to Aster API docs: totalParams = queryString + requestBody
                all_params = {**params, **data}
                signature = self._generate_signature(all_params)
                all_params['signature'] = signature

                async with session.post(url, data=all_params, headers=headers) as response:
                    result = await response.json()
                    if response.status != 200:
                        raise Exception(f"API request failed: {result}")
                    return result
            elif method.upper() == 'DELETE':
                # For DELETE requests, signature is based on query parameters only
                signature = self._generate_signature(params)
                params['signature'] = signature

                async with session.delete(url, params=params, headers=headers) as response:
                    result = await response.json()
                    if response.status != 200:
                        raise Exception(f"API request failed: {result}")
                    return result

    async def connect(self) -> None:
        """Connect to Aster WebSocket."""
        # Initialize WebSocket manager
        self.ws_manager = AsterWebSocketManager(
            config=self.config,
            api_key=self.api_key,
            secret_key=self.secret_key,
            order_update_callback=self._handle_websocket_order_update
        )

        # Set logger for WebSocket manager
        self.ws_manager.set_logger(self.logger)

        try:
            # Start WebSocket connection in background task
            asyncio.create_task(self.ws_manager.connect())
            # Wait a moment for connection to establish
            await asyncio.sleep(2)
        except Exception as e:
            self.logger.log(f"Error connecting to Aster WebSocket: {e}", "ERROR")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Aster."""
        try:
            if hasattr(self, 'ws_manager') and self.ws_manager:
                await self.ws_manager.disconnect()
        except Exception as e:
            self.logger.log(f"Error during Aster disconnect: {e}", "ERROR")

    def get_exchange_name(self) -> str:
        """Get the exchange name."""
        return "aster"

    def setup_order_update_handler(self, handler) -> None:
        """Setup order update handler for WebSocket."""
        self._order_update_handler = handler

    async def _handle_websocket_order_update(self, order_data: Dict[str, Any]):
        """Handle order updates from WebSocket."""
        try:
            if self._order_update_handler:
                self._order_update_handler(order_data)
        except Exception as e:
            self.logger.log(f"Error handling WebSocket order update: {e}", "ERROR")

    @query_retry(default_return=(0, 0))
    async def fetch_bbo_prices(self, contract_id: str) -> Tuple[Decimal, Decimal]:
        """Fetch best bid and ask prices from Aster."""
        result = await self._make_request('GET', '/fapi/v1/ticker/bookTicker', {'symbol': contract_id})

        best_bid = Decimal(result.get('bidPrice', 0))
        best_ask = Decimal(result.get('askPrice', 0))

        return best_bid, best_ask

    async def get_order_price(self, direction: str) -> Decimal:
        """Get the price of an order with Aster using official SDK."""
        best_bid, best_ask = await self.fetch_bbo_prices(self.config.contract_id)
        if best_bid <= 0 or best_ask <= 0:
            self.logger.log("Invalid bid/ask prices", "ERROR")
            raise ValueError("Invalid bid/ask prices")

        if direction == 'buy':
            # For buy orders, place slightly below best ask to ensure execution
            order_price = best_ask - self.config.tick_size
        else:
            # For sell orders, place slightly above best bid to ensure execution
            order_price = best_bid + self.config.tick_size
        return order_price

    async def place_open_order(self, contract_id: str, quantity: Decimal, direction: str) -> OrderResult:
        """Place an open order with Aster."""
        attempt = 0
        while True:
            attempt += 1
            if attempt % 5 == 0:
                self.logger.log(f"[OPEN] Attempt {attempt} to place order", "INFO")
                active_orders = await self.get_active_orders(contract_id)
                active_open_orders = 0
                for order in active_orders:
                    if order.side == self.config.direction:
                        active_open_orders += 1
                if active_open_orders > 1:
                    self.logger.log(f"[OPEN] ERROR: Active open orders abnormal: {active_open_orders}", "ERROR")
                    raise Exception(f"[OPEN] ERROR: Active open orders abnormal: {active_open_orders}")

            best_bid, best_ask = await self.fetch_bbo_prices(contract_id)

            if best_bid <= 0 or best_ask <= 0:
                return OrderResult(success=False, error_message='Invalid bid/ask prices')

            # Determine order side and price
            if direction == 'buy':
                # For buy orders, place slightly below best ask to ensure execution
                price = best_ask - self.config.tick_size
            elif direction == 'sell':
                # For sell orders, place slightly above best bid to ensure execution
                price = best_bid + self.config.tick_size
            else:
                raise Exception(f"[OPEN] Invalid direction: {direction}")

            # Place the order
            order_data = {
                'symbol': contract_id,
                'side': direction.upper(),
                'type': 'LIMIT',
                'quantity': str(quantity),
                'price': str(price),
                'timeInForce': 'GTX'  # GTX is Good Till Crossing (Post Only)
            }

            result = await self._make_request('POST', '/fapi/v1/order', data=order_data)
            order_status = result.get('status', '')
            order_id = result.get('orderId', '')

            start_time = time.time()
            while order_status == 'NEW' and time.time() - start_time < 2:
                await asyncio.sleep(0.1)
                order_info = await self.get_order_info(order_id)
                if order_info is not None:
                    order_status = order_info.status

            if order_status in ['NEW', 'PARTIALLY_FILLED']:
                return OrderResult(success=True, order_id=order_id, side=direction, size=quantity, price=price, status='OPEN')
            elif order_status == 'FILLED':
                return OrderResult(success=True, order_id=order_id, side=direction, size=quantity, price=price, status='FILLED')
            elif order_status == 'EXPIRED':
                continue
            else:
                return OrderResult(success=False, error_message='Unknown order status: ' + order_status)

    async def _get_active_close_orders(self, contract_id: str) -> int:
        """Get active close orders for a contract using official SDK."""
        active_orders = await self.get_active_orders(contract_id)
        active_close_orders = 0
        for order in active_orders:
            if order.side == self.config.close_order_side:
                active_close_orders += 1
        return active_close_orders

    async def place_close_order(self, contract_id: str, quantity: Decimal, price: Decimal, side: str) -> OrderResult:
        """Place a close order with Aster."""
        attempt = 0
        active_close_orders = await self._get_active_close_orders(contract_id)
        while True:
            attempt += 1
            if attempt % 5 == 0:
                self.logger.log(f"[CLOSE] Attempt {attempt} to place order", "INFO")
                current_close_orders = await self._get_active_close_orders(contract_id)

                if current_close_orders - active_close_orders > 1:
                    self.logger.log(f"[CLOSE] ERROR: Active close orders abnormal: "
                                    f"{active_close_orders}, {current_close_orders}", "ERROR")
                    raise Exception(f"[CLOSE] ERROR: Active close orders abnormal: "
                                    f"{active_close_orders}, {current_close_orders}")
                else:
                    active_close_orders = current_close_orders
            # Get current market prices to adjust order price if needed
            best_bid, best_ask = await self.fetch_bbo_prices(contract_id)

            if best_bid <= 0 or best_ask <= 0:
                return OrderResult(success=False, error_message='No bid/ask data available')

            # Adjust order price based on market conditions and side
            adjusted_price = price
            if side.lower() == 'sell':
                order_side = 'SELL'
                # For sell orders, ensure price is above best bid to be a maker order
                if price <= best_bid:
                    adjusted_price = best_bid + self.config.tick_size
            elif side.lower() == 'buy':
                order_side = 'BUY'
                # For buy orders, ensure price is below best ask to be a maker order
                if price >= best_ask:
                    adjusted_price = best_ask - self.config.tick_size

            adjusted_price = self.round_to_tick(adjusted_price)

            # Place the order
            order_data = {
                'symbol': contract_id,
                'side': order_side,
                'type': 'LIMIT',
                'quantity': str(quantity),
                'price': str(adjusted_price),
                'timeInForce': 'GTX'  # GTX is Good Till Crossing (Post Only)
            }

            result = await self._make_request('POST', '/fapi/v1/order', data=order_data)
            order_status = result.get('status', '')
            order_id = result.get('orderId', '')

            start_time = time.time()
            while order_status == 'NEW' and time.time() - start_time < 2:
                await asyncio.sleep(0.1)
                order_info = await self.get_order_info(order_id)
                if order_info is not None:
                    order_status = order_info.status

            if order_status in ['NEW', 'PARTIALLY_FILLED']:
                return OrderResult(success=True, order_id=order_id, side=order_side.lower(),
                                   size=quantity, price=adjusted_price, status='OPEN')
            elif order_status == 'FILLED':
                return OrderResult(success=True, order_id=order_id, side=order_side.lower(),
                                   size=quantity, price=adjusted_price, status='FILLED')
            elif order_status == 'EXPIRED':
                continue
            else:
                return OrderResult(success=False, error_message='Unknown order status: ' + order_status)

    async def place_market_order(self, contract_id: str, quantity: Decimal, direction: str) -> OrderResult:
        """Place a market order with Aster."""
        # Validate direction
        if direction.lower() not in ['buy', 'sell']:
            return OrderResult(success=False, error_message=f'Invalid direction: {direction}')

        # Place the market order
        order_data = {
            'symbol': contract_id,
            'side': direction.upper(),
            'type': 'MARKET',
            'quantity': str(quantity)
        }

        result = await self._make_request('POST', '/fapi/v1/order', data=order_data)
        order_status = result.get('status', '')
        order_id = result.get('orderId', '')

        start_time = time.time()
        while order_status != 'FILLED' and time.time() - start_time < 10:
            await asyncio.sleep(0.2)
            order_info = await self.get_order_info(order_id)
            if order_info is not None:
                order_status = order_info.status

        if order_status != 'FILLED':
            self.logger.log(f"Market order failed with status: {order_status}", "ERROR")
            sys.exit(1)
        # For market orders, we expect them to be filled immediately
        else:
            return OrderResult(
                success=True,
                order_id=order_id,
                side=direction.lower(),
                size=quantity,
                price=order_info.price,
                status='FILLED'
            )

    async def cancel_order(self, order_id: str) -> OrderResult:
        """Cancel an order with Aster."""
        try:
            result = await self._make_request('DELETE', '/fapi/v1/order', {
                'symbol': self.config.contract_id,
                'orderId': order_id
            })

            if 'orderId' in result:
                return OrderResult(success=True, filled_size=Decimal(result.get('executedQty', 0)))
            else:
                return OrderResult(success=False, error_message=result.get('msg', 'Unknown error'))

        except Exception as e:
            return OrderResult(success=False, error_message=str(e))

    @query_retry()
    async def get_order_info(self, order_id: str) -> Optional[OrderInfo]:
        """Get order information from Aster."""
        result = await self._make_request('GET', '/fapi/v1/order', {
            'symbol': self.config.contract_id,
            'orderId': order_id
        })

        order_type = result.get('type', '')
        if order_type == 'MARKET':
            price = Decimal(result.get('avgPrice', 0))
        else:
            price = Decimal(result.get('price', 0))

        if 'orderId' in result:
            return OrderInfo(
                order_id=str(result['orderId']),
                side=result.get('side', '').lower(),
                size=Decimal(result.get('origQty', 0)),
                price=price,
                status=result.get('status', ''),
                filled_size=Decimal(result.get('executedQty', 0)),
                remaining_size=Decimal(result.get('origQty', 0)) - Decimal(result.get('executedQty', 0))
            )
        return None

    @query_retry(default_return=[])
    async def get_active_orders(self, contract_id: str) -> List[OrderInfo]:
        """Get active orders for a contract from Aster."""
        result = await self._make_request('GET', '/fapi/v1/openOrders', {'symbol': contract_id})

        orders = []
        for order in result:
            orders.append(OrderInfo(
                order_id=str(order['orderId']),
                side=order.get('side', '').lower(),
                size=Decimal(order.get('origQty', 0)) - Decimal(order.get('executedQty', 0)),
                price=Decimal(order.get('price', 0)),
                status=order.get('status', ''),
                filled_size=Decimal(order.get('executedQty', 0)),
                remaining_size=Decimal(order.get('origQty', 0)) - Decimal(order.get('executedQty', 0))
            ))

        return orders

    @query_retry(reraise=True)
    async def get_account_positions(self) -> Decimal:
        """Get account positions from Aster."""
        result = await self._make_request('GET', '/fapi/v2/positionRisk', {'symbol': self.config.contract_id})

        for position in result:
            if position.get('symbol') == self.config.contract_id:
                position_amt = abs(Decimal(position.get('positionAmt', 0)))
                return position_amt

        return Decimal(0)

    @query_retry(reraise=True)
    async def get_real_position(self) -> Decimal:
        """Get real account positions from Aster with sign based on positionSide.
        
        Returns:
            Decimal: Position amount with sign (negative for SHORT, positive for LONG)
        """
        result = await self._make_request('GET', '/fapi/v2/positionRisk', {'symbol': self.config.contract_id})

        for position in result:
            if position.get('symbol') == self.config.contract_id:
                position_amt = Decimal(position.get('positionAmt', 0))
                position_side = position.get('positionSide', 'BOTH')
                
                # 根据 positionSide 确定正负号
                if position_side == 'SHORT':
                    # 空仓返回负数
                    return -abs(position_amt)
                elif position_side == 'LONG':
                    # 多仓返回正数
                    return abs(position_amt)
                else:
                    # BOTH 模式或其他情况，使用原始值
                    return position_amt

        return Decimal(0)

    async def get_contract_attributes(self) -> Tuple[str, Decimal]:
        """Get contract ID and tick size for a ticker."""
        ticker = self.config.ticker
        if len(ticker) == 0:
            self.logger.log("Ticker is empty", "ERROR")
            raise ValueError("Ticker is empty")

        try:
            result = await self._make_request('GET', '/fapi/v1/exchangeInfo')

            for symbol_info in result['symbols']:
                if (symbol_info.get('status') == 'TRADING' and
                        symbol_info.get('baseAsset') == ticker and
                        symbol_info.get('quoteAsset') == 'USDT'):

                    self.config.contract_id = symbol_info.get('symbol', '')

                    # Get tick size from filters
                    for filter_info in symbol_info.get('filters', []):
                        if filter_info.get('filterType') == 'PRICE_FILTER':
                            self.config.tick_size = Decimal(filter_info['tickSize'].strip('0'))
                            break

                    # Get minimum quantity
                    min_quantity = Decimal(0)
                    for filter_info in symbol_info.get('filters', []):
                        if filter_info.get('filterType') == 'LOT_SIZE':
                            min_quantity = Decimal(filter_info.get('minQty', 0))
                            break

                    if self.config.quantity < min_quantity:
                        self.logger.log(
                            f"Order quantity is less than min quantity: "
                            f"{self.config.quantity} < {min_quantity}", "ERROR"
                        )
                        raise ValueError(
                            f"Order quantity is less than min quantity: "
                            f"{self.config.quantity} < {min_quantity}"
                        )

                    if self.config.tick_size == 0:
                        self.logger.log("Failed to get tick size for ticker", "ERROR")
                        raise ValueError("Failed to get tick size for ticker")

                    return self.config.contract_id, self.config.tick_size

            self.logger.log("Failed to get contract ID for ticker", "ERROR")
            raise ValueError("Failed to get contract ID for ticker")

        except Exception as e:
            self.logger.log(f"Error getting contract attributes: {e}", "ERROR")
            raise
