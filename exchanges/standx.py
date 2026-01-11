"""
StandX exchange client implementation.
Custom HTTP/WebSocket implementation since no Python SDK exists.
"""

import os
import asyncio
import json
import time
import uuid
import base64
import aiohttp
import websockets
from decimal import Decimal
from typing import Dict, Any, List, Optional, Tuple, Callable
from nacl.signing import SigningKey
from nacl.encoding import RawEncoder
import nacl.bindings
from dotenv import load_dotenv

from eth_account import Account as EthAccount
from eth_account.messages import encode_defunct

from .base import BaseExchangeClient, OrderResult, OrderInfo, query_retry
from helpers.logger import TradingLogger

# Load environment variables from .env file
load_dotenv()


def base58_encode(data: bytes) -> str:
    """Encode bytes to base58 string."""
    alphabet = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
    n = int.from_bytes(data, 'big')
    result = ''
    while n > 0:
        n, remainder = divmod(n, 58)
        result = alphabet[remainder] + result
    # Handle leading zeros
    for byte in data:
        if byte == 0:
            result = '1' + result
        else:
            break
    return result or '1'


class StandXAuth:
    """StandX authentication handler with JWT and ed25519 body signing."""

    def __init__(self, wallet_address: str, private_key: str, chain: str = "bsc"):
        self.wallet_address = wallet_address
        self.private_key = private_key
        self.chain = chain
        self.base_url = "https://api.standx.com"
        self.perps_url = "https://perps.standx.com"

        # Generate ed25519 key pair for body signing
        self.ed25519_private_key = SigningKey.generate()
        self.ed25519_public_key = self.ed25519_private_key.verify_key
        self.request_id = base58_encode(bytes(self.ed25519_public_key))

        # JWT token (populated after login)
        self.token: Optional[str] = None
        self.session_id = str(uuid.uuid4())

    def _parse_jwt(self, token: str) -> Dict[str, Any]:
        """Parse JWT payload without verification."""
        parts = token.split('.')
        if len(parts) != 3:
            raise ValueError("Invalid JWT format")

        # Base64 decode the payload (middle part)
        payload_b64 = parts[1]
        # Add padding if needed
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += '=' * padding
        # Handle URL-safe base64
        payload_b64 = payload_b64.replace('-', '+').replace('_', '/')
        payload_bytes = base64.b64decode(payload_b64)
        return json.loads(payload_bytes.decode('utf-8'))

    async def authenticate(self) -> str:
        """Complete authentication flow and return JWT token."""
        async with aiohttp.ClientSession() as session:
            # Step 1: Prepare sign-in
            signed_data_jwt = await self._prepare_signin(session)

            # Step 2: Parse JWT to get message
            payload = self._parse_jwt(signed_data_jwt)
            message = payload.get('message', '')

            # Step 3: Sign message with wallet private key (BSC/ETH compatible)
            signature = self._sign_message_with_wallet(message)

            # Step 4: Login to get access token
            self.token = await self._login(session, signature, signed_data_jwt)

            return self.token

    async def _prepare_signin(self, session: aiohttp.ClientSession) -> str:
        """Request signature data from server."""
        url = f"{self.base_url}/v1/offchain/prepare-signin?chain={self.chain}"
        data = {
            "address": self.wallet_address,
            "requestId": self.request_id
        }

        async with session.post(url, json=data) as response:
            result = await response.json()
            if not result.get('success'):
                raise ValueError(f"Failed to prepare sign-in: {result}")
            return result['signedData']

    def _sign_message_with_wallet(self, message: str) -> str:
        """Sign message with BSC/ETH wallet private key."""
        account = EthAccount.from_key(self.private_key)
        message_encoded = encode_defunct(text=message)
        signed = account.sign_message(message_encoded)
        return signed.signature.hex()

    async def _login(self, session: aiohttp.ClientSession, signature: str, signed_data: str) -> str:
        """Login to get access token."""
        url = f"{self.base_url}/v1/offchain/login?chain={self.chain}"
        data = {
            "signature": signature if signature.startswith('0x') else f"0x{signature}",
            "signedData": signed_data,
            "expiresSeconds": 604800  # 7 days
        }

        async with session.post(url, json=data) as response:
            result = await response.json()
            if 'token' not in result:
                raise ValueError(f"Login failed: {result}")
            return result['token']

    def sign_request(self, payload: str) -> Dict[str, str]:
        """Sign request body with ed25519 and return headers."""
        version = "v1"
        request_id = str(uuid.uuid4())
        timestamp = int(time.time() * 1000)

        # Build message: "{version},{id},{timestamp},{payload}"
        message = f"{version},{request_id},{timestamp},{payload}"
        message_bytes = message.encode('utf-8')

        # Sign with ed25519
        signed = self.ed25519_private_key.sign(message_bytes, encoder=RawEncoder)
        signature = base64.b64encode(signed.signature).decode('utf-8')

        return {
            "x-request-sign-version": version,
            "x-request-id": request_id,
            "x-request-timestamp": str(timestamp),
            "x-request-signature": signature
        }

    def get_auth_headers(self, payload: str = "") -> Dict[str, str]:
        """Get full authentication headers for a request."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
            "x-session-id": self.session_id
        }
        if payload:
            headers.update(self.sign_request(payload))
        return headers


class StandXWebSocketManager:
    """WebSocket manager for StandX market data and order updates."""

    def __init__(self, symbol: str, token: str, session_id: str,
                 depth_callback: Optional[Callable] = None,
                 order_callback: Optional[Callable] = None,
                 price_callback: Optional[Callable] = None):
        self.symbol = symbol
        self.token = token
        self.session_id = session_id
        self.depth_callback = depth_callback
        self.order_callback = order_callback
        self.price_callback = price_callback
        self.ws_url = "wss://perps.standx.com/ws-stream/v1"
        self.websocket = None
        self.running = False
        self.logger = None

        # Cache for BBO prices
        self.best_bid: Decimal = Decimal('0')
        self.best_ask: Decimal = Decimal('0')

        # Cache for mark price
        self.mark_price: Decimal = Decimal('0')

    async def connect(self):
        """Connect to StandX WebSocket and subscribe to channels."""
        while self.running:
            try:
                if self.logger:
                    self.logger.log("Connecting to StandX WebSocket", "INFO")

                async with websockets.connect(self.ws_url) as ws:
                    self.websocket = ws

                    # Authenticate and subscribe to order channel
                    auth_msg = {
                        "auth": {
                            "token": self.token,
                            "streams": [{"channel": "order"}]
                        }
                    }
                    await ws.send(json.dumps(auth_msg))

                    # Subscribe to depth book for BBO prices
                    depth_msg = {
                        "subscribe": {
                            "channel": "depth_book",
                            "symbol": self.symbol
                        }
                    }
                    await ws.send(json.dumps(depth_msg))

                    # Subscribe to price channel for mark price
                    price_msg = {
                        "subscribe": {
                            "channel": "price",
                            "symbol": self.symbol
                        }
                    }
                    await ws.send(json.dumps(price_msg))

                    if self.logger:
                        self.logger.log(f"Subscribed to depth_book, price, and order channels for {self.symbol}", "INFO")

                    # Listen for messages
                    await self._listen(ws)

            except websockets.exceptions.ConnectionClosed as e:
                if self.logger:
                    self.logger.log(f"WebSocket connection closed: {e}", "WARNING")
                if self.running:
                    await asyncio.sleep(5)  # Reconnect delay
            except Exception as e:
                if self.logger:
                    self.logger.log(f"WebSocket error: {e}", "ERROR")
                if self.running:
                    await asyncio.sleep(5)

    async def _listen(self, ws):
        """Listen for WebSocket messages."""
        async for message in ws:
            if not self.running:
                break

            try:
                data = json.loads(message)
                await self._handle_message(data)
            except json.JSONDecodeError as e:
                if self.logger:
                    self.logger.log(f"Failed to parse WebSocket message: {e}", "ERROR")
            except Exception as e:
                if self.logger:
                    self.logger.log(f"Error handling message: {e}", "ERROR")

    async def _handle_message(self, data: Dict[str, Any]):
        """Handle incoming WebSocket messages."""
        channel = data.get('channel', '')

        if channel == 'depth_book':
            await self._handle_depth_book(data.get('data', {}))
        elif channel == 'price':
            await self._handle_price(data.get('data', {}))
        elif channel == 'order':
            await self._handle_order_update(data.get('data', {}))
        elif channel == 'auth':
            code = data.get('data', {}).get('code', 0)
            if code != 200:
                if self.logger:
                    self.logger.log(f"WebSocket auth failed: {data}", "ERROR")
            else:
                if self.logger:
                    self.logger.log("WebSocket authenticated successfully", "INFO")

    async def _handle_depth_book(self, data: Dict[str, Any]):
        """Handle depth book updates for BBO prices."""
        bids = data.get('bids', [])
        asks = data.get('asks', [])

        if bids:
            self.best_bid = Decimal(bids[0][0])
        if asks:
            self.best_ask = Decimal(asks[0][0])

        if self.depth_callback:
            await self.depth_callback(self.best_bid, self.best_ask)

    async def _handle_price(self, data: Dict[str, Any]):
        """Handle price channel updates for mark price."""
        mark_price_str = data.get('mark_price', '')
        if mark_price_str:
            self.mark_price = Decimal(mark_price_str)

        if self.price_callback:
            await self.price_callback(self.mark_price)

    async def _handle_order_update(self, data: Dict[str, Any]):
        """Handle order update messages."""
        if self.order_callback:
            await self.order_callback(data)

    async def disconnect(self):
        """Disconnect from WebSocket."""
        self.running = False
        if self.websocket:
            await self.websocket.close()
            if self.logger:
                self.logger.log("WebSocket disconnected", "INFO")

    def set_logger(self, logger):
        """Set the logger instance."""
        self.logger = logger


class StandXClient(BaseExchangeClient):
    """StandX exchange client implementation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize StandX client."""
        super().__init__(config)

        # StandX credentials from environment
        self.wallet_address = os.getenv('STANDX_WALLET_ADDRESS')
        self.private_key = os.getenv('STANDX_PRIVATE_KEY')
        self.chain = os.getenv('STANDX_CHAIN', 'bsc')

        if not self.wallet_address or not self.private_key:
            raise ValueError("STANDX_WALLET_ADDRESS and STANDX_PRIVATE_KEY must be set")

        # Initialize auth handler
        self.auth = StandXAuth(self.wallet_address, self.private_key, self.chain)
        self.base_url = "https://perps.standx.com"

        self._order_update_handler = None
        self.ws_manager: Optional[StandXWebSocketManager] = None
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.logger: Optional[TradingLogger] = None

    def _validate_config(self) -> None:
        """Validate StandX configuration."""
        required_env_vars = ['STANDX_WALLET_ADDRESS', 'STANDX_PRIVATE_KEY']
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")

    async def connect(self) -> None:
        """Connect to StandX (authenticate and start WebSocket)."""
        # Initialize logger
        self.logger = TradingLogger(exchange="standx", ticker=self.config.ticker, log_to_console=False)

        try:
            # Authenticate to get JWT token
            self.logger.log("Authenticating with StandX...", "INFO")
            await self.auth.authenticate()
            self.logger.log("Authentication successful", "INFO")

            # Create HTTP session
            self.http_session = aiohttp.ClientSession()

            # Initialize and start WebSocket manager
            self.ws_manager = StandXWebSocketManager(
                symbol=self.config.contract_id,
                token=self.auth.token,
                session_id=self.auth.session_id,
                order_callback=self._handle_websocket_order_update
            )
            self.ws_manager.set_logger(self.logger)
            self.ws_manager.running = True

            # Start WebSocket in background
            asyncio.create_task(self.ws_manager.connect())
            await asyncio.sleep(2)

        except Exception as e:
            self.logger.log(f"Error connecting to StandX: {e}", "ERROR")
            raise

    async def disconnect(self) -> None:
        """Disconnect from StandX."""
        try:
            if self.ws_manager:
                await self.ws_manager.disconnect()
            if self.http_session:
                await self.http_session.close()
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error during StandX disconnect: {e}", "ERROR")

    def get_exchange_name(self) -> str:
        """Get the exchange name."""
        return "standx"

    def setup_order_update_handler(self, handler) -> None:
        """Setup order update handler for WebSocket."""
        self._order_update_handler = handler

    async def _handle_websocket_order_update(self, order_data: Dict[str, Any]):
        """Handle order updates from WebSocket."""
        try:
            symbol = order_data.get('symbol', '')

            # Only process orders for our symbol
            if symbol != self.config.contract_id:
                return

            order_id = str(order_data.get('id', ''))
            side = order_data.get('side', '').lower()
            quantity = order_data.get('qty', '0')
            price = order_data.get('price', '0')
            fill_qty = order_data.get('fill_qty', '0')
            status = order_data.get('status', '').upper()
            reduce_only = order_data.get('reduce_only', False)

            # Map StandX status to our standard status
            # Official statuses: open, canceled, filled, rejected, untriggered
            status_map = {
                'OPEN': 'OPEN',
                'CANCELED': 'CANCELED',
                'FILLED': 'FILLED',
                'REJECTED': 'CANCELED',
                'UNTRIGGERED': 'OPEN'  # For stop/trigger orders not yet activated
            }
            mapped_status = status_map.get(status, status)

            # Determine order type (OPEN vs CLOSE)
            is_close_order = reduce_only or (side == self.config.close_order_side)
            order_type = "CLOSE" if is_close_order else "OPEN"

            if self._order_update_handler:
                self._order_update_handler({
                    'order_id': order_id,
                    'side': side,
                    'order_type': order_type,
                    'status': mapped_status,
                    'size': quantity,
                    'price': price,
                    'contract_id': symbol,
                    'filled_size': fill_qty
                })

        except Exception as e:
            if self.logger:
                self.logger.log(f"Error handling WebSocket order update: {e}", "ERROR")

    async def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None,
                            params: Optional[Dict] = None, signed: bool = False) -> Dict[str, Any]:
        """Make HTTP request to StandX API."""
        # Auto-create http_session if not exists
        if self.http_session is None:
            self.http_session = aiohttp.ClientSession()

        url = f"{self.base_url}{endpoint}"
        payload = json.dumps(data) if data else ""

        if signed:
            headers = self.auth.get_auth_headers(payload)
        else:
            headers = {"Content-Type": "application/json"}
            if self.auth.token:
                headers["Authorization"] = f"Bearer {self.auth.token}"

        try:
            if method == "GET":
                async with self.http_session.get(url, headers=headers, params=params) as resp:
                    return await resp.json()
            elif method == "POST":
                async with self.http_session.post(url, headers=headers, data=payload) as resp:
                    return await resp.json()
        except Exception as e:
            if self.logger:
                self.logger.log(f"HTTP request error: {e}", "ERROR")
            raise

    @query_retry(default_return=(Decimal('0'), Decimal('0')))
    async def fetch_bbo_prices(self, contract_id: str) -> Tuple[Decimal, Decimal]:
        """Get BBO prices from WebSocket cache or HTTP fallback."""
        # Try WebSocket cache first
        if self.ws_manager and self.ws_manager.best_bid > 0 and self.ws_manager.best_ask > 0:
            return self.ws_manager.best_bid, self.ws_manager.best_ask

        # Fallback to HTTP
        result = await self._make_request("GET", "/api/query_depth_book",
                                          params={"symbol": contract_id})
        bids = result.get('bids', [])
        asks = result.get('asks', [])

        best_bid = Decimal(bids[0][0]) if bids else Decimal('0')
        best_ask = Decimal(asks[0][0]) if asks else Decimal('0')

        return best_bid, best_ask

    @query_retry(default_return=Decimal('0'))
    async def fetch_mark_price(self, contract_id: str) -> Decimal:
        """Get mark price from WebSocket cache or HTTP fallback."""
        # Try WebSocket cache first
        if self.ws_manager and self.ws_manager.mark_price > 0:
            return self.ws_manager.mark_price

        # Fallback to HTTP - use query_symbol_price endpoint
        result = await self._make_request("GET", "/api/query_symbol_price",
                                          params={"symbol": contract_id})
        mark_price = Decimal(result.get('mark_price', '0'))
        return mark_price

    async def get_order_price(self, direction: str) -> Decimal:
        """Get the price for placing an order."""
        best_bid, best_ask = await self.fetch_bbo_prices(self.config.contract_id)
        if best_bid <= 0 or best_ask <= 0:
            raise ValueError("Invalid bid/ask prices")

        if direction == 'buy':
            order_price = best_ask - self.config.tick_size
        else:
            order_price = best_bid + self.config.tick_size

        return self.round_to_tick(order_price)

    async def place_open_order(self, contract_id: str, quantity: Decimal, direction: str) -> OrderResult:
        """Place an open order with StandX."""
        max_retries = 15
        retry_count = 0

        while retry_count < max_retries:
            retry_count += 1

            best_bid, best_ask = await self.fetch_bbo_prices(contract_id)
            if best_bid <= 0 or best_ask <= 0:
                return OrderResult(success=False, error_message='Invalid bid/ask prices')

            if direction == 'buy':
                order_price = best_ask - self.config.tick_size
                side = 'buy'
            else:
                order_price = best_bid + self.config.tick_size
                side = 'sell'

            order_price = self.round_to_tick(order_price)

            payload = {
                "symbol": contract_id,
                "side": side,
                "order_type": "limit",
                "qty": str(quantity),
                "price": str(order_price),
                "time_in_force": "alo",  # Add-liquidity-only (post-only)
                "reduce_only": False
            }

            result = await self._make_request("POST", "/api/new_order",
                                              data=payload, signed=True)

            if result.get('code', -1) != 0:
                message = result.get('message', 'Unknown error')
                self.logger.log(f"[OPEN] Order rejected: {message}", "WARNING")
                continue

            return OrderResult(
                success=True,
                order_id=result.get('request_id', ''),
                side=side,
                size=quantity,
                price=order_price,
                status='New'
            )

        return OrderResult(success=False, error_message='Max retries exceeded')

    async def place_market_order(self, contract_id: str, quantity: Decimal, direction: str) -> OrderResult:
        """Place a market order with StandX."""
        side = 'buy' if direction == 'buy' else 'sell'

        payload = {
            "symbol": contract_id,
            "side": side,
            "order_type": "market",
            "qty": str(quantity),
            "time_in_force": "ioc",  # Immediate or cancel
            "reduce_only": False
        }

        result = await self._make_request("POST", "/api/new_order",
                                          data=payload, signed=True)

        if result.get('code', -1) != 0:
            return OrderResult(success=False, error_message=result.get('message', 'Order failed'))

        return OrderResult(
            success=True,
            order_id=result.get('request_id', ''),
            side=side,
            size=quantity,
            status='FILLED'
        )

    async def place_close_order(self, contract_id: str, quantity: Decimal, price: Decimal, side: str) -> OrderResult:
        """Place a close order with StandX."""
        max_retries = 15
        retry_count = 0

        while retry_count < max_retries:
            retry_count += 1

            best_bid, best_ask = await self.fetch_bbo_prices(contract_id)
            if best_bid <= 0 or best_ask <= 0:
                return OrderResult(success=False, error_message='Invalid bid/ask prices')

            adjusted_price = price
            if side.lower() == 'sell':
                if price <= best_bid:
                    adjusted_price = best_bid + self.config.tick_size
            elif side.lower() == 'buy':
                if price >= best_ask:
                    adjusted_price = best_ask - self.config.tick_size

            adjusted_price = self.round_to_tick(adjusted_price)

            payload = {
                "symbol": contract_id,
                "side": side.lower(),
                "order_type": "limit",
                "qty": str(quantity),
                "price": str(adjusted_price),
                "time_in_force": "alo",
                "reduce_only": True
            }

            result = await self._make_request("POST", "/api/new_order",
                                              data=payload, signed=True)

            if result.get('code', -1) != 0:
                message = result.get('message', 'Unknown error')
                self.logger.log(f"[CLOSE] Order rejected: {message}", "WARNING")
                continue

            return OrderResult(
                success=True,
                order_id=result.get('request_id', ''),
                side=side.lower(),
                size=quantity,
                price=adjusted_price,
                status='New'
            )

        return OrderResult(success=False, error_message='Max retries exceeded for close order')

    async def cancel_order(self, order_id: str) -> OrderResult:
        """Cancel an order with StandX."""
        try:
            payload = {"order_id": int(order_id)}
            result = await self._make_request("POST", "/api/cancel_order",
                                              data=payload, signed=True)

            if result.get('code', -1) != 0:
                return OrderResult(success=False, error_message=result.get('message', 'Cancel failed'))

            return OrderResult(success=True)

        except Exception as e:
            return OrderResult(success=False, error_message=str(e))

    @query_retry()
    async def get_order_info(self, order_id: str) -> Optional[OrderInfo]:
        """Get order information from StandX."""
        result = await self._make_request("GET", "/api/query_order",
                                          params={"order_id": order_id})

        if not result or 'id' not in result:
            return None

        # Official statuses: open, canceled, filled, rejected, untriggered
        status_map = {
            'open': 'OPEN',
            'canceled': 'CANCELED',
            'filled': 'FILLED',
            'rejected': 'CANCELED',
            'untriggered': 'OPEN'
        }

        return OrderInfo(
            order_id=str(result.get('id', '')),
            side=result.get('side', '').lower(),
            size=Decimal(result.get('qty', '0')),
            price=Decimal(result.get('price', '0')),
            status=status_map.get(result.get('status', '').lower(), 'OPEN'),
            filled_size=Decimal(result.get('fill_qty', '0')),
            remaining_size=Decimal(result.get('qty', '0')) - Decimal(result.get('fill_qty', '0'))
        )

    @query_retry(default_return=[])
    async def get_active_orders(self, contract_id: str) -> List[OrderInfo]:
        """Get active orders for a contract."""
        result = await self._make_request("GET", "/api/query_open_orders",
                                          params={"symbol": contract_id})

        orders = []
        order_list = result.get('result', [])

        for order in order_list:
            orders.append(OrderInfo(
                order_id=str(order.get('id', '')),
                side=order.get('side', '').lower(),
                size=Decimal(order.get('qty', '0')),
                price=Decimal(order.get('price', '0')),
                status='OPEN',
                filled_size=Decimal(order.get('fill_qty', '0')),
                remaining_size=Decimal(order.get('qty', '0')) - Decimal(order.get('fill_qty', '0'))
            ))

        return orders

    @query_retry(default_return=Decimal('0'))
    async def get_account_positions(self) -> Decimal:
        """Get account positions."""
        result = await self._make_request("GET", "/api/query_positions",
                                          params={"symbol": self.config.contract_id})

        for position in result if isinstance(result, list) else []:
            if position.get('symbol', '') == self.config.contract_id:
                qty = Decimal(position.get('qty', '0'))
                # StandX uses positive qty, side determines direction
                return qty

        return Decimal('0')

    async def get_contract_attributes(self) -> Tuple[str, Decimal]:
        """Get contract ID and tick size for a ticker."""
        ticker = self.config.ticker

        result = await self._make_request("GET", "/api/query_symbol_info",
                                          params={"symbol": f"{ticker}-USD"})

        symbol_info = result[0] if isinstance(result, list) and result else result

        if not symbol_info:
            raise ValueError(f"Symbol info not found for {ticker}")

        contract_id = symbol_info.get('symbol', f"{ticker}-USD")
        # Tick size from price_tick_decimals
        price_decimals = int(symbol_info.get('price_tick_decimals', 2))
        tick_size = Decimal(10) ** (-price_decimals)

        # Validate min quantity
        min_qty = Decimal(symbol_info.get('min_order_qty', '0'))
        if self.config.quantity < min_qty:
            raise ValueError(f"Order quantity less than minimum: {self.config.quantity} < {min_qty}")

        self.config.contract_id = contract_id
        self.config.tick_size = tick_size

        return contract_id, tick_size
