# Multi-Exchange Trading Bot

A modular trading bot that supports multiple exchanges including EdgeX and Backpack. The bot implements an automated strategy that places orders and automatically closes them at a profit.

## Follow Me

- **X (Twitter)**: [@yourQuantGuy](https://x.com/yourQuantGuy)

## Referral Links

### EdgeX Exchange

Use my referral link to sign up:  
👉 [https://pro.edgex.exchange/referral/QUANT](https://pro.edgex.exchange/referral/QUANT)

By using my EdgeX referral, you will enjoy the following benefits:

1. **Instant VIP 1 Trading Fees** – get upgraded directly to VIP 1 fee rates.
2. **10% Fee Rebate** – automatically settled every 24 hours and claimable directly on the EdgeX website.
   - This rebate is on top of the VIP 1 fee rate, which means your effective trading fee becomes:
     ```
     0.013% * 0.9 = 0.0117%
     ```
3. **10% Bonus Points** – extra points credited to your account.

### Backpack Exchange

**30% Fee Rebates** – By using my Backpack referral, you will get 30% fee rebates on all your trading fees:
👉 [https://backpack.exchange/join/quant](https://backpack.exchange/join/quant)

## Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd perp-dex-tools
   ```

2. **Create and activate virtual environment**:

   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Use the env_example.txt to create a `.env` file in the project root.

## Sample commands:

### EdgeX Exchange:

ETH:

```bash
python runbot.py --exchange edgex --ticker ETH --quantity 0.1 --take-profit 0.02 --max-orders 40 --wait-time 450
```

ETH (with grid step control):

```bash
python runbot.py --exchange edgex --ticker ETH --quantity 0.1 --take-profit 0.02 --max-orders 40 --wait-time 450 --grid-step 0.5
```

BTC:

```bash
python runbot.py --exchange edgex --ticker BTC --quantity 0.05 --take-profit 0.02 --max-orders 40 --wait-time 450
```

### Backpack Exchange:

ETH Perpetual:

```bash
python runbot.py --exchange backpack --ticker ETH --quantity 0.1 --take-profit 0.02 --max-orders 40 --wait-time 450
```

ETH Perpetual (with grid step control):

```bash
python runbot.py --exchange backpack --ticker ETH --quantity 0.1 --take-profit 0.02 --max-orders 40 --wait-time 450 --grid-step 0.3
```

## Configuration

### Environment Variables

#### EdgeX Configuration

- `EDGEX_ACCOUNT_ID`: Your EdgeX account ID
- `EDGEX_STARK_PRIVATE_KEY`: Your EdgeX api private key
- `EDGEX_BASE_URL`: EdgeX API base URL (default: https://pro.edgex.exchange)
- `EDGEX_WS_URL`: EdgeX WebSocket URL (default: wss://quote.edgex.exchange)

#### Backpack Configuration

- `BACKPACK_PUBLIC_KEY`: Your Backpack public key
- `BACKPACK_SECRET_KEY`: Your Backpack secret key (base64 encoded)

### Command Line Arguments

- `--exchange`: Exchange to use: 'edgex' or 'backpack' (default: edgex)
- `--ticker`: Base asset symbol (e.g., ETH, BTC, SOL). Contract ID is auto-resolved.
- `--quantity`: Order quantity (default: 0.1)
- `--take-profit`: Take profit percent (e.g., 0.02 means 0.02%)
- `--direction`: Trading direction: 'buy' or 'sell' (default: buy)
- `--max-orders`: Maximum number of active orders (default: 40)
- `--wait-time`: Wait time between orders in seconds (default: 450)
- `--grid-step`: Minimum distance in percentage to the next close order price (default: -100, means no restriction)

## Trading Strategy

The bot implements a simple scalping strategy:

1. **Order Placement**: Places a limit order slightly above/below market price
2. **Order Monitoring**: Waits for the order to be filled
3. **Close Order**: Automatically places a close order at the take profit level
4. **Position Management**: Monitors positions and active orders
5. **Risk Management**: Limits maximum number of concurrent orders
6. **Grid Step Control**: Controls minimum price distance between new orders and existing close orders via `--grid-step` parameter

### Grid Step Feature

The `--grid-step` parameter controls the minimum distance between new order close prices and existing close order prices:

- **Default -100**: No grid step restriction, executes original strategy
- **Positive value (e.g., 0.5)**: New order close price must maintain at least 0.5% distance from the nearest close order price
- **Purpose**: Prevents close orders from being too dense, improving fill probability and risk management

For example, when Long and `--grid-step 0.5`:
- If existing close order price is 2000 USDT
- New order close price must be lower than 1990 USDT (2000 × (1 - 0.5%))
- This prevents close orders from being too close together, improving overall strategy effectiveness

## Architecture

The bot is built with a modular architecture supporting multiple exchanges:

### 1. Exchange Clients

#### EdgeX Client (Official SDK)

- REST API client for EdgeX using the official SDK
- Handles authentication and API requests
- Manages order placement, cancellation, and status queries
- Position and account information retrieval

#### Backpack Client (Official SDK)

- REST API client for Backpack using the official BPX SDK
- Handles authentication and API requests
- Manages order placement, cancellation, and status queries
- Position and account information retrieval

### 2. WebSocket Managers

#### EdgeX WebSocket Manager (Official SDK)

- WebSocket connection management using the official SDK
- Real-time market data streaming
- Order update notifications
- Automatic connection handling

#### Backpack WebSocket Manager

- WebSocket connection management for Backpack
- Real-time order update notifications
- ED25519 signature authentication
- Automatic connection handling

### 3. Main Trading Bot (`runbot.py`)

- Core scalping logic
- Order placement and monitoring
- Position management
- Main trading loop
- Multi-exchange support

## Logging

The bot provides comprehensive logging:

- **Transaction Logs**: CSV files with order details
- **Debug Logs**: Detailed activity logs with timestamps
- **Console Output**: Real-time status updates
- **Error Handling**: Comprehensive error logging and handling

## Safety Features

- **Order Limits**: Configurable maximum order count
- **Timeout Handling**: Automatic order cancellation on timeouts
- **Position Monitoring**: Continuous position and order status checking
- **Error Recovery**: Graceful handling of API errors and disconnections

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This software is for educational and research purposes only. Trading cryptocurrencies involves significant risk and can result in substantial financial losses. Use at your own risk and never trade with money you cannot afford to lose.

## Support

For issues related to:

- **EdgeX API**: Check the [EdgeX API documentation](https://docs.edgex.exchange)
- **EdgeX SDK**: Check the [EdgeX Python SDK documentation](https://github.com/edgex-Tech/edgex-python-sdk)
- **This Bot**: Open an issue in this repository
