##### Follow Me - **X (Twitter)**: [@yourQuantGuy](https://x.com/yourQuantGuy)
## Multi-Exchange Trading Bot

A modular trading bot that supports multiple exchanges including EdgeX and Backpack. The bot implements an automated strategy that places orders and automatically closes them at a profit.

## Referral Links (Enjoy fee rebates and benefits)

#### EdgeX: [https://pro.edgex.exchange/referral/QUANT](https://pro.edgex.exchange/referral/QUANT)
Instant VIP 1 Trading Fees; 10% Fee Rebate; 10% Bonus Points

#### Backpack Exchange: [https://backpack.exchange/join/quant](https://backpack.exchange/join/quant)
You will get 30% fee rebates on all your trading fees

#### Paradex Exchange: [https://app.paradex.trade/r/quant](https://app.paradex.trade/r/quant)
You will get 10% taker fee discount rebates and potential future benefits

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

## Strategy Overview

### How the Bot Works

This trading bot specifically designed for perpetual contracts. Here's a detailed explanation of the strategy:

#### 🎯 Core Strategy
1. **Auto Order Placement**: The bot automatically places limit orders near the current market price
2. **Wait for Fill**: Monitors order status and waits for orders to be filled by the market
3. **Auto Close**: Once filled, immediately places a close order at the preset take-profit price
4. **Loop Execution**: Repeats the process continuously for ongoing trading

#### 📊 Trading Flow Example
Assuming current ETH price is $2000 with take-profit set to 0.02%:

1. **Open Position**: Places buy order at $2000.40 (slightly above market price)
2. **Fill**: Order gets filled by the market, acquiring long position
3. **Close Position**: Immediately places sell order at $2000.80 (take-profit price)
4. **Complete**: Close order gets filled, earning 0.02% profit
5. **Repeat**: Continues to the next trading cycle

#### ⚙️ Key Parameters
- **quantity**: Trading amount per order
- **take-profit**: Take-profit percentage (e.g., 0.02 means 0.02%)
- **max-orders**: Maximum concurrent active orders (risk control)
- **wait-time**: Wait time between orders (prevents overtrading)
- **grid-step**: Grid step control (prevents close orders from being too dense)

#### 🛡️ Risk Management
- **Order Limits**: Limits maximum concurrent orders via `max-orders`
- **Grid Control**: Ensures reasonable spacing between close orders via `grid-step`
- **Timeout Handling**: Automatically cancels orders that remain unfilled too long
- **Real-time Monitoring**: Continuously monitors positions and order status
- **⚠️ No Stop Loss**: This strategy does not include stop-loss functionality and may face significant losses in adverse market conditions

#### 💡 Best Use Cases
- **Sideways Markets**: Profiting from price oscillations within a range
- **Low Volatility**: Accumulating volumes and profits through frequent small trades
- **Automated Trading**: 24/7 trading without manual intervention

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
- `BACKPACK_SECRET_KEY`: Your Backpack secret key

#### Paradex Configuration

- `PARADEX_L1_ADDRESS`: Your L1 wallet address
- `PARADEX_L2_PRIVATE_KEY`: Your Paradex L2 private key

### Command Line Arguments

- `--exchange`: Exchange to use: 'edgex', 'backpack', or 'paradex' (default: edgex)
- `--ticker`: Base asset symbol (e.g., ETH, BTC, SOL). Contract ID is auto-resolved.
- `--quantity`: Order quantity (default: 0.1)
- `--take-profit`: Take profit percent (e.g., 0.02 means 0.02%)
- `--direction`: Trading direction: 'buy' or 'sell' (default: buy)
- `--max-orders`: Maximum number of active orders (default: 40)
- `--wait-time`: Wait time between orders in seconds (default: 450)
- `--grid-step`: Minimum distance in percentage to the next close order price (default: -100, means no restriction)

## Trading Strategy

The bot implements a simple strategy:

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

## Logging

The bot provides comprehensive logging:

- **Transaction Logs**: CSV files with order details
- **Debug Logs**: Detailed activity logs with timestamps
- **Console Output**: Real-time status updates
- **Error Handling**: Comprehensive error logging and handling

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
