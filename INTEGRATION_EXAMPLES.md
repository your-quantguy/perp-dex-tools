# 🔧 Integration Examples

This document shows how to integrate security modules into your existing code.

## 📋 Table of Contents

1. [Quick Integration (5 minutes)](#quick-integration)
2. [Trading Bot Integration](#trading-bot-integration)
3. [Hedge Mode Integration](#hedge-mode-integration)
4. [Exchange Client Integration](#exchange-client-integration)

---

## Quick Integration

### Step 1: Add Security Import

```python
from security import get_security

# Initialize once at startup
security = get_security(
    use_encrypted_keys=True,  # Use encrypted key storage
    initial_capital=10000      # Your starting capital
)
```

### Step 2: Use in Your Code

```python
# Get API keys
api_key = security.get_api_key('EDGEX_ACCOUNT_ID')

# Check if can trade (circuit breaker)
if security.can_trade('edgex'):
    # Safe to place orders
    pass

# Record trades
security.record_trade('edgex', amount=1000, pnl=25.50)
```

---

## Trading Bot Integration

### Original Code (`trading_bot.py`)

```python
class TradingBot:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = TradingLogger(...)
        self.exchange_client = ExchangeFactory.create_exchange(...)
```

### With Security Integration

```python
from security import get_security

class TradingBot:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = TradingLogger(...)

        # Add security module
        self.security = get_security(
            use_encrypted_keys=True,
            initial_capital=10000
        )

        # Create exchange with secure keys
        self.exchange_client = ExchangeFactory.create_exchange(...)

    async def place_order(self, symbol, side, price, size):
        """Place order with security checks"""

        # 1. Check circuit breaker
        if not self.security.can_trade(self.config.exchange):
            self.logger.log("Circuit breaker OPEN - skipping trade", "WARNING")
            return None

        # 2. Check risk limits
        allowed, reason = self.security.check_risk_limits(
            exchange=self.config.exchange,
            symbol=symbol,
            side=side,
            entry_price=price,
            size_usd=size * price
        )

        if not allowed:
            self.logger.log(f"Trade rejected: {reason}", "WARNING")
            return None

        # 3. Place order with rate limiting (handled by exchange client)
        try:
            order = await self.exchange_client.place_order(...)

            # 4. Record trade
            if order.success:
                pnl = self._calculate_pnl(order)
                self.security.record_trade(
                    self.config.exchange,
                    amount=size * price,
                    pnl=pnl
                )

            return order

        except Exception as e:
            self.logger.log(f"Order failed: {e}", "ERROR")
            return None
```

---

## Hedge Mode Integration

### Original Code (`hedge_mode_edgex.py`)

```python
class HedgeBot:
    def __init__(self, ticker, order_quantity, ...):
        self.ticker = ticker
        self.order_quantity = order_quantity
        # ...

    async def run(self):
        while not self.stop_flag:
            # Place orders
            await self.place_edgex_order()
            await self.place_lighter_hedge()
```

### With Security Integration

```python
from security import get_security

class HedgeBot:
    def __init__(self, ticker, order_quantity, ...):
        self.ticker = ticker
        self.order_quantity = order_quantity

        # Add security
        self.security = get_security(use_encrypted_keys=True, initial_capital=10000)

        # Get API keys from encrypted storage
        self.edgex_account_id = self.security.get_api_key('EDGEX_ACCOUNT_ID')
        self.edgex_stark_private_key = self.security.get_api_key('EDGEX_STARK_PRIVATE_KEY')
        self.lighter_account_index = int(self.security.get_api_key('LIGHTER_ACCOUNT_INDEX'))

    async def run(self):
        while not self.stop_flag:
            # Check circuit breakers before trading
            if not self.security.can_trade('edgex'):
                self.logger.info("⚠️ edgeX circuit breaker OPEN - pausing")
                await asyncio.sleep(60)  # Wait 1 minute
                continue

            if not self.security.can_trade('lighter'):
                self.logger.info("⚠️ Lighter circuit breaker OPEN - pausing")
                await asyncio.sleep(60)
                continue

            # Place orders
            try:
                edgex_result = await self.place_edgex_order()

                if edgex_result.success:
                    lighter_result = await self.place_lighter_hedge()

                    # Calculate PnL
                    pnl = self._calculate_hedge_pnl(edgex_result, lighter_result)

                    # Record trades
                    self.security.record_trade('edgex', amount=float(self.order_quantity), pnl=pnl/2)
                    self.security.record_trade('lighter', amount=float(self.order_quantity), pnl=pnl/2)

            except Exception as e:
                self.logger.error(f"Hedge cycle failed: {e}")
                # Circuit breaker will open automatically if too many failures

    def _calculate_hedge_pnl(self, edgex_result, lighter_result):
        """Calculate profit from hedge"""
        # Your PnL calculation here
        edgex_price = float(edgex_result.price)
        lighter_price = float(lighter_result.price)
        quantity = float(self.order_quantity)

        # Assuming buy on edgex, sell on lighter
        gross_pnl = (lighter_price - edgex_price) * quantity

        # Account for fees
        edgex_fee = edgex_price * quantity * 0.0005  # 0.05%
        lighter_fee = lighter_price * quantity * 0.0002  # 0.02%

        net_pnl = gross_pnl - edgex_fee - lighter_fee
        return net_pnl
```

---

## Exchange Client Integration

### Original Code (`exchanges/edgex.py`)

```python
class EdgeXClient(BaseExchangeClient):
    async def place_order(self, symbol, side, price, size):
        order = await self.client.create_order(...)
        return order
```

### With Rate Limiting

```python
from security.rate_limiter import get_manager

class EdgeXClient(BaseExchangeClient):
    def __init__(self, config):
        super().__init__(config)
        self.rate_manager = get_manager()

    async def place_order(self, symbol, side, price, size):
        # Wrap with rate limiting and circuit breaker
        order = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.rate_manager.execute_with_protection(
                'edgex',
                self._create_order_sync,
                symbol, side, price, size
            )
        )
        return order

    def _create_order_sync(self, symbol, side, price, size):
        """Synchronous order creation for rate limiter"""
        return self.client.create_order(
            symbol=symbol,
            side=side,
            price=price,
            size=size
        )
```

---

## Using Decorators

### Rate Limiting Decorator

```python
from security import with_rate_limit

class ExchangeClient:
    @with_rate_limit('edgex')
    def get_orderbook(self, symbol):
        return self.client.get_orderbook(symbol)

    @with_rate_limit('edgex')
    def get_balance(self):
        return self.client.get_balance()
```

### Circuit Breaker Decorator

```python
from security import require_circuit_closed

class TradingBot:
    @require_circuit_closed('edgex')
    async def execute_strategy(self):
        # This will only run if circuit is closed
        await self.place_orders()
```

---

## Complete Example: Enhanced Hedge Bot

```python
import asyncio
from decimal import Decimal
from security import get_security
from exchanges import create_exchange

class EnhancedHedgeBot:
    def __init__(self, ticker: str, quantity: Decimal, initial_capital: float = 10000):
        # Initialize security
        self.security = get_security(
            use_encrypted_keys=True,
            initial_capital=initial_capital
        )

        # Configuration
        self.ticker = ticker
        self.quantity = quantity

        # Initialize exchanges with secure keys
        self.edgex = self._init_edgex()
        self.lighter = self._init_lighter()

        # Statistics
        self.total_trades = 0
        self.profitable_trades = 0

    def _init_edgex(self):
        """Initialize EdgeX with encrypted keys"""
        config = {
            'account_id': self.security.get_api_key('EDGEX_ACCOUNT_ID'),
            'stark_private_key': self.security.get_api_key('EDGEX_STARK_PRIVATE_KEY'),
        }
        return create_exchange('edgex', config)

    def _init_lighter(self):
        """Initialize Lighter with encrypted keys"""
        config = {
            'account_index': int(self.security.get_api_key('LIGHTER_ACCOUNT_INDEX')),
            'api_key_private_key': self.security.get_api_key('API_KEY_PRIVATE_KEY'),
        }
        return create_exchange('lighter', config)

    async def run(self, max_iterations: int = None):
        """Main trading loop with full security"""
        iteration = 0

        while True:
            if max_iterations and iteration >= max_iterations:
                break

            try:
                # Check circuit breakers
                if not self._check_circuits():
                    await asyncio.sleep(60)
                    continue

                # Check risk limits
                if not self._check_risk_limits():
                    await asyncio.sleep(60)
                    continue

                # Execute hedge cycle
                await self._execute_hedge_cycle()

                # Print status periodically
                if iteration % 10 == 0:
                    self.security.print_status()

                iteration += 1

            except KeyboardInterrupt:
                print("\n🛑 Stopping...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(5)

        # Final summary
        self._print_summary()

    def _check_circuits(self) -> bool:
        """Check if both exchange circuits are closed"""
        if not self.security.can_trade('edgex'):
            print("⚠️ EdgeX circuit breaker OPEN")
            return False

        if not self.security.can_trade('lighter'):
            print("⚠️ Lighter circuit breaker OPEN")
            return False

        return True

    def _check_risk_limits(self) -> bool:
        """Check if risk limits allow trading"""
        metrics = self.security.risk_manager.get_portfolio_metrics()

        # Check daily loss
        if metrics['daily_pnl'] < 0:
            daily_loss_pct = abs(metrics['daily_pnl']) / metrics['current_capital'] * 100
            if daily_loss_pct >= 5.0:  # 5% daily loss limit
                print(f"⚠️ Daily loss limit reached: {daily_loss_pct:.1f}%")
                return False

        # Check drawdown
        if metrics['drawdown_pct'] >= 10.0:  # 10% max drawdown
            print(f"⚠️ Max drawdown reached: {metrics['drawdown_pct']:.1f}%")
            return False

        return True

    async def _execute_hedge_cycle(self):
        """Execute one hedge cycle with security"""
        # Get best prices
        edgex_book = await self.edgex.get_orderbook(self.ticker)
        lighter_book = await self.lighter.get_orderbook(self.ticker)

        # Calculate opportunity
        edgex_ask = edgex_book['asks'][0][0]
        lighter_bid = lighter_book['bids'][0][0]

        # Check profitability
        gross_profit = (lighter_bid - edgex_ask) * float(self.quantity)
        fees = (edgex_ask * 0.0005 + lighter_bid * 0.0002) * float(self.quantity)
        net_profit = gross_profit - fees

        if net_profit < 0.50:  # Minimum $0.50 profit
            return

        # Place orders
        print(f"💰 Opportunity: ${net_profit:.2f}")

        edgex_order = await self.edgex.place_order(
            symbol=self.ticker,
            side='buy',
            price=edgex_ask,
            size=self.quantity
        )

        if edgex_order.success:
            lighter_order = await self.lighter.place_order(
                symbol=self.ticker,
                side='sell',
                price=lighter_bid,
                size=self.quantity
            )

            # Record results
            if lighter_order.success:
                self.total_trades += 1
                if net_profit > 0:
                    self.profitable_trades += 1

                # Update security tracking
                self.security.record_trade('edgex', float(self.quantity) * edgex_ask, net_profit / 2)
                self.security.record_trade('lighter', float(self.quantity) * lighter_bid, net_profit / 2)

                print(f"✅ Hedge complete: ${net_profit:.2f}")

    def _print_summary(self):
        """Print final summary"""
        print("\n" + "=" * 60)
        print("📊 TRADING SUMMARY")
        print("=" * 60)

        print(f"\nTotal Trades: {self.total_trades}")
        if self.total_trades > 0:
            win_rate = (self.profitable_trades / self.total_trades) * 100
            print(f"Win Rate: {win_rate:.1f}%")

        self.security.print_status()


# Run the bot
if __name__ == "__main__":
    bot = EnhancedHedgeBot(
        ticker='BTC-USD',
        quantity=Decimal('0.01'),
        initial_capital=10000
    )

    asyncio.run(bot.run(max_iterations=100))
```

---

## Testing Integration

### Test Security Features

```python
# test_security_integration.py

from security import get_security

def test_security_integration():
    # Initialize
    security = get_security(use_encrypted_keys=False, initial_capital=10000)

    # Test position sizing
    size = security.get_position_size(win_rate=0.6, avg_win=2.0, avg_loss=1.0)
    print(f"✅ Position size: ${size:.2f}")
    assert size > 0

    # Test risk checks
    allowed, reason = security.check_risk_limits(
        exchange='edgex',
        symbol='BTC-USD',
        side='long',
        entry_price=50000,
        size_usd=1000
    )
    print(f"✅ Risk check: {allowed} - {reason}")
    assert allowed == True

    # Test circuit breaker
    can_trade = security.can_trade('edgex')
    print(f"✅ Can trade: {can_trade}")
    assert can_trade == True

    # Test trade recording
    security.record_trade('edgex', amount=1000, pnl=25.50)
    security.record_trade('edgex', amount=1000, pnl=-10.00)

    # Print stats
    security.print_status()

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_security_integration()
```

---

## Migration Checklist

When integrating into existing code:

- [ ] Install security modules (`./install_security.sh`)
- [ ] Migrate API keys to encrypted storage
- [ ] Add security import to your files
- [ ] Initialize SecurityIntegration in `__init__`
- [ ] Add circuit breaker checks before trading
- [ ] Add risk limit checks before placing orders
- [ ] Record trades for monitoring
- [ ] Test with small amounts first
- [ ] Monitor logs for rate limit or circuit breaker events
- [ ] Gradually increase position sizes

---

## Troubleshooting

### "Circuit breaker OPEN"

The circuit breaker automatically opens when:
- 5+ consecutive failures
- 5%+ losses in 5 minutes

**Solution**: Wait for timeout (60 seconds) or investigate the root cause.

### "Daily loss limit reached"

Your bot has lost more than the configured daily limit (default 5%).

**Solution**: Review trades, adjust strategy, or increase limit in config.

### "Rate limit timeout"

You're making too many API calls too quickly.

**Solution**: The system handles this automatically. If persistent, reduce trading frequency.

---

## Best Practices

1. **Always check circuit breakers** before trading
2. **Record all trades** for monitoring
3. **Start with conservative limits** (small positions, strict stop-loss)
4. **Monitor regularly** using `security.print_status()`
5. **Test thoroughly** before going live
6. **Use encrypted keys** in production

---

For more information, see:
- `QUICK_START_GUIDE.md` - Setup instructions
- `SECURITY_IMPROVEMENTS.md` - Technical details
- `security/integration_helper.py` - Source code
