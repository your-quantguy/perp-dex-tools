# 🚀 Quick Start Guide - Security Improvements

**Time to complete**: 15-30 minutes

## 📋 Overview

This guide will help you integrate the security improvements into your Perp-DEX-Tools project quickly and safely.

## ✅ Prerequisites

- Python 3.8+
- Existing Perp-DEX-Tools installation
- API keys for your exchanges (in `.env` file)

---

## 🎯 Step 1: Run Automatic Installation (5 minutes)

The easiest way to get started is using the installation script:

```bash
./install_security.sh
```

This script will:
- ✅ Check your environment
- ✅ Install dependencies (`cryptography`)
- ✅ Verify security modules
- ✅ Create configuration files
- ✅ Run tests
- ✅ Optionally migrate your API keys to encrypted storage

**Follow the prompts** - the script will guide you through each step.

---

## 🎯 Step 2: Migrate API Keys to Encrypted Storage (5 minutes)

### Option A: During Installation (Recommended)

If you ran `install_security.sh`, you'll be prompted to migrate keys automatically.

### Option B: Manual Migration

```bash
# 1. Initialize encrypted storage
python3 -m security.secure_key_manager init

# You'll be prompted to create a master password
# ⚠️ REMEMBER THIS PASSWORD - there's no recovery!

# 2. Migrate from .env
python3 -m security.secure_key_manager migrate .env

# This will:
# - Backup .env to .env.backup
# - Encrypt all keys
# - Store in .keys.enc
```

### Verify Migration

```bash
# List encrypted keys
python3 -m security.secure_key_manager list

# Should show keys like:
# - EDGEX_ACCOUNT_ID
# - EDGEX_STARK_PRIVATE_KEY
# - LIGHTER_ACCOUNT_INDEX
# etc.
```

---

## 🎯 Step 3: Configure Arbitrage Settings (3 minutes)

Edit `arbitrage_config.json` to match your trading strategy:

```json
{
  "exchanges": ["edgex", "lighter", "backpack"],
  "symbols": ["BTC-USD", "ETH-USD"],
  "min_profit_bps": 10,  // Minimum 0.1% profit
  "max_position_size_usd": 5000,  // Max $5k per trade
  "risk": {
    "max_daily_loss_pct": 5.0,  // Stop if 5% daily loss
    "max_drawdown_pct": 10.0,  // Stop if 10% drawdown
    "use_kelly_criterion": true,  // Smart position sizing
    "default_stop_loss_pct": 1.0,  // 1% stop-loss
    "default_take_profit_pct": 2.0  // 2% take-profit
  }
}
```

**Key Parameters to Adjust:**

| Parameter | Description | Safe Value | Aggressive Value |
|-----------|-------------|------------|------------------|
| `min_profit_bps` | Minimum profit to execute | 20 (0.2%) | 5 (0.05%) |
| `max_position_size_usd` | Max per trade | $1,000 | $10,000 |
| `max_daily_loss_pct` | Daily loss limit | 2% | 10% |
| `use_kelly_criterion` | Smart sizing | true | true |

---

## 🎯 Step 4: Test the Improved Engine (5 minutes)

### Dry Run (No Real Trades)

First, test without executing real trades:

```bash
# Edit improved_arbitrage_engine.py
# Set DRY_RUN = True at the top

python3 improved_arbitrage_engine.py 60
# Runs for 60 seconds
```

You should see output like:

```
🚀 Starting arbitrage engine...
✅ Initialized edgex
✅ Initialized lighter
💰 Opportunity: BTC-USD edgex→lighter Net profit=0.12% ($6.00) Score=75.3
```

### Live Test (Small Amount)

If dry run looks good:

1. **Start with small capital**: Edit config to `max_position_size_usd: 100`
2. **Run for short duration**:
   ```bash
   python3 improved_arbitrage_engine.py 300  # 5 minutes
   ```
3. **Monitor carefully**: Watch for errors or unexpected behavior

---

## 🎯 Step 5: Integrate with Existing Code (10 minutes)

### For `trading_bot.py`

Add security imports:

```python
from security.rate_limiter import get_manager as get_rate_limit_manager
from security.risk_manager import RiskManager, RiskConfig
from security.secure_key_manager import SecureKeyManager

class TradingBot:
    def __init__(self, ...):
        # Add security modules
        self.rate_limit_manager = get_rate_limit_manager()
        self.risk_manager = RiskManager(initial_capital=10000)

        # Optional: use encrypted keys
        # self.key_manager = SecureKeyManager()
        # self.key_manager.load_keys()
```

### Wrap Exchange Calls with Rate Limiting

```python
# Before (vulnerable to rate limits):
orderbook = exchange.get_orderbook(symbol)

# After (protected):
orderbook = self.rate_limit_manager.execute_with_protection(
    exchange_name,
    exchange.get_orderbook,
    symbol
)
```

### Add Risk Checks Before Trading

```python
# Before placing order:
position_size = self.risk_manager.calculate_position_size(
    win_rate=0.65,  # Your historical win rate
    avg_win=2.0,  # Average win %
    avg_loss=1.0  # Average loss %
)

# Check if should open position
position = self.risk_manager.open_position(
    exchange="edgex",
    symbol="BTC-USD",
    side="long",
    entry_price=50000.0,
    size_usd=position_size
)

if position:
    # Safe to place order
    pass
else:
    # Risk limits exceeded
    logger.warning("Position rejected by risk manager")
```

### For `hedge_mode.py` Files

Add circuit breaker checks:

```python
from security.rate_limiter import get_manager

rate_manager = get_manager()
circuit = rate_manager.get_circuit_breaker(exchange_name)

if not circuit.is_closed():
    logger.error(f"Circuit breaker OPEN for {exchange_name}")
    return  # Skip this exchange

# Proceed with trading...
```

---

## 🎯 Step 6: Verify Everything Works (5 minutes)

### Run Built-in Tests

```bash
# Test rate limiter
python3 -c "
from security.rate_limiter import RateLimiter, RateLimitConfig
config = RateLimitConfig(max_requests=10, window_seconds=1)
limiter = RateLimiter(config)
print('✅ Rate limiter OK' if limiter.acquire() else '❌ Failed')
"

# Test risk manager
python3 -c "
from security.risk_manager import RiskManager
manager = RiskManager(initial_capital=10000)
size = manager.calculate_position_size(0.6, 2.0, 1.0)
print(f'✅ Risk manager OK - Position size: \${size:.2f}')
"
```

### Check Encrypted Keys

```bash
# Retrieve a key (will prompt for password)
python3 -m security.secure_key_manager get EDGEX_ACCOUNT_ID
```

---

## 📊 Monitoring Your System

### View Rate Limiter Stats

```python
from security.rate_limiter import get_manager

manager = get_manager()
stats = manager.get_all_stats()

for exchange, data in stats.items():
    print(f"\n{exchange}:")
    print(f"  Rate: {data['rate_limiter']['current_rate']} req/s")
    print(f"  Circuit: {data['circuit_breaker']['state']}")
```

### View Risk Metrics

```python
from security.risk_manager import RiskManager

manager = RiskManager()
metrics = manager.get_portfolio_metrics()

print(f"Capital: ${metrics['current_capital']:.2f}")
print(f"Total PnL: ${metrics['total_pnl']:.2f}")
print(f"Win Rate: {metrics['win_rate']:.1f}%")
```

---

## ⚠️ Important Security Notes

### 1. Master Password

- **Choose a strong password** (12+ characters)
- **Write it down** securely (password manager recommended)
- **No recovery possible** if you forget it

### 2. File Security

Add to `.gitignore`:

```
.env
.env.backup
.keys.enc
.salt
*.log
```

Verify:

```bash
# These should show as ignored:
git status --ignored
```

### 3. Never Commit Secrets

```bash
# Remove from git history if accidentally committed:
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch .env .keys.enc' \
  --prune-empty --tag-name-filter cat -- --all
```

---

## 🐛 Troubleshooting

### "Failed to decrypt keys"

**Cause**: Wrong master password

**Fix**:
```bash
# Verify password by listing keys
python3 -m security.secure_key_manager list

# If truly forgotten, re-migrate from .env.backup:
python3 -m security.secure_key_manager init --force
python3 -m security.secure_key_manager migrate .env.backup
```

### "Circuit breaker OPEN"

**Cause**: Too many failures or excessive losses

**Fix**:
```python
from security.rate_limiter import get_manager

manager = get_manager()
circuit = manager.get_circuit_breaker('edgex')

# Check status
print(circuit.get_stats())

# Manual reset if safe:
circuit.reset()
```

### "Rate limit timeout"

**Cause**: Too many requests too quickly

**Fix**: The system handles this automatically. If persistent:
- Reduce `check_interval_seconds` in config
- Reduce number of exchanges/symbols

---

## 📈 Next Steps

1. **Monitor for 24-48 hours** with small positions
2. **Review logs daily** for errors or warnings
3. **Gradually increase** position sizes if stable
4. **Set up alerts** (Telegram/Lark bot) for critical events
5. **Backtest** different risk parameters

---

## 🆘 Getting Help

If you encounter issues:

1. **Check logs**: Look in terminal output for errors
2. **Verify config**: Review `arbitrage_config.json`
3. **Test modules**: Run individual tests from this guide
4. **Review docs**: See `SECURITY_AUDIT_REPORT.md` for details

---

## ✅ Success Checklist

Before going live:

- [ ] Installed security modules
- [ ] Migrated API keys to encrypted storage
- [ ] Configured `arbitrage_config.json`
- [ ] Tested with dry run (no real trades)
- [ ] Tested with small amount ($100)
- [ ] Integrated with existing bot code
- [ ] Added to `.gitignore`
- [ ] Verified rate limiting works
- [ ] Verified circuit breakers work
- [ ] Set up monitoring/alerts
- [ ] Backed up master password securely

**When all boxes checked**: You're ready to scale up! 🚀
