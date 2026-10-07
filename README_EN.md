# Perpetual Trading Bot

[中文说明](README.md)

This project supports **Bulk, Arcus, and Lighter** in single-exchange mode. Hedge and spread modes can pair Bulk/Arcus, Bulk/Lighter, or Arcus/Lighter: Bulk and Arcus currently support the maker role, and all three support the taker role. `runbot.py` places a limit entry order on the selected exchange, then places a take-profit order after a fill. It monitors orders, positions, and market data. Single-exchange mode has no stop loss or automatic position close on exit.

Entropy (Hyperliquid's `io` perpetual DEX) is available as a taker in two-leg modes, including Arcus maker + Entropy taker where both venues list the same asset. It is not a single-venue or maker adapter.

All three exchanges use the same order semantics: an order ID can be queried and canceled; size is the original quantity, filled size is cumulative, and remaining size is unfilled quantity; normalized statuses are `OPEN`, `PARTIALLY_FILLED`, `FILLED`, `CANCELED`, and `REJECTED`. A submission acknowledgment means the exchange accepted the request, while fills come from order queries or updates. Cancellation succeeds only after the terminal status and cumulative fill are confirmed. If the outcome cannot be confirmed, automatic ordering stops for manual inspection. Lighter uses the client order index as its order ID, and its history queries are subject to the exchange's retention limits.

## Install

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). `uv.lock` pins the resolved dependencies:

```bash
uv sync --locked
```

Copy [env_example.txt](env_example.txt) to `.env` and fill in credentials for the exchange you intend to use. You can also pass an external file with `--env-file`. Existing shell environment variables take precedence over the env file, so clear old values when switching accounts.

```bash
uv run --locked python runbot.py --help
uv run --locked python runbot.py --exchange bulk --ticker ETH --direction buy --quantity 0.1 --take-profit 0.02 --max-orders 40 --wait-time 450
```

Specify `--exchange` as `bulk`, `arcus`, or `lighter`. `--take-profit 0.02` means a **0.02%** price change, not 0.02 USDT or net profit after fees. `--quantity` is the amount of the underlying asset; exchange lot size and notional limits still apply.

## Exchange setup

### Bulk

Mainnet is the default. Choose one signing mode in `.env`:

- Owner wallet: set `BULK_PRIVATE_KEY` (Base58). Optional `BULK_ACCOUNT` must match the key.
- Agent wallet: authorize the agent with the owner wallet first, then set `BULK_AGENT_PRIVATE_KEY` and the target trading account public key in `BULK_ACCOUNT`. Do not also set `BULK_PRIVATE_KEY`.

`BULK_ACCOUNT` is the trading account, not the agent address. The bot does not register or revoke agents. For testnet, set both `BULK_API_URL` and `BULK_WS_URL`; see [env_example.txt](env_example.txt). Entry and take-profit orders are post-only limits, and take-profit orders are reduce-only. After a disconnect, the bot reconciles market data and orders before resuming. An uncertain order outcome stops new entries and requires manual review.

### Arcus

Uses a separate Ed25519 API Signing Key rather than the owner wallet private key. Mainnet, testnet, and subaccounts are supported. Entry and take-profit orders use `LIMIT + ALO`; take-profit is reduce-only. See the [Arcus setup guide](docs/arcus-setup.md) for configuration and testnet validation steps.

### Lighter

Set `API_KEY_PRIVATE_KEY`, `LIGHTER_ACCOUNT_INDEX`, and `LIGHTER_API_KEY_INDEX`. This adapter currently connects to Lighter mainnet.

```bash
uv run --locked python runbot.py --exchange lighter --ticker ETH --direction buy --quantity 0.1 --take-profit 0.02 --max-orders 40 --wait-time 450
```

### Entropy

Authorize a separate Hyperliquid API wallet, then set `ENTROPY_API_PRIVATE_KEY` to its private key and `ENTROPY_ACCOUNT_ADDRESS` to the main account address. Set `ENTROPY_NETWORK=mainnet` when needed. The adapter uses the official SDK for price-bounded IOC hedge orders. Spread risk currently supports standard accounts only; unified account and portfolio margin require a different policy. Set the Entropy market's isolated leverage at or below the spread strategy's `max_leverage` and verify actual fees and rebates before trading. See the [spread guide](docs/spread-strategy.md).

## Hedge mode

`hedge_mode.py` places maker orders on Bulk or Arcus and uses an equal, opposite immediate order on the selected taker exchange for each confirmed fill. It cancels an unfilled maker order after `--fill-timeout`, reconciles the final fill, and checks both positions before placing another order. If a known order ID loses its submit acknowledgement, a read briefly fails, or a cancel acknowledgement is lost, the bot reconciles orders and positions within `--recovery-timeout` and continues once confirmed. An unresolved outcome stops new orders and triggers configured Telegram or Lark alerts; orders are never blindly resubmitted.

Select the hedge leg with `--hedge-exchange`; it defaults to `lighter`, and can also be `bulk`, `arcus`, or `entropy` (different from the maker). A hedge adapter must validate exact order quantities, submit an immediate order with a slippage bound that cannot remain active, and expose market data, active orders, and positions. The strategy still confirms the resulting position after an order acknowledgment. Hedge cycling does not apply the spread strategy's leverage cap.

Both legs must use the same network; a Lighter leg requires a mainnet counterparty. Both accounts must start flat with no active orders in the selected market. Configure credentials for both exchanges and verify market availability, fees, account limits, and sufficient margin on both sides.

```bash
uv run --locked python hedge_mode.py --exchange bulk --ticker ETH --size 0.1 --iter 2
uv run --locked python hedge_mode.py --exchange arcus --hedge-exchange lighter --ticker ETH --size 0.1 --iter 2 --max-position 0.5
uv run --locked python hedge_mode.py --exchange bulk --hedge-exchange arcus --ticker ETH --size 0.1 --iter 2
```

`--size` is the underlying amount per maker order; `--iter` is the cycle count. With the default `--max-position 0`, each cycle buys from flat and sells back to flat. With a positive value, the maker position grows to `+max-position`, then reverses to `-max-position`; the final cycle returns to flat. The two legs remain opposite. `max-position` must be an exact multiple of `size`. `--max-slippage` bounds each taker order by 0.5% by default, `--target-timeout` defaults to 600 seconds per position target, and `--recovery-timeout` defaults to 15 seconds for order reconciliation. On exit, the bot does not force close positions or cancel orders whose outcomes are uncertain; review both accounts manually.

## Strategy architecture

`runbot.py` runs `TakeProfitStrategy`, and `hedge_mode.py` runs `HedgeCycleStrategy`; the former `TradingBot` and `HedgeBot` names remain importable. Both strategies use `StrategyRunner` for venue lifecycle and failure handling, plus `Notifier` for Telegram and Lark alerts. The hedge strategy uses `MakerFirstPairExecutor` to reach position targets, hedge partial fills, and reconcile uncertain orders. Exchange sessions use `VenueConfig` instead of placeholder take-profit settings.

The spread-convergence strategy uses the same pair executor with separate signal and risk logic. Venue pairs, including Entropy as taker when both markets list the asset, are selected by capabilities; see the [capability map](docs/exchange-capabilities.md).

For now, run only one strategy per exchange account and market. Future funding strategies can share the runner and pair executor while defining their own entry, hold, and exit decisions. An execution policy with a different order sequence will need its own executor. The program does not automatically take over positions after a crash; inspect both accounts before restarting.

## Parameters and risk

- `--direction buy|sell`: entry direction.
- `--max-orders`: active take-profit order count limit, not a strict exposure or leverage cap.
- `--wait-time`: base delay between entries, adjusted by the number of active take-profit orders.
- `--grid-step`: minimum percentage distance from existing take-profit prices; the default `-100` disables it.
- `--stop-price`: stops **new trading and exits the process** at the threshold. It does not close positions or cancel take-profit orders.
- `--pause-price`: pauses new entries at the threshold.

The strategy has no stop loss. Exposure can accumulate during an adverse price move. Fees, funding, and slippage can exceed the take-profit spread. Validate order, cancel, and position handling in a test environment first; check live orders and positions after exits or disconnects.

Telegram notifications remain available through `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`; see the [Telegram guide](docs/telegram-bot-setup-en.md). Logs are written to `logs/`.

## Development checks

```bash
uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked pyright
```

Pytest runs the existing unittest suite. Pyright checks the whole project, including exchange adapters, trading logic, and tests. Ruff formatting can be applied to files as they are changed; the existing code has not been formatted wholesale.
