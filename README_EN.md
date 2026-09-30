# Perpetual Trading Bot

[中文说明](README.md)

This project supports **Bulk, Arcus, and Lighter** in single-exchange mode, plus **Bulk↔Lighter and Arcus↔Lighter** hedge mode. `runbot.py` places a limit entry order on the selected exchange, then places a take-profit order after a fill. It monitors orders, positions, and market data. Single-exchange mode has no stop loss or automatic position close on exit.

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

## Hedge mode

`hedge_mode.py` places maker orders on Bulk or Arcus and uses an equal, opposite market order on Lighter for each confirmed fill. It cancels an unfilled maker order after `--fill-timeout`, reconciles the final fill, and checks both positions before placing another order. An uncertain order or fill stops trading and triggers configured Telegram or Lark alerts.

Lighter currently connects only to mainnet, so the maker exchange **must also be on mainnet**. Both accounts must start flat with no active orders in the selected market. Configure credentials for both exchanges and verify market availability, fees, account limits, and sufficient Lighter margin.

```bash
uv run --locked python hedge_mode.py --exchange bulk --ticker ETH --size 0.1 --iter 2
uv run --locked python hedge_mode.py --exchange arcus --ticker ETH --size 0.1 --iter 2 --max-position 0.5
```

`--size` is the underlying amount per maker order; `--iter` is the cycle count. With the default `--max-position 0`, each cycle buys from flat and sells back to flat. With a positive value, the maker position grows to `+max-position`, then reverses to `-max-position`; the final cycle returns to flat. The two legs remain opposite. `max-position` must be an exact multiple of `size`. `--max-slippage` limits each Lighter market hedge to 0.5% by default, and `--target-timeout` defaults to 600 seconds per position target. On exit, the bot does not force close positions or cancel orders whose outcomes are uncertain; review both accounts manually.

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

Pytest runs the existing unittest suite. Pyright currently checks the exchange base interface and hedge entry point; its scope will grow as the code is refactored. Ruff formatting can be applied to files as they are changed; the existing code has not been formatted wholesale.
