# 永续合约交易机器人

[English](README_EN.md)

本项目支持 **Bulk、Arcus、Lighter** 的单交易所模式，以及 **Bulk↔Lighter、Arcus↔Lighter** 对冲模式。`runbot.py` 在所选交易所挂开仓限价单，成交后按指定百分比挂止盈单，并监控订单、仓位和行情。单交易所模式没有止损或自动平仓退出机制。

三家交易所的订单接口使用相同语义：订单 ID 可用于查询和撤单；订单数量是原始委托量，成交量是累计成交量，剩余量是尚未成交的数量；订单状态统一为 `OPEN`、`PARTIALLY_FILLED`、`FILLED`、`CANCELED` 或 `REJECTED`。下单回执只表示交易所接受了请求，实际成交以订单查询或推送为准。撤单只有在确认终态和累计成交量后才返回成功；无法确认的操作会停止自动下单并要求人工核查。Lighter 的订单 ID 使用客户端订单编号，其历史查询受交易所保留范围限制。

## 安装

使用 Python 3.12 和 [uv](https://docs.astral.sh/uv/)；`uv.lock` 固定了已解析的依赖版本：

```bash
uv sync --locked
```

复制 [env_example.txt](env_example.txt) 为 `.env`，只填写要运行的交易所凭据。也可用 `--env-file` 指向仓库外的配置文件。已有的 shell 环境变量优先于 env 文件中的值，切换账户前应清理旧变量。

```bash
uv run --locked python runbot.py --help
uv run --locked python runbot.py --exchange bulk --ticker ETH --direction buy --quantity 0.1 --take-profit 0.02 --max-orders 40 --wait-time 450
```

`--exchange` 必须明确指定为 `bulk`、`arcus` 或 `lighter`。`--take-profit 0.02` 表示 **0.02%** 的价格变化，并非 0.02 USDT 或扣除费用后的净利润。`--quantity` 是合约标的数量；交易所的最小数量、步长和名义金额限制仍适用。

## 交易所配置

### Bulk

默认连接主网。在 `.env` 中选择一种签名方式：

- 主钱包：设置 `BULK_PRIVATE_KEY`（Base58 私钥）。可选的 `BULK_ACCOUNT` 必须与该私钥对应。
- Agent wallet：先在主钱包授权 agent，再设置 `BULK_AGENT_PRIVATE_KEY` 和目标交易账户公钥 `BULK_ACCOUNT`；不要同时设置 `BULK_PRIVATE_KEY`。

`BULK_ACCOUNT` 是交易账户，并非 agent 地址。程序不会注册或撤销 agent。测试网需同时设置 `BULK_API_URL` 和 `BULK_WS_URL`；具体变量见 [env_example.txt](env_example.txt)。开仓与止盈使用只挂单限价单，止盈单为 reduce-only。断线后程序会重新核对行情和订单；订单结果无法确认时停止继续开仓，需要人工检查。

### Arcus

使用独立的 Ed25519 API Signing Key，不需要主钱包私钥。支持主网、测试网和子账户。开仓与止盈使用 `LIMIT + ALO`，止盈单为 reduce-only。测试网配置、只读探测与首次部署步骤见 [Arcus 接入说明](docs/arcus-setup.md)。

### Lighter

设置 `API_KEY_PRIVATE_KEY`、`LIGHTER_ACCOUNT_INDEX` 和 `LIGHTER_API_KEY_INDEX`。当前适配连接 Lighter 主网。示例：

```bash
uv run --locked python runbot.py --exchange lighter --ticker ETH --direction buy --quantity 0.1 --take-profit 0.02 --max-orders 40 --wait-time 450
```

## 对冲模式

`hedge_mode.py` 在 Bulk 或 Arcus 挂 maker 单，每次确认成交量后在 Lighter 下等量反向市价单。maker 单到 `--fill-timeout` 后会撤单，并依据确认的最终成交量补齐对冲；两腿仓位核对通过后才开始下一笔。已知订单 ID 的提交回执丢失、短暂查询失败或撤单回执丢失时，会在 `--recovery-timeout` 内核对订单与仓位，确认后继续；结果仍不明时停止新单并通过已配置的 Telegram/飞书渠道告警，不会盲目重发订单。

对冲腿由 `--hedge-exchange` 指定，默认且目前仅支持 `lighter`。对冲腿适配器需要支持精确数量校验、带滑点上限且不会留下活动挂单的即时对冲单，以及行情、活动订单和仓位查询；下单回执仍需由策略核对最终仓位。

Lighter 目前只连接主网，因此对冲模式要求 Bulk 或 Arcus **同样使用主网**。启动时要求两个账户在该市场无仓位、无活动订单。请分别配置两家的凭据，并先确认市场、额度、费用与 Lighter 保证金足以执行反向单。

```bash
uv run --locked python hedge_mode.py --exchange bulk --ticker ETH --size 0.1 --iter 2
uv run --locked python hedge_mode.py --exchange arcus --hedge-exchange lighter --ticker ETH --size 0.1 --iter 2 --max-position 0.5
```

`--size` 是每笔 maker 订单的标的数量；`--iter` 是循环次数。默认 `--max-position 0` 时，每轮从零仓位买入并对冲，再卖出回零。设置正值后，maker 仓位逐笔累积到 `+max-position`，再逐笔反向到 `-max-position`；最后一轮结束后回到零，两腿始终保持反向。`max-position` 必须是 `size` 的整数倍。`--max-slippage` 默认限制 Lighter 单笔市价对冲的价格偏离为 0.5%；`--target-timeout` 默认允许每个目标仓位最多等待 600 秒；`--recovery-timeout` 默认允许订单恢复核对 15 秒。程序退出时不会自动撤销未确认订单或强制平仓，需人工核对两个账户。

## 策略架构

`runbot.py` 运行 `TakeProfitStrategy`，`hedge_mode.py` 运行 `HedgeCycleStrategy`；原有 `TradingBot` 和 `HedgeBot` 名称仍可导入。两个策略共用 `StrategyRunner` 管理交易所连接、断开和异常处理，共用 `Notifier` 发送 Telegram/飞书通知。对冲策略通过 `MakerFirstPairExecutor` 执行目标仓位、确认部分成交和核对反向腿。交易所会话配置使用 `VenueConfig`，不再借用止盈策略的参数。

现阶段每个交易所账户、每个市场只运行一个策略。未来的价差收敛和 funding 策略可以复用运行器及两腿执行器，并分别实现机会判断、持仓和退出规则；若需要其他下单顺序，应增加相应执行器。现有程序不会在崩溃后自动接管仓位，重启前仍需人工核查。

## 主要参数与风险

- `--direction buy|sell`：开仓方向。
- `--max-orders`：活跃止盈单数量限制，不是严格的总敞口或杠杆上限。
- `--wait-time`：新开仓之间的基础等待时间；程序会依据活跃止盈单数量调整。
- `--grid-step`：新止盈价与已有止盈价的最小百分比距离；默认 `-100` 表示关闭限制。
- `--stop-price`：达到条件时停止**继续交易并退出程序**，不会自动平掉已有仓位或撤销止盈单。
- `--pause-price`：达到条件时暂停新开仓。

策略没有止损。价格长期朝不利方向移动时，仓位和风险可能持续累积。手续费、资金费和滑点也可能高于设定的止盈价差。首次使用先在测试环境验证订单、撤单和仓位核对；退出或断线后要检查交易所上的真实订单和仓位。

Telegram 通知仍然保留：配置 `TELEGRAM_BOT_TOKEN` 和 `TELEGRAM_CHAT_ID`，详见 [Telegram 指南](docs/telegram-bot-setup.md)。日志写入 `logs/`。

## 开发检查

```bash
uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked pyright
```

现有测试可由 pytest 直接运行。Pyright 检查整个项目，包括交易所适配器、交易逻辑和测试。Ruff 格式化可用于新改动的文件；现有代码尚未整体格式化。
