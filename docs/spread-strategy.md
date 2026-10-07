# 价差收敛策略（首版）

当前支持 Bulk/Lighter、Arcus/Lighter、Bulk/Arcus 三组账户；Bulk 或 Arcus 可以做 maker，三家都可以做 taker，两个账户必须在相同网络。`venue_a` 和 `venue_b` 可交换顺序，但会改变价差的正负号。价差始终为 `(A 中间价 / B 中间价 - 1) × 10000` bps。每个账户与市场只能同时运行一个策略。交易所要求由[能力接口](exchange-capabilities.md)描述。

复制并编辑 [示例配置](../spread_config.example.json)。**示例数值仅说明格式，不能直接作为交易阈值。**先用历史数据和真实费率确认中枢、开平仓区间、维护保证金和成本预留。`center_bps` 在本版由人工固定；`entry_band_bps` 必须大于 `exit_band_bps`。

```bash
uv run --locked python spread_mode.py --config spread_config.example.json
uv run --locked python -m scripts.compare_spread_exits --config spread_config.example.json --observations state/bulk_lighter_eth_observations.jsonl
```

策略在价差落到下轨时做多 A/做空 B，升到上轨时反向。它先检查两家账户的总名义仓位与权益，限制新增仓位不超过 `max_leverage`（默认 2.5）和 `max_position`；taker 腿的多档盘口还限制单次可对冲数量。开仓的预计收益必须覆盖 maker/taker 手续费、退出预留、至少一天的 carry 预留和最低利润。maker 单使用明确报价与 post-only，超时后撤单，确认部分成交就对冲；下一笔重新读取盘口和信号。

`exit_rule` 默认 `center`，价差靠近中枢后仍需通过预计净利润检查才平仓。也可配置 `opposite_band` 研究另一种退出信号。盈亏判断使用保守的 taker 滑点保护价格，`carry_reserve_bps` 按持仓天数计提，至少计一天。正常离场仍是 maker + taker。退出旧方向、确认双腿回零后，下一轮才可能开反向仓。

`warning_buffer_fraction` 和 `critical_buffer_fraction` 是账户权益扣除维护保证金后，除以总名义仓位的比例。Bulk 与 Lighter 使用账户提供的维护保证金；Arcus 用配置的 `maintenance_fraction_a/b` 中对应的值估算，必须按实际市场规则填写。首版只接受全仓保证金账户。达到警戒线会通过已配置的 Telegram/飞书通知并停止加仓；达到紧急线会尝试对 maker 腿使用 reduce-only IOC，再同步减掉 taker 腿，最多核对重试三次。订单结果、活跃订单或仓位无法确认时停止自动操作并通知人工核查。紧急平仓不等待利润条件，也不保证交易所一定成交。

持仓成本与数量写入 `state_file`。重启若发现持仓，会要求人工核对两家账户和账本，再显式加 `--resume`；账本仓位不一致就拒绝启动。程序退出不会自动平仓。运行中的价差、账户风险和双方向执行报价写入 `observation_file`，对比脚本在相同入场信号上用记录的可交易数量、保护价格、费用和 carry 预留比较两种退出条件；**它没有模拟 maker 成交、队列优先级或实际 funding，不能当成回测收益。**当前成本账本采用下单时的保守价格边界和按日递增的 funding 预留，未读取实际成交均价和实际 funding；对冲发生在 maker 成交后，行情跳变仍可能使实际成本高于预估。应在小额实盘验证成交与风险读数后再扩大规模。
