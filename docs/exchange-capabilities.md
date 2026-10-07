# 交易所能力与策略要求

交易所适配器通过 `exchanges/capabilities.py` 中的 Python `Protocol` 表示能力。策略构造时检查能力组合，不按交易所名称决定角色。`BaseExchangeClient` 只定义连接和基础行情；订单方法必须由适配器实际实现，因此继承基类不会让未实现的能力误通过检查。

| 策略或角色 | 必需能力组合 | 当前适配器 |
| --- | --- | --- |
| 单交易所开仓止盈 `TakeProfitStrategy` | `TakeProfitVenue`：连接、报价、仓位、活动订单、订单查询与撤单、限价开平仓 | Bulk、Arcus、Lighter |
| 对冲循环 maker `HedgeCycleStrategy` | `PairMaker`：连接、网络标识、仓位、订单核对、post-only 开平仓 | Bulk、Arcus |
| 对冲循环 taker | `PairTaker`：连接、网络标识、仓位、活动订单、精确数量校验、带滑点上限的即时单 | Bulk、Arcus、Lighter |
| 价差收敛 maker `SpreadStrategy` | `SpreadMaker`：`PairMaker` 加多档盘口、数量步长、账户风险、紧急 reduce-only IOC | Bulk、Arcus |
| 价差收敛 taker | `SpreadTaker`：`PairTaker` 加多档盘口、数量步长、账户风险 | Bulk、Arcus、Lighter |

当前两腿策略支持 Bulk/Arcus、Bulk/Lighter、Arcus/Lighter。Bulk 与 Arcus 可以互换 maker/taker 角色；Lighter 目前只支持 taker 角色，因为尚未实现明确报价的 post-only 能力。两腿必须在同一网络，且不能是同一交易所。交易所的合约可用性、数量步长、最小名义价值和保证金由连接后的市场数据与下单校验决定。

`BoundedTaker.place_hedge_market_order` 的语义是按指定滑点计算价格保护，提交不会留下活动挂单的即时订单；它不保证全部成交。Bulk 和 Arcus 用限价 IOC 实现，Lighter 使用其现有的带滑点控制下单方式。调用方仍须核对目标仓位；部分成交、未知回执或无法确认的订单会停止后续自动交易。`PostOnlyMaker` 要求明确报价只挂单或拒单，不能偷偷转成 taker。`AuthoritativeOrderRecovery` 是可选能力，有该能力时订单恢复优先读取权威来源。Telegram/飞书通知由策略层 `Notifier` 负责，不属于交易所能力。

新增交易所时，先实现真实订单语义和所需的细粒度接口，再通过对应策略的能力检查。至少应测试 IOC 的价格边界、部分成交、未知回执、订单状态与仓位核对。未来 funding 套利需要 funding 费率、结算时间和已结算费用等新能力；信号与持仓规则属于策略，不应塞进交易所接口。
