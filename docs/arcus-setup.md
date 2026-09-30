# Arcus 单交易所接入

支持 Arcus **永续合约**。开仓与止盈均为 `LIMIT + ALO`（只做 maker），止盈为 `reduceOnly`。

## 环境

使用 Python 3.12 和项目统一依赖：

```bash
cd ~/perp-dex-tools
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python runbot.py --help
python scripts/arcus_probe.py --network testnet --ticker ETH
```

## 授权与配置

在 [Arcus 测试网](https://testnet.arcus.xyz/) 或 [主网](https://app.arcus.xyz/)的 API Keys 页面，用主钱包连接、生成并授权 API key。保存仅显示一次的 **API Signing Key**；账户与子账户必须和授权一致。机器人不生成、不注册、不撤销密钥，也不需要主钱包私钥。

在仓库外建立专用 env 文件，例如 `~/.config/perp-dex-tools/arcus.env`，权限设为 `600`。替换以下占位符：

```dotenv
ARCUS_API_SIGNING_KEY=这里填API_Signing_Key的64位十六进制字符串
ARCUS_ADDRESS=0x这里填主钱包的40位十六进制地址
ARCUS_ACCOUNT_INDEX=0
ARCUS_NETWORK=testnet
# ARCUS_API_KEY=可选的64位十六进制API公钥
```

`ARCUS_API_KEY` 是公钥，可从 Signing Key 自动派生；**只提供公钥无法下单**。`ARCUS_ADDRESS` 是交易账户主钱包地址，不是 API 公钥。`ARCUS_NETWORK` 默认主网，首次请显式使用 `testnet`；一个变量同步选择 REST 和 WebSocket 网络。

API Signing Key 泄露也可能导致恶意交易及亏损。不要上传真实 env、私钥或签名。`runbot.py` 优先读取 shell 已有变量，切换时先清除旧的 `ARCUS_*` 变量再启动；修改 env 不影响已运行进程。

## 测试网启动

先在网页为选定子账户充值测试资金。程序检查密钥注册状态、有效期、子账户权限及账户可读性；不会自动修改杠杆或保证金模式。

以下是**测试网功能验证**命令，不是实盘资金建议。先确认 env 为 `ARCUS_NETWORK=testnet`：

```bash
python runbot.py \
  --exchange arcus \
  --env-file "$HOME/.config/perp-dex-tools/arcus.env" \
  --ticker ETH \
  --direction buy \
  --quantity 0.02 \
  --take-profit 0.02 \
  --max-orders 1 \
  --wait-time 450
```

数量规则从 `/v1/markets` 读取，检查步长、最小开仓数量、最小开仓名义金额和最大订单量。不要照搬 Bulk 的最低金额。Reduce-only 单豁免最小开仓名义金额，但仍受步长、最大数量及服务器其他规则约束。

`--max-orders` 是原有策略的挂单数参数，不是严格的美元敞口/杠杆上限。`--take-profit 0.02` 为 **0.02%**，不是净利润，手续费和资金费可能超过价差。切换主网须重新授权、充值并确认实盘费用、数量和总敞口。

## 日志与恢复

- 日志：`logs/arcus_ETH_activity.log`；成交 CSV 使用现有 `TradingLogger`。执行 `tail -n 100 logs/arcus_ETH_activity.log` 查看。
- `202/ACK` 只代表受理：下单等待订单状态；撤单等待终态，使用已确认的累计成交量挂对应止盈。写请求结果不明时不会重试，而是停止继续开仓并要求人工核对。
- 止盈单若已确认以 `POST_ONLY_WOULD_CROSS` 拒绝且零成交，按最新盘口最多提交 5 次（含首次），卖价不低于原止盈价、买价不高于原止盈价；其他拒绝或结果不明仍停止并要求核对。
- 自动重连、重新订阅和核对本进程跟踪的订单，行情和订单就绪后恢复；旧成交不会完成当前订单，重复状态不会重复回调。
- 分档价格按对应 tick 对齐，签名始终使用基础 `tickSize`；有效期为提交起 40 天，以满足官方至少一个月要求。
- 退出/断线**不自动撤单或平仓**。重启允许保留原止盈单；存在同方向旧开仓单时拒绝启动，要求先确认成交与止盈并撤销未成交开仓单。不要同时运行同一子账户同一市场的多份机器人，也不要运行中从网页修改它跟踪的订单。
- 返回 1000 笔活动订单可能已截断；适配器拒绝将其用于仓位核对。

## 验证范围

已做离线签名、精度、状态竞态、撤单、重连、账户及共用策略回归测试，并只读探测公开市场接口。**尚未使用用户 API key 做测试网或实盘下单验收**。首次部署先观察测试网上一笔开仓→止盈闭环与撤单/重连日志，再考虑实盘。

协议依据：[签名](https://docs.arcus.xyz/api-reference/authentication)、[下单](https://docs.arcus.xyz/api-reference/exchange/place-order)、[WebSocket](https://docs.arcus.xyz/api-reference/websocket)、[子账户](https://docs.arcus.xyz/concepts/subaccounts)。
