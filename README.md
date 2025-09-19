##### 关注我 **X (Twitter)**: [@yourQuantGuy](https://x.com/yourQuantGuy)
---
🌍 Language / 语言： **English speakers**: Please read README_EN.md for the English version of this documentation.

## 交易机器人
一个支持多个交易所（目前包括 EdgeX 和 Backpack）的模块化交易机器人。该机器人实现了自动下单并在盈利时自动平仓的策略。


## 邀请链接 (获得返佣以及福利)

#### EdgeX 交易所: [https://pro.edgex.exchange/referral/QUANT](https://pro.edgex.exchange/referral/QUANT)
永久享受 VIP 1 费率；额外 10% 手续费返佣；10% 额外奖励积分

#### Backpack 交易所: [https://backpack.exchange/join/quant](https://backpack.exchange/join/quant)
使用我的推荐链接获得 30% 手续费返佣

#### Paradex 交易所: [https://app.paradex.trade/r/quant](https://app.paradex.trade/r/quant)
使用我的推荐链接获得 10% 手续费返佣以及潜在未来福利

## 安装

1. **克隆仓库**：

   ```bash
   git clone <repository-url>
   cd perp-dex-tools
   ```

2. **创建并激活虚拟环境**：

   ```bash
   python3 -m venv env
   source env/bin/activate  # Windows: env\Scripts\activate
   ```

3. **安装依赖**：

   ```bash
   pip install -r requirements.txt
   ```

   **Paradex 用户**：如果您想使用 Paradex 交易所，需要额外安装 Paradex 专用依赖：

   ```bash
   pip install -r para_requirements.txt
   ```

4. **设置环境变量**：
   使用 env_example.txt 在项目根目录创建`.env`文件。

## 策略概述

### 机器人工作原理

本交易机器人专门设计用于永续合约交易。以下是策略的详细说明：

#### 🎯 核心策略
1. **自动下单**：机器人会在当前市场价格附近自动下限价单
2. **等待成交**：监控订单状态，等待订单被市场成交
3. **自动平仓**：订单成交后，立即在预设的止盈价格下平仓单
4. **循环执行**：重复上述过程，持续进行交易

#### 📊 交易流程示例
假设当前 ETH 价格为 $2000，设置止盈为 0.02%：

1. **开仓**：在 $2000.40 下买单（略高于市价）
2. **成交**：订单被市场成交，获得多头仓位
3. **平仓**：立即在 $2000.80 下卖单（止盈价格）
4. **完成**：平仓单成交，获得 0.02% 利润
5. **重复**：继续下一轮交易

#### ⚙️ 关键参数
- **quantity**: 每笔订单的交易数量
- **take-profit**: 止盈百分比（如 0.02 表示 0.02%）
- **max-orders**: 最大同时活跃订单数（风险控制）
- **wait-time**: 订单间等待时间（避免过于频繁交易）
- **grid-step**: 网格步长控制（防止平仓订单过于密集）

#### 🛡️ 风险控制
- **订单限制**：通过 `max-orders` 限制最大并发订单数
- **网格控制**：通过 `grid-step` 确保平仓订单有合理间距
- **超时处理**：长时间未成交的订单会被自动取消
- **实时监控**：持续监控持仓和订单状态
- **⚠️ 无止损机制**：此策略不包含止损功能，在不利市场条件下可能面临较大损失

#### 💡 适用场景
- **震荡市场**：在价格区间内反复交易获利
- **低波动环境**：通过频繁小额交易积累交易量和利润
- **自动化交易**：无需人工干预的24/7交易

## 示例命令：

### EdgeX 交易所：

ETH：

```bash
python runbot.py --exchange edgex --ticker ETH --quantity 0.1 --take-profit 0.02 --max-orders 40 --wait-time 450
```

ETH（带网格步长控制）：

```bash
python runbot.py --exchange edgex --ticker ETH --quantity 0.1 --take-profit 0.02 --max-orders 40 --wait-time 450 --grid-step 0.5
```

BTC：

```bash
python runbot.py --exchange edgex --ticker BTC --quantity 0.05 --take-profit 0.02 --max-orders 40 --wait-time 450
```

### Backpack 交易所：

ETH 永续合约：

```bash
python runbot.py --exchange backpack --ticker ETH --quantity 0.1 --take-profit 0.02 --max-orders 40 --wait-time 450
```

ETH 永续合约（带网格步长控制）：

```bash
python runbot.py --exchange backpack --ticker ETH --quantity 0.1 --take-profit 0.02 --max-orders 40 --wait-time 450 --grid-step 0.3
```

## 配置

### 环境变量

#### EdgeX 配置

- `EDGEX_ACCOUNT_ID`: 您的 EdgeX 账户 ID
- `EDGEX_STARK_PRIVATE_KEY`: 您的 EdgeX API 私钥
- `EDGEX_BASE_URL`: EdgeX API 基础 URL（默认：https://pro.edgex.exchange）
- `EDGEX_WS_URL`: EdgeX WebSocket URL（默认：wss://quote.edgex.exchange）

#### Backpack 配置

- `BACKPACK_PUBLIC_KEY`: 您的 Backpack API Key
- `BACKPACK_SECRET_KEY`: 您的 Backpack API Secret

#### Paradex 配置

- `PARADEX_L1_ADDRESS`: L1钱包地址
- `PARADEX_L2_PRIVATE_KEY`: L2钱包私钥（点击头像，钱包，“复制paradex私钥”）

### 命令行参数

- `--exchange`: 使用的交易所：'edgex'、'backpack'或'paradex'（默认：edgex）
- `--ticker`: 标的资产符号（例如：ETH、BTC、SOL）。合约 ID 自动解析。
- `--quantity`: 订单数量（默认：0.1）
- `--take-profit`: 止盈百分比（例如 0.02 表示 0.02%）
- `--direction`: 交易方向：'buy'或'sell'（默认：buy）
- `--max-orders`: 最大活跃订单数（默认：40）
- `--wait-time`: 订单间等待时间（秒）（默认：450）
- `--grid-step`: 与下一个平仓订单价格的最小距离百分比（默认：-100，表示无限制）

## 交易策略

**重要提醒**：大家一定要先理解了这个脚本的逻辑和风险，这样你就能设置更适合你自己的参数，或者你也可能觉得这不是一个好策略，根本不想用这个策略来刷交易量。我在推特也说过，我不是为了分享而写这些脚本，而是我真的在用这个脚本，所以才写了，然后才顺便分享出来。
这个脚本主要还是要看长期下来的磨损，只要脚本持续开单，如果一个月后价格到你被套的最高点，那么你这一个月的交易量就都是零磨损的了。所以我认为如果把`--quantity`和`--wait-time`设置的太小，并不是一个好的长期的策略，但确实适合短期内高强度冲交易量。我自己一般用40到60的quantity，450到650的wait-time，以此来保证即使市场和你的判断想法，脚本依然能够持续稳定地下单，直到价格回到你的开单点，实现零磨损刷了交易量。

该机器人实现了简单的交易策略：

1. **订单下单**：在市场价格附近下限价单
2. **订单监控**：等待订单成交
3. **平仓订单**：在止盈水平自动下平仓单
4. **持仓管理**：监控持仓和活跃订单
5. **风险管理**：限制最大并发订单数
6. **网格步长控制**：通过 `--grid-step` 参数控制新订单与现有平仓订单之间的最小价格距离

### 网格步长功能

`--grid-step` 参数用于控制新订单的平仓价格与现有平仓订单之间的最小距离：

- **默认值 -100**：无网格步长限制，按原策略执行
- **正值（如 0.5）**：新订单的平仓价格必须与最近的平仓订单价格保持至少 0.5% 的距离
- **作用**：防止平仓订单过于密集，提高成交概率和风险管理

例如，当看多且 `--grid-step 0.5` 时：
- 如果现有平仓订单价格为 2000 USDT
- 新订单的平仓价格必须低于 1990 USDT（2000 × (1 - 0.5%)）
- 这样可以避免平仓订单过于接近，提高整体策略效果

## 日志记录

该机器人提供全面的日志记录：

- **交易日志**：包含订单详情的 CSV 文件
- **调试日志**：带时间戳的详细活动日志
- **控制台输出**：实时状态更新
- **错误处理**：全面的错误日志记录和处理

## 安全功能

- **订单限制**：可配置的最大订单数量
- **超时处理**：超时时自动取消订单
- **持仓监控**：持续监控持仓和订单状态
- **错误恢复**：优雅处理 API 错误和断开连接

## 贡献

1. Fork 仓库
2. 创建功能分支
3. 进行更改
4. 如适用，添加测试
5. 提交拉取请求

## 许可证

本项目采用 MIT 许可证 - 详情请参阅[LICENSE](LICENSE)文件。

## 免责声明

本软件仅供教育和研究目的。加密货币交易涉及重大风险，可能导致重大财务损失。使用风险自负，切勿用您无法承受损失的资金进行交易。
