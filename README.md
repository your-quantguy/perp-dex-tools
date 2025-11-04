##### 关注我 **X (Twitter)**: [@yourQuantGuy](https://x.com/yourQuantGuy)

---

**English speakers**: Please read README_EN.md for the English version of this documentation.

## 📢 分享说明

**欢迎分享本项目！** 如果您要分享或修改此代码，请务必包含对原始仓库的引用。我们鼓励开源社区的发展，但请保持对原作者工作的尊重和认可。

---

## 自动交易机器人

一个支持多个交易所（目前包括 EdgeX, Backpack, Paradex, Aster, Lighter, grvt, Extended）的模块化交易机器人。该机器人实现了自动下单并在盈利时自动平仓的策略，主要目的是取得高交易量。

## 邀请链接 (获得返佣以及福利)

#### EdgeX: [https://pro.edgex.exchange/referral/QUANT](https://pro.edgex.exchange/referral/QUANT)

永久享受 VIP 1 费率；额外 10% 手续费返佣；10% 额外奖励积分

#### Backpack: [https://backpack.exchange/join/quant](https://backpack.exchange/join/quant)

使用我的推荐链接获得 35% 手续费返佣

#### Paradex: [https://app.paradex.trade/r/quant](https://app.paradex.trade/r/quant)

使用我的推荐链接获得 10% 手续费返佣以及潜在未来福利

#### Aster: [https://www.asterdex.com/zh-CN/referral/5191B1](https://www.asterdex.com/zh-CN/referral/5191B1)

使用我的推荐链接获得 30% 手续费返佣以及积分加成

#### grvt: [https://grvt.io/exchange/sign-up?ref=QUANT](https://grvt.io/exchange/sign-up?ref=QUANT)

获得 1.3x 全网最高的积分加成，未来的手续费返佣（官方预计 10 月中上线），以及即将开始的专属交易竞赛

#### Extended: [https://app.extended.exchange/join/QUANT](https://app.extended.exchange/join/QUANT)
10%的即时手续费减免；积分加成（官方未公布具体加成公式，但文档里有明确说明，通过官方大使邀请能拿到比自己小号邀请自己更多的分数）

#### ApeX: [https://join.omni.apex.exchange/quant]( https://join.omni.apex.exchange/quant)
30%返佣; 5%手续费减免; 积分加成; 有资格参与10月20日至11月2日的社区专属交易竞赛，总奖金高达$5500

## 安装

Python 版本要求（最佳选项是 Python 3.10 - 3.12）：

- grvt 要求 python 版本在 3.10 及以上
- Paradex 要求 python 版本在 3.9 - 3.12
- 其他交易所需要 python 版本在 3.8 及以上

1. **克隆仓库**：

   ```bash
   git clone <repository-url>
   cd perp-dex-tools
   ```

2. **创建并激活虚拟环境**：

   首先确保你目前不在任何虚拟环境中：

   ```bash
   deactivate
   ```

   创建虚拟环境：

   ```bash
   python3 -m venv env
   ```

   激活虚拟环境（每次使用脚本时，都需要激活虚拟环境）：

   ```bash
   source env/bin/activate  # Windows: env\Scripts\activate
   ```

3. **安装依赖**：
   首先确保你目前不在任何虚拟环境中：

   ```bash
   deactivate
   ```

   激活虚拟环境（每次使用脚本时，都需要激活虚拟环境）：

   ```bash
   source env/bin/activate  # Windows: env\Scripts\activate
   ```

   ```bash
   pip install -r requirements.txt
   ```

   **grvt 用户**：如果您想使用 grvt 交易所，需要额外安装 grvt 专用依赖：
   激活虚拟环境（每次使用脚本时，都需要激活虚拟环境）：

   ```bash
   source env/bin/activate  # Windows: env\Scripts\activate
   ```

   ```bash
   pip install grvt-pysdk
   ```

   **Paradex 用户**：如果您想使用 Paradex 交易所，需要额外创建一个虚拟环境并安装 Paradex 专用依赖：

   首先确保你目前不在任何虚拟环境中：

   ```bash
   deactivate
   ```

   创建 Paradex 专用的虚拟环境（名称为 para_env）：

   ```bash
   python3 -m venv para_env
   ```

   激活虚拟环境（每次使用脚本时，都需要激活虚拟环境）：

   ```bash
   source para_env/bin/activate  # Windows: para_env\Scripts\activate
   ```

   安装 Paradex 依赖

   ```bash
   pip install -r para_requirements.txt
   ```

   **apex 用户**：如果您想使用 apex 交易所，需要额外安装 apex 专用依赖：
   激活虚拟环境（每次使用脚本时，都需要激活虚拟环境）：

   ```bash
   source env/bin/activate  # Windows: env\Scripts\activate
   ```

   ```bash
   pip install -r apex_requirements.txt
   ```

4. **设置环境变量**：
   在项目根目录创建`.env`文件，并使用 env_example.txt 作为样本，修改为你的 api 密匙。

5. **Telegram 机器人设置（可选）**：
   如需接收交易通知，请参考 [Telegram 机器人设置指南](docs/telegram-bot-setup.md) 配置 Telegram 机器人。

## 策略概述

**重要提醒**：大家一定要先理解了这个脚本的逻辑和风险，这样你就能设置更适合你自己的参数，或者你也可能觉得这不是一个好策略，根本不想用这个策略来刷交易量。我在推特也说过，我不是为了分享而写这些脚本，而是我真的在用这个脚本，所以才写了，然后才顺便分享出来。
这个脚本主要还是要看长期下来的磨损，只要脚本持续开单，如果一个月后价格到你被套的最高点，那么你这一个月的交易量就都是零磨损的了。所以我认为如果把`--quantity`和`--wait-time`设置的太小，并不是一个好的长期的策略，但确实适合短期内高强度冲交易量。我自己一般用 40 到 60 的 quantity，450 到 650 的 wait-time，以此来保证即使市场和你的判断想法，脚本依然能够持续稳定地下单，直到价格回到你的开单点，实现零磨损刷了交易量。

该机器人实现了简单的交易策略：

1. **订单下单**：在市场价格附近下限价单
2. **订单监控**：等待订单成交
3. **平仓订单**：在止盈水平自动下平仓单
4. **持仓管理**：监控持仓和活跃订单
5. **风险管理**：限制最大并发订单数
6. **网格步长控制**：通过 `--grid-step` 参数控制新订单与现有平仓订单之间的最小价格距离
7. **停止交易控制**：通过 `--stop-price` 参数控制停止交易的的价格条件

#### ⚙️ 关键参数

- **quantity**: 每笔订单的交易数量
- **direction**: 脚本交易的方向，buy 表示看多，sell 表示看空
- **take-profit**: 止盈百分比（如 0.02 表示 0.02%）
- **max-orders**: 最大同时活跃订单数（风险控制）
- **wait-time**: 订单间等待时间（避免过于频繁交易）
- **grid-step**: 网格步长控制（防止平仓订单过于密集）
- **stop-price**: 当市场价格达到该价格时退出脚本
- **pause-price**: 当市场价格达到该价格时暂停脚本

#### 网格步长功能详解

`--grid-step` 参数用于控制新订单的平仓价格与现有平仓订单之间的最小距离：

- **默认值 -100**：无网格步长限制，按原策略执行
- **正值（如 0.5）**：新订单的平仓价格必须与最近的平仓订单价格保持至少 0.5% 的距离
- **作用**：防止平仓订单过于密集，提高成交概率和风险管理

例如，当看多且 `--grid-step 0.5` 时：

- 如果现有平仓订单价格为 2000 USDT
- 新订单的平仓价格必须低于 1990 USDT（2000 × (1 - 0.5%)）
- 这样可以避免平仓订单过于接近，提高整体策略效果

#### 📊 交易流程示例

假设当前 ETH 价格为 $2000，设置止盈为 0.02%：

1. **开仓**：在 $2000.40 下买单（略高于市价）
2. **成交**：订单被市场成交，获得多头仓位
3. **平仓**：立即在 $2000.80 下卖单（止盈价格）
4. **完成**：平仓单成交，获得 0.02% 利润
5. **重复**：继续下一轮交易

#### 🛡️ 风险控制

- **订单限制**：通过 `max-orders` 限制最大并发订单数
- **网格控制**：通过 `grid-step` 确保平仓订单有合理间距
- **下单频率控制**：通过 `wait-time` 确保下单的时间间隔，防止短时间内被套
- **实时监控**：持续监控持仓和订单状态
- **⚠️ 无止损机制**：此策略不包含止损功能，在不利市场条件下可能面临较大损失

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

ETH（带停止交易的价格控制）：

```bash
python runbot.py --exchange edgex --ticker ETH --quantity 0.1 --take-profit 0.02 --max-orders 40 --wait-time 450 --stop-price 5500
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

ETH 永续合约（启用 Boost 模式）：

```bash
python runbot.py --exchange backpack --ticker ETH --direction buy --quantity 0.1 --boost
```

### Aster 交易所：

ETH：

```bash
python runbot.py --exchange aster --ticker ETH --quantity 0.1 --take-profit 0.02 --max-orders 40 --wait-time 450
```

ETH（启用 Boost 模式）：

```bash
python runbot.py --exchange aster --ticker ETH --direction buy --quantity 0.1 --boost
```

### GRVT 交易所：

BTC：

```bash
python runbot.py --exchange grvt --ticker BTC --quantity 0.05 --take-profit 0.02 --max-orders 40 --wait-time 450
```

### Extended 交易所：

ETH：

```bash
python runbot.py --exchange extended --ticker ETH --quantity 0.1 --take-profit 0 --max-orders 40 --wait-time 450 --grid-step 0.1
```

## 🆕 对冲模式 (Hedge Mode)

新增的对冲模式 (`hedge_mode.py`) 是一个新的交易策略，通过同时在两个交易所进行对冲交易来降低风险：

### 对冲模式工作原理

1. **开仓阶段**：在选定交易所（如 Backpack）下 maker 订单
2. **对冲阶段**：订单成交后，立即在 Lighter 下市价订单进行对冲
3. **平仓阶段**：在选定交易所下另一个 maker 订单平仓
4. **对冲平仓**：在 Lighter 下市价订单平仓

### 对冲模式优势

- **风险降低**：通过同时持有相反头寸，降低单边市场风险
- **交易量提升**：在两个交易所同时产生交易量
- **套利机会**：利用两个交易所之间的价差
- **自动化执行**：全自动化的对冲交易流程

### 对冲模式使用示例

```bash
# 运行 BTC 对冲模式（Backpack）
python hedge_mode.py --exchange backpack --ticker BTC --size 0.05 --iter 20

# 运行 ETH 对冲模式（Extended）
python hedge_mode.py --exchange extended --ticker ETH --size 0.1 --iter 20

# 运行 BTC 对冲模式（Apex）
python hedge_mode.py --exchange apex --ticker BTC --size 0.05 --iter 20

# 运行 BTC 对冲模式（GRVT）
python hedge_mode.py --exchange grvt --ticker BTC --size 0.05 --iter 20

# 运行 BTC 对冲模式（edgeX）
python hedge_mode.py --exchange edgex --ticker BTC --size 0.001 --iter 20
```

### 对冲模式参数

- `--exchange`: 主要交易所（支持 'backpack', 'extended', 'apex', 'grvt', 'edgex'）
- `--ticker`: 交易对符号（如 BTC, ETH）
- `--size`: 每笔订单数量
- `--iter`: 交易循环次数
- `--fill-timeout`: maker 订单填充超时时间（秒，默认 5）
- `--sleep`: 每一笔交易之后的暂停时间，增加持仓时间（秒，默认 0）

## 配置

### 环境变量

#### 通用配置

- `ACCOUNT_NAME`: 环境变量中当前账号的名称，用于多账号日志区分，可自定义，非必须

#### Telegram 配置（可选）

- `TELEGRAM_BOT_TOKEN`: Telegram 机器人令牌
- `TELEGRAM_CHAT_ID`: Telegram 对话 ID

#### EdgeX 配置

- `EDGEX_ACCOUNT_ID`: 您的 EdgeX 账户 ID
- `EDGEX_STARK_PRIVATE_KEY`: 您的 EdgeX API 私钥
- `EDGEX_BASE_URL`: EdgeX API 基础 URL（默认：https://pro.edgex.exchange）
- `EDGEX_WS_URL`: EdgeX WebSocket URL（默认：wss://quote.edgex.exchange）

#### Backpack 配置

- `BACKPACK_PUBLIC_KEY`: 您的 Backpack API Key
- `BACKPACK_SECRET_KEY`: 您的 Backpack API Secret

#### Paradex 配置

- `PARADEX_L1_ADDRESS`: L1 钱包地址
- `PARADEX_L2_PRIVATE_KEY`: L2 钱包私钥（点击头像，钱包，"复制 paradex 私钥"）

#### Aster 配置

- `ASTER_API_KEY`: 您的 Aster API Key
- `ASTER_SECRET_KEY`: 您的 Aster API Secret

#### Lighter 配置

- `API_KEY_PRIVATE_KEY`: Lighter API 私钥
- `LIGHTER_ACCOUNT_INDEX`: Lighter 账户索引
- `LIGHTER_API_KEY_INDEX`: Lighter API 密钥索引

#### GRVT 配置

- `GRVT_TRADING_ACCOUNT_ID`: 您的 GRVT 交易账户 ID
- `GRVT_PRIVATE_KEY`: 您的 GRVT 私钥
- `GRVT_API_KEY`: 您的 GRVT API 密钥

#### Extended 配置

- `EXTENDED_API_KEY`: Extended API Key
- `EXTENDED_STARK_KEY_PUBLIC`: 创建API后显示的 Stark 公钥
- `EXTENDED_STARK_KEY_PRIVATE`: 创建API后显示的 Stark 私钥
- `EXTENDED_VAULT`: 创建API后显示的 Extended Vault ID

#### Apex 配置

- `APEX_API_KEY`: 您的 Apex API 密钥
- `APEX_API_KEY_PASSPHRASE`: 您的 Apex API 密钥密码
- `APEX_API_KEY_SECRET`: 您的 Apex API 密钥私钥
- `APEX_OMNI_KEY_SEED`: 您的 Apex Omni 密钥种子

**获取 LIGHTER_ACCOUNT_INDEX 的方法**：

1. 在下面的网址最后加上你的钱包地址：

   ```
   https://mainnet.zklighter.elliot.ai/api/v1/account?by=l1_address&value=
   ```

2. 在浏览器中打开这个网址

3. 在结果中搜索 "account_index" - 如果你有子账户，会有多个 account_index，短的那个是你主账户的，长的是你的子账户。

### 命令行参数

- `--exchange`: 使用的交易所：'edgex'、'backpack'、'paradex'、'aster'、'lighter'、'grvt' 或 'extended'（默认：edgex）
- `--ticker`: 标的资产符号（例如：ETH、BTC、SOL）。合约 ID 自动解析。
- `--quantity`: 订单数量（默认：0.1）
- `--take-profit`: 止盈百分比（例如 0.02 表示 0.02%）
- `--direction`: 交易方向：'buy'或'sell'（默认：buy）
- `--env-file`: 账户配置文件 (默认：.env)
- `--max-orders`: 最大活跃订单数（默认：40）
- `--wait-time`: 订单间等待时间（秒）（默认：450）
- `--grid-step`: 与下一个平仓订单价格的最小距离百分比（默认：-100，表示无限制）
- `--stop-price`: 当 `direction` 是 'buy' 时，当 price >= stop-price 时停止交易并退出程序；'sell' 逻辑相反（默认：-1，表示不会因为价格原因停止交易），参数的目的是防止订单被挂在”你认为的开多高点或开空低点“。
- `--pause-price`: 当 `direction` 是 'buy' 时，当 price >= pause-price 时暂停交易，并在价格回到 pause-price 以下时重新开始交易；'sell' 逻辑相反（默认：-1，表示不会因为价格原因停止交易），参数的目的是防止订单被挂在”你认为的开多高点或开空低点“。
- `--boost`: 启用 Boost 模式进行交易量提升（仅适用于 aster 和 backpack 交易所）
  Boost 模式的下单逻辑：下 maker 单开仓，成交后立即用 taker 单关仓，以此循环。磨损为一单 maker，一单 taker 的手续费，以及滑点。

## 日志记录

该机器人提供全面的日志记录：

- **交易日志**：包含订单详情的 CSV 文件
- **调试日志**：带时间戳的详细活动日志
- **控制台输出**：实时状态更新
- **错误处理**：全面的错误日志记录和处理

## Q & A

### 如何在同一设备配置同一交易所的多个账号？

1. 为每个账户创建一个 .env 文件，如 account_1.env, account_2.env
2. 在每个账户的 .env 文件中设置 `ACCOUNT_NAME=`, 如`ACCOUNT_NAME=MAIN`。
3. 在每个文件中配置好每个账户的 API key 或是密匙
4. 通过更改命令行中的 `--env-file` 参数来开始不同的账户，如 `python runbot.py --env-file account_1.env [其他参数...]`

### 如何在同一设备配置不同交易所的多个账号？

将不同交易所的账号都配置在同一 `.env` 文件后，通过更改命令行中的 `--exchange` 参数来开始不同的交易所，如 `python runbot.py --exchange backpack [其他参数...]`

### 如何在同一设备用同一账号配置同一交易所的多个合约？

将账号配置在 `.env` 文件后，通过更改命令行中的 `--ticker` 参数来开始不同的合约，如 `python runbot.py --ticker ETH [其他参数...]`

## 贡献

1. Fork 仓库
2. 创建功能分支
3. 进行更改
4. 如适用，添加测试
5. 提交拉取请求

## 许可证

本项目采用非商业许可证 - 详情请参阅[LICENSE](LICENSE)文件。

**重要提醒**：本软件仅供个人学习和研究使用，严禁用于任何商业用途。如需商业使用，请联系作者获取商业许可证。

## 免责声明

本软件仅供教育和研究目的。加密货币交易涉及重大风险，可能导致重大财务损失。使用风险自负，切勿用您无法承受损失的资金进行交易。
