# 多交易对配置说明

此文档说明如何使用 JSON 配置文件同时运行多个交易对。

## 快速开始

### 1. 创建配置文件

复制 `config_example.json` 并根据需要修改：

```bash
cp config_example.json my_config.json
```

### 2. 运行多交易对机器人

```bash
python runbot_multi.py --config my_config.json
```

## 配置文件结构

### 完整配置示例

```json
{
  "exchange": "backpack",
  "env_file": ".env",
  "common_params": {
    "take_profit": 0.02,
    "direction": "sell",
    "max_orders": 10,
    "wait_time": 450,
    "grid_step": 0.2,
    "stop_price": -1,
    "pause_price": -1,
    "boost_mode": false
  },
  "tickers": [
    {
      "ticker": "SOL",
      "quantity": 0.5,
      "instance_id": "SOL_hao"
    },
    {
      "ticker": "BTC",
      "quantity": 0.01,
      "instance_id": "BTC_hao",
      "take_profit": 0.03,
      "direction": "buy"
    }
  ]
}
```

### 字段说明

#### 顶层字段

- **exchange** (必填): 交易所名称
  - 可选值: `backpack`, `edgex`, `paradex`, `aster`, `lighter`, `grvt`, `extended`, `ethereal`, `nado`, `standx`, `apex`

- **env_file** (可选): 环境变量文件路径
  - 默认: `.env`
  
#### common_params (公共参数)

这些参数会应用到所有交易对，除非在具体交易对中被覆盖：

- **take_profit** (可选): 止盈百分比
  - 默认: `0.02` (0.02%)
  
- **direction** (可选): 交易方向
  - 可选值: `buy` 或 `sell`
  - 默认: `buy`

- **max_orders** (可选): 最大活跃订单数
  - 默认: `40`

- **wait_time** (可选): 订单间等待时间（秒）
  - 默认: `450`

- **grid_step** (可选): 网格步长百分比
  - 默认: `-100` (无限制)
  - 正值表示最小价格间距

- **stop_price** (可选): 停止价格
  - 默认: `-1` (不停止)
  - buy: price >= stop_price 时停止
  - sell: price <= stop_price 时停止

- **pause_price** (可选): 暂停价格
  - 默认: `-1` (不暂停)
  - buy: price >= pause_price 时暂停
  - sell: price <= pause_price 时暂停

- **boost_mode** (可选): Boost 模式
  - 默认: `false`
  - 仅支持 `aster` 和 `backpack` 交易所

#### tickers (交易对列表)

每个交易对可以有以下字段：

- **ticker** (必填): 交易对符号
  - 示例: `"SOL"`, `"BTC"`, `"ETH"`

- **quantity** (必填): 订单数量
  - 示例: `0.5`, `0.01`, `1.0`

- **instance_id** (必填): 实例标识符
  - 用于区分不同的交易实例
  - 影响日志文件命名
  - 示例: `"SOL_hao"`, `"BTC_zhang"`

- **其他参数** (可选): 可以覆盖 `common_params` 中的任何参数
  - `take_profit`
  - `direction`
  - `max_orders`
  - `wait_time`
  - `grid_step`
  - `stop_price`
  - `pause_price`
  - `boost_mode`

## 使用示例

### 示例 1: 基本配置（所有交易对使用相同参数）

```json
{
  "exchange": "backpack",
  "common_params": {
    "take_profit": 0.02,
    "direction": "sell",
    "max_orders": 10,
    "wait_time": 450,
    "grid_step": 0.2
  },
  "tickers": [
    {"ticker": "SOL", "quantity": 0.5, "instance_id": "SOL_1"},
    {"ticker": "BTC", "quantity": 0.01, "instance_id": "BTC_1"},
    {"ticker": "ETH", "quantity": 0.1, "instance_id": "ETH_1"}
  ]
}
```

### 示例 2: 混合策略（不同交易对使用不同参数）

```json
{
  "exchange": "backpack",
  "common_params": {
    "take_profit": 0.02,
    "max_orders": 10,
    "wait_time": 450
  },
  "tickers": [
    {
      "ticker": "SOL",
      "quantity": 0.5,
      "instance_id": "SOL_sell",
      "direction": "sell",
      "grid_step": 0.2
    },
    {
      "ticker": "BTC",
      "quantity": 0.01,
      "instance_id": "BTC_buy",
      "direction": "buy",
      "take_profit": 0.03,
      "grid_step": 0.1
    },
    {
      "ticker": "ETH",
      "quantity": 0.1,
      "instance_id": "ETH_sell",
      "direction": "sell",
      "max_orders": 15,
      "grid_step": 0.25
    }
  ]
}
```

### 示例 3: 带价格控制

```json
{
  "exchange": "backpack",
  "common_params": {
    "take_profit": 0.02,
    "direction": "buy",
    "max_orders": 10,
    "wait_time": 450,
    "stop_price": 100000,
    "pause_price": 95000
  },
  "tickers": [
    {
      "ticker": "BTC",
      "quantity": 0.01,
      "instance_id": "BTC_conservative",
      "stop_price": 98000
    },
    {
      "ticker": "SOL",
      "quantity": 0.5,
      "instance_id": "SOL_aggressive",
      "stop_price": 200
    }
  ]
}
```

### 示例 4: Boost 模式（仅 Backpack 和 Aster）

```json
{
  "exchange": "backpack",
  "common_params": {
    "take_profit": 0.02,
    "direction": "buy",
    "boost_mode": true
  },
  "tickers": [
    {"ticker": "SOL", "quantity": 0.5, "instance_id": "SOL_boost"},
    {"ticker": "BTC", "quantity": 0.01, "instance_id": "BTC_boost"}
  ]
}
```

## 运行命令

### 基本运行

```bash
python runbot_multi.py --config my_config.json
```

### 带调试日志

```bash
python runbot_multi.py --config my_config.json --log-level DEBUG
```

### 可用日志级别

- `DEBUG`: 详细调试信息
- `INFO`: 一般信息
- `WARNING`: 警告信息（默认）
- `ERROR`: 仅错误信息

## 日志文件

每个交易对会生成独立的日志文件：

```
logs/
├── backpack_SOL_hao_orders.csv
├── backpack_BTC_hao_orders.csv
├── backpack_ETH_hao_orders.csv
└── backpack_SUI_hao_orders.csv
```

日志文件命名格式: `{exchange}_{ticker}_{instance_id}_orders.csv`

## 飞书通知

每个交易对的飞书通知都会包含对应的 `instance_id`，方便区分：

- 启动通知: 包含实例ID和配置信息
- 错误通知: 包含实例ID和错误详情
- 每小时统计: 包含实例ID和统计数据

## 停止运行

使用 `Ctrl+C` 停止所有交易对。程序会优雅地关闭所有实例。

## 注意事项

1. **资源消耗**: 同时运行多个交易对会占用更多 CPU 和内存
2. **API 限制**: 注意交易所的 API 速率限制
3. **实例ID**: 确保每个交易对使用不同的 `instance_id`
4. **Boost 模式**: 仅支持 `aster` 和 `backpack` 交易所
5. **环境文件**: 所有交易对共享同一个环境文件中的 API 凭证
6. **启动延迟**: 各交易对启动之间有 2 秒延迟，避免冲突

## 故障排除

### 问题: 配置文件解析错误

确保 JSON 格式正确，可以使用在线 JSON 验证器检查。

### 问题: 某个交易对启动失败

检查该交易对的配置参数是否正确，特别是 `quantity` 和 `ticker`。

### 问题: Boost 模式错误

确保使用的交易所是 `backpack` 或 `aster`。

### 问题: 环境文件未找到

检查 `env_file` 路径是否正确，或使用默认的 `.env` 文件。

## 从单个命令行迁移

如果你之前使用：

```bash
python runbot.py --exchange backpack --ticker SOL --quantity 0.5 --take-profit 0.02 --direction sell --max-orders 10 --wait-time 450 --grid-step 0.2 --instance-id SOL_hao
```

现在可以创建配置文件 `sol_config.json`:

```json
{
  "exchange": "backpack",
  "tickers": [
    {
      "ticker": "SOL",
      "quantity": 0.5,
      "instance_id": "SOL_hao",
      "take_profit": 0.02,
      "direction": "sell",
      "max_orders": 10,
      "wait_time": 450,
      "grid_step": 0.2
    }
  ]
}
```

然后运行：

```bash
python runbot_multi.py --config sol_config.json
```
