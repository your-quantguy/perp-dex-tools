#!/usr/bin/env python3
"""
运行Lighter和Paradex之间的对冲策略
"""

import asyncio
import argparse
from decimal import Decimal
from hedge_strategy import HedgeStrategy
from dotenv import load_dotenv
from types import SimpleNamespace


async def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='运行Lighter和Paradex之间的对冲策略')
    parser.add_argument('--env-file', type=str, default='.env', help='环境变量文件路径（默认 .env）')
    parser.add_argument('--ticker', type=str, default='ETH-PERP', help='交易对名称')
    parser.add_argument('--quantity', type=str, default='0.03', help='交易数量')
    parser.add_argument('--side', type=str, default='buy', choices=['buy', 'sell'], help='在Lighter上的交易方向')
    parser.add_argument('--price-offset-ticks', type=int, default=1, help='BBO价格偏移的tick数量')
    parser.add_argument('--order-timeout', type=int, default=60, help='订单超时取消时间（秒）')
    parser.add_argument('--max-retries', type=int, default=3, help='取消后最大重试次数')
    args = parser.parse_args()

    # 加载环境变量
    try:
        load_dotenv(args.env_file)
    except Exception:
        # 容错：即使加载失败也继续，后续客户端会校验并报错
        pass

    # 配置策略
    # Normalize symbols for different exchanges
    base_symbol = args.ticker.split('-')[0] if '-PERP' in args.ticker else args.ticker

    # Base config
    config = {
        "ticker": args.ticker,
        "quantity": args.quantity,
        "side": args.side,
        "price_offset_ticks": args.price_offset_ticks,
        "order_timeout_seconds": args.order_timeout,
        "max_retries": args.max_retries,
    }

    # Exchange-specific configs as attribute objects (expected by clients)
    # lighter: uses ticker, tick_size, close_order_side
    lighter_cfg = SimpleNamespace(
        ticker=base_symbol,
        tick_size=Decimal("0.01"),
        close_order_side=args.side
    )
    # paradex: uses ticker, tick_size, quantity, direction, close_order_side
    paradex_direction = 'sell' if args.side == 'buy' else 'buy'
    paradex_cfg = SimpleNamespace(
        ticker=base_symbol,
        tick_size=Decimal("0.01"),
        quantity=Decimal(str(args.quantity)),
        direction=paradex_direction,
        close_order_side=paradex_direction
    )
    config["lighter"] = lighter_cfg
    config["paradex"] = paradex_cfg
    
    # 创建并运行策略
    strategy = HedgeStrategy(config)
    result = await strategy.run()
    
    if result:
        print(f"对冲策略成功完成: {args.side} {args.quantity} {args.ticker} 在Lighter，并在Paradex上对冲")
    else:
        print("对冲策略执行失败，请查看日志了解详情")


if __name__ == "__main__":
    asyncio.run(main())