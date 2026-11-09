#!/usr/bin/env python3
"""
Multi-Ticker Trading Bot - Run multiple trading pairs from a JSON config file
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
import sys
import dotenv
from decimal import Decimal
from typing import List, Dict, Any
from trading_bot import TradingBot, TradingConfig
from exchanges import ExchangeFactory


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    config_path = Path(config_file)
    if not config_path.exists():
        print(f"配置文件不存在: {config_path.resolve()}")
        sys.exit(1)
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError as e:
        print(f"JSON 配置文件解析错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"读取配置文件失败: {e}")
        sys.exit(1)


def create_trading_config(ticker_config: Dict[str, Any], common_params: Dict[str, Any], 
                         exchange: str) -> TradingConfig:
    """Create a TradingConfig from ticker configuration and common parameters."""
    # Start with common parameters
    params = common_params.copy()
    
    # Override with ticker-specific parameters
    params.update(ticker_config)
    
    # Ensure required fields exist
    if 'ticker' not in params:
        raise ValueError("Ticker configuration must include 'ticker' field")
    if 'quantity' not in params:
        raise ValueError(f"Ticker {params['ticker']} must include 'quantity' field")
    if 'instance_id' not in params:
        params['instance_id'] = f"{params['ticker']}_default"
    
    # Create TradingConfig
    config = TradingConfig(
        ticker=params['ticker'].upper(),
        contract_id='',  # will be set in the bot's run method
        tick_size=Decimal(0),
        quantity=Decimal(str(params['quantity'])),
        take_profit=Decimal(str(params.get('take_profit', 0.02))),
        direction=params.get('direction', 'buy').lower(),
        max_orders=params.get('max_orders', 40),
        wait_time=params.get('wait_time', 450),
        exchange=exchange.lower(),
        grid_step=Decimal(str(params.get('grid_step', -100))),
        instance_id=params['instance_id'],
        stop_price=Decimal(str(params.get('stop_price', -1))),
        pause_price=Decimal(str(params.get('pause_price', -1))),
        stop_loss=Decimal(str(params.get('stop-loss', params.get('stop_loss', -1)))),
        boost_mode=params.get('boost_mode', False)
    )
    
    return config


def setup_logging(log_level: str):
    """Setup global logging configuration."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    root_logger.setLevel(level)
    
    if log_level.upper() != 'DEBUG':
        logging.getLogger('websockets').setLevel(logging.WARNING)
    
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('lighter').setLevel(logging.WARNING)
    
    if log_level.upper() != 'DEBUG':
        root_logger.setLevel(logging.WARNING)


async def run_single_bot(config: TradingConfig, bot_number: int, total_bots: int):
    """Run a single trading bot instance."""
    bot = TradingBot(config)
    try:
        print(f"[{bot_number}/{total_bots}] 启动交易对: {config.ticker} "
              f"({config.direction}, 数量: {config.quantity}, "
              f"实例ID: {config.instance_id})")
        await bot.run()
    except Exception as e:
        print(f"[{bot_number}/{total_bots}] 交易对 {config.ticker} 执行失败: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Multi-Ticker Trading Bot - Run multiple trading pairs from a JSON config file'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to JSON configuration file')
    parser.add_argument('--log-level', type=str, default='WARNING',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: WARNING)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load configuration
    config_data = load_config(args.config)
    
    # Validate configuration
    if 'exchange' not in config_data:
        print("错误: 配置文件必须包含 'exchange' 字段")
        sys.exit(1)
    
    if 'tickers' not in config_data or not config_data['tickers']:
        print("错误: 配置文件必须包含至少一个 'tickers' 条目")
        sys.exit(1)
    
    exchange = config_data['exchange']
    supported_exchanges = ExchangeFactory.get_supported_exchanges()
    if exchange.lower() not in supported_exchanges:
        print(f"错误: 不支持的交易所 '{exchange}'. "
              f"支持的交易所: {', '.join(supported_exchanges)}")
        sys.exit(1)
    
    # Load environment file
    env_file = config_data.get('env_file', '.env')
    env_path = Path(env_file)
    if not env_path.exists():
        print(f"环境文件不存在: {env_path.resolve()}")
        sys.exit(1)
    dotenv.load_dotenv(env_file)
    
    # Get common parameters
    common_params = config_data.get('common_params', {})
    
    # Validate boost mode
    for ticker_config in config_data['tickers']:
        boost_mode = ticker_config.get('boost_mode', common_params.get('boost_mode', False))
        if boost_mode and exchange.lower() not in ['aster', 'backpack']:
            ticker = ticker_config.get('ticker', 'unknown')
            print(f"错误: Boost 模式只能用于 'aster' 或 'backpack' 交易所. "
                  f"交易对 {ticker} 的配置无效")
            sys.exit(1)
    
    # Create trading configurations
    trading_configs = []
    for ticker_config in config_data['tickers']:
        try:
            config = create_trading_config(ticker_config, common_params, exchange)
            trading_configs.append(config)
        except Exception as e:
            print(f"创建配置失败: {e}")
            sys.exit(1)
    
    # Print summary
    print("=" * 60)
    print(f"多交易对交易机器人")
    print(f"交易所: {exchange}")
    print(f"交易对数量: {len(trading_configs)}")
    print("=" * 60)
    for i, config in enumerate(trading_configs, 1):
        print(f"{i}. {config.ticker}: {config.direction} | "
              f"数量={config.quantity} | 止盈={config.take_profit} | "
              f"实例ID={config.instance_id}")
    print("=" * 60)
    print()
    
    # Create tasks for all bots
    tasks = []
    for i, config in enumerate(trading_configs, 1):
        task = asyncio.create_task(
            run_single_bot(config, i, len(trading_configs))
        )
        tasks.append(task)
        # Add a small delay between starting bots to avoid race conditions
        await asyncio.sleep(2)
    
    # Wait for all bots to complete (or run indefinitely)
    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        print("\n收到中断信号，正在停止所有交易对...")
        # Cancel all tasks
        for task in tasks:
            task.cancel()
        # Wait for all tasks to be cancelled
        await asyncio.gather(*tasks, return_exceptions=True)
        print("所有交易对已停止")
    except Exception as e:
        print(f"执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序已终止")
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            import asyncio
            loop = asyncio.get_event_loop()
            loop.run_until_complete(main())
        else:
            raise
