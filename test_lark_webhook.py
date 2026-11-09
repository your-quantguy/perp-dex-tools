#!/usr/bin/env python3
"""
测试飞书 Webhook 通知

用法:
    python test_lark_webhook.py
"""

import asyncio
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from helpers.lark_webhook import get_lark_webhook_bot, CardColor
import dotenv

async def main():
    """测试飞书 webhook 通知"""
    # 加载环境变量
    dotenv.load_dotenv()
    
    # 获取机器人实例
    bot = get_lark_webhook_bot()
    
    if not bot:
        print("❌ 未配置 LARK_WEBHOOK_URL，无法发送测试消息")
        print("请在 .env 文件中配置:")
        print("  LARK_WEBHOOK_URL=https://open.feishu.cn/open-apis/bot/v2/hook/xxx")
        print("  LARK_WEBHOOK_KEYWORD=你的关键词  # 可选")
        return
    
    print("✅ 飞书 Webhook 配置已加载")
    print("=" * 60)
    
    # 测试1: 纯文本消息
    print("\n📝 测试1: 发送纯文本消息...")
    success = await bot.send_text("🧪 这是一条测试消息！")
    if success:
        print("✅ 纯文本消息发送成功")
    else:
        print("❌ 纯文本消息发送失败")
    
    await asyncio.sleep(2)
    
    # 测试2: 交易信号通知
    print("\n📈 测试2: 发送交易信号通知...")
    success = await bot.send_trade_signal(
        signal='LONG',
        ticker='BTC',
        price=50000.0,
        quantity=0.01,
        strategy='Direction=buy | Qty=0.01 | TP=0.02% | MaxOrders=10',
        instance_id='test_instance',
        exchange='backpack',
        timeframe='4h',
        extra_info={
            '测试参数': '这是一个测试',
            '当前时间': '2026-02-11'
        }
    )
    if success:
        print("✅ 交易信号通知发送成功")
    else:
        print("❌ 交易信号通知发送失败")
    
    await asyncio.sleep(2)
    
    # 测试3: 错误告警
    print("\n⚠️ 测试3: 发送错误告警...")
    success = await bot.send_error_alert(
        error_type='测试错误',
        error_message='这是一个测试错误消息',
        instance='test_instance',
        exchange='backpack',
        ticker='BTC',
        strategy='Direction=buy | Qty=0.01 | TP=0.02%',
        traceback_preview='File "test.py", line 1\n  print("test")\nSyntaxError: invalid syntax',
        error_count=1
    )
    if success:
        print("✅ 错误告警发送成功")
    else:
        print("❌ 错误告警发送失败")
    
    await asyncio.sleep(2)
    
    # 测试4: 系统状态通知
    print("\n📊 测试4: 发送系统状态通知...")
    success = await bot.send_system_status(
        status_type='startup',
        message='交易机器人已启动',
        metrics={
            '数量': '0.01',
            '止盈': '0.02%',
            '方向': 'BUY',
            '最大订单': '10',
            '等待时间': '450秒',
            '网格步长': '0.2%'
        },
        instance_id='test_instance',
        exchange='backpack',
        ticker='BTC',
        strategy='Direction=buy | Qty=0.01 | TP=0.02% | MaxOrders=10',
        color=CardColor.GREEN
    )
    if success:
        print("✅ 系统状态通知发送成功")
    else:
        print("❌ 系统状态通知发送失败")
    
    await asyncio.sleep(2)
    
    # 测试5: 每小时统计
    print("\n📈 测试5: 发送每小时统计...")
    success = await bot.send_system_status(
        status_type='resource',
        message='每小时交易统计',
        metrics={
            '仓位操作数': '15',
            '平仓操作数': '8',
            '成功成交数': '20',
            '取消订单数': '3',
            '预估手续费': '0.0023',
            '当前仓位': '0.05',
            '活跃平仓': '0.05',
            '统计周期': '14:00 - 15:00'
        },
        instance_id='test_instance',
        exchange='backpack',
        ticker='BTC',
        strategy='Direction=buy | Qty=0.01 | TP=0.02%',
        color=CardColor.BLUE
    )
    if success:
        print("✅ 每小时统计发送成功")
    else:
        print("❌ 每小时统计发送失败")
    
    print("\n" + "=" * 60)
    print("✅ 测试完成！请检查飞书群查看通知效果")
    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("飞书 Webhook 通知测试")
    print("=" * 60)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ 测试已中断")
    except Exception as e:
        print(f"\n\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
