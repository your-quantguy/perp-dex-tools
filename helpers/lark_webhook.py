"""
飞书自定义机器人通知（Webhook方式）

使用飞书群自定义机器人发送交互式卡片消息。
官方文档: https://open.feishu.cn/document/client-docs/bot-v3/add-custom-bot

特点:
1. 直接发送到群机器人
2. 支持交互式卡片（颜色、图标、模块化）
3. 代码中直接控制样式
4. 无需后台配置和解析

使用场景:
- 交易信号通知（买入/卖出/平仓）
- 错误告警通知
- 系统状态通知
- 每日交易摘要

配置:
在.env文件中配置webhook URL:
    LARK_WEBHOOK_URL=https://open.feishu.cn/open-apis/bot/v2/hook/xxx
"""

import aiohttp
import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum


class MessageType(Enum):
    """消息类型枚举"""
    TEXT = "text"           # 纯文本
    POST = "post"           # 富文本
    INTERACTIVE = "interactive"  # 交互式卡片（推荐）
    IMAGE = "image"         # 图片


class CardColor(Enum):
    """卡片颜色主题"""
    BLUE = "blue"       # 蓝色 - 信息/状态
    GREEN = "green"     # 绿色 - 成功/买入
    RED = "red"         # 红色 - 错误/卖出
    ORANGE = "orange"   # 橙色 - 警告
    GREY = "grey"       # 灰色 - 中性/平仓


class LarkWebhookBot:
    """
    飞书Webhook机器人

    Example:
        >>> bot = LarkWebhookBot(webhook_url='https://...')
        >>>
        >>> # 发送交易信号
        >>> await bot.send_trade_signal(
        ...     signal='LONG',
        ...     ticker='BTC',
        ...     price=50000.0,
        ...     quantity=0.001,
        ...     strategy='KAMA'
        ... )
        >>>
        >>> # 发送错误告警
        >>> await bot.send_error_alert(
        ...     error_type='API超时',
        ...     error_message='连接backpack超时',
        ...     instance='account_a_backpack_BTC'
        ... )
    """

    def __init__(self, webhook_url: str, enable_at_all: bool = False, keyword: Optional[str] = None):
        """
        初始化飞书Webhook机器人

        Args:
            webhook_url: 飞书群自定义机器人的Webhook URL
            enable_at_all: 是否@所有人（仅用于紧急告警）
            keyword: 飞书机器人关键词（如果配置了关键词验证，需提供）
        """
        self.webhook_url = webhook_url
        self.enable_at_all = enable_at_all
        self.keyword = keyword  # 🆕 关键词配置

        # 日志
        self.logger = logging.getLogger("LarkWebhookBot")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    async def _send_message(self, message: Dict[str, Any], max_retries: int = 2) -> bool:
        """
        发送消息到飞书群（统一接口，带重试机制）

        Args:
            message: 消息体（符合飞书API格式）
            max_retries: 最大重试次数（默认2次）

        Returns:
            是否发送成功
        """
        # 🆕 自动添加关键词（如果配置了）
        if self.keyword:
            if message.get("msg_type") == "interactive":
                # 交互式卡片：在第一个元素前添加关键词
                if "card" in message and "elements" in message["card"]:
                    keyword_element = self._build_text_module(
                        f"🔑 {self.keyword}",
                        is_markdown=False
                    )
                    # 插入到第一个位置（在header之后）
                    message["card"]["elements"].insert(0, keyword_element)
            elif message.get("msg_type") == "text":
                # 纯文本：在开头添加关键词
                content = message.get("content", {})
                original_text = content.get("text", "")
                content["text"] = f"🔑 {self.keyword}\n{original_text}"

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.webhook_url,
                        json=message,
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        result = await response.json()

                        if result.get('code') == 0:
                            if attempt > 0:
                                self.logger.debug(f"✓ 飞书消息发送成功 (重试 {attempt} 次后)")
                            else:
                                self.logger.debug(f"✓ 飞书消息发送成功")
                            return True
                        else:
                            error_msg = result.get('msg', 'Unknown error')
                            self.logger.error(
                                f"✗ 飞书消息发送失败 (尝试 {attempt+1}/{max_retries+1}): {error_msg}"
                            )
                            last_error = error_msg
                            
                            # 如果不是最后一次尝试，等待后重试
                            if attempt < max_retries:
                                await asyncio.sleep(1 * (attempt + 1))  # 递增延迟
                                continue
                            return False

            except asyncio.TimeoutError:
                self.logger.error(f"✗ 飞书消息发送超时 (尝试 {attempt+1}/{max_retries+1})")
                last_error = "发送超时"
                if attempt < max_retries:
                    await asyncio.sleep(1 * (attempt + 1))
                    continue
                return False
            except Exception as e:
                self.logger.error(f"✗ 飞书消息发送异常 (尝试 {attempt+1}/{max_retries+1}): {e}")
                last_error = str(e)
                if attempt < max_retries:
                    await asyncio.sleep(1 * (attempt + 1))
                    continue
                return False
        
        # 所有重试都失败
        self.logger.error(f"✗ 飞书消息发送失败，已重试 {max_retries} 次，最后错误: {last_error}")
        return False

    def _build_card_header(self, title: str, subtitle: Optional[str] = None,
                          color: CardColor = CardColor.BLUE) -> Dict:
        """构建卡片头部"""
        header = {
            "template": color.value,
            "title": {
                "tag": "plain_text",
                "content": title
            }
        }

        if subtitle:
            header["subtitle"] = {
                "tag": "plain_text",
                "content": subtitle
            }

        return header

    def _build_field_module(self, fields: List[Dict[str, str]]) -> Dict:
        """
        构建字段模块（两列展示）

        Args:
            fields: 字段列表，每项包含 {'name': '字段名', 'value': '字段值'}
        """
        return {
            "tag": "div",
            "fields": [
                {
                    "is_short": True,
                    "text": {
                        "tag": "lark_md",
                        "content": f"**{field['name']}**\n{field['value']}"
                    }
                }
                for field in fields
            ]
        }

    def _build_text_module(self, text: str, is_markdown: bool = True) -> Dict:
        """构建文本模块"""
        return {
            "tag": "div",
            "text": {
                "tag": "lark_md" if is_markdown else "plain_text",
                "content": text
            }
        }

    def _build_divider(self) -> Dict:
        """构建分割线"""
        return {"tag": "hr"}

    def _build_note_module(self, text: str) -> Dict:
        """构建备注模块（小字灰色）"""
        return {
            "tag": "note",
            "elements": [
                {
                    "tag": "plain_text",
                    "content": text
                }
            ]
        }

    def _build_base_info_module(self, instance_id: str, exchange: str, ticker: str, strategy: str) -> Dict:
        """
        🆕 构建基础信息模块（统一显示在所有卡片中）

        Args:
            instance_id: 实例ID
            exchange: 交易所
            ticker: 交易对
            strategy: 策略名称（包含完整参数）

        Returns:
            字段模块
        """
        return self._build_field_module([
            {"name": "实例ID", "value": instance_id},
            {"name": "交易所", "value": exchange.upper()},
            {"name": "交易对", "value": ticker},
            {"name": "策略名称", "value": strategy},
        ])

    async def send_trade_signal(self,
                                signal: str,
                                ticker: str,
                                price: float,
                                quantity: float,
                                strategy: str,
                                instance_id: str,
                                exchange: str = 'backpack',
                                timeframe: str = '4h',
                                extra_info: Optional[Dict] = None) -> bool:
        """
        发送交易信号通知

        Args:
            signal: 信号类型 ('LONG', 'SHORT', 'CLOSE')
            ticker: 交易对 (如 'BTC')
            price: 价格
            quantity: 数量
            strategy: 策略名称（完整参数，如 KAMAStrategy(period=35, fast=2, slow=30)）
            instance_id: 实例ID
            exchange: 交易所
            timeframe: 时间周期
            extra_info: 额外信息（可选）

        Returns:
            是否发送成功
        """
        # 根据信号类型选择颜色和图标
        if signal == 'LONG':
            color = CardColor.GREEN
            icon = "📈"
            signal_text = "做多信号"
        elif signal == 'SHORT':
            color = CardColor.RED
            icon = "📉"
            signal_text = "做空信号"
        elif signal == 'CLOSE':
            color = CardColor.GREY
            icon = "🔄"
            signal_text = "平仓信号"
        else:
            color = CardColor.BLUE
            icon = "📊"
            signal_text = signal

        # 构建卡片
        card = {
            "msg_type": "interactive",
            "card": {
                "header": self._build_card_header(
                    title=f"{icon} {signal_text}",
                    subtitle=f"{ticker} {timeframe}",
                    color=color
                ),
                "elements": [
                    # 🆕 基础信息（统一模块）
                    self._build_base_info_module(
                        instance_id=instance_id,
                        exchange=exchange,
                        ticker=ticker,
                        strategy=strategy
                    ),

                    self._build_divider(),

                    # 交易信息
                    self._build_field_module([
                        {"name": "价格", "value": f"${price:,.2f}"},
                        {"name": "数量", "value": f"{quantity}"},
                        {"name": "周期", "value": timeframe},
                        {"name": "时间", "value": datetime.now().strftime('%H:%M:%S')},
                    ]),
                ]
            }
        }

        # 添加额外信息
        if extra_info:
            card["card"]["elements"].append(self._build_divider())
            extra_fields = [
                {"name": k, "value": str(v)}
                for k, v in extra_info.items()
            ]
            card["card"]["elements"].append(self._build_field_module(extra_fields))

        # 添加备注
        card["card"]["elements"].append(
            self._build_note_module(
                f"📍 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                f"自动交易系统"
            )
        )

        # @所有人（如果启用）
        if self.enable_at_all:
            card["card"]["elements"].append(
                self._build_text_module("<at id=all></at>")
            )

        return await self._send_message(card)

    async def send_error_alert(self,
                               error_type: str,
                               error_message: str,
                               instance: str,
                               exchange: Optional[str] = None,
                               ticker: Optional[str] = None,
                               strategy: Optional[str] = None,
                               traceback_preview: Optional[str] = None,
                               error_count: int = 1) -> bool:
        """
        发送错误告警通知

        Args:
            error_type: 错误类型
            error_message: 错误消息
            instance: 实例名称
            exchange: 交易所（可选）
            ticker: 交易对（可选）
            strategy: 策略名称（可选，完整参数）
            traceback_preview: 堆栈预览（可选，截取前200字符）
            error_count: 错误次数

        Returns:
            是否发送成功
        """
        # 构建卡片
        card = {
            "msg_type": "interactive",
            "card": {
                "header": self._build_card_header(
                    title="⚠️ 系统异常告警",
                    subtitle=f"错误次数: {error_count}",
                    color=CardColor.RED
                ),
                "elements": []
            }
        }

        # 🆕 如果有完整的基础信息，显示统一模块
        if exchange and ticker and strategy:
            card["card"]["elements"].append(
                self._build_base_info_module(
                    instance_id=instance,
                    exchange=exchange,
                    ticker=ticker,
                    strategy=strategy
                )
            )
            card["card"]["elements"].append(self._build_divider())

        # 错误信息
        card["card"]["elements"].append(
            self._build_text_module(
                f"**错误类型**: {error_type}\n"
                f"**错误消息**: {error_message}"
            )
        )

        # 如果没有完整基础信息，显示简化信息
        if not (exchange and ticker and strategy):
            card["card"]["elements"].append(self._build_divider())
            error_context = [{"name": "实例", "value": instance}]
            if exchange:
                error_context.append({"name": "交易所", "value": exchange})
            if ticker:
                error_context.append({"name": "交易对", "value": ticker})
            error_context.append({"name": "错误次数", "value": str(error_count)})
            card["card"]["elements"].append(self._build_field_module(error_context))

        # 添加堆栈预览
        if traceback_preview:
            card["card"]["elements"].extend([
                self._build_divider(),
                self._build_text_module(
                    f"**堆栈预览**:\n```\n{traceback_preview[:200]}\n```"
                )
            ])

        # 添加备注
        card["card"]["elements"].append(
            self._build_note_module(
                f"🚨 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                f"请及时处理"
            )
        )

        # 错误告警@所有人
        card["card"]["elements"].append(
            self._build_text_module("<at id=all></at>")
        )

        return await self._send_message(card)

    async def send_system_status(self,
                                 status_type: str,
                                 message: str,
                                 metrics: Dict[str, Any],
                                 instance_id: Optional[str] = None,
                                 exchange: Optional[str] = None,
                                 ticker: Optional[str] = None,
                                 strategy: Optional[str] = None,
                                 color: CardColor = CardColor.BLUE) -> bool:
        """
        发送系统状态通知

        Args:
            status_type: 状态类型（健康检查/资源监控/启动/停止）
            message: 状态消息
            metrics: 指标数据（移除"更新间隔"等不必要字段）
            instance_id: 实例ID（可选）
            exchange: 交易所（可选）
            ticker: 交易对（可选）
            strategy: 策略名称（可选，完整参数）
            color: 卡片颜色

        Returns:
            是否发送成功
        """
        # 图标映射
        icons = {
            'health_check': '💊',
            'resource': '📊',
            'startup': '🚀',
            'shutdown': '🛑'
        }
        icon = icons.get(status_type, '📌')

        # 构建卡片
        card = {
            "msg_type": "interactive",
            "card": {
                "header": self._build_card_header(
                    title=f"{icon} 系统状态",
                    subtitle=message,
                    color=color
                ),
                "elements": []
            }
        }

        # 🆕 如果有基础信息，显示统一模块
        if instance_id and exchange and ticker and strategy:
            card["card"]["elements"].append(
                self._build_base_info_module(
                    instance_id=instance_id,
                    exchange=exchange,
                    ticker=ticker,
                    strategy=strategy
                )
            )
            card["card"]["elements"].append(self._build_divider())

        # 指标数据（过滤掉不需要的字段）
        filtered_metrics = {
            k: v for k, v in metrics.items()
            if k not in ['更新间隔', 'update_interval', '检查周期']  # 🔧 过滤不需要的字段
        }

        if filtered_metrics:
            card["card"]["elements"].append(
                self._build_field_module([
                    {"name": k, "value": str(v)}
                    for k, v in filtered_metrics.items()
                ])
            )

        # 备注
        card["card"]["elements"].append(
            self._build_note_module(
                f"📍 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
        )

        return await self._send_message(card)

    async def send_daily_summary(self,
                                 date: str,
                                 instances: List[Dict[str, Any]],
                                 total_signals: int,
                                 total_trades: int,
                                 total_errors: int) -> bool:
        """
        发送每日交易摘要

        Args:
            date: 日期（YYYY-MM-DD）
            instances: 实例统计列表
            total_signals: 总信号数
            total_trades: 总交易数
            total_errors: 总错误数

        Returns:
            是否发送成功
        """
        # 构建卡片
        card = {
            "msg_type": "interactive",
            "card": {
                "header": self._build_card_header(
                    title="📅 每日交易摘要",
                    subtitle=date,
                    color=CardColor.GREEN
                ),
                "elements": [
                    # 总体统计
                    self._build_text_module(
                        f"**总体情况**"
                    ),
                    self._build_field_module([
                        {"name": "信号总数", "value": str(total_signals)},
                        {"name": "交易总数", "value": str(total_trades)},
                        {"name": "错误总数", "value": str(total_errors)},
                        {"name": "活跃实例", "value": str(len(instances))},
                    ]),

                    self._build_divider(),

                    # 实例详情
                    self._build_text_module("**实例详情**"),
                ]
            }
        }

        # 添加每个实例的统计
        for inst in instances[:5]:  # 最多显示5个实例
            card["card"]["elements"].append(
                self._build_field_module([
                    {"name": "实例", "value": inst['name']},
                    {"name": "信号", "value": str(inst.get('signals', 0))},
                    {"name": "交易", "value": str(inst.get('trades', 0))},
                    {"name": "状态", "value": inst.get('status', 'unknown')},
                ])
            )

        # 备注
        card["card"]["elements"].append(
            self._build_note_module(
                f"📊 数据时间: {date} | 自动生成"
            )
        )

        return await self._send_message(card)

    async def send_text(self, text: str) -> bool:
        """
        发送纯文本消息（简单场景）

        Args:
            text: 文本内容

        Returns:
            是否发送成功
        """
        message = {
            "msg_type": "text",
            "content": {
                "text": text
            }
        }

        return await self._send_message(message)


def get_lark_webhook_bot(webhook_url: Optional[str] = None,
                         enable_at_all: bool = False,
                         keyword: Optional[str] = None) -> Optional[LarkWebhookBot]:
    """
    工厂函数: 创建飞书Webhook机器人

    Args:
        webhook_url: Webhook URL（如果未提供，从环境变量读取）
        enable_at_all: 是否启用@所有人
        keyword: 飞书关键词（如果未提供，从环境变量LARK_WEBHOOK_KEYWORD读取）

    Returns:
        LarkWebhookBot实例，如果未配置则返回None

    Example:
        >>> import os
        >>> bot = get_lark_webhook_bot()
        >>> if bot:
        ...     await bot.send_trade_signal(...)
    """
    import os

    if not webhook_url:
        webhook_url = os.getenv('LARK_WEBHOOK_URL')

    if not webhook_url:
        logging.warning("[LarkWebhook] 未配置LARK_WEBHOOK_URL，飞书通知已禁用")
        return None

    # 🆕 从环境变量读取关键词
    if not keyword:
        keyword = os.getenv('LARK_WEBHOOK_KEYWORD')

    return LarkWebhookBot(webhook_url, enable_at_all, keyword)


__all__ = [
    'LarkWebhookBot',
    'MessageType',
    'CardColor',
    'get_lark_webhook_bot'
]
