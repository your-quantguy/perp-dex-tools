"""
在 Lighter 与 Paradex 交易所之间执行对冲策略。
策略会在 Lighter 上挂出限价单并监控直到成交，
随后在 Paradex 上下反向订单进行对冲。
限价单价格使用订单簿的 BBO（最优买卖）价格。
"""

import asyncio
import os
import time
from decimal import Decimal
from typing import Dict, Any, Optional, Tuple

from exchanges.lighter import LighterClient
from exchanges.paradex import ParadexClient
from helpers.logger import TradingLogger


class HedgeStrategy:
    """
    在 Lighter 与 Paradex 之间的对冲策略。
    在 Lighter 上挂限价单并监控直到成交，
    然后在 Paradex 上下反向订单进行对冲。
    限价单价格基于订单簿的 BBO（最优买卖）价格。
    """

    def __init__(self, config: Dict[str, Any]):
        """使用配置初始化对冲策略。"""
        self.config = config
        self.logger = TradingLogger(
            exchange="hedge_strategy",
            ticker=self.config.get("ticker", "ETH-PERP"),
            log_to_console=True
        )
        
        # Initialize exchange clients
        self.lighter_client = LighterClient(config.get("lighter", {}))
        self.paradex_client = ParadexClient(config.get("paradex", {}))
        
        # Strategy parameters
        self.ticker = config.get("ticker", "ETH-PERP")
        self.quantity = Decimal(config.get("quantity", "0.1"))
        self.side = config.get("side", "buy")  # Initial side on Lighter
        
        # Order parameters
        self.price_offset_ticks = config.get("price_offset_ticks", 0)  # Ticks away from BBO
        self.order_timeout_seconds = config.get("order_timeout_seconds", 5)  # Time before canceling order (changed to 5 seconds)
        self.max_retries = config.get("max_retries", 10)  # Maximum number of order placement retries (increased for more attempts)
        
        # 跟踪变量
        self.lighter_order_id = None
        self.paradex_order_id = None
        self.is_running = False
        self.lighter_order_filled = False
        self.order_placement_time = None
        self.current_retry_count = 0

        # 对冲周期配置与状态
        self.hedge_cycle_seconds = int(self.config.get("hedge_cycle_seconds", 3600))
        self.cycle_start_time: Optional[float] = None
        self.cycle_count = 0
        self.cycle_history = []
        
        # 自动取消并替换配置
        self.auto_cancel_enabled = config.get('auto_cancel_enabled', True)
        self.price_check_interval = config.get('price_check_interval', 5)  # 秒
        self.price_tolerance = config.get('price_tolerance', 0.001)  # 0.1% 容忍度

        # 风险控制配置
        self.risk_enabled = config.get('risk_enabled', True)
        # 当当前价格距离强平价在该百分比内时触发风险退出
        self.risk_threshold_pct = Decimal(str(config.get('risk_threshold_pct', 0.10)))
        self.risk_check_interval = config.get('risk_check_interval', 2)  # 秒

    async def initialize(self):
        """初始化与交易所的连接。"""
        self.logger.log("正在初始化对冲策略", "INFO")
        
        try:
            # Ensure contract attributes (contract_id, tick_size) are resolved first
            self.logger.log("正在解析合约属性", "INFO")
            await self.lighter_client.get_contract_attributes()
            await self.paradex_client.get_contract_attributes()
            
            # 连接到各交易所
            
            await self.lighter_client.connect()
            await self.paradex_client.connect()
            self.logger.log("已成功连接到交易所", "INFO")
            return True
        except Exception as e:
            self.logger.log(f"初始化策略失败: {e}", "ERROR")
            return False

    async def shutdown(self):
        """关闭与交易所的连接。"""
        self.logger.log("正在关闭对冲策略", "INFO")
        self.is_running = False
        
        # 断开交易所连接（分别处理错误）
        try:
            await self.lighter_client.disconnect()
        except Exception as e:
            self.logger.log(f"断开 Lighter 连接出错: {e}", "ERROR")
            
        try:
            await self.paradex_client.disconnect()
        except Exception as e:
            self.logger.log(f"断开 Paradex 连接出错: {e}", "ERROR")
            
        self.logger.log("对冲策略关闭完成", "INFO")

    async def get_bbo_price(self) -> Tuple[Decimal, Decimal]:
        """使用 Lighter 客户端获取最优买价与卖价（BBO）。"""
        try:
            contract_id = self.lighter_client.config.contract_id
            best_bid, best_ask = await self.lighter_client.fetch_bbo_prices(contract_id)
            self.logger.log(f"BBO 价格 - 买价: {best_bid}, 卖价: {best_ask}", "INFO")
            return best_bid, best_ask
        except Exception as e:
            self.logger.log(f"获取 BBO 价格出错: {e}", "ERROR")
            raise
    
    async def calculate_limit_price(self) -> Decimal:
        """基于 BBO 与订单方向计算限价价格，靠近盘口且不跨价：
        - 买单：在买价之上偏移 1-2 tick，但不超过卖价-1tick
        - 卖单：在卖价之下偏移 1-2 tick，但不低于买价+1tick
        """
        best_bid, best_ask = await self.get_bbo_price()
        tick_size = self.lighter_client.config.tick_size
        ticks = max(0, int(self.price_offset_ticks or 0))

        if self.side == "buy":
            target = best_bid + (tick_size * ticks)
            # 不跨价：最多到卖价-1tick；若价差过小，则使用买价
            upper_bound = best_ask - tick_size
            if target >= upper_bound:
                price = max(best_bid, upper_bound)
            else:
                price = target
        else:
            target = best_ask - (tick_size * ticks)
            # 不跨价：至少到买价+1tick；若价差过小，则使用卖价
            lower_bound = best_bid + tick_size
            if target <= lower_bound:
                price = min(best_ask, lower_bound)
            else:
                price = target

        # 按 tick 对齐
        price = self.lighter_client.round_to_tick(price)
        self.logger.log(f"{self.side} 的限价计算结果: {price}（BBO: bid={best_bid}, ask={best_ask}, ticks={ticks}）", "INFO")
        return price

    def _maker_safe_price_for_side(self, side: str, best_bid: Decimal, best_ask: Decimal, offset_ticks: Optional[int] = None) -> Decimal:
        """根据 BBO 为给定方向返回对做市友好的限价（盘口内靠近对手盘，但不跨价）。
        - `buy`：选 `best_bid + n*tick`，并限制不超过 `best_ask - 1*tick`
        - `sell`：选 `best_ask - n*tick`，并限制不低于 `best_bid + 1*tick`
        """
        ticks = int(offset_ticks) if offset_ticks is not None else int(self.price_offset_ticks or 0)
        ticks = max(0, ticks)
        tick_size = self.lighter_client.config.tick_size
        if side.lower() == 'buy':
            target = best_bid + (tick_size * ticks)
            upper_bound = best_ask - tick_size
            price = target if target < upper_bound else max(best_bid, upper_bound)
        else:
            target = best_ask - (tick_size * ticks)
            lower_bound = best_bid + tick_size
            price = target if target > lower_bound else min(best_ask, lower_bound)
        return self.lighter_client.round_to_tick(price)
    
    def _aggressive_cross_price_for_side(self, side: str, best_bid: Decimal, best_ask: Decimal) -> Decimal:
        """返回激进跨价的限价以实现市价效果：
        - `sell` 使用买一价保证吃单
        - `buy` 使用卖一价保证吃单
        """
        price = best_bid if side.lower() == 'sell' else best_ask
        try:
            return self.lighter_client.round_to_tick(price)
        except Exception:
            return price
    
    async def should_cancel_and_replace_order(self):
        """根据价格变化检查当前订单是否需要取消并重新下单。"""
        if not self.auto_cancel_enabled or not self.lighter_order_id:
            return False
            
        try:
            # 获取当前 BBO 价格
            bid_price, ask_price = await self.get_bbo_price()
            
            # 根据订单方向选择合适的 BBO 价格
            if self.side == "buy":
                current_bbo_price = bid_price
            else:
                current_bbo_price = ask_price
                
            if not current_bbo_price:
                self.logger.log("获取当前 BBO 价格失败，无法进行订单替换检查", "WARNING")
                return False
            
            # 获取当前订单详情
            orders = await self.lighter_client.get_active_orders(self.lighter_client.config.contract_id)
            current_order = None
            for order in orders:
                if order.order_id == self.lighter_order_id:
                    current_order = order
                    break
            
            if not current_order:
                self.logger.log(f"在活跃订单中未找到当前订单 {self.lighter_order_id}", "DEBUG")
                return False
            
            # 计算价格差异
            order_price = current_order.price
            price_diff = abs(current_bbo_price - order_price) / order_price
            
            self.logger.log(f"价格检查: 订单价={order_price}, 当前 BBO={current_bbo_price}, "
                          f"差值={price_diff:.4f}, 容忍度={self.price_tolerance}", "DEBUG")
            
            # 检查价格是否超出容忍度
            if price_diff > self.price_tolerance:
                self.logger.log(f"价格移动超过容忍度 ({price_diff:.4f} > {self.price_tolerance})，将取消并重新下单。", "INFO")
                return True
                
            return False
            
        except Exception as e:
            self.logger.log(f"检查是否需要替换订单时出错: {e}", "ERROR")
            return False

    async def cancel_and_replace_lighter_order(self):
        """取消当前 Lighter 订单，并根据最新 BBO 价格重新下单。"""
        if not self.lighter_order_id:
            self.logger.log("没有可取消并替换的活跃订单", "WARNING")
            return False
            
        try:
            # 取消当前订单
            self.logger.log(f"正在取消订单 {self.lighter_order_id} 以进行替换", "INFO")
            cancel_result = await self.cancel_lighter_order()
            
            if not cancel_result:
                self.logger.log("取消订单失败，无法进行替换", "ERROR")
                return False
            
            # 等待片刻以便取消生效
            await asyncio.sleep(0.5)
            
            # 使用当前 BBO 价格重新下单
            self.logger.log("正在按最新 BBO 价格下替换订单", "INFO")
            placement_result = await self.place_lighter_limit_order()
            
            if placement_result:
                self.logger.log("替换订单下单成功", "INFO")
                return True
            else:
                self.logger.log("替换订单下单失败", "ERROR")
                return False
                
        except Exception as e:
            self.logger.log(f"取消并替换过程中发生错误: {e}", "ERROR")
            return False

    async def cancel_lighter_order(self):
        """如存在则取消当前 Lighter 订单。"""
        if not self.lighter_order_id:
            return True
            
        try:
            self.logger.log(f"正在取消 Lighter 订单 {self.lighter_order_id}", "INFO")
            result = await self.lighter_client.cancel_order(self.lighter_order_id)
            
            if result:
                self.logger.log(f"成功取消订单 {self.lighter_order_id}", "INFO")
                self.lighter_order_id = None
                return True
            else:
                self.logger.log(f"取消订单 {self.lighter_order_id} 失败", "WARNING")
                return False
        except Exception as e:
            self.logger.log(f"取消订单时发生错误: {e}", "ERROR")
            return False
    
    async def place_lighter_limit_order(self):
        """使用 BBO 价格在 Lighter 交易所下限价单。"""
        try:
            # 基于 BBO 计算限价
            price = await self.calculate_limit_price()
            
            self.logger.log(f"在 Lighter 上下 {self.side} 限价单，数量 {self.quantity} {self.ticker}，价格 {price}", "INFO")
            
            # 在 Lighter 上通过 place_close_order 提交限价单
            direction = "long" if self.side == "buy" else "short"
            order_result = await self.lighter_client.place_close_order(
                contract_id=self.lighter_client.config.contract_id,
                quantity=self.quantity,
                price=price,
                side=self.side
            )
            
            if order_result.success:
                self.lighter_order_id = order_result.order_id
                self.order_placement_time = time.time()
                # 新订单重置成交状态
                self.lighter_order_filled = False
                # 在本周期首次成功下单时记录周期开始时间
                if not self.cycle_start_time:
                    self.cycle_start_time = self.order_placement_time
                    self.logger.log(f"周期开始时间: {self.cycle_start_time}", "DEBUG")
                self.logger.log(f"Lighter 订单下单成功: {self.lighter_order_id}（尝试次数 {self.current_retry_count + 1}）", "INFO")
                return True
            else:
                self.logger.log(f"Lighter 下单失败: {order_result.error_message}", "ERROR")
                return False
        except Exception as e:
            self.logger.log(f"在 Lighter 下单时发生错误: {e}", "ERROR")
            return False

    async def monitor_lighter_order(self):
        """监控 Lighter 订单直到成交或超时。"""
        self.logger.log(f"正在监控 Lighter 订单 {self.lighter_order_id}", "INFO")
        
        # 若可用则设置 WebSocket 的订单更新回调
        try:
            self.lighter_client.setup_order_update_handler(self.on_lighter_order_update)
            self.logger.log("已成功设置 WebSocket 订单更新回调", "INFO")
        except Exception as e:
            self.logger.log(f"设置 WebSocket 更新失败，回退为轮询: {e}", "WARNING")
        
        # 轮询循环
        poll_count = 0
        last_price_check_time = time.time()
        last_risk_check_time = time.time()
        
        while self.is_running and not self.lighter_order_filled:
            try:
                poll_count += 1
                self.logger.log(f"正在轮询 Lighter 订单状态（第 {poll_count} 次）", "DEBUG")

                current_time = time.time()

                # 风险控制：监控逼近强平，如触发则退出
                if self.risk_enabled and (poll_count == 1 or (current_time - last_risk_check_time >= self.risk_check_interval)):
                    risk_triggered = await self.monitor_liquidation_risk()
                    if risk_triggered:
                        self.logger.log("风险阈值触发。尝试在两个交易所同时平仓并停止策略。", "ERROR")
                        try:
                            await self.cancel_lighter_order()
                        except Exception as e:
                            self.logger.log(f"风险退出过程中取消 Lighter 订单出错: {e}", "ERROR")
                        await self.shutdown()
                        return False
                    last_risk_check_time = time.time()

                # 检查是否发生订单超时
                if self.order_placement_time and (current_time - self.order_placement_time > self.order_timeout_seconds):
                    self.logger.log(f"订单已达到超时时间（{self.order_timeout_seconds} 秒）。将取消并替换订单。", "INFO")
                    await self.cancel_lighter_order()
                    
                    # 检查是否需要重试
                    if self.current_retry_count < self.max_retries:
                        self.current_retry_count += 1
                        self.logger.log(f"因超时自动替换订单（第 {self.current_retry_count}/{self.max_retries} 次）", "INFO")
                        if await self.place_lighter_limit_order():
                            # 为新订单重置超时设置
                            last_price_check_time = time.time()  # 重置价格检查计时器
                            continue
                        else:
                            self.logger.log("替换订单下单失败", "ERROR")
                            return False
                    else:
                        self.logger.log(f"已达到最大重试次数（{self.max_retries}）。停止重试。", "WARNING")
                        return False
                
                # 检查是否因价格变动取消并替换订单
                if (current_time - last_price_check_time >= self.price_check_interval):
                    self.logger.log("检查是否因价格变动需要取消并替换订单", "DEBUG")
                    if await self.should_cancel_and_replace_order():
                        replace_result = await self.cancel_and_replace_lighter_order()
                        if replace_result:
                            # 为新订单重置计时器
                            last_price_check_time = time.time()
                            self.order_placement_time = time.time()
                            continue
                        else:
                            self.logger.log("替换订单失败，继续使用当前订单", "WARNING")
                    last_price_check_time = time.time()
                
                # 优先检查 WebSocket 的 current_order（如可用）
                if hasattr(self.lighter_client, 'current_order') and self.lighter_client.current_order:
                    current_order = self.lighter_client.current_order
                    if current_order.order_id == self.lighter_order_id:
                        status_upper = str(current_order.status).upper()
                        self.logger.log(f"来自 WebSocket 的当前订单状态: {status_upper}, 成交: {current_order.filled_size}/{current_order.size}", "DEBUG")
                        
                        if status_upper == "FILLED":
                            self.lighter_order_filled = True
                            self.logger.log(f"Lighter 订单 {self.lighter_order_id} 已成交（WebSocket）", "INFO")
                            return True
                        elif status_upper in ("CANCELED", "EXPIRED"):
                            self.logger.log(f"Lighter 订单 {self.lighter_order_id} {status_upper}（WebSocket）", "WARNING")
                            return False
                        elif current_order.filled_size > 0:
                            self.logger.log(f"订单 {self.lighter_order_id} 部分成交: {current_order.filled_size}/{current_order.size}（WebSocket）", "INFO")
                
                # 回退方案：通过 Lighter API 获取当前活跃订单
                self.logger.log(f"获取合约 {self.lighter_client.config.contract_id} 的活跃订单", "DEBUG")
                orders = await self.lighter_client.get_active_orders(self.lighter_client.config.contract_id)
                self.logger.log(f"发现 {len(orders)} 个活跃订单", "DEBUG")

                found = False
                for order in orders:
                    if order.order_id == self.lighter_order_id:
                        found = True
                        status_upper = str(order.status).upper()
                        self.logger.log(f"订单 {order.order_id} 状态: {status_upper}, 成交: {order.filled_size}/{order.size}", "DEBUG")
                        
                        if status_upper == "FILLED":
                            self.lighter_order_filled = True
                            self.logger.log(f"Lighter 订单 {self.lighter_order_id} 已成交", "INFO")
                            return True
                        elif status_upper in ("CANCELED", "EXPIRED"):
                            self.logger.log(f"Lighter 订单 {self.lighter_order_id} {status_upper}", "WARNING")
                            return False
                        elif order.filled_size > 0:
                            self.logger.log(f"订单 {self.lighter_order_id} 部分成交: {order.filled_size}/{order.size}", "INFO")
                        break

                # 若活跃订单中未找到，最终回退为查询订单详情
                if not found:
                    self.logger.log(f"活跃订单中未找到 {self.lighter_order_id}，改查订单详情", "DEBUG")
                    info = await self.lighter_client.get_order_info(self.lighter_order_id)
                    if info:
                        self.logger.log(f"订单详情状态: {info.status}", "DEBUG")
                        if info.status == "FILLED":
                            self.lighter_order_filled = True
                            self.logger.log(f"Lighter 订单 {self.lighter_order_id} 已成交（来自订单详情）", "INFO")
                            return True
                    else:
                        self.logger.log(f"无法获取订单 {self.lighter_order_id} 的详情", "WARNING")
                
                # 等待后再次检查
                self.logger.log("等待 2 秒后进行下一次状态检查", "DEBUG")
                await asyncio.sleep(2)
            except Exception as e:
                self.logger.log(f"监控 Lighter 订单时发生错误: {e}", "ERROR")
                self.logger.log("发生错误后等待 5 秒再重试", "DEBUG")
                await asyncio.sleep(5)  # Longer wait on error

        self.logger.log(f"退出监控循环，lighter_order_filled: {self.lighter_order_filled}", "DEBUG")
        return self.lighter_order_filled

    async def monitor_liquidation_risk(self) -> bool:
        """Check liquidation proximity across exchanges and trigger dual-side exit if needed.
        Returns True if risk threshold is triggered and exit actions should be taken.
        """
        try:
            # Fetch positions from both exchanges
            lighter_pos = await self.lighter_client.get_account_positions()
            paradex_pos = await self.paradex_client.get_account_positions()

            # If no positions open, no risk
            if lighter_pos == Decimal(0) and paradex_pos == Decimal(0):
                return False

            # Obtain current mark/mid price using BBO
            best_bid, best_ask = await self.get_bbo_price()
            current_price = (best_bid + best_ask) / Decimal(2)

            # Try to obtain liquidation prices if available via clients; if not, approximate or skip
            lighter_liq = getattr(self.lighter_client.config, 'liquidation_price', None)
            paradex_liq = getattr(self.paradex_client.config, 'liquidation_price', None)

            # If not provided, attempt rough heuristic using entry price when available (not implemented in clients)
            # Fallback: no direct liq price available; skip triggering to avoid false positives
            liq_prices = []
            if lighter_liq is not None:
                liq_prices.append(Decimal(str(lighter_liq)))
            if paradex_liq is not None:
                liq_prices.append(Decimal(str(paradex_liq)))

            if not liq_prices:
                # No liquidation price available from either exchange
                return False

            # If any liquidation price is within threshold of current price, trigger exit
            threshold = current_price * self.risk_threshold_pct
            for liq in liq_prices:
                if abs(current_price - liq) <= threshold:
                    # Trigger dual-side close and stop
                    await self.dual_side_close_and_stop()
                    return True

            return False
        except Exception as e:
            self.logger.log(f"监控强平风险时发生错误: {e}", "ERROR")
            return False

    async def dual_side_close_and_stop(self):
        """根据实际持仓在两个交易所同时尝试平仓，然后停止。"""
        try:
            # 先取消任何在途订单
            try:
                await self.cancel_lighter_order()
            except Exception as e:
                self.logger.log(f"在平仓前取消 Lighter 订单时发生错误: {e}", "ERROR")

            # 获取当前 BBO 用于定价
            try:
                best_bid, best_ask = await self.get_bbo_price()
            except Exception as e:
                self.logger.log(f"获取平仓定价所需 BBO 时发生错误: {e}", "ERROR")
                best_bid, best_ask = Decimal(0), Decimal(0)

            # Lighter 侧根据实际持仓方向进行平仓
            try:
                lighter_pos = await self.lighter_client.get_account_positions()
                lighter_close_side: Optional[str] = None
                if lighter_pos > 0:
                    lighter_close_side = 'sell'
                elif lighter_pos < 0:
                    lighter_close_side = 'buy'
                if lighter_close_side:
                    # 使用 maker-safe 价格确保挂单限价（避免吃单）
                    price = self._maker_safe_price_for_side(lighter_close_side, best_bid, best_ask)
                    await self.lighter_client.place_close_order(
                        self.lighter_client.config.contract_id,
                        self.quantity,
                        price,
                        lighter_close_side
                    )
            except Exception as e:
                self.logger.log(f"在 Lighter 下平仓单时发生错误: {e}", "ERROR")

            # Paradex 侧仅在存在持仓时进行平仓；若可获取持仓详情则据此确定方向
            try:
                paradex_close_side: Optional[str] = None
                # Prefer detailed position info when available
                if hasattr(self.paradex_client, '_fetch_positions_with_retry'):
                    try:
                        positions = await self.paradex_client._fetch_positions_with_retry()
                    except Exception:
                        positions = []
                    if positions:
                        # 使用首个持仓的方向
                        pos = positions[0]
                        side_str = str(pos.get('side', '')).upper()
                        if side_str == 'LONG':
                            paradex_close_side = 'sell'
                        elif side_str == 'SHORT':
                            paradex_close_side = 'buy'
                # 回退为仅根据持仓数量进行判断
                if paradex_close_side is None:
                    try:
                        paradex_size = await self.paradex_client.get_account_positions()
                    except Exception:
                        paradex_size = Decimal(0)
                    if paradex_size == Decimal(0):
                        paradex_close_side = None
                    else:
                        # 若无法确定方向，则回退为与策略方向相反的一侧
                        paradex_close_side = 'sell' if self.side == 'buy' else 'buy'

                if paradex_close_side:
                    # Paradex 使用当前 BBO 进行限价平仓；用户偏好将主要平仓放在 Lighter
                    price_para = best_bid if paradex_close_side == 'sell' else best_ask
                    price_para = self.paradex_client.round_to_tick(price_para)
                    await self.paradex_client.place_close_order(
                        self.paradex_client.config.contract_id,
                        self.quantity,
                        price_para,
                        paradex_close_side
                    )
            except Exception as e:
                self.logger.log(f"在 Paradex 下平仓单时发生错误: {e}", "ERROR")

        finally:
            await self.shutdown()

    async def dual_side_flatten_for_cycle_end(self) -> Dict[str, Any]:
        """在周期结束时先处理 Lighter 平仓，Paradex 平仓延后到 Lighter 完全生效之后。
        - Lighter：在反向侧使用接近 BBO 的激进限价进行平仓，并记录订单信息用于后续监控与重挂。
        - Paradex：不在本方法中执行；将等待 Lighter 完全清算后再在运行循环中下市价平仓。
        返回包含操作细节的汇总字典（此处 Paradex 部分暂为占位）。
        """
        summary: Dict[str, Any] = {
            "lighter": {"success": False},
            "paradex": {"success": False},
        }
        try:
            # Lighter 平仓：与当前方向相反，使用接近 BBO 的激进价格
            best_bid, best_ask = await self.get_bbo_price()
            lighter_close_side = 'sell' if self.side == 'buy' else 'buy'
            # 使用 maker-safe 限价以避免吃单
            lighter_price = self._maker_safe_price_for_side(
                lighter_close_side, best_bid, best_ask, offset_ticks=self.price_offset_ticks or 1
            )
            self.logger.log(f"周期结束：在 Lighter 下平仓单 {lighter_close_side} 数量 {self.quantity} 价格 {lighter_price}", "INFO")
            try:
                # 重置重试计数，用于周期结束平仓的监控与重挂
                self.current_retry_count = 0
                lr = await self.lighter_client.place_close_order(
                    self.lighter_client.config.contract_id,
                    self.quantity,
                    lighter_price,
                    lighter_close_side
                )
                # 记录平仓订单以便监控与重挂
                self.lighter_order_id = getattr(lr, "order_id", None)
                self.order_placement_time = time.time()
                summary["lighter"] = {
                    "success": bool(getattr(lr, "success", False)),
                    "order_id": getattr(lr, "order_id", None),
                    "side": lighter_close_side,
                    "price": str(lighter_price),
                    "quantity": str(self.quantity),
                }
            except Exception as e:
                summary["lighter"] = {"success": False, "error": str(e)}
                self.logger.log(f"周期结束在 Lighter 下平仓单时发生错误: {e}", "ERROR")

            # Paradex 平仓延后：等待运行循环中在 Lighter 完全清算后再执行

        except Exception as e:
            self.logger.log(f"周期结束扁平化过程中出现意外错误: {e}", "ERROR")
        return summary
    
    def on_lighter_order_update(self, order_update):
        """Callback for Lighter order updates via WebSocket."""
        if not order_update or not hasattr(order_update, 'order_id'):
            return
            
        if order_update.order_id == self.lighter_order_id:
            if order_update.status == "filled":
                self.lighter_order_filled = True
                self.logger.log(f"WebSocket：Lighter 订单 {self.lighter_order_id} 成交", "INFO")
            elif order_update.status == "canceled" or order_update.status == "expired":
                self.logger.log(f"WebSocket：Lighter 订单 {self.lighter_order_id} {order_update.status}", "WARNING")

    async def place_paradex_hedge_order(self):
        """在 Paradex 交易所下对冲订单（市价单）。"""
        # Determine opposite side for hedging
        hedge_side = "sell" if self.side == "buy" else "buy"
        self.logger.log(f"在 Paradex 下 {hedge_side} 对冲单，数量 {self.quantity} 标的 {self.ticker}", "INFO")
        
        try:
            # Place market order on Paradex for immediate execution
            order_result = await self.paradex_client.place_market_order(
                contract_id=self.paradex_client.config.contract_id,
                quantity=self.quantity,
                direction=hedge_side
            )
            
            if order_result.success:
                self.paradex_order_id = order_result.order_id
                self.logger.log(f"Paradex 对冲订单下单成功: {self.paradex_order_id}", "INFO")
                return True
            else:
                self.logger.log(f"Paradex 对冲订单下单失败: {order_result.error_message}", "ERROR")
                # 回退为跨价差的激进限价对冲
                self.logger.log("回退为在 Paradex 使用激进限价进行对冲", "WARNING")
                order_result = await self.paradex_client.place_aggressive_limit_order(
                    contract_id=self.paradex_client.config.contract_id,
                    quantity=self.quantity,
                    direction=hedge_side
                )
                if order_result and order_result.success:
                    self.paradex_order_id = order_result.order_id
                    self.logger.log(f"通过限价在 Paradex 下对冲订单成功: {self.paradex_order_id}", "INFO")
                    return True
                return False
        except Exception as e:
            self.logger.log(f"在 Paradex 下对冲订单时发生错误: {e}", "ERROR")
            return False

    async def _cycle_start_fallback_flatten(self) -> Dict[str, Any]:
        """在每个周期开始时兜底：取消两侧所有挂单并平掉遗留持仓。
        - Lighter 无市价单，用激进跨价限价实现市价效果
        - Paradex 使用市价单
        完成后等待两侧清算。"""
        summary: Dict[str, Any] = {"cancel": {}, "flatten": {}}
        # 取消 Lighter 活跃订单
        try:
            active_lighter = await self.lighter_client.get_active_orders(self.lighter_client.config.contract_id)
            cancelled_ids: list = []
            for od in active_lighter:
                try:
                    if await self.lighter_client.cancel_order(od.order_id):
                        cancelled_ids.append(od.order_id)
                except Exception:
                    pass
            summary["cancel"]["lighter"] = {"count": len(cancelled_ids), "order_ids": cancelled_ids}
            if cancelled_ids:
                self.logger.log(f"周期开始：取消 Lighter 挂单 {cancelled_ids}", "INFO")
        except Exception as e:
            summary["cancel"]["lighter"] = {"error": str(e)}
            self.logger.log(f"周期开始：取消 Lighter 挂单失败：{e}", "ERROR")

        # 取消 Paradex 活跃订单
        try:
            active_para = await self.paradex_client.get_active_orders(self.paradex_client.config.contract_id)
            cancelled_p: list = []
            for od in active_para:
                try:
                    if await self.paradex_client.cancel_order(od.order_id):
                        cancelled_p.append(od.order_id)
                except Exception:
                    pass
            summary["cancel"]["paradex"] = {"count": len(cancelled_p), "order_ids": cancelled_p}
            if cancelled_p:
                self.logger.log(f"周期开始：取消 Paradex 挂单 {cancelled_p}", "INFO")
        except Exception as e:
            summary["cancel"]["paradex"] = {"error": str(e)}
            self.logger.log(f"周期开始：取消 Paradex 挂单失败：{e}", "ERROR")

        # 获取 BBO 用于 Lighter 激进价
        try:
            best_bid, best_ask = await self.get_bbo_price()
        except Exception as e:
            self.logger.log(f"周期开始：获取 BBO 失败：{e}", "ERROR")
            best_bid, best_ask = Decimal(0), Decimal(0)

        # Lighter 平仓（使用激进跨价限价）
        try:
            lighter_pos = await self.lighter_client.get_account_positions()
            lighter_qty = abs(Decimal(lighter_pos))
            if lighter_qty > Decimal('0'):
                lighter_close_side = 'sell' if lighter_pos > 0 else 'buy'
                price = self._aggressive_cross_price_for_side(lighter_close_side, best_bid, best_ask)
                lr = await self.lighter_client.place_close_order(
                    self.lighter_client.config.contract_id,
                    lighter_qty,
                    price,
                    lighter_close_side
                )
                self.lighter_order_id = getattr(lr, "order_id", None)
                summary["flatten"]["lighter"] = {
                    "side": lighter_close_side,
                    "quantity": str(lighter_qty),
                    "price": str(price),
                    "order_id": self.lighter_order_id,
                }
                self.logger.log(f"周期开始：Lighter 激进跨价平仓 {lighter_close_side} 数量 {lighter_qty} 价格 {price}", "INFO")
            else:
                summary["flatten"]["lighter"] = {"skipped": True}
        except Exception as e:
            summary["flatten"]["lighter"] = {"error": str(e)}
            self.logger.log(f"周期开始：Lighter 平仓失败：{e}", "ERROR")

        # Paradex 平仓（市价单）
        try:
            paradex_close_side: Optional[str] = None
            paradex_qty: Decimal = Decimal('0')
            if hasattr(self.paradex_client, '_fetch_positions_with_retry'):
                try:
                    positions = await self.paradex_client._fetch_positions_with_retry()
                except Exception:
                    positions = []
                if positions:
                    pos = positions[0]
                    side_str = str(pos.get('side', '')).upper()
                    paradex_close_side = 'sell' if side_str == 'LONG' else ('buy' if side_str == 'SHORT' else None)
                    size_val = pos.get('size') or pos.get('position') or Decimal('0')
                    try:
                        paradex_qty = abs(Decimal(str(size_val)))
                    except Exception:
                        paradex_qty = Decimal('0')
            if paradex_close_side is None or paradex_qty == Decimal('0'):
                try:
                    size_abs = await self.paradex_client.get_account_positions()
                except Exception:
                    size_abs = Decimal('0')
                if size_abs and size_abs > Decimal('0'):
                    paradex_qty = abs(Decimal(size_abs))
                    paradex_close_side = 'sell' if self.side == 'buy' else 'buy'
            if paradex_close_side and paradex_qty > Decimal('0'):
                pr = await self.paradex_client.place_market_order(
                    contract_id=self.paradex_client.config.contract_id,
                    quantity=paradex_qty,
                    direction=paradex_close_side
                )
                self.paradex_order_id = getattr(pr, "order_id", None)
                summary["flatten"]["paradex"] = {
                    "side": paradex_close_side,
                    "quantity": str(paradex_qty),
                    "order_id": self.paradex_order_id,
                }
                self.logger.log(f"周期开始：Paradex 市价平仓 {paradex_close_side} 数量 {paradex_qty}", "INFO")
            else:
                summary["flatten"]["paradex"] = {"skipped": True}
        except Exception as e:
            summary["flatten"]["paradex"] = {"error": str(e)}
            self.logger.log(f"周期开始：Paradex 平仓失败：{e}", "ERROR")

        # 等待两侧清算
        try:
            await self._wait_until_cleared(timeout_seconds=30)
        except Exception as e:
            summary["wait"] = {"error": str(e)}
            self.logger.log(f"周期开始：等待清算失败：{e}", "ERROR")
        return summary

    async def run(self):
        """运行带 BBO 定价、自动替换与周期性平仓的对冲策略。
        若 `hedge_cycle_seconds` > 0，则策略会循环运行直到停止。
        """
        self.is_running = True
        self.logger.log("启动带周期的对冲策略", "INFO")

        try:
            # 仅初始化一次连接
            self.logger.log("正在初始化交易所连接", "DEBUG")
            if not await self.initialize():
                self.logger.log("初始化失败，退出。", "ERROR")
                return False

            # 循环运行周期直到停止
            while self.is_running:
                # 重置每个周期的状态
                self.current_retry_count = 0
                self.lighter_order_id = None
                self.paradex_order_id = None
                self.lighter_order_filled = False
                self.order_placement_time = None
                self.cycle_start_time = None
                
                # 周期开始兜底：取消两侧挂单并市价效果平仓
                try:
                    await self._cycle_start_fallback_flatten()
                except Exception as e:
                    self.logger.log(f"周期开始兜底平仓出错：{e}", "ERROR")

                # 为新周期在 Lighter 下初始限价单
                self.logger.log("为新周期在 Lighter 下初始限价单", "DEBUG")
                placed = await self.place_lighter_limit_order()
                if not placed:
                    self.logger.log("本周期 Lighter 初始下单失败，下一周期重试。", "ERROR")
                    await asyncio.sleep(2)
                    continue

                # 监控 Lighter 订单直到成交或超时/取消
                self.logger.log("监控 Lighter 订单成交", "DEBUG")
                monitor_result = await self.monitor_lighter_order()
                self.logger.log(f"Lighter 订单监控完成，结果: {monitor_result}", "DEBUG")

                if not monitor_result:
                    self.logger.log("Lighter 订单未成交；进入下一个周期。", "WARNING")
                    await asyncio.sleep(1)
                    continue

                # 在 Paradex 下对冲订单
                self.logger.log("为本周期在 Paradex 下对冲订单", "DEBUG")
                hedge_result = await self.place_paradex_hedge_order()
                if not hedge_result:
                    self.logger.log("Paradex 对冲订单下单失败；进入下一个周期。", "ERROR")
                    await asyncio.sleep(1)
                    continue

                # 等待至周期结束
                if self.hedge_cycle_seconds > 0 and self.cycle_start_time:
                    deadline = self.cycle_start_time + self.hedge_cycle_seconds
                    self.logger.log(f"周期运行至 {deadline}（Unix 时间戳秒）", "DEBUG")
                    while self.is_running and time.time() < deadline:
                        # 周期内可选的风险定期监控
                        if self.risk_enabled:
                            try:
                                if await self.monitor_liquidation_risk():
                                    self.logger.log("周期内触发风险退出；停止策略。", "ERROR")
                                    return False
                            except Exception as e:
                                self.logger.log(f"周期内风险监控出错: {e}", "ERROR")
                        await asyncio.sleep(1)

                # 周期结束：先在 Lighter 平仓并等待其完全清算（支持超时取消重挂）
                end_ts = time.time()
                actions = await self.dual_side_flatten_for_cycle_end()
                try:
                    lighter_close_side = 'sell' if self.side == 'buy' else 'buy'
                    await self._ensure_cycle_end_cleared(lighter_close_side)
                except Exception as e:
                    self.logger.log(f"等待 Lighter 清算时发生错误: {e}", "ERROR")

                # 仅当 Lighter 完全清算后，再在 Paradex 下市价平仓并等待其清算
                try:
                    # Paradex 的平仓方向应为与其开仓相反，即与 self.side 相同
                    paradex_close_side = self.side
                    self.logger.log(f"周期结束：在 Paradex 下市价平仓 {paradex_close_side} 数量 {self.quantity}", "INFO")
                    pr = await self.paradex_client.place_market_order(
                        contract_id=self.paradex_client.config.contract_id,
                        quantity=self.quantity,
                        direction=paradex_close_side
                    )
                    self.paradex_order_id = getattr(pr, "order_id", None)
                    # 阻塞等待 Paradex 清算完成
                    while self.is_running and not await self._is_paradex_cleared():
                        await asyncio.sleep(1)
                except Exception as e:
                    self.logger.log(f"周期结束在 Paradex 下市价平仓或等待清算时发生错误: {e}", "ERROR")

                # 清算后记录周期
                self.cycle_count += 1
                cycle_record = {
                    "cycle_index": self.cycle_count,
                    "start_time": self.cycle_start_time,
                    "end_time": end_ts,
                    "duration_seconds": (end_ts - self.cycle_start_time) if self.cycle_start_time else None,
                    "lighter_order_id": self.lighter_order_id,
                    "paradex_order_id": self.paradex_order_id,
                    "actions": actions,
                }
                self.cycle_history.append(cycle_record)
                self.logger.log(f"周期 {self.cycle_count} 结束并记录: {cycle_record}", "INFO")

                # 自动进入下一个周期
                continue

        except Exception as e:
            self.logger.log(f"对冲策略发生错误: {e}", "ERROR")
            self.logger.log("处理错误时尝试取消所有未完成订单", "DEBUG")
            try:
                await self.cancel_lighter_order()
            except Exception:
                pass
            await self.shutdown()
            return False

    async def _is_lighter_cleared(self) -> bool:
        """检查 Lighter 侧是否已清算：无活跃平仓订单且仓位为零。"""
        try:
            # 无活跃的平仓订单
            active_orders = await self.lighter_client.get_active_orders(self.lighter_client.config.contract_id)
            for order in active_orders:
                if order.side == ('sell' if self.side == 'buy' else 'buy') and order.status in ['OPEN', 'PARTIALLY_FILLED']:
                    return False
            # 无持仓
            pos = await self.lighter_client.get_account_positions()
            if abs(pos) > Decimal('0'):
                return False
            return True
        except Exception:
            return False

    async def _is_paradex_cleared(self) -> bool:
        """检查 Paradex 侧是否已清算：无活跃平仓订单且仓位为零。"""
        try:
            active_orders = await self.paradex_client.get_active_orders(self.paradex_client.config.contract_id)
            for order in active_orders:
                # 周期结束时的 Paradex 平仓方向与 self.side 相同（与其开仓相反）
                paradex_close_side = self.side
                if order.side == paradex_close_side and order.status in ['OPEN', 'PARTIALLY_FILLED']:
                    return False
            pos = await self.paradex_client.get_account_positions()
            if abs(pos) > Decimal('0'):
                return False
            return True
        except Exception:
            return False

    async def _wait_until_cleared(self, timeout_seconds: int = 30) -> None:
        """轮询直到 Lighter 和 Paradex 都已清算或超时。"""
        start = time.time()
        while self.is_running and (time.time() - start) < timeout_seconds:
            lighter_ok = await self._is_lighter_cleared()
            paradex_ok = await self._is_paradex_cleared()
            if lighter_ok and paradex_ok:
                self.logger.log("两侧已清算。开始下一周期。", "INFO")
                return
            await asyncio.sleep(1)
        self.logger.log("等待清算超时；继续下一周期。", "WARNING")

    async def _ensure_cycle_end_cleared(self, lighter_close_side: str) -> None:
        """在周期结束时仅确保 Lighter 完全清算：
        - 若 Lighter 平仓限价单超时且启用自动替换，则取消并按最新 BBO 重挂（遵循 max_retries）。
        - 不等待 Paradex；Paradex 将在 Lighter 完全清算后另行下市价并单独等待。
        该方法不设整体超时，直到 Lighter 清算后才返回。
        """
        while self.is_running:
            try:
                lighter_ok = await self._is_lighter_cleared()

                if lighter_ok:
                    self.logger.log("Lighter 已清算。", "INFO")
                    return

                # 仅在 Lighter 未清算时处理超时取消并重挂
                if not lighter_ok:
                    # 当存在挂单且达到超时阈值时，取消并按最新 BBO 重挂
                    if (
                        self.order_placement_time is not None
                        and self.order_timeout_seconds > 0
                        and (time.time() - self.order_placement_time) >= self.order_timeout_seconds
                    ):
                        if self.auto_cancel_enabled and self.current_retry_count < self.max_retries:
                            try:
                                self.logger.log(
                                    f"周期结束平仓：订单超时，取消并重新挂单（第 {self.current_retry_count + 1}/{self.max_retries} 次）",
                                    "INFO",
                                )
                                await self.cancel_lighter_order()

                                # 取当前持仓方向（带符号），反向挂单平仓
                                pos_signed = await self.lighter_client.get_account_positions()
                                if pos_signed == Decimal(0):
                                    self.logger.log("Lighter 无持仓，无需再挂平仓单", "INFO")
                                    return

                                best_bid, best_ask = await self.get_bbo_price()
                                new_price = self._maker_safe_price_for_side(
                                    lighter_close_side,
                                    best_bid,
                                    best_ask,
                                    offset_ticks=self.price_offset_ticks or 1,
                                )
                                lr = await self.lighter_client.place_close_order(
                                    self.lighter_client.config.contract_id,
                                    abs(pos_signed),  # 用实际持仓数量（绝对值）
                                    new_price,
                                    lighter_close_side,
                                )
                                self.lighter_order_id = getattr(lr, "order_id", None)
                                self.order_placement_time = time.time()
                                self.current_retry_count += 1
                                # 明确记录已重挂的新平仓单信息，便于观察不是“只在取消”
                                self.logger.log(
                                    f"周期结束平仓：已重挂 Lighter 订单 {self.lighter_order_id} 价格 {new_price}（第 {self.current_retry_count}/{self.max_retries} 次）",
                                    "INFO",
                                )
                            except Exception as e:
                                self.logger.log(f"周期结束平仓监控时发生错误: {e}", "ERROR")
                        else:
                            # 达到最大重试或未启用自动替换：触发激进跨价在 Lighter 平仓，避免无限等待
                            try:
                                self.logger.log(
                                    "周期结束平仓：达到最大重试或未启用自动替换；改用激进跨价在 Lighter 平仓。",
                                    "WARNING",
                                )
                                # 获取当前 Lighter 持仓与 BBO
                                pos_signed = await self.lighter_client.get_account_positions()
                                if pos_signed == Decimal(0):
                                    self.logger.log("Lighter 当前无持仓，无需激进平仓。", "INFO")
                                else:
                                    lighter_close_side = "sell" if pos_signed > 0 else "buy"
                                    best_bid, best_ask = await self.get_bbo_price()
                                    cross_price = self._aggressive_cross_price_for_side(
                                        lighter_close_side, best_bid, best_ask
                                    )
                                    lr = await self.lighter_client.place_close_order(
                                        self.lighter_client.config.contract_id,
                                        abs(pos_signed),
                                        cross_price,
                                        lighter_close_side,
                                    )
                                    self.lighter_order_id = getattr(lr, "order_id", None)
                                    self.order_placement_time = time.time()
                                    self.logger.log(
                                        f"周期结束平仓：已使用激进跨价在 Lighter 平仓 {lighter_close_side} 数量 {abs(pos_signed)} 价格 {cross_price} 订单 {self.lighter_order_id}",
                                        "INFO",
                                    )
                            except Exception as e:
                                self.logger.log(f"周期结束平仓：激进跨价下单失败: {e}", "ERROR")
                            # 继续循环等待 Lighter 清算；清算完成后 run() 会触发 Paradex 市价平仓

                await asyncio.sleep(1)
            except Exception as e:
                self.logger.log(f"确保周期结束清算时发生错误: {e}", "ERROR")
                await asyncio.sleep(2)


async def main():
    """运行对冲策略的主函数。"""
    strategy = None
    try:
        # 加载配置
        config = {
            "ticker": "ETH-PERP",
            "quantity": "0.1",
            "price": "3000",  # 设置为 0 表示市价
            "side": "buy",    # 在 Lighter 的初始方向
            "lighter": {
                "ticker": "ETH-PERP",
                "tick_size": Decimal("0.01")
            },
            "paradex": {
                "ticker": "ETH-PERP",
                "tick_size": Decimal("0.01")
            }
        }
        
        # 创建并运行策略
        strategy = HedgeStrategy(config)
        await strategy.run()
    except KeyboardInterrupt:
        print("\n收到中断信号，正在关闭...")
    except Exception as e:
        print(f"主函数错误: {e}")
    finally:
        # 确保正确清理
        if strategy:
            try:
                await strategy.shutdown()
            except Exception as e:
                print(f"关闭过程中发生错误: {e}")


if __name__ == "__main__":
    asyncio.run(main())