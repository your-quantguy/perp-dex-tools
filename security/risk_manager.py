"""
Risk Manager for Perp-DEX-Tools
Manages position sizing, stop-loss, take-profit, and risk metrics
"""

import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
import time

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classification"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"


@dataclass
class RiskConfig:
    """Risk management configuration"""
    # Portfolio limits
    max_position_size_usd: float = 10000  # Max position size in USD
    max_total_exposure_usd: float = 50000  # Max total exposure
    max_leverage: float = 5.0  # Max leverage

    # Risk per trade
    max_risk_per_trade_pct: float = 2.0  # Max 2% risk per trade
    max_daily_loss_pct: float = 5.0  # Max 5% daily loss
    max_drawdown_pct: float = 10.0  # Max 10% drawdown

    # Position management
    use_kelly_criterion: bool = True  # Use Kelly for position sizing
    kelly_fraction: float = 0.25  # Quarter Kelly (conservative)

    # Stop-loss / Take-profit
    default_stop_loss_pct: float = 1.0  # 1% stop-loss
    default_take_profit_pct: float = 2.0  # 2% take-profit (1:2 ratio)
    trailing_stop_enabled: bool = True
    trailing_stop_pct: float = 0.5  # 0.5% trailing stop

    # Arbitrage specific
    min_profit_bps: int = 10  # Minimum 10 bps profit (0.1%)
    max_execution_time_ms: int = 5000  # Max 5s execution time
    slippage_tolerance_bps: int = 20  # 20 bps slippage tolerance


@dataclass
class Position:
    """Trading position"""
    exchange: str
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    size: float  # Position size in base currency
    size_usd: float  # Position size in USD
    timestamp: float = field(default_factory=time.time)

    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None

    # PnL tracking
    current_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


class RiskManager:
    """
    Comprehensive risk management system

    Features:
    - Kelly Criterion position sizing
    - Dynamic stop-loss/take-profit
    - Risk level assessment
    - Portfolio risk metrics
    - Arbitrage opportunity scoring
    """

    def __init__(self, config: RiskConfig = None, initial_capital: float = 10000):
        """
        Initialize risk manager

        Args:
            config: Risk configuration
            initial_capital: Starting capital in USD
        """
        self.config = config or RiskConfig()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital

        # Portfolio tracking
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []

        # Performance metrics
        self.total_pnl = 0.0
        self.daily_pnl = 0.0
        self.peak_capital = initial_capital
        self.daily_reset_time = time.time()

        logger.info(f"RiskManager initialized with ${initial_capital:,.2f} capital")

    def calculate_position_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        max_size_usd: Optional[float] = None
    ) -> float:
        """
        Calculate optimal position size using Kelly Criterion

        Kelly % = (Win% × Avg Win - Loss% × Avg Loss) / Avg Win

        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning amount (%)
            avg_loss: Average losing amount (%)
            max_size_usd: Maximum position size override

        Returns:
            Recommended position size in USD
        """
        if not self.config.use_kelly_criterion:
            # Fixed position sizing
            return min(
                self.config.max_position_size_usd,
                self.current_capital * (self.config.max_risk_per_trade_pct / 100)
            )

        # Kelly Criterion
        loss_rate = 1 - win_rate

        if avg_win <= 0 or win_rate <= 0:
            logger.warning("Invalid Kelly inputs, using minimum position size")
            return self.current_capital * 0.01  # 1% minimum

        kelly_pct = ((win_rate * avg_win) - (loss_rate * avg_loss)) / avg_win

        # Apply Kelly fraction (conservative)
        kelly_pct = max(0, kelly_pct * self.config.kelly_fraction)

        # Calculate position size
        position_size = self.current_capital * kelly_pct

        # Apply limits
        max_size = max_size_usd or self.config.max_position_size_usd
        position_size = min(position_size, max_size)

        # Risk per trade limit
        max_risk_size = self.current_capital * (self.config.max_risk_per_trade_pct / 100)
        position_size = min(position_size, max_risk_size)

        logger.info(
            f"Kelly position size: ${position_size:,.2f} "
            f"(Kelly%={kelly_pct*100:.2f}%, WinRate={win_rate*100:.1f}%)"
        )

        return position_size

    def calculate_stop_loss_take_profit(
        self,
        entry_price: float,
        side: str,
        custom_sl_pct: Optional[float] = None,
        custom_tp_pct: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate stop-loss and take-profit prices

        Args:
            entry_price: Entry price
            side: 'long' or 'short'
            custom_sl_pct: Custom stop-loss percentage
            custom_tp_pct: Custom take-profit percentage

        Returns:
            (stop_loss_price, take_profit_price)
        """
        sl_pct = custom_sl_pct or self.config.default_stop_loss_pct
        tp_pct = custom_tp_pct or self.config.default_take_profit_pct

        if side.lower() == 'long':
            stop_loss = entry_price * (1 - sl_pct / 100)
            take_profit = entry_price * (1 + tp_pct / 100)
        else:  # short
            stop_loss = entry_price * (1 + sl_pct / 100)
            take_profit = entry_price * (1 - tp_pct / 100)

        return stop_loss, take_profit

    def update_trailing_stop(self, position: Position, current_price: float) -> Optional[float]:
        """
        Update trailing stop-loss

        Args:
            position: Position to update
            current_price: Current market price

        Returns:
            New trailing stop price or None
        """
        if not self.config.trailing_stop_enabled:
            return None

        # Calculate new trailing stop
        trailing_pct = self.config.trailing_stop_pct / 100

        if position.side.lower() == 'long':
            # Long position: trail stop upward
            new_trailing_stop = current_price * (1 - trailing_pct)

            if position.trailing_stop is None:
                position.trailing_stop = new_trailing_stop
            else:
                # Only move stop upward
                position.trailing_stop = max(position.trailing_stop, new_trailing_stop)

        else:  # short
            # Short position: trail stop downward
            new_trailing_stop = current_price * (1 + trailing_pct)

            if position.trailing_stop is None:
                position.trailing_stop = new_trailing_stop
            else:
                # Only move stop downward
                position.trailing_stop = min(position.trailing_stop, new_trailing_stop)

        return position.trailing_stop

    def should_close_position(self, position: Position, current_price: float) -> Tuple[bool, str]:
        """
        Check if position should be closed (SL/TP hit)

        Args:
            position: Position to check
            current_price: Current market price

        Returns:
            (should_close, reason)
        """
        # Update position
        position.current_price = current_price

        if position.side.lower() == 'long':
            # Check stop-loss
            if position.stop_loss and current_price <= position.stop_loss:
                return True, "stop_loss"

            # Check trailing stop
            if position.trailing_stop and current_price <= position.trailing_stop:
                return True, "trailing_stop"

            # Check take-profit
            if position.take_profit and current_price >= position.take_profit:
                return True, "take_profit"

        else:  # short
            # Check stop-loss
            if position.stop_loss and current_price >= position.stop_loss:
                return True, "stop_loss"

            # Check trailing stop
            if position.trailing_stop and current_price >= position.trailing_stop:
                return True, "trailing_stop"

            # Check take-profit
            if position.take_profit and current_price <= position.take_profit:
                return True, "take_profit"

        return False, ""

    def open_position(
        self,
        exchange: str,
        symbol: str,
        side: str,
        entry_price: float,
        size_usd: float,
        custom_sl_pct: Optional[float] = None,
        custom_tp_pct: Optional[float] = None
    ) -> Optional[Position]:
        """
        Open a new position with risk management

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            side: 'long' or 'short'
            entry_price: Entry price
            size_usd: Position size in USD
            custom_sl_pct: Custom stop-loss percentage
            custom_tp_pct: Custom take-profit percentage

        Returns:
            Position object or None if rejected
        """
        # Check daily loss limit
        if self.daily_pnl < 0 and abs(self.daily_pnl) >= self.current_capital * (self.config.max_daily_loss_pct / 100):
            logger.error(
                f"Daily loss limit reached: ${abs(self.daily_pnl):,.2f} "
                f"({self.config.max_daily_loss_pct}%)"
            )
            return None

        # Check drawdown limit
        drawdown_pct = ((self.peak_capital - self.current_capital) / self.peak_capital) * 100
        if drawdown_pct >= self.config.max_drawdown_pct:
            logger.error(f"Max drawdown reached: {drawdown_pct:.2f}%")
            return None

        # Check total exposure
        total_exposure = sum(p.size_usd for p in self.positions.values())
        if total_exposure + size_usd > self.config.max_total_exposure_usd:
            logger.error(
                f"Total exposure limit exceeded: ${total_exposure + size_usd:,.2f} "
                f"(max ${self.config.max_total_exposure_usd:,.2f})"
            )
            return None

        # Calculate stop-loss and take-profit
        stop_loss, take_profit = self.calculate_stop_loss_take_profit(
            entry_price, side, custom_sl_pct, custom_tp_pct
        )

        # Create position
        position_id = f"{exchange}_{symbol}_{int(time.time())}"
        size = size_usd / entry_price  # Convert USD to base currency

        position = Position(
            exchange=exchange,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            size=size,
            size_usd=size_usd,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        self.positions[position_id] = position

        logger.info(
            f"✅ Opened {side} position: {symbol} @ ${entry_price:.4f}, "
            f"Size=${size_usd:.2f}, SL=${stop_loss:.4f}, TP=${take_profit:.4f}"
        )

        return position

    def close_position(self, position_id: str, exit_price: float, reason: str = "manual") -> Optional[float]:
        """
        Close a position and calculate PnL

        Args:
            position_id: Position identifier
            exit_price: Exit price
            reason: Reason for closing

        Returns:
            Realized PnL or None
        """
        if position_id not in self.positions:
            logger.warning(f"Position not found: {position_id}")
            return None

        position = self.positions[position_id]

        # Calculate PnL
        if position.side.lower() == 'long':
            pnl = (exit_price - position.entry_price) * position.size
        else:  # short
            pnl = (position.entry_price - exit_price) * position.size

        position.realized_pnl = pnl
        position.current_price = exit_price

        # Update capital
        self.current_capital += pnl
        self.total_pnl += pnl
        self.daily_pnl += pnl

        # Update peak capital
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital

        # Move to closed positions
        self.closed_positions.append(position)
        del self.positions[position_id]

        pnl_pct = (pnl / position.size_usd) * 100

        logger.info(
            f"✅ Closed {position.side} position: {position.symbol} @ ${exit_price:.4f}, "
            f"PnL=${pnl:.2f} ({pnl_pct:+.2f}%), Reason={reason}"
        )

        return pnl

    def assess_risk_level(
        self,
        position_size_usd: float,
        leverage: float = 1.0,
        volatility: Optional[float] = None
    ) -> RiskLevel:
        """
        Assess risk level of a potential trade

        Args:
            position_size_usd: Position size in USD
            leverage: Leverage used
            volatility: Asset volatility (optional)

        Returns:
            RiskLevel enum
        """
        # Calculate risk score (0-100)
        risk_score = 0

        # Position size risk
        size_pct = (position_size_usd / self.current_capital) * 100
        if size_pct > 20:
            risk_score += 40
        elif size_pct > 10:
            risk_score += 30
        elif size_pct > 5:
            risk_score += 20
        else:
            risk_score += 10

        # Leverage risk
        if leverage > 5:
            risk_score += 40
        elif leverage > 3:
            risk_score += 30
        elif leverage > 2:
            risk_score += 20
        else:
            risk_score += 10

        # Volatility risk
        if volatility:
            if volatility > 5:
                risk_score += 20
            elif volatility > 3:
                risk_score += 15
            elif volatility > 1:
                risk_score += 10
            else:
                risk_score += 5

        # Map score to level
        if risk_score >= 80:
            return RiskLevel.EXTREME
        elif risk_score >= 60:
            return RiskLevel.VERY_HIGH
        elif risk_score >= 40:
            return RiskLevel.HIGH
        elif risk_score >= 25:
            return RiskLevel.MEDIUM
        elif risk_score >= 15:
            return RiskLevel.LOW
        else:
            return RiskLevel.VERY_LOW

    def score_arbitrage_opportunity(
        self,
        buy_price: float,
        sell_price: float,
        buy_fee_pct: float,
        sell_fee_pct: float,
        size_usd: float,
        execution_time_ms: int,
        estimated_slippage_bps: int = 0
    ) -> Dict:
        """
        Score an arbitrage opportunity with comprehensive risk analysis

        Args:
            buy_price: Buy price
            sell_price: Sell price
            buy_fee_pct: Buy fee percentage
            sell_fee_pct: Sell fee percentage
            size_usd: Trade size in USD
            execution_time_ms: Expected execution time
            estimated_slippage_bps: Estimated slippage in bps

        Returns:
            Opportunity score and details
        """
        # Calculate gross profit
        gross_profit_pct = ((sell_price - buy_price) / buy_price) * 100

        # Calculate fees
        total_fee_pct = buy_fee_pct + sell_fee_pct

        # Calculate slippage
        slippage_pct = estimated_slippage_bps / 100

        # Calculate net profit
        net_profit_pct = gross_profit_pct - total_fee_pct - slippage_pct
        net_profit_usd = size_usd * (net_profit_pct / 100)

        # Minimum profit check
        min_profit_pct = self.config.min_profit_bps / 100
        if net_profit_pct < min_profit_pct:
            return {
                'score': 0,
                'executable': False,
                'reason': f'Net profit {net_profit_pct:.3f}% < min {min_profit_pct:.3f}%',
                'gross_profit_pct': gross_profit_pct,
                'net_profit_pct': net_profit_pct,
                'net_profit_usd': net_profit_usd,
            }

        # Execution time check
        if execution_time_ms > self.config.max_execution_time_ms:
            return {
                'score': 0,
                'executable': False,
                'reason': f'Execution time {execution_time_ms}ms > max {self.config.max_execution_time_ms}ms',
                'gross_profit_pct': gross_profit_pct,
                'net_profit_pct': net_profit_pct,
                'net_profit_usd': net_profit_usd,
            }

        # Slippage check
        max_slippage_bps = self.config.slippage_tolerance_bps
        if estimated_slippage_bps > max_slippage_bps:
            return {
                'score': 50,  # Risky but maybe executable
                'executable': True,
                'warning': f'High slippage {estimated_slippage_bps}bps > {max_slippage_bps}bps',
                'gross_profit_pct': gross_profit_pct,
                'net_profit_pct': net_profit_pct,
                'net_profit_usd': net_profit_usd,
            }

        # Calculate opportunity score (0-100)
        score = 0

        # Profit score (max 40 points)
        profit_score = min(40, net_profit_pct * 100)  # 0.4% = 40 points
        score += profit_score

        # Size score (max 30 points)
        size_score = min(30, (net_profit_usd / 100) * 30)  # $100 = 30 points
        score += size_score

        # Speed score (max 20 points)
        speed_score = max(0, 20 - (execution_time_ms / 100))  # Faster = better
        score += speed_score

        # Slippage score (max 10 points)
        slippage_score = max(0, 10 - estimated_slippage_bps / 2)
        score += slippage_score

        return {
            'score': min(100, score),
            'executable': True,
            'gross_profit_pct': round(gross_profit_pct, 4),
            'net_profit_pct': round(net_profit_pct, 4),
            'net_profit_usd': round(net_profit_usd, 2),
            'total_fee_pct': round(total_fee_pct, 4),
            'slippage_pct': round(slippage_pct, 4),
            'execution_time_ms': execution_time_ms,
            'size_usd': size_usd,
        }

    def get_portfolio_metrics(self) -> Dict:
        """
        Get comprehensive portfolio risk metrics

        Returns:
            Dictionary of metrics
        """
        # Reset daily PnL if new day
        if time.time() - self.daily_reset_time > 86400:
            self.daily_pnl = 0
            self.daily_reset_time = time.time()

        # Calculate total exposure
        total_exposure = sum(p.size_usd for p in self.positions.values())

        # Calculate unrealized PnL
        unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())

        # Calculate drawdown
        drawdown = self.peak_capital - self.current_capital
        drawdown_pct = (drawdown / self.peak_capital * 100) if self.peak_capital > 0 else 0

        # Calculate win rate from closed positions
        if self.closed_positions:
            winning_trades = [p for p in self.closed_positions if p.realized_pnl > 0]
            win_rate = len(winning_trades) / len(self.closed_positions)

            avg_win = sum(p.realized_pnl for p in winning_trades) / len(winning_trades) if winning_trades else 0
            losing_trades = [p for p in self.closed_positions if p.realized_pnl < 0]
            avg_loss = sum(abs(p.realized_pnl) for p in losing_trades) / len(losing_trades) if losing_trades else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0

        return {
            'initial_capital': self.initial_capital,
            'current_capital': round(self.current_capital, 2),
            'total_pnl': round(self.total_pnl, 2),
            'total_pnl_pct': round((self.total_pnl / self.initial_capital) * 100, 2),
            'daily_pnl': round(self.daily_pnl, 2),
            'unrealized_pnl': round(unrealized_pnl, 2),
            'peak_capital': round(self.peak_capital, 2),
            'drawdown': round(drawdown, 2),
            'drawdown_pct': round(drawdown_pct, 2),
            'total_exposure': round(total_exposure, 2),
            'open_positions': len(self.positions),
            'closed_positions': len(self.closed_positions),
            'win_rate': round(win_rate * 100, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
        }
