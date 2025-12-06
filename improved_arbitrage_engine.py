"""
Improved Arbitrage Engine for Perp-DEX-Tools
Integrates security modules for safe and profitable arbitrage trading
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
from dataclasses import dataclass, asdict
import json

from security.secure_key_manager import SecureKeyManager
from security.rate_limiter import get_manager as get_rate_limit_manager, CircuitState
from security.risk_manager import RiskManager, RiskConfig, RiskLevel

from exchanges.factory import create_exchange

logger = logging.getLogger(__name__)


@dataclass
class ArbitrageOpportunity:
    """Arbitrage opportunity details"""
    buy_exchange: str
    sell_exchange: str
    symbol: str
    buy_price: float
    sell_price: float
    gross_profit_pct: float
    net_profit_pct: float
    net_profit_usd: float
    size_usd: float
    score: float
    timestamp: float
    executable: bool
    reason: str = ""
    warning: str = ""


@dataclass
class ArbitrageTrade:
    """Completed arbitrage trade"""
    opportunity: ArbitrageOpportunity
    executed: bool
    buy_order_id: Optional[str] = None
    sell_order_id: Optional[str] = None
    actual_buy_price: Optional[float] = None
    actual_sell_price: Optional[float] = None
    actual_profit_usd: Optional[float] = None
    execution_time_ms: Optional[int] = None
    error: Optional[str] = None


class ImprovedArbitrageEngine:
    """
    Advanced arbitrage engine with comprehensive security and risk management

    Features:
    - Encrypted API key management
    - Rate limiting and circuit breakers
    - Risk-based position sizing
    - Real profit calculation (fees + slippage)
    - Opportunity scoring
    - Parallel order execution
    - Comprehensive statistics
    """

    def __init__(
        self,
        config_file: str = "arbitrage_config.json",
        use_encrypted_keys: bool = True,
        initial_capital: float = 10000
    ):
        """
        Initialize arbitrage engine

        Args:
            config_file: Configuration file path
            use_encrypted_keys: Use encrypted key storage
            initial_capital: Starting capital in USD
        """
        self.config = self._load_config(config_file)
        self.use_encrypted_keys = use_encrypted_keys

        # Security modules
        self.key_manager = SecureKeyManager() if use_encrypted_keys else None
        self.rate_limit_manager = get_rate_limit_manager()
        self.risk_manager = RiskManager(
            config=RiskConfig(**self.config.get('risk', {})),
            initial_capital=initial_capital
        )

        # Exchange clients
        self.exchanges: Dict[str, Any] = {}

        # Trading state
        self.opportunities: List[ArbitrageOpportunity] = []
        self.trades: List[ArbitrageTrade] = []
        self.active = False

        # Statistics
        self.stats = {
            'opportunities_found': 0,
            'opportunities_executed': 0,
            'total_profit_usd': 0.0,
            'total_loss_usd': 0.0,
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0,
        }

        logger.info("ImprovedArbitrageEngine initialized")

    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from file"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_file}, using defaults")
            return self._default_config()

    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'exchanges': ['edgex', 'lighter', 'backpack'],
            'symbols': ['BTC-USD', 'ETH-USD'],
            'min_profit_bps': 10,  # 0.1%
            'max_position_size_usd': 5000,
            'check_interval_seconds': 1,
            'risk': {
                'max_position_size_usd': 5000,
                'max_risk_per_trade_pct': 1.0,
                'use_kelly_criterion': True,
                'min_profit_bps': 10,
            },
            'fees': {
                'edgex': 0.05,  # 0.05% maker
                'lighter': 0.02,  # 0.02% maker
                'backpack': 0.02,
                'paradex': 0.05,
                'aster': 0.03,
                'grvt': 0.02,
                'extended': 0.05,
                'apex': 0.05,
            }
        }

    async def initialize_exchanges(self):
        """Initialize exchange clients"""
        if self.use_encrypted_keys:
            logger.info("Loading encrypted API keys...")
            self.key_manager.load_keys()

        for exchange_name in self.config['exchanges']:
            try:
                # Get API credentials
                if self.use_encrypted_keys:
                    credentials = self._get_exchange_credentials(exchange_name)
                else:
                    import os
                    credentials = self._get_exchange_credentials_from_env(exchange_name)

                # Create exchange client
                exchange = create_exchange(exchange_name, credentials)
                self.exchanges[exchange_name] = exchange

                logger.info(f"✅ Initialized {exchange_name}")

            except Exception as e:
                logger.error(f"Failed to initialize {exchange_name}: {e}")

    def _get_exchange_credentials(self, exchange: str) -> Dict:
        """Get credentials from encrypted storage"""
        creds = {}

        # Map exchange to required keys
        key_mapping = {
            'edgex': ['EDGEX_ACCOUNT_ID', 'EDGEX_STARK_PRIVATE_KEY'],
            'lighter': ['API_KEY_PRIVATE_KEY', 'LIGHTER_ACCOUNT_INDEX'],
            'backpack': ['BACKPACK_PUBLIC_KEY', 'BACKPACK_SECRET_KEY'],
            'paradex': ['PARADEX_L1_ADDRESS', 'PARADEX_L2_PRIVATE_KEY'],
            'aster': ['ASTER_API_KEY', 'ASTER_SECRET_KEY'],
            'grvt': ['GRVT_TRADING_ACCOUNT_ID', 'GRVT_PRIVATE_KEY'],
            'extended': ['EXTENDED_API_KEY', 'EXTENDED_STARK_KEY_PAIR'],
            'apex': ['APEX_API_KEY', 'APEX_OMNI_KEY_SEED'],
        }

        for key in key_mapping.get(exchange, []):
            value = self.key_manager.get_key(key)
            if value:
                creds[key] = value

        return creds

    def _get_exchange_credentials_from_env(self, exchange: str) -> Dict:
        """Get credentials from environment variables"""
        import os
        from dotenv import load_dotenv
        load_dotenv()

        creds = {}
        key_mapping = {
            'edgex': ['EDGEX_ACCOUNT_ID', 'EDGEX_STARK_PRIVATE_KEY'],
            'lighter': ['API_KEY_PRIVATE_KEY', 'LIGHTER_ACCOUNT_INDEX'],
            'backpack': ['BACKPACK_PUBLIC_KEY', 'BACKPACK_SECRET_KEY'],
        }

        for key in key_mapping.get(exchange, []):
            value = os.getenv(key)
            if value:
                creds[key] = value

        return creds

    async def get_best_prices(self, symbol: str) -> Dict[str, Tuple[float, float]]:
        """
        Get best bid/ask prices from all exchanges

        Args:
            symbol: Trading symbol

        Returns:
            Dict of {exchange: (best_bid, best_ask)}
        """
        prices = {}

        async def fetch_price(exchange_name: str, client: Any):
            try:
                # Execute with rate limiting and circuit breaker
                orderbook = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.rate_limit_manager.execute_with_protection(
                        exchange_name,
                        client.get_orderbook,
                        symbol
                    )
                )

                if orderbook and 'bids' in orderbook and 'asks' in orderbook:
                    best_bid = float(orderbook['bids'][0][0]) if orderbook['bids'] else 0
                    best_ask = float(orderbook['asks'][0][0]) if orderbook['asks'] else 0
                    prices[exchange_name] = (best_bid, best_ask)

            except Exception as e:
                logger.warning(f"Failed to get price from {exchange_name}: {e}")

        # Fetch prices in parallel
        tasks = [
            fetch_price(name, client)
            for name, client in self.exchanges.items()
        ]
        await asyncio.gather(*tasks)

        return prices

    def find_arbitrage_opportunities(self, prices: Dict[str, Tuple[float, float]], symbol: str) -> List[ArbitrageOpportunity]:
        """
        Find arbitrage opportunities from price data

        Args:
            prices: Price data from exchanges
            symbol: Trading symbol

        Returns:
            List of arbitrage opportunities
        """
        opportunities = []

        # Compare all exchange pairs
        exchange_names = list(prices.keys())

        for i, buy_exchange in enumerate(exchange_names):
            for sell_exchange in exchange_names[i + 1:]:
                # Get prices
                buy_bid, buy_ask = prices[buy_exchange]
                sell_bid, sell_ask = prices[sell_exchange]

                # Check both directions
                # Direction 1: Buy from buy_exchange, sell to sell_exchange
                if buy_ask > 0 and sell_bid > buy_ask:
                    opp = self._create_opportunity(
                        buy_exchange, sell_exchange, symbol,
                        buy_ask, sell_bid
                    )
                    if opp and opp.executable:
                        opportunities.append(opp)

                # Direction 2: Buy from sell_exchange, sell to buy_exchange
                if sell_ask > 0 and buy_bid > sell_ask:
                    opp = self._create_opportunity(
                        sell_exchange, buy_exchange, symbol,
                        sell_ask, buy_bid
                    )
                    if opp and opp.executable:
                        opportunities.append(opp)

        return opportunities

    def _create_opportunity(
        self,
        buy_exchange: str,
        sell_exchange: str,
        symbol: str,
        buy_price: float,
        sell_price: float
    ) -> Optional[ArbitrageOpportunity]:
        """Create and score arbitrage opportunity"""

        # Get fees
        buy_fee = self.config['fees'].get(buy_exchange, 0.05)
        sell_fee = self.config['fees'].get(sell_exchange, 0.05)

        # Position size
        size_usd = min(
            self.config['max_position_size_usd'],
            self.risk_manager.current_capital * 0.1  # 10% max
        )

        # Estimate slippage (1-5 bps depending on size)
        slippage_bps = int(5 * (size_usd / 10000))

        # Score opportunity
        score_result = self.risk_manager.score_arbitrage_opportunity(
            buy_price=buy_price,
            sell_price=sell_price,
            buy_fee_pct=buy_fee,
            sell_fee_pct=sell_fee,
            size_usd=size_usd,
            execution_time_ms=1000,  # Estimate 1s
            estimated_slippage_bps=slippage_bps
        )

        if not score_result['executable']:
            return None

        return ArbitrageOpportunity(
            buy_exchange=buy_exchange,
            sell_exchange=sell_exchange,
            symbol=symbol,
            buy_price=buy_price,
            sell_price=sell_price,
            gross_profit_pct=score_result['gross_profit_pct'],
            net_profit_pct=score_result['net_profit_pct'],
            net_profit_usd=score_result['net_profit_usd'],
            size_usd=size_usd,
            score=score_result['score'],
            timestamp=time.time(),
            executable=True,
            warning=score_result.get('warning', '')
        )

    async def execute_arbitrage(self, opportunity: ArbitrageOpportunity) -> ArbitrageTrade:
        """
        Execute arbitrage trade

        Args:
            opportunity: Arbitrage opportunity

        Returns:
            Trade result
        """
        start_time = time.time()
        trade = ArbitrageTrade(opportunity=opportunity, executed=False)

        # Check circuit breakers
        buy_circuit = self.rate_limit_manager.get_circuit_breaker(opportunity.buy_exchange)
        sell_circuit = self.rate_limit_manager.get_circuit_breaker(opportunity.sell_exchange)

        if not buy_circuit.is_closed():
            trade.error = f"Circuit breaker open for {opportunity.buy_exchange}"
            return trade

        if not sell_circuit.is_closed():
            trade.error = f"Circuit breaker open for {opportunity.sell_exchange}"
            return trade

        try:
            # Calculate order size
            size = opportunity.size_usd / opportunity.buy_price

            # Execute orders in parallel
            buy_task = asyncio.create_task(
                self._place_order(
                    opportunity.buy_exchange,
                    opportunity.symbol,
                    'buy',
                    opportunity.buy_price,
                    size
                )
            )

            sell_task = asyncio.create_task(
                self._place_order(
                    opportunity.sell_exchange,
                    opportunity.symbol,
                    'sell',
                    opportunity.sell_price,
                    size
                )
            )

            # Wait for both orders
            buy_result, sell_result = await asyncio.gather(buy_task, sell_task)

            # Record results
            trade.buy_order_id = buy_result.get('order_id')
            trade.sell_order_id = sell_result.get('order_id')
            trade.actual_buy_price = buy_result.get('price', opportunity.buy_price)
            trade.actual_sell_price = sell_result.get('price', opportunity.sell_price)

            # Calculate actual profit
            actual_profit = (trade.actual_sell_price - trade.actual_buy_price) * size
            trade.actual_profit_usd = actual_profit
            trade.executed = True

            # Record execution time
            trade.execution_time_ms = int((time.time() - start_time) * 1000)

            # Update risk manager
            self.risk_manager.current_capital += actual_profit

            # Record trade in circuit breaker
            buy_circuit.record_trade(opportunity.size_usd, actual_profit / 2)
            sell_circuit.record_trade(opportunity.size_usd, actual_profit / 2)

            # Update statistics
            self._update_stats(trade)

            logger.info(
                f"✅ Arbitrage executed: {opportunity.symbol} "
                f"Buy@{trade.actual_buy_price} {opportunity.buy_exchange} → "
                f"Sell@{trade.actual_sell_price} {opportunity.sell_exchange}, "
                f"Profit=${actual_profit:.2f}"
            )

        except Exception as e:
            trade.error = str(e)
            logger.error(f"Arbitrage execution failed: {e}")

        return trade

    async def _place_order(
        self,
        exchange: str,
        symbol: str,
        side: str,
        price: float,
        size: float
    ) -> Dict:
        """Place order on exchange"""
        client = self.exchanges.get(exchange)
        if not client:
            raise Exception(f"Exchange not initialized: {exchange}")

        # Execute with rate limiting
        order = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.rate_limit_manager.execute_with_protection(
                exchange,
                client.create_order,
                symbol=symbol,
                side=side,
                order_type='limit',
                price=price,
                size=size
            )
        )

        return order

    def _update_stats(self, trade: ArbitrageTrade):
        """Update trading statistics"""
        if not trade.executed or trade.actual_profit_usd is None:
            return

        self.stats['opportunities_executed'] += 1

        profit = trade.actual_profit_usd

        if profit > 0:
            self.stats['total_profit_usd'] += profit
            if profit > self.stats['best_trade']:
                self.stats['best_trade'] = profit
        else:
            self.stats['total_loss_usd'] += abs(profit)
            if profit < self.stats['worst_trade']:
                self.stats['worst_trade'] = profit

        # Calculate win rate
        profitable_trades = [t for t in self.trades if t.executed and t.actual_profit_usd and t.actual_profit_usd > 0]
        self.stats['win_rate'] = len(profitable_trades) / len([t for t in self.trades if t.executed])

        # Average profit/loss
        if profitable_trades:
            self.stats['avg_profit'] = sum(t.actual_profit_usd for t in profitable_trades) / len(profitable_trades)

        losing_trades = [t for t in self.trades if t.executed and t.actual_profit_usd and t.actual_profit_usd < 0]
        if losing_trades:
            self.stats['avg_loss'] = sum(t.actual_profit_usd for t in losing_trades) / len(losing_trades)

    async def run(self, duration_seconds: Optional[int] = None):
        """
        Run arbitrage engine

        Args:
            duration_seconds: Run duration (None = indefinite)
        """
        self.active = True
        start_time = time.time()

        logger.info("🚀 Starting arbitrage engine...")

        await self.initialize_exchanges()

        if not self.exchanges:
            logger.error("No exchanges initialized. Exiting.")
            return

        try:
            while self.active:
                # Check duration
                if duration_seconds and (time.time() - start_time) >= duration_seconds:
                    logger.info("Duration reached. Stopping...")
                    break

                # Scan for opportunities
                for symbol in self.config['symbols']:
                    try:
                        # Get prices
                        prices = await self.get_best_prices(symbol)

                        if len(prices) < 2:
                            continue

                        # Find opportunities
                        opportunities = self.find_arbitrage_opportunities(prices, symbol)

                        for opp in opportunities:
                            self.stats['opportunities_found'] += 1
                            self.opportunities.append(opp)

                            logger.info(
                                f"💰 Opportunity: {opp.symbol} "
                                f"{opp.buy_exchange}→{opp.sell_exchange} "
                                f"Net profit={opp.net_profit_pct:.3f}% (${opp.net_profit_usd:.2f}) "
                                f"Score={opp.score:.1f}"
                            )

                            # Execute if score is high enough
                            if opp.score >= 60:
                                trade = await self.execute_arbitrage(opp)
                                self.trades.append(trade)

                    except Exception as e:
                        logger.error(f"Error scanning {symbol}: {e}")

                # Wait before next scan
                await asyncio.sleep(self.config['check_interval_seconds'])

        except KeyboardInterrupt:
            logger.info("Stopped by user")
        finally:
            self.active = False
            self.print_summary()

    def print_summary(self):
        """Print trading summary"""
        print("\n" + "=" * 60)
        print("📊 ARBITRAGE TRADING SUMMARY")
        print("=" * 60)

        print(f"\n🔍 Opportunities Found: {self.stats['opportunities_found']}")
        print(f"✅ Trades Executed: {self.stats['opportunities_executed']}")

        if self.stats['opportunities_executed'] > 0:
            print(f"\n💰 Profit: ${self.stats['total_profit_usd']:.2f}")
            print(f"📉 Loss: ${self.stats['total_loss_usd']:.2f}")
            net_pnl = self.stats['total_profit_usd'] - self.stats['total_loss_usd']
            print(f"💵 Net PnL: ${net_pnl:.2f}")

            print(f"\n📈 Win Rate: {self.stats['win_rate'] * 100:.1f}%")
            print(f"💚 Avg Win: ${self.stats['avg_profit']:.2f}")
            print(f"💔 Avg Loss: ${self.stats['avg_loss']:.2f}")
            print(f"🏆 Best Trade: ${self.stats['best_trade']:.2f}")
            print(f"📉 Worst Trade: ${self.stats['worst_trade']:.2f}")

        print(f"\n💼 Portfolio Metrics:")
        metrics = self.risk_manager.get_portfolio_metrics()
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        print("\n" + "=" * 60)

    def stop(self):
        """Stop the engine"""
        self.active = False


async def main():
    """Main entry point"""
    import sys

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Parse command line args
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else None

    # Create and run engine
    engine = ImprovedArbitrageEngine(
        use_encrypted_keys=False,  # Set to True after migrating keys
        initial_capital=10000
    )

    await engine.run(duration_seconds=duration)


if __name__ == "__main__":
    asyncio.run(main())
