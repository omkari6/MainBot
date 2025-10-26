import os
import time
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import asyncio
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
import numpy as np
from dataclasses import dataclass
import threading
from queue import Queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dcm_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class BotConfig:
    """Configuration class for DCM Bot"""
    pair: str = "SPXUSDT"
    ma_period: int = 100
    initial_order_size: float = 20.0
    dcm_percent: float = 3.5
    tp_percent: float = 2.5
    sl_percent: float = 2.5
    maturation_bars: int = 50
    max_series: int = 100
    k1: float = 1000.0
    exp: float = 1.1
    k2: float = 10.0
    max_order_size: float = 100.0
    timeframe: str = "5m"

@dataclass
class PositionData:
    """Data class to track position information"""
    side: str  # 'LONG' or 'SHORT'
    series_count: int = 0
    total_size: float = 0.0
    total_cost: float = 0.0
    average_cost: float = 0.0
    is_mature: bool = False
    tp_order_id: Optional[str] = None
    sl_order_id: Optional[str] = None
    consecutive_dcm_breaks: int = 0
    last_dcm_break_bar: int = -1

class DCMTradingBot:
    """Professional DCM Contrarian Trading Bot for Binance Futures"""

    def __init__(self, config: BotConfig):
        self.config = config
        self.client = None
        self.position = PositionData(side="NONE")
        self.price_data = []
        self.ma_values = []
        self.bar_count = 0
        self.running = False
        self.last_kline_close_time = 0

        # Trade history
        self.trade_history = []
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0
        }

        # Rate limiting
        self.request_queue = Queue()
        self.last_request_time = 0
        self.request_count = 0

        logger.info(f"DCM Bot initialized with config: {self.config}")

    def initialize_client(self, api_key: str, api_secret: str):
        """Initialize Binance client with error handling"""
        try:
            self.client = Client(api_key, api_secret, testnet=False)
            self.client.API_URL = 'https://fapi.binance.com'  # Futures API

            # Test connection
            account_info = self.client.futures_account()
            logger.info("Successfully connected to Binance Futures API")
            logger.info(f"Account balance: {account_info.get('totalWalletBalance', 'N/A')} USDT")

            # Set leverage
            self.client.futures_change_leverage(symbol=self.config.pair, leverage=1)
            logger.info(f"Set leverage to 1x for {self.config.pair}")

        except Exception as e:
            logger.error(f"Failed to initialize Binance client: {e}")
            raise

    def rate_limit_handler(self):
        """Handle rate limiting to avoid API limits"""
        current_time = time.time()

        # Reset counter every minute
        if current_time - self.last_request_time > 60:
            self.request_count = 0
            self.last_request_time = current_time

        # Binance allows 1200 requests per minute, we'll use 1000 to be safe
        if self.request_count >= 1000:
            sleep_time = 60 - (current_time - self.last_request_time)
            if sleep_time > 0:
                logger.warning(f"Rate limit approaching, sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
                self.request_count = 0
                self.last_request_time = time.time()

        self.request_count += 1

    def safe_api_call(self, func, *args, max_retries=3, **kwargs):
        """Wrapper for API calls with retry logic and rate limiting"""
        self.rate_limit_handler()

        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except BinanceAPIException as e:
                if e.code == -1021:  # Timestamp error
                    logger.warning("Timestamp error, syncing time...")
                    time.sleep(1)
                elif e.code == -2019:  # Margin insufficient
                    logger.error("Insufficient margin for order")
                    return None
                elif attempt == max_retries - 1:
                    logger.error(f"API call failed after {max_retries} attempts: {e}")
                    raise
                else:
                    wait_time = 2 ** attempt
                    logger.warning(f"API call failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Unexpected error in API call: {e}")
                    raise
                else:
                    wait_time = 2 ** attempt
                    logger.warning(f"Unexpected error (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)

    def get_historical_data(self) -> pd.DataFrame:
        """Get historical kline data for MA calculation"""
        try:
            klines = self.safe_api_call(
                self.client.futures_klines,
                symbol=self.config.pair,
                interval=self.config.timeframe,
                limit=self.config.ma_period + 50
            )

            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])

            df['close'] = df['close'].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            return df
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            raise

    def calculate_moving_average(self, prices: List[float]) -> float:
        """Calculate Simple Moving Average"""
        if len(prices) < self.config.ma_period:
            return sum(prices) / len(prices) if prices else 0
        return sum(prices[-self.config.ma_period:]) / self.config.ma_period

    def calculate_dcm_levels(self, ma: float) -> Tuple[float, float]:
        """Calculate DCM levels"""
        dcm_multiplier = self.config.dcm_percent / 100
        upper_dcm = ma * (1 + dcm_multiplier)
        lower_dcm = ma * (1 - dcm_multiplier)
        return upper_dcm, lower_dcm

    def calculate_position_size(self, close_price: float) -> float:
        """Calculate dynamic position size using the exponential formula"""
        if self.position.series_count == 0 or self.position.average_cost == 0:
            return self.config.initial_order_size

        # Formula: (Initial Order Size + (|Average Cost - Close Price| / Average Cost) * k1)^exp * k2
        cost_divergence = abs(self.position.average_cost - close_price) / self.position.average_cost
        size = ((self.config.initial_order_size + (cost_divergence * self.config.k1)) ** self.config.exp) * self.config.k2

        # Apply maximum order size limit
        size = min(size, self.config.max_order_size)

        logger.debug(f"Calculated position size: {size:.4f} USDT (cost_divergence: {cost_divergence:.4f})")
        return round(size, 4)

    def get_symbol_info(self):
        """Get symbol information for precision"""
        try:
            exchange_info = self.safe_api_call(self.client.futures_exchange_info)
            for symbol in exchange_info['symbols']:
                if symbol['symbol'] == self.config.pair:
                    return symbol
            return None
        except Exception as e:
            logger.error(f"Error getting symbol info: {e}")
            return None

    def place_market_order(self, side: str, quantity_usdt: float, current_price: float) -> Optional[Dict]:
        """Place market order by calculating quantity in base asset terms"""
        try:
            # Get symbol info for precision
            symbol_info = self.get_symbol_info()
            if not symbol_info:
                logger.error("Could not get symbol information")
                return None
            
            # Find quantity precision
            quantity_precision = 0
            for filter_info in symbol_info['filters']:
                if filter_info['filterType'] == 'LOT_SIZE':
                    step_size = float(filter_info['stepSize'])
                    quantity_precision = len(str(step_size).split('.')[-1]) if '.' in str(step_size) else 0
                    break
            
            # Calculate quantity in base asset terms
            quantity = quantity_usdt / current_price
            
            # Round to proper precision
            quantity = round(quantity, quantity_precision)
            
            order = self.safe_api_call(
                self.client.futures_create_order,
                symbol=self.config.pair,
                side=side,
                type='MARKET',
                quantity=quantity
            )

            logger.info(f"Market order placed: {side} {quantity} {self.config.pair} (~{quantity_usdt} USDT)")
            return order
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            return None

    def place_tp_sl_orders(self, entry_price: float):
        """Place TP and SL orders after maturation"""
        try:
            position_info = self.get_position_info()
            if not position_info or position_info['positionAmt'] == '0':
                logger.warning("No position found when trying to place TP/SL orders")
                return

            position_size = abs(float(position_info['positionAmt']))
            side = 'LONG' if float(position_info['positionAmt']) > 0 else 'SHORT'

            if side == 'LONG':
                tp_price = entry_price * (1 + self.config.tp_percent / 100)
                sl_price = entry_price * (1 - self.config.sl_percent / 100)
                tp_side = 'SELL'
                sl_side = 'SELL'
            else:
                tp_price = entry_price * (1 - self.config.tp_percent / 100)
                sl_price = entry_price * (1 + self.config.sl_percent / 100)
                tp_side = 'BUY'
                sl_side = 'BUY'

            # Place TP order (limit order)
            tp_order = self.safe_api_call(
                self.client.futures_create_order,
                symbol=self.config.pair,
                side=tp_side,
                type='LIMIT',
                timeInForce='GTC',
                quantity=position_size,
                price=f"{tp_price:.4f}"
            )

            if tp_order:
                self.position.tp_order_id = tp_order['orderId']
                logger.info(f"TP order placed at {tp_price:.4f}")

            # Place SL order (stop market order)
            sl_order = self.safe_api_call(
                self.client.futures_create_order,
                symbol=self.config.pair,
                side=sl_side,
                type='STOP_MARKET',
                quantity=position_size,
                stopPrice=f"{sl_price:.4f}"
            )

            if sl_order:
                self.position.sl_order_id = sl_order['orderId']
                logger.info(f"SL order placed at {sl_price:.4f}")

        except Exception as e:
            logger.error(f"Error placing TP/SL orders: {e}")

    def cancel_tp_sl_orders(self):
        """Cancel existing TP/SL orders"""
        try:
            if self.position.tp_order_id:
                self.safe_api_call(
                    self.client.futures_cancel_order,
                    symbol=self.config.pair,
                    orderId=self.position.tp_order_id
                )
                logger.info(f"Cancelled TP order: {self.position.tp_order_id}")
                self.position.tp_order_id = None

            if self.position.sl_order_id:
                self.safe_api_call(
                    self.client.futures_cancel_order,
                    symbol=self.config.pair,
                    orderId=self.position.sl_order_id
                )
                logger.info(f"Cancelled SL order: {self.position.sl_order_id}")
                self.position.sl_order_id = None

        except Exception as e:
            logger.error(f"Error cancelling TP/SL orders: {e}")

    def get_position_info(self) -> Optional[Dict]:
        """Get current position information"""
        try:
            positions = self.safe_api_call(
                self.client.futures_position_information,
                symbol=self.config.pair
            )

            if positions:
                return positions[0]
            return None
        except Exception as e:
            logger.error(f"Error getting position info: {e}")
            return None

    def close_all_positions(self):
        """Close all positions and cancel orders"""
        try:
            position_info = self.get_position_info()
            if position_info and position_info['positionAmt'] != '0':
                position_size = abs(float(position_info['positionAmt']))
                side = 'SELL' if float(position_info['positionAmt']) > 0 else 'BUY'

                # Close position with market order
                close_order = self.safe_api_call(
                    self.client.futures_create_order,
                    symbol=self.config.pair,
                    side=side,
                    type='MARKET',
                    quantity=position_size
                )

                if close_order:
                    logger.info(f"Position closed: {side} {position_size} {self.config.pair}")
                    self.record_trade_close(close_order)

            # Cancel any existing orders
            self.cancel_tp_sl_orders()

            # Reset position data
            self.reset_position()

        except Exception as e:
            logger.error(f"Error closing positions: {e}")

    def reset_position(self):
        """Reset position tracking data"""
        self.position = PositionData(side="NONE")
        logger.debug("Position data reset")

    def record_trade_close(self, order: Dict):
        """Record completed trade for performance tracking"""
        try:
            position_info = self.get_position_info()
            if position_info:
                pnl = float(position_info.get('unRealizedProfit', 0))

                trade_record = {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': self.config.pair,
                    'side': self.position.side,
                    'size': self.position.total_size,
                    'avg_cost': self.position.average_cost,
                    'close_price': float(order.get('avgPrice', 0)),
                    'pnl': pnl,
                    'series_count': self.position.series_count
                }

                self.trade_history.append(trade_record)
                self.update_performance_metrics(pnl)
                self.save_trade_history()

                logger.info(f"Trade recorded: PnL: {pnl:.4f} USDT")

        except Exception as e:
            logger.error(f"Error recording trade: {e}")

    def update_performance_metrics(self, pnl: float):
        """Update performance tracking metrics"""
        self.performance_metrics['total_trades'] += 1
        self.performance_metrics['total_pnl'] += pnl

        if pnl > 0:
            self.performance_metrics['winning_trades'] += 1
        else:
            self.performance_metrics['losing_trades'] += 1

        # Update drawdown
        if pnl < 0:
            self.performance_metrics['current_drawdown'] += abs(pnl)
            self.performance_metrics['max_drawdown'] = max(
                self.performance_metrics['max_drawdown'],
                self.performance_metrics['current_drawdown']
            )
        else:
            self.performance_metrics['current_drawdown'] = max(0, self.performance_metrics['current_drawdown'] - pnl)

    def save_trade_history(self):
        """Save trade history to file"""
        try:
            filename = f"dcm_trade_history_{datetime.now().strftime('%Y%m%d')}.json"
            with open(filename, 'w') as f:
                json.dump({
                    'trade_history': self.trade_history,
                    'performance_metrics': self.performance_metrics
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving trade history: {e}")

    def log_monitoring_data(self, close_price: float, ma: float):
        """Log all required monitoring data"""
        position_info = self.get_position_info()
        unrealized_pnl = float(position_info.get('unRealizedProfit', 0)) if position_info else 0

        divergence = ((close_price - ma) / ma) * 100 if ma > 0 else 0

        monitoring_data = {
            'timestamp': datetime.now().isoformat(),
            'price': close_price,
            'close_price': close_price,
            'moving_average': ma,
            'total_position_profit_pct': (unrealized_pnl / self.position.total_cost * 100) if self.position.total_cost > 0 else 0,
            'close_price_ma_divergence': divergence,
            'series_count': self.position.series_count,
            'total_position_size': self.position.total_size,
            'total_position_cost': self.position.total_cost,
            'average_cost_per_piece': self.position.average_cost,
            'average_profit_per_piece': unrealized_pnl / self.position.total_size if self.position.total_size > 0 else 0
        }

        logger.info(f"Monitor: Price: {close_price:.4f}, MA: {ma:.4f}, Div: {divergence:.2f}%, "
                   f"Series: {self.position.series_count}, PnL: {unrealized_pnl:.4f}")

        # Save detailed monitoring data
        try:
            filename = f"dcm_monitoring_{datetime.now().strftime('%Y%m%d')}.json"
            with open(filename, 'a') as f:
                f.write(json.dumps(monitoring_data) + '\n')
        except Exception as e:
            logger.error(f"Error saving monitoring data: {e}")

    def process_new_bar(self, close_price: float):
        """Process new bar close and execute trading logic"""
        self.bar_count += 1
        self.price_data.append(close_price)

        # Calculate moving average
        ma = self.calculate_moving_average(self.price_data)
        self.ma_values.append(ma)

        # Calculate DCM levels
        upper_dcm, lower_dcm = self.calculate_dcm_levels(ma)

        # Log monitoring data
        self.log_monitoring_data(close_price, ma)

        logger.debug(f"Bar {self.bar_count}: Price: {close_price:.4f}, MA: {ma:.4f}, "
                    f"Upper DCM: {upper_dcm:.4f}, Lower DCM: {lower_dcm:.4f}")

        # Check for MA cross (closes all positions in any regime)
        if self.position.side != "NONE":
            if ((self.position.side == "LONG" and close_price >= ma) or
                (self.position.side == "SHORT" and close_price <= ma)):
                logger.info(f"MA cross detected, closing all {self.position.side} positions")
                self.close_all_positions()
                return

        # Main trading logic
        if self.position.side == "NONE":
            # No position, check for DCM level breach
            divergence_pct = abs((close_price - ma) / ma) * 100

            if divergence_pct >= self.config.dcm_percent:
                if close_price < lower_dcm:
                    # Start LONG series
                    self.start_new_series("LONG", close_price)
                elif close_price > upper_dcm:
                    # Start SHORT series
                    self.start_new_series("SHORT", close_price)

        else:
            # We have a position, manage it
            self.manage_existing_position(close_price, upper_dcm, lower_dcm)

    def start_new_series(self, side: str, close_price: float):
        """Start a new trading series"""
        try:
            order_size = self.config.initial_order_size
            market_side = "BUY" if side == "LONG" else "SELL"

            order = self.place_market_order(market_side, order_size)
            if order:
                self.position.side = side
                self.position.series_count = 1
                self.position.total_cost += order_size

                # Get actual fill price and quantity
                fill_price = float(order.get('avgPrice', close_price))
                fill_qty = float(order.get('executedQty', order_size / close_price))

                self.position.total_size = fill_qty
                self.position.average_cost = fill_price
                self.position.consecutive_dcm_breaks = 0

                logger.info(f"Started {side} series: Size: {order_size} USDT at {fill_price:.4f}")

        except Exception as e:
            logger.error(f"Error starting new series: {e}")

    def manage_existing_position(self, close_price: float, upper_dcm: float, lower_dcm: float):
        """Manage existing position based on DCM strategy"""
        try:
            # Check if we should continue accumulating
            should_accumulate = False
            is_in_accumulation_zone = False

            # Determine if price is in accumulation zone (outside DCM lines)
            if self.position.side == "LONG" and close_price < lower_dcm:
                is_in_accumulation_zone = True
                should_accumulate = True
            elif self.position.side == "SHORT" and close_price > upper_dcm:
                is_in_accumulation_zone = True
                should_accumulate = True

            # Check for consecutive DCM breaks (ONLY before maturation)
            if not self.position.is_mature:
                is_between_dcm = lower_dcm <= close_price <= upper_dcm

                if is_between_dcm:
                    if self.position.last_dcm_break_bar != self.bar_count - 1:
                        self.position.consecutive_dcm_breaks = 1
                    else:
                        self.position.consecutive_dcm_breaks += 1

                    self.position.last_dcm_break_bar = self.bar_count

                    if self.position.consecutive_dcm_breaks >= 2:
                        logger.info(f"Two consecutive DCM breaks before maturation, closing {self.position.side} position")
                        self.close_all_positions()
                        return
                else:
                    self.position.consecutive_dcm_breaks = 0

            # After maturation: Check maximum series limit
            if self.position.is_mature and should_accumulate:
                if self.position.series_count >= self.config.max_series:
                    logger.info(f"Maximum series limit ({self.config.max_series}) reached, stopping accumulation")
                    should_accumulate = False

            # Accumulate if conditions are met
            if should_accumulate:
                order_size = self.calculate_position_size(close_price)
                market_side = "BUY" if self.position.side == "LONG" else "SELL"

                order = self.place_market_order(market_side, order_size, close_price)
                if order:
                    fill_price = float(order.get('avgPrice', close_price))
                    fill_qty = float(order.get('executedQty', order_size / close_price))

                    # Update position data
                    old_total_cost = self.position.total_cost
                    old_total_size = self.position.total_size

                    self.position.total_cost += order_size
                    self.position.total_size += fill_qty
                    self.position.average_cost = ((old_total_cost * self.position.average_cost) +
                                                (order_size * fill_price)) / self.position.total_cost

                    logger.info(f"Accumulated {self.position.side}: +{order_size} USDT at {fill_price:.4f}, "
                              f"Avg Cost: {self.position.average_cost:.4f}")

            # Increment series count
            self.position.series_count += 1

            # Check for maturation
            if not self.position.is_mature and self.position.series_count >= self.config.maturation_bars:
                self.position.is_mature = True
                self.place_tp_sl_orders(self.position.average_cost)
                logger.info(f"Position matured at bar {self.position.series_count}, TP/SL activated")
                logger.info(f"Accumulation will continue when in accumulation zone until max series ({self.config.max_series})")

        except Exception as e:
            logger.error(f"Error managing existing position: {e}")

    def check_order_fills(self):
        """Check if TP/SL orders have been filled"""
        try:
            if self.position.tp_order_id or self.position.sl_order_id:
                orders = self.safe_api_call(
                    self.client.futures_get_open_orders,
                    symbol=self.config.pair
                )

                open_order_ids = [order['orderId'] for order in orders] if orders else []

                # Check if TP order was filled
                if (self.position.tp_order_id and
                    str(self.position.tp_order_id) not in open_order_ids):
                    logger.info("TP order filled, position closed")
                    self.record_trade_close({'avgPrice': 0})  # Will get actual price from position info
                    self.reset_position()
                    return

                # Check if SL order was filled
                if (self.position.sl_order_id and
                    str(self.position.sl_order_id) not in open_order_ids):
                    logger.info("SL order filled, position closed")
                    self.record_trade_close({'avgPrice': 0})  # Will get actual price from position info
                    self.reset_position()
                    return

        except Exception as e:
            logger.error(f"Error checking order fills: {e}")

    def run(self):
        """Main bot execution loop"""
        logger.info("Starting DCM Trading Bot...")
        self.running = True

        try:
            # Get initial historical data
            df = self.get_historical_data()
            self.price_data = df['close'].tolist()

            logger.info(f"Loaded {len(self.price_data)} historical bars")

            # Main loop
            while self.running:
                try:
                    # Get latest kline
                    klines = self.safe_api_call(
                        self.client.futures_klines,
                        symbol=self.config.pair,
                        interval=self.config.timeframe,
                        limit=1
                    )

                    if klines:
                        latest_kline = klines[0]
                        close_time = int(latest_kline[6])  # Close time
                        close_price = float(latest_kline[4])  # Close price

                        # Check if this is a new completed bar
                        if close_time > self.last_kline_close_time:
                            self.last_kline_close_time = close_time

                            # Process the new bar
                            self.process_new_bar(close_price)

                            # Check for order fills
                            if self.position.is_mature:
                                self.check_order_fills()

                    # Sleep for a short interval
                    time.sleep(10)  # Check every 10 seconds

                except KeyboardInterrupt:
                    logger.info("Received interrupt signal, shutting down...")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    time.sleep(30)  # Wait before retrying

        except Exception as e:
            logger.error(f"Critical error in bot execution: {e}")
        finally:
            self.shutdown()

    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down DCM Trading Bot...")
        self.running = False

        try:
            # Close any open positions
            if self.position.side != "NONE":
                logger.info("Closing positions before shutdown...")
                self.close_all_positions()

            # Save final trade history
            self.save_trade_history()

            # Print performance summary
            self.print_performance_summary()

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

        logger.info("DCM Trading Bot shutdown complete")

    def print_performance_summary(self):
        """Print performance summary"""
        metrics = self.performance_metrics
        win_rate = (metrics['winning_trades'] / metrics['total_trades'] * 100) if metrics['total_trades'] > 0 else 0

        logger.info("=== PERFORMANCE SUMMARY ===")
        logger.info(f"Total Trades: {metrics['total_trades']}")
        logger.info(f"Winning Trades: {metrics['winning_trades']}")
        logger.info(f"Losing Trades: {metrics['losing_trades']}")
        logger.info(f"Win Rate: {win_rate:.2f}%")
        logger.info(f"Total PnL: {metrics['total_pnl']:.4f} USDT")
        logger.info(f"Max Drawdown: {metrics['max_drawdown']:.4f} USDT")
        logger.info("===========================")

def main():
    """Main function to run the DCM Trading Bot"""
    # Load configuration
    config = BotConfig()

    # Initialize bot
    bot = DCMTradingBot(config)

    try:
        # Get API credentials from environment
        API_KEY = os.environ.get('BINANCE_API_KEY')
        API_SECRET = os.environ.get('BINANCE_API_SECRET')

        if not API_KEY or not API_SECRET:
            raise ValueError("Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables.")

        # Initialize client
        bot.initialize_client(API_KEY, API_SECRET)

        # Run bot
        bot.run()

    except KeyboardInterrupt:
        logger.info("Bot interrupted by user")
    except Exception as e:
        logger.error(f"Bot failed with error: {e}")
    finally:
        if 'bot' in locals():
            bot.shutdown()

if __name__ == "__main__":
    main()