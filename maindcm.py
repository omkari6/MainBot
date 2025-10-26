#!/usr/bin/env python3
"""
DCM Trading Bot - Binance Futures Perpetual Trading Bot
Implements Dollar Cost Mean (DCM) contrarian strategy
"""

import os
import time
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
import requests

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dcm_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class BotConfig:
    """Configuration class for DCM Trading Bot"""
    pair: str = "ASTERUSDT"
    ma_period: int = 100
    initial_order_size: float = 10.0
    dcm_percent: float = 0.5
    tp_percent: float = 0.7
    sl_percent: float = 0.7
    maturation_bars: int = 20
    max_series: int = 30  # Maximum series limit
    k1: float = 1000.0
    exp: float = 1.1
    k2: float = 10.0
    max_order_size: float = 100.0
    timeframe: str = "5m"

@dataclass
class Position:
    """Position tracking class"""
    side: str  # 'LONG' or 'SHORT'
    series_count: int = 0
    total_size: float = 0.0
    total_cost: float = 0.0
    avg_price: float = 0.0
    matured: bool = False
    tp_order_id: Optional[str] = None
    sl_order_id: Optional[str] = None
    consecutive_between_dcm: int = 0
    last_accumulation_bar: int = 0

class DCMTradingBot:
    """DCM Trading Bot Implementation"""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.client = None
        self.position: Optional[Position] = None
        self.price_history: List[float] = []
        self.ma_history: List[float] = []
        self.current_bar = 0
        self.trade_history: List[Dict] = []
        self.symbol_info = None
        
        # Initialize Binance client
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Binance Futures client"""
        try:
            api_key = os.getenv('BINANCE_API_KEY')
            api_secret = os.getenv('BINANCE_API_SECRET')
            
            if not api_key or not api_secret:
                raise ValueError("Binance API credentials not found in environment variables")
            
            self.client = Client(api_key, api_secret)
            self.client.API_URL = 'https://fapi.binance.com'
            
            # Test connection
            account_info = self.client.futures_account()
            logger.info("Successfully connected to Binance Futures API")
            
            # Log account balance
            balance = float(account_info['totalWalletBalance'])
            logger.info(f"Account balance: {balance} USDT")
            
            # Get symbol info for precision
            self._get_symbol_info()
            
            # Set leverage to 1x
            self.client.futures_change_leverage(symbol=self.config.pair, leverage=1)
            logger.info(f"Set leverage to 1x for {self.config.pair}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Binance client: {e}")
            raise
    
    def _get_symbol_info(self):
        """Get symbol information for precision"""
        try:
            exchange_info = self.client.futures_exchange_info()
            for symbol in exchange_info['symbols']:
                if symbol['symbol'] == self.config.pair:
                    self.symbol_info = symbol
                    break
            
            if not self.symbol_info:
                raise ValueError(f"Symbol {self.config.pair} not found")
            
        except Exception as e:
            logger.error(f"Failed to get symbol info: {e}")
            raise
    
    def _round_quantity(self, quantity: float) -> float:
        """Round quantity according to symbol precision"""
        if not self.symbol_info:
            return round(quantity, 6)
        
        for filter_info in self.symbol_info['filters']:
            if filter_info['filterType'] == 'LOT_SIZE':
                step_size = float(filter_info['stepSize'])
                precision = len(str(step_size).split('.')[-1].rstrip('0'))
                return round(quantity, precision)
        
        return round(quantity, 6)
    
    def get_current_price(self) -> float:
        """Get current price for the trading pair"""
        try:
            ticker = self.client.futures_symbol_ticker(symbol=self.config.pair)
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"Failed to get current price: {e}")
            raise
    
    def get_historical_data(self, limit: int = 150) -> pd.DataFrame:
        """Get historical kline data"""
        try:
            klines = self.client.futures_klines(
                symbol=self.config.pair,
                interval=self.config.timeframe,
                limit=limit
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            df['close'] = df['close'].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
        
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            raise
    
    def calculate_ma(self, prices: List[float], period: int) -> float:
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return sum(prices) / len(prices)
        return sum(prices[-period:]) / period
    
    def calculate_dcm_levels(self, ma: float) -> Tuple[float, float]:
        """Calculate DCM levels"""
        dcm_range = ma * (self.config.dcm_percent / 100)
        upper_dcm = ma + dcm_range
        lower_dcm = ma - dcm_range
        return upper_dcm, lower_dcm
    
    def calculate_position_size(self, current_price: float) -> float:
        """Calculate position size using DCM formula"""
        if not self.position or self.position.total_size == 0:
            return self.config.initial_order_size
        
        avg_cost = self.position.avg_price
        price_diff_ratio = abs(avg_cost - current_price) / avg_cost
        
        size_usd = (self.config.initial_order_size + (price_diff_ratio * self.config.k1)) ** self.config.exp * self.config.k2
        
        return min(size_usd, self.config.max_order_size)
    
    def place_market_order(self, side: str, size_usd: float, current_price: float) -> Dict:
        """Place market order with proper quantity calculation"""
        try:
            # Calculate quantity in base asset terms
            quantity = size_usd / current_price
            quantity = self._round_quantity(quantity)
            
            order = self.client.futures_create_order(
                symbol=self.config.pair,
                side=side,
                type='MARKET',
                quantity=quantity
            )
            
            logger.info(f"Market order placed: {side} {quantity} {self.config.pair} at ~{current_price}")
            
            # Debug: Log the full order response
            logger.debug(f"Order response: {order}")
            
            # Check if order was filled
            if 'executedQty' not in order or float(order['executedQty']) == 0:
                # Try to get order status
                try:
                    order_status = self.client.futures_get_order(
                        symbol=self.config.pair,
                        orderId=order['orderId']
                    )
                    logger.debug(f"Order status: {order_status}")
                    if 'executedQty' in order_status:
                        order['executedQty'] = order_status['executedQty']
                except Exception as e:
                    logger.warning(f"Could not get order status: {e}")
            
            return order
        
        except BinanceAPIException as e:
            logger.error(f"Binance API error placing market order: {e}")
            raise
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            raise
    
    def place_tp_sl_orders(self):
        """Place Take Profit and Stop Loss orders"""
        if not self.position:
            return
        
        try:
            current_price = self.get_current_price()
            
            if self.position.side == 'LONG':
                tp_price = self.position.avg_price * (1 + self.config.tp_percent / 100)
                sl_price = self.position.avg_price * (1 - self.config.sl_percent / 100)
                tp_side = 'SELL'
                sl_side = 'SELL'
            else:  # SHORT
                tp_price = self.position.avg_price * (1 - self.config.tp_percent / 100)
                sl_price = self.position.avg_price * (1 + self.config.sl_percent / 100)
                tp_side = 'BUY'
                sl_side = 'BUY'
            
            quantity = self._round_quantity(self.position.total_size)
            
            # Place TP order
            tp_order = self.client.futures_create_order(
                symbol=self.config.pair,
                side=tp_side,
                type='LIMIT',
                timeInForce='GTC',
                quantity=quantity,
                price=f"{tp_price:.6f}"
            )
            self.position.tp_order_id = tp_order['orderId']
            
            # Place SL order
            sl_order = self.client.futures_create_order(
                symbol=self.config.pair,
                side=sl_side,
                type='STOP_MARKET',
                quantity=quantity,
                stopPrice=f"{sl_price:.6f}"
            )
            self.position.sl_order_id = sl_order['orderId']
            
            logger.info(f"TP/SL orders placed: TP@{tp_price:.6f}, SL@{sl_price:.6f}")
        
        except Exception as e:
            logger.error(f"Error placing TP/SL orders: {e}")
    
    def cancel_tp_sl_orders(self):
        """Cancel existing TP/SL orders"""
        if not self.position:
            return
        
        try:
            if self.position.tp_order_id:
                self.client.futures_cancel_order(
                    symbol=self.config.pair,
                    orderId=self.position.tp_order_id
                )
                self.position.tp_order_id = None
            
            if self.position.sl_order_id:
                self.client.futures_cancel_order(
                    symbol=self.config.pair,
                    orderId=self.position.sl_order_id
                )
                self.position.sl_order_id = None
        
        except Exception as e:
            logger.warning(f"Error canceling TP/SL orders: {e}")
    
    def close_position(self, reason: str = "Manual"):
        """Close current position"""
        if not self.position:
            return
        
        try:
            current_price = self.get_current_price()
            
            # Cancel TP/SL orders first
            self.cancel_tp_sl_orders()
            
            # Close position with market order
            close_side = 'SELL' if self.position.side == 'LONG' else 'BUY'
            quantity = self._round_quantity(self.position.total_size)
            
            if quantity <= 0:
                logger.error(f"Cannot close position: invalid quantity {quantity}")
                self.position = None  # Reset position to prevent further errors
                return
            
            order = self.client.futures_create_order(
                symbol=self.config.pair,
                side=close_side,
                type='MARKET',
                quantity=quantity
            )
            
            # Calculate PnL
            if self.position.side == 'LONG':
                pnl = (current_price - self.position.avg_price) * self.position.total_size
            else:
                pnl = (self.position.avg_price - current_price) * self.position.total_size
            
            pnl_percent = (pnl / self.position.total_cost) * 100
            
            # Log trade
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'side': self.position.side,
                'series_count': self.position.series_count,
                'total_size': self.position.total_size,
                'avg_price': self.position.avg_price,
                'close_price': current_price,
                'pnl': pnl,
                'pnl_percent': pnl_percent,
                'reason': reason
            }
            
            self.trade_history.append(trade_record)
            self.save_trade_history()
            
            logger.info(f"Position closed: {reason} - PnL: {pnl:.4f} USDT ({pnl_percent:.2f}%)")
            
            # Reset position
            self.position = None
        
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    def start_new_series(self, side: str, current_price: float):
        """Start a new trading series"""
        try:
            # Convert position side to Binance API side
            api_side = 'BUY' if side == 'LONG' else 'SELL'
            
            # Place initial order
            order = self.place_market_order(api_side, self.config.initial_order_size, current_price)
            
            # Initialize position
            executed_qty = float(order.get('executedQty', 0))
            if executed_qty <= 0:
                logger.error(f"Invalid executed quantity: {executed_qty}. Order response: {order}")
                # Try to get actual position from account
                try:
                    positions = self.client.futures_position_information(symbol=self.config.pair)
                    for pos in positions:
                        if float(pos['positionAmt']) != 0:
                            executed_qty = abs(float(pos['positionAmt']))
                            logger.info(f"Retrieved position size from account: {executed_qty}")
                            break
                except Exception as e:
                    logger.error(f"Could not retrieve position from account: {e}")
                
                if executed_qty <= 0:
                    logger.error("Could not determine valid position size")
                    return
            
            self.position = Position(
                side=side,
                series_count=1,
                total_size=executed_qty,
                total_cost=self.config.initial_order_size,
                avg_price=current_price,
                last_accumulation_bar=self.current_bar
            )
            
            logger.info(f"Started new {side} series - Size: {self.position.total_size:.6f}")
        
        except Exception as e:
            logger.error(f"Error starting new series: {e}")
    
    def manage_existing_position(self, current_price: float, ma: float, upper_dcm: float, lower_dcm: float):
        """Manage existing position"""
        if not self.position:
            return
        
        # Check if maximum series limit reached
        if self.position.series_count >= self.config.max_series:
            logger.info(f"Maximum series limit ({self.config.max_series}) reached - no more accumulation")
            return
        
        # Increment series count each bar
        self.position.series_count += 1
        
        # Check maturation
        if not self.position.matured and self.position.series_count >= self.config.maturation_bars:
            self.position.matured = True
            self.place_tp_sl_orders()
            logger.info(f"Position matured at bar {self.position.series_count}")
        
        # Check for price crossing MA (closes position in both regimes)
        if ((self.position.side == 'LONG' and current_price >= ma) or 
            (self.position.side == 'SHORT' and current_price <= ma)):
            self.close_position("Price crossed MA")
            return
        
        # Check accumulation conditions
        should_accumulate = False
        
        if self.position.side == 'LONG' and current_price <= lower_dcm:
            should_accumulate = True
            self.position.consecutive_between_dcm = 0
        elif self.position.side == 'SHORT' and current_price >= upper_dcm:
            should_accumulate = True
            self.position.consecutive_between_dcm = 0
        else:
            # Price is between DCM levels
            self.position.consecutive_between_dcm += 1
            
            # Before maturation: 2 consecutive closes between DCM levels closes position
            if not self.position.matured and self.position.consecutive_between_dcm >= 2:
                self.close_position("2 consecutive closes between DCM levels (pre-maturation)")
                return
        
        # Accumulate if conditions are met
        if should_accumulate:
            try:
                size_usd = self.calculate_position_size(current_price)
                side = 'BUY' if self.position.side == 'LONG' else 'SELL'
                
                order = self.place_market_order(side, size_usd, current_price)
                
                # Update position
                new_quantity = float(order['executedQty'])
                new_cost = size_usd
                
                if new_quantity <= 0:
                    logger.error(f"Invalid executed quantity in accumulation: {new_quantity}")
                    return
                
                total_quantity = self.position.total_size + new_quantity
                total_cost = self.position.total_cost + new_cost
                
                if total_quantity <= 0:
                    logger.error(f"Invalid total quantity: {total_quantity}")
                    return
                
                self.position.avg_price = total_cost / total_quantity
                self.position.total_size = total_quantity
                self.position.total_cost = total_cost
                self.position.last_accumulation_bar = self.current_bar
                
                logger.info(f"Accumulated {side}: +{new_quantity:.6f} at {current_price:.6f}")
            
            except Exception as e:
                logger.error(f"Error accumulating position: {e}")
    
    def check_tp_sl_filled(self):
        """Check if TP or SL orders are filled"""
        if not self.position:
            return
        
        try:
            # Check TP order
            if self.position.tp_order_id:
                tp_order = self.client.futures_get_order(
                    symbol=self.config.pair,
                    orderId=self.position.tp_order_id
                )
                if tp_order['status'] == 'FILLED':
                    logger.info("Take Profit order filled - closing entire position")
                    self.close_position("Take Profit hit")
                    return True
            
            # Check SL order
            if self.position.sl_order_id:
                sl_order = self.client.futures_get_order(
                    symbol=self.config.pair,
                    orderId=self.position.sl_order_id
                )
                if sl_order['status'] == 'FILLED':
                    logger.info("Stop Loss order filled - closing entire position")
                    self.close_position("Stop Loss hit")
                    return True
        
        except Exception as e:
            logger.warning(f"Error checking TP/SL orders: {e}")
        
        return False
    
    def calculate_pnl(self, current_price: float) -> float:
        """Calculate current PnL"""
        if not self.position or self.position.total_size == 0:
            return 0.0
        
        if self.position.side == 'LONG':
            return (current_price - self.position.avg_price) * self.position.total_size
        else:
            return (self.position.avg_price - current_price) * self.position.total_size
    
    def check_pnl_based_exit(self, current_price: float) -> bool:
        """Check if position should be closed based on PnL thresholds"""
        if not self.position or not self.position.matured:
            return False
        
        current_pnl = self.calculate_pnl(current_price)
        pnl_percent = (current_pnl / self.position.total_cost) * 100
        
        # Check TP threshold
        if pnl_percent >= self.config.tp_percent:
            logger.info(f"PnL-based TP hit: {pnl_percent:.2f}% >= {self.config.tp_percent}%")
            self.close_position(f"PnL-based TP: {pnl_percent:.2f}%")
            return True
        
        # Check SL threshold
        if pnl_percent <= -self.config.sl_percent:
            logger.info(f"PnL-based SL hit: {pnl_percent:.2f}% <= -{self.config.sl_percent}%")
            self.close_position(f"PnL-based SL: {pnl_percent:.2f}%")
            return True
        
        return False
    
    def save_trade_history(self):
        """Save trade history to file"""
        try:
            with open('trade_history.json', 'w') as f:
                json.dump(self.trade_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving trade history: {e}")
    
    def run(self):
        """Main trading loop"""
        logger.info("Starting DCM Trading Bot...")
        
        try:
            # Load initial historical data
            df = self.get_historical_data()
            self.price_history = df['close'].tolist()
            logger.info(f"Loaded {len(self.price_history)} historical bars")
            
            while True:
                try:
                    # Get current price
                    current_price = self.get_current_price()
                    
                    # Update price history
                    self.price_history.append(current_price)
                    if len(self.price_history) > 200:  # Keep last 200 bars
                        self.price_history.pop(0)
                    
                    # Calculate MA
                    ma = self.calculate_ma(self.price_history, self.config.ma_period)
                    self.ma_history.append(ma)
                    
                    # Calculate DCM levels
                    upper_dcm, lower_dcm = self.calculate_dcm_levels(ma)
                    
                    # Calculate divergence
                    divergence = ((current_price - ma) / ma) * 100
                    
                    # Increment bar counter
                    self.current_bar += 1
                    
                    # Check TP/SL orders first
                    if self.check_tp_sl_filled():
                        continue  # Position was closed, skip to next iteration
                    
                    # Check PnL-based exit conditions
                    if self.check_pnl_based_exit(current_price):
                        continue  # Position was closed, skip to next iteration
                    
                    # Trading logic
                    if self.position is None:
                        # No position - check for entry signals
                        if current_price <= lower_dcm:
                            # Start LONG series
                            self.start_new_series('LONG', current_price)
                        elif current_price >= upper_dcm:
                            # Start SHORT series
                            self.start_new_series('SHORT', current_price)
                    else:
                        # Manage existing position
                        self.manage_existing_position(current_price, ma, upper_dcm, lower_dcm)
                    
                    # Calculate current PnL
                    current_pnl = self.calculate_pnl(current_price)
                    
                    # Log monitoring info
                    series_count = self.position.series_count if self.position else 0
                    logger.info(
                        f"Monitor: Price: {current_price:.4f}, MA: {ma:.4f}, "
                        f"Div: {divergence:.2f}%, Series: {series_count}, PnL: {current_pnl:.4f}"
                    )
                    
                    # Wait for next bar (5 minutes for 5m timeframe)
                    time.sleep(300)
                
                except KeyboardInterrupt:
                    logger.info("Bot stopped by user")
                    if self.position:
                        self.close_position("Manual stop")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    time.sleep(10)  # Wait before retrying
        
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            raise

def main():
    """Main function"""
    # Configuration
    config = BotConfig(
        pair="ASTERUSDT",
        ma_period=100,
        initial_order_size=10.0,
        dcm_percent=0.5,
        tp_percent=0.7,
        sl_percent=0.7,
        maturation_bars=20,
        max_series=30,
        k1=1000.0,
        exp=1.1,
        k2=10.0,
        max_order_size=30.0,
        timeframe="2m"
    )
    
    logger.info(f"DCM Bot initialized with config: {config}")
    
    # Create and run bot
    bot = DCMTradingBot(config)
    bot.run()

if __name__ == "__main__":
    main()
