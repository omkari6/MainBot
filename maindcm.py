import os
import time
import logging
import json
from datetime import datetime, timedelta
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import requests.exceptions
import socket

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{datetime.now().strftime("%Y-%m-%d")}logs.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BotConfig:
    def __init__(self):
        # Strategy parameters from document
        self.ma_period = 100
        self.symbol = "ASTERUSDT"  # Changed from SPXUSDT as per logs
        self.initial_order_size = 20.0  # USDT
        self.dcm_percent = 0.035  # 3.5%
        self.tp_percent = 0.025  # 2.5%
        self.sl_percent = 0.025  # 2.5%
        self.maturation_bars = 50
        self.k1 = 1000
        self.exp = 1.1
        self.k2 = 10
        self.max_order_size = 100.0  # USDT
        self.max_series = 50  # Maximum series limit
        self.timeframe = "5m"
        
        # Risk management
        self.max_position_value = 5000.0  # Maximum position value in USDT
        self.max_daily_loss = 500.0  # Maximum daily loss in USDT
        
        # Connection settings
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        self.connection_timeout = 30  # seconds
        
        # Rate limiting
        self.api_call_delay = 0.1  # seconds between API calls
        self.last_api_call = 0

class TradeHistory:
    def __init__(self, filename="trade_history.json"):
        self.filename = filename
        self.trades = []
        self.load_history()
        
    def load_history(self):
        """Load trade history from file"""
        try:
            if os.path.exists(self.filename):
                with open(self.filename, 'r') as f:
                    self.trades = json.load(f)
                logger.info(f"Loaded {len(self.trades)} trades from history")
        except Exception as e:
            logger.error(f"Error loading trade history: {e}")
            self.trades = []
            
    def save_history(self):
        """Save trade history to file"""
        try:
            with open(self.filename, 'w') as f:
                json.dump(self.trades, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving trade history: {e}")
            
    def add_trade(self, trade_data):
        """Add a trade to history"""
        trade_data['timestamp'] = datetime.now().isoformat()
        self.trades.append(trade_data)
        self.save_history()
        
    def get_daily_pnl(self, date=None):
        """Get daily PnL"""
        if date is None:
            date = datetime.now().date()
            
        daily_trades = [t for t in self.trades 
                       if datetime.fromisoformat(t['timestamp']).date() == date]
        
        return sum(t.get('pnl', 0) for t in daily_trades)

class DCMBot:
    def __init__(self, config):
        self.config = config
        self.trade_history = TradeHistory()
        
        # Initialize Binance client with retry logic
        self.client = None
        self.initialize_client()
        
        # Position tracking
        self.position_side = None  # 'LONG' or 'SHORT'
        self.position_size = 0.0  # Total position size in base asset
        self.position_cost = 0.0  # Total cost in USDT
        self.series_count = 0
        self.matured = False
        self.tp_order_id = None
        self.sl_order_id = None
        
        # Price data
        self.prices = []
        self.ma_values = []
        
        # Consecutive closes tracking
        self.consecutive_neutral_closes = 0
        
        # Symbol info
        self.symbol_info = None
        self.price_precision = 4
        self.qty_precision = 1
        self.min_qty = 0.001
        self.tick_size = 0.0001
        
        # Risk management
        self.daily_start_balance = 0.0
        self.session_start_time = datetime.now()
        
        # Connection management
        self.last_successful_connection = datetime.now()
        self.connection_errors = 0
        
        self.initialize_symbol_info()
        self.initialize_account_info()
        
    def initialize_client(self):
        """Initialize Binance client with error handling"""
        try:
            self.client = Client(
                api_key=os.getenv('BINANCE_API_KEY'),
                api_secret=os.getenv('BINANCE_SECRET_KEY'),
                requests_params={'timeout': self.config.connection_timeout}
            )
            
            # Test connection
            self.client.futures_ping()
            logger.info("Binance client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Binance client: {e}")
            raise
            
    def rate_limit_check(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last_call = current_time - self.config.last_api_call
        
        if time_since_last_call < self.config.api_call_delay:
            sleep_time = self.config.api_call_delay - time_since_last_call
            time.sleep(sleep_time)
            
        self.config.last_api_call = time.time()
        
    def api_call_with_retry(self, func, *args, **kwargs):
        """Execute API call with retry logic"""
        for attempt in range(self.config.max_retries):
            try:
                self.rate_limit_check()
                result = func(*args, **kwargs)
                self.connection_errors = 0
                self.last_successful_connection = datetime.now()
                return result
                
            except (BinanceRequestException, requests.exceptions.RequestException, 
                    socket.timeout, ConnectionError) as e:
                self.connection_errors += 1
                logger.warning(f"API call failed (attempt {attempt + 1}/{self.config.max_retries}): {e}")
                
                if attempt < self.config.max_retries - 1:
                    sleep_time = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"API call failed after {self.config.max_retries} attempts")
                    raise
                    
            except BinanceAPIException as e:
                if e.code in [-1021, -1001]:  # Timestamp or connectivity issues
                    logger.warning(f"Timestamp/connectivity error: {e}")
                    if attempt < self.config.max_retries - 1:
                        time.sleep(self.config.retry_delay)
                        continue
                raise
                
    def initialize_symbol_info(self):
        """Get symbol information for precision"""
        try:
            exchange_info = self.api_call_with_retry(self.client.futures_exchange_info)
            
            for symbol in exchange_info['symbols']:
                if symbol['symbol'] == self.config.symbol:
                    self.symbol_info = symbol
                    
                    # Get precision info
                    for filter_info in symbol['filters']:
                        if filter_info['filterType'] == 'PRICE_FILTER':
                            self.tick_size = float(filter_info['tickSize'])
                            self.price_precision = len(str(self.tick_size).rstrip('0').split('.')[-1])
                        elif filter_info['filterType'] == 'LOT_SIZE':
                            step_size = float(filter_info['stepSize'])
                            self.qty_precision = len(str(step_size).rstrip('0').split('.')[-1])
                            self.min_qty = float(filter_info['minQty'])
                    break
                    
            logger.info(f"Symbol info initialized - Price precision: {self.price_precision}, "
                       f"Qty precision: {self.qty_precision}, Min qty: {self.min_qty}")
            
        except Exception as e:
            logger.error(f"Error getting symbol info: {e}")
            # Use defaults if API fails
            self.price_precision = 4
            self.qty_precision = 1
            self.min_qty = 0.001
            
    def initialize_account_info(self):
        """Initialize account information"""
        try:
            account_info = self.api_call_with_retry(self.client.futures_account)
            self.daily_start_balance = float(account_info['totalWalletBalance'])
            logger.info(f"Account initialized - Balance: {self.daily_start_balance:.2f} USDT")
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            
    def get_current_price(self):
        """Get current price with error handling"""
        try:
            ticker = self.api_call_with_retry(
                self.client.futures_symbol_ticker, 
                symbol=self.config.symbol
            )
            return float(ticker['price'])
            
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return None
            
    def get_klines(self):
        """Get historical klines with error handling"""
        try:
            klines = self.api_call_with_retry(
                self.client.futures_klines,
                symbol=self.config.symbol,
                interval=self.config.timeframe,
                limit=self.config.ma_period + 10
            )
            
            closes = [float(kline[4]) for kline in klines]
            return closes
            
        except Exception as e:
            logger.error(f"Error getting klines: {e}")
            return None
            
    def calculate_ma(self, prices):
        """Calculate Simple Moving Average"""
        if len(prices) >= self.config.ma_period:
            return sum(prices[-self.config.ma_period:]) / self.config.ma_period
        return None
        
    def calculate_dcm_levels(self, ma):
        """Calculate DCM levels"""
        upper_dcm = ma * (1 + self.config.dcm_percent)
        lower_dcm = ma * (1 - self.config.dcm_percent)
        return upper_dcm, lower_dcm
        
    def calculate_order_size(self, current_price):
        """Calculate order size using exponential formula"""
        if self.series_count == 0:
            return self.config.initial_order_size
            
        # Calculate average cost per piece
        if self.position_size > 0:
            avg_cost = self.position_cost / self.position_size
            
            # Exponential formula: (Initial + (|avg_cost - current_price| / avg_cost) * k1)^exp * k2
            price_diff_ratio = abs(avg_cost - current_price) / avg_cost
            order_size = ((self.config.initial_order_size + price_diff_ratio * self.config.k1) ** self.config.exp) * self.config.k2
            
            # Cap at maximum order size
            order_size = min(order_size, self.config.max_order_size)
            
            # Risk management: check if new position would exceed limits
            potential_position_value = self.position_cost + order_size
            if potential_position_value > self.config.max_position_value:
                order_size = max(0, self.config.max_position_value - self.position_cost)
                logger.warning(f"Order size reduced due to position limit: {order_size:.2f}")
            
            logger.info(f"Exponential calculation: avg_cost={avg_cost:.6f}, "
                       f"price_diff_ratio={price_diff_ratio:.6f}, order_size={order_size:.2f}")
            return order_size
        else:
            return self.config.initial_order_size
            
    def check_risk_limits(self):
        """Check if risk limits are exceeded"""
        # Check daily loss limit
        daily_pnl = self.trade_history.get_daily_pnl()
        if daily_pnl < -self.config.max_daily_loss:
            logger.error(f"Daily loss limit exceeded: {daily_pnl:.2f} USDT")
            return False
            
        # Check connection health
        time_since_last_connection = datetime.now() - self.last_successful_connection
        if time_since_last_connection > timedelta(minutes=10):
            logger.error("Connection health check failed")
            return False
            
        return True
        
    def round_price(self, price):
        """Round price to proper precision"""
        return round(price / self.tick_size) * self.tick_size
        
    def round_quantity(self, quantity):
        """Round quantity to proper precision"""
        return round(quantity, self.qty_precision)
        
    def place_market_order(self, side, size_usd, current_price):
        """Place market order with comprehensive error handling"""
        try:
            # Calculate quantity in base asset
            quantity = size_usd / current_price
            quantity = self.round_quantity(quantity)
            
            if quantity < self.min_qty:
                logger.error(f"Quantity {quantity} below minimum {self.min_qty}")
                return None
                
            logger.info(f"Placing {side} order: {quantity} {self.config.symbol} "
                       f"(${size_usd:.2f} at ~{current_price})")
            
            order = self.api_call_with_retry(
                self.client.futures_create_order,
                symbol=self.config.symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )
            
            # Get executed quantity and price
            executed_qty = float(order.get('executedQty', 0))
            
            if executed_qty > 0:
                # Get fill price from order info
                order_info = self.api_call_with_retry(
                    self.client.futures_get_order,
                    symbol=self.config.symbol,
                    orderId=order['orderId']
                )
                
                if order_info['status'] == 'FILLED':
                    avg_price = float(order_info['avgPrice'])
                    logger.info(f"Order filled: {executed_qty} at {avg_price}")
                    
                    # Save to trade history
                    trade_data = {
                        'symbol': self.config.symbol,
                        'side': side,
                        'quantity': executed_qty,
                        'price': avg_price,
                        'value': executed_qty * avg_price,
                        'order_id': order['orderId'],
                        'series_count': self.series_count
                    }
                    self.trade_history.add_trade(trade_data)
                    
                    return {
                        'quantity': executed_qty,
                        'price': avg_price,
                        'side': side,
                        'orderId': order['orderId']
                    }
                else:
                    logger.error(f"Order not filled: {order_info['status']}")
                    return None
            else:
                logger.error(f"Invalid executed quantity: {executed_qty}")
                return None
                
        except BinanceAPIException as e:
            logger.error(f"Binance API error placing market order: {e}")
            return None
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            return None
            
    def update_position(self, order_result):
        """Update position tracking with validation"""
        if not order_result:
            return
            
        quantity = order_result['quantity']
        price = order_result['price']
        side = order_result['side']
        
        # Validate order result
        if quantity <= 0 or price <= 0:
            logger.error(f"Invalid order result: qty={quantity}, price={price}")
            return
            
        if side == 'BUY':
            self.position_size += quantity
            self.position_cost += quantity * price
            logger.info(f"Accumulated BUY: +{quantity:.6f} at {price:.6f}")
        elif side == 'SELL':
            self.position_size -= quantity
            self.position_cost -= quantity * price
            logger.info(f"Accumulated SELL: -{quantity:.6f} at {price:.6f}")
            
        # Log position summary
        if self.position_size != 0:
            avg_cost = self.position_cost / self.position_size
            logger.info(f"Position Summary: Size={self.position_size:.6f}, "
                       f"Cost={self.position_cost:.2f}, AvgCost={avg_cost:.6f}")
            
    def get_position_pnl(self, current_price):
        """Calculate position PnL with validation"""
        if self.position_size == 0:
            return 0.0
            
        if self.position_side == 'LONG':
            pnl = (current_price * self.position_size) - self.position_cost
        else:  # SHORT
            pnl = self.position_cost - (current_price * self.position_size)
            
        return pnl
        
    def place_tp_sl_orders(self, current_price):
        """Place TP and SL orders after maturation with proper error handling"""
        try:
            if self.position_size == 0:
                return
                
            avg_cost = self.position_cost / self.position_size
            
            if self.position_side == 'LONG':
                tp_price = avg_cost * (1 + self.config.tp_percent)
                sl_price = avg_cost * (1 - self.config.sl_percent)
                tp_side = 'SELL'
                sl_side = 'SELL'
            else:  # SHORT
                tp_price = avg_cost * (1 - self.config.tp_percent)
                sl_price = avg_cost * (1 + self.config.sl_percent)
                tp_side = 'BUY'
                sl_side = 'BUY'
                
            # Round prices to proper precision
            tp_price = self.round_price(tp_price)
            sl_price = self.round_price(sl_price)
            quantity = self.round_quantity(abs(self.position_size))
            
            # Validate prices
            if tp_price <= 0 or sl_price <= 0 or quantity <= 0:
                logger.error(f"Invalid TP/SL parameters: tp={tp_price}, sl={sl_price}, qty={quantity}")
                return
                
            # Place TP order (limit)
            try:
                tp_order = self.api_call_with_retry(
                    self.client.futures_create_order,
                    symbol=self.config.symbol,
                    side=tp_side,
                    type='LIMIT',
                    timeInForce='GTC',
                    quantity=quantity,
                    price=tp_price
                )
                self.tp_order_id = tp_order['orderId']
                logger.info(f"TP order placed: {tp_side} {quantity} at {tp_price} (ID:{self.tp_order_id})")
                
            except Exception as e:
                logger.error(f"Error placing TP order: {e}")
                
            # Place SL order (stop market)
            try:
                sl_order = self.api_call_with_retry(
                    self.client.futures_create_order,
                    symbol=self.config.symbol,
                    side=sl_side,
                    type='STOP_MARKET',
                    quantity=quantity,
                    stopPrice=sl_price
                )
                self.sl_order_id = sl_order['orderId']
                logger.info(f"SL order placed: {sl_side} {quantity} at {sl_price} (ID:{self.sl_order_id})")
                
            except Exception as e:
                logger.error(f"Error placing SL order: {e}")
                
        except Exception as e:
            logger.error(f"Error in place_tp_sl_orders: {e}")
            
    def check_tp_sl_hit(self):
        """Check if TP or SL orders were filled"""
        try:
            if not self.tp_order_id and not self.sl_order_id:
                return False
                
            # Check TP order
            if self.tp_order_id:
                try:
                    tp_order = self.api_call_with_retry(
                        self.client.futures_get_order,
                        symbol=self.config.symbol,
                        orderId=self.tp_order_id
                    )
                    if tp_order['status'] == 'FILLED':
                        logger.info(f"TP HIT! Order {self.tp_order_id} filled at {tp_order['avgPrice']}")
                        self.close_all_positions("TP Hit")
                        return True
                except Exception as e:
                    logger.error(f"Error checking TP order: {e}")
                    
            # Check SL order
            if self.sl_order_id:
                try:
                    sl_order = self.api_call_with_retry(
                        self.client.futures_get_order,
                        symbol=self.config.symbol,
                        orderId=self.sl_order_id
                    )
                    if sl_order['status'] == 'FILLED':
                        logger.info(f"SL HIT! Order {self.sl_order_id} filled at {sl_order['avgPrice']}")
                        self.close_all_positions("SL Hit")
                        return True
                except Exception as e:
                    logger.error(f"Error checking SL order: {e}")
                    
            return False
            
        except Exception as e:
            logger.error(f"Error in check_tp_sl_hit: {e}")
            return False
            
    def close_all_positions(self, reason):
        """Close all positions and reset with comprehensive error handling"""
        try:
            if self.position_size != 0:
                # Cancel existing TP/SL orders
                if self.tp_order_id:
                    try:
                        self.api_call_with_retry(
                            self.client.futures_cancel_order,
                            symbol=self.config.symbol,
                            orderId=self.tp_order_id
                        )
                        logger.info(f"TP order {self.tp_order_id} cancelled")
                    except Exception as e:
                        logger.warning(f"Could not cancel TP order: {e}")
                        
                if self.sl_order_id:
                    try:
                        self.api_call_with_retry(
                            self.client.futures_cancel_order,
                            symbol=self.config.symbol,
                            orderId=self.sl_order_id
                        )
                        logger.info(f"SL order {self.sl_order_id} cancelled")
                    except Exception as e:
                        logger.warning(f"Could not cancel SL order: {e}")
                
                # Close position with market order
                current_price = self.get_current_price()
                if current_price:
                    pnl = self.get_position_pnl(current_price)
                    pnl_percent = (pnl / abs(self.position_cost)) * 100 if self.position_cost != 0 else 0
                    
                    side = 'SELL' if self.position_size > 0 else 'BUY'
                    quantity = self.round_quantity(abs(self.position_size))
                    
                    if quantity >= self.min_qty:
                        try:
                            close_order = self.api_call_with_retry(
                                self.client.futures_create_order,
                                symbol=self.config.symbol,
                                side=side,
                                type='MARKET',
                                quantity=quantity
                            )
                            
                            logger.info(f"Position closed: {reason} - PnL: {pnl:.4f} USDT ({pnl_percent:.2f}%)")
                            
                            # Save closing trade to history
                            close_trade_data = {
                                'symbol': self.config.symbol,
                                'side': side,
                                'quantity': quantity,
                                'price': current_price,
                                'value': quantity * current_price,
                                'pnl': pnl,
                                'pnl_percent': pnl_percent,
                                'reason': reason,
                                'series_count': self.series_count,
                                'type': 'close'
                            }
                            self.trade_history.add_trade(close_trade_data)
                            
                        except Exception as e:
                            logger.error(f"Error closing position: {e}")
                    else:
                        logger.warning(f"Position size {quantity} too small to close")
                        
            # Reset all tracking
            self.position_side = None
            self.position_size = 0.0
            self.position_cost = 0.0
            self.series_count = 0
            self.matured = False
            self.tp_order_id = None
            self.sl_order_id = None
            self.consecutive_neutral_closes = 0
            
            logger.info("Position tracking reset")
            
        except Exception as e:
            logger.error(f"Error in close_all_positions: {e}")
            
    def start_new_series(self, side, current_price):
        """Start a new trading series with validation"""
        try:
            # Risk checks
            if not self.check_risk_limits():
                logger.warning("Risk limits exceeded, not starting new series")
                return
                
            order_size = self.config.initial_order_size
            api_side = 'BUY' if side == 'LONG' else 'SELL'
            
            order_result = self.place_market_order(api_side, order_size, current_price)
            if order_result:
                self.position_side = side
                self.series_count = 1
                self.matured = False
                self.consecutive_neutral_closes = 0
                self.update_position(order_result)
                logger.info(f"Started new {side} series at {current_price}")
                
        except Exception as e:
            logger.error(f"Error starting new series: {e}")
            
    def accumulate_position(self, current_price):
        """Accumulate existing position with validation"""
        try:
            if self.series_count >= self.config.max_series:
                logger.info(f"Maximum series limit ({self.config.max_series}) reached")
                return
                
            # Risk checks
            if not self.check_risk_limits():
                logger.warning("Risk limits exceeded, not accumulating")
                return
                
            order_size = self.calculate_order_size(current_price)
            if order_size <= 0:
                logger.warning("Calculated order size is 0 or negative")
                return
                
            api_side = 'BUY' if self.position_side == 'LONG' else 'SELL'
            
            order_result = self.place_market_order(api_side, order_size, current_price)
            if order_result:
                self.update_position(order_result)
                
        except Exception as e:
            logger.error(f"Error accumulating position: {e}")
            
    def log_system_status(self):
        """Log system health and status"""
        try:
            # Account info
            account_info = self.api_call_with_retry(self.client.futures_account)
            balance = float(account_info['totalWalletBalance'])
            
            # Daily PnL
            daily_pnl = self.trade_history.get_daily_pnl()
            
            # Connection health
            time_since_last_connection = datetime.now() - self.last_successful_connection
            
            logger.info(f"System Status - Balance: {balance:.2f} USDT, "
                       f"Daily PnL: {daily_pnl:.2f} USDT, "
                       f"Connection Errors: {self.connection_errors}, "
                       f"Last Connection: {time_since_last_connection.total_seconds():.0f}s ago")
                       
        except Exception as e:
            logger.error(f"Error logging system status: {e}")
            
    def run(self):
        """Main bot loop with comprehensive error handling"""
        logger.info("DCM Bot started")
        
        # Log initial system status
        self.log_system_status()
        
        loop_count = 0
        
        try:
            while True:
                loop_count += 1
                
                # Log system status every 60 loops (approximately 1 hour)
                if loop_count % 60 == 0:
                    self.log_system_status()
                
                # Get current data
                current_price = self.get_current_price()
                if not current_price:
                    logger.warning("Could not get current price, retrying...")
                    time.sleep(30)
                    continue
                    
                closes = self.get_klines()
                if not closes:
                    logger.warning("Could not get klines, retrying...")
                    time.sleep(30)
                    continue
                    
                ma = self.calculate_ma(closes)
                if not ma:
                    logger.warning("Could not calculate MA, retrying...")
                    time.sleep(30)
                    continue
                    
                upper_dcm, lower_dcm = self.calculate_dcm_levels(ma)
                divergence = ((current_price - ma) / ma) * 100
                
                # Check if TP/SL hit (after maturation)
                if self.matured and (self.tp_order_id or self.sl_order_id):
                    if self.check_tp_sl_hit():
                        continue  # Position was closed, continue monitoring
                        
                # Calculate PnL
                pnl = self.get_position_pnl(current_price)
                
                # Log current status with additional metrics
                position_value = abs(self.position_size * current_price) if self.position_size != 0 else 0
                logger.info(f"Monitor: Price: {current_price:.4f}, MA: {ma:.4f}, "
                           f"Div: {divergence:.2f}%, Series: {self.series_count}, "
                           f"PnL: {pnl:.4f}, PosValue: {position_value:.2f}")
                
                # Check for price crossing MA (closes all positions)
                if self.position_side:
                    if ((self.position_side == 'LONG' and current_price >= ma) or 
                        (self.position_side == 'SHORT' and current_price <= ma)):
                        self.close_all_positions("Price crossed MA")
                        continue
                        
                # Check if price is in neutral zone
                in_neutral_zone = lower_dcm <= current_price <= upper_dcm
                
                if self.position_side:
                    # Existing position logic
                    self.series_count += 1
                    
                    if not self.matured and self.series_count >= self.config.maturation_bars:
                        self.matured = True
                        logger.info(f"Position matured at bar {self.series_count}")
                        self.place_tp_sl_orders(current_price)
                        
                    if not self.matured:
                        # Before maturation: check for 2 consecutive neutral closes
                        if in_neutral_zone:
                            self.consecutive_neutral_closes += 1
                            logger.info(f"Neutral close {self.consecutive_neutral_closes}/2")
                            if self.consecutive_neutral_closes >= 2:
                                self.close_all_positions("2 consecutive neutral closes")
                                continue
                        else:
                            self.consecutive_neutral_closes = 0
                            
                    # Continue accumulating if conditions met
                    if ((self.position_side == 'LONG' and current_price < lower_dcm) or
                        (self.position_side == 'SHORT' and current_price > upper_dcm)):
                        self.accumulate_position(current_price)
                        
                else:
                    # No position: check for new series start
                    if current_price < lower_dcm:
                        logger.info(f"Price {current_price:.4f} below lower DCM {lower_dcm:.4f}")
                        self.start_new_series('LONG', current_price)
                    elif current_price > upper_dcm:
                        logger.info(f"Price {current_price:.4f} above upper DCM {upper_dcm:.4f}")
                        self.start_new_series('SHORT', current_price)
                        
                time.sleep(60)  # 1 minute intervals for 5m timeframe
                
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
            if self.position_side:
                self.close_all_positions("Manual stop")
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            if self.position_side:
                self.close_all_positions("Error stop")
            raise

if __name__ == "__main__":
    try:
        config = BotConfig()
        bot = DCMBot(config)
        bot.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
