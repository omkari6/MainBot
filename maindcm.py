import os
import time
import logging
import json
from datetime import datetime
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
import numpy as np

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

class DCMBot:
    def __init__(self, config):
        self.config = config
        self.client = Client(
            api_key=os.getenv('BINANCE_API_KEY'),
            api_secret=os.getenv('BINANCE_SECRET_KEY')
        )
        
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
        
        self.initialize_symbol_info()
        
    def initialize_symbol_info(self):
        """Get symbol information for precision"""
        try:
            exchange_info = self.client.futures_exchange_info()
            for symbol in exchange_info['symbols']:
                if symbol['symbol'] == self.config.symbol:
                    self.symbol_info = symbol
                    
                    # Get precision info
                    for filter_info in symbol['filters']:
                        if filter_info['filterType'] == 'PRICE_FILTER':
                            tick_size = float(filter_info['tickSize'])
                            self.price_precision = len(str(tick_size).rstrip('0').split('.')[-1])
                        elif filter_info['filterType'] == 'LOT_SIZE':
                            step_size = float(filter_info['stepSize'])
                            self.qty_precision = len(str(step_size).rstrip('0').split('.')[-1])
                    break
                    
            logger.info(f"Symbol info initialized - Price precision: {self.price_precision}, Qty precision: {self.qty_precision}")
            
        except Exception as e:
            logger.error(f"Error getting symbol info: {e}")
            
    def get_current_price(self):
        """Get current price"""
        try:
            ticker = self.client.futures_symbol_ticker(symbol=self.config.symbol)
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return None
            
    def get_klines(self):
        """Get historical klines"""
        try:
            klines = self.client.futures_klines(
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
            
            logger.info(f"Exponential calculation: avg_cost={avg_cost:.6f}, price_diff_ratio={price_diff_ratio:.6f}, order_size={order_size:.2f}")
            return order_size
        else:
            return self.config.initial_order_size
            
    def place_market_order(self, side, size_usd, current_price):
        """Place market order with proper quantity calculation"""
        try:
            # Calculate quantity in base asset
            quantity = size_usd / current_price
            quantity = round(quantity, self.qty_precision)
            
            if quantity <= 0:
                logger.error(f"Invalid quantity calculated: {quantity}")
                return None
                
            logger.info(f"Placing {side} order: {quantity} {self.config.symbol} (${size_usd:.2f} at ~{current_price})")
            
            order = self.client.futures_create_order(
                symbol=self.config.symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )
            
            # Get executed quantity and price
            executed_qty = float(order.get('executedQty', 0))
            
            if executed_qty > 0:
                # Get fill price from order info
                order_info = self.client.futures_get_order(
                    symbol=self.config.symbol,
                    orderId=order['orderId']
                )
                
                if order_info['status'] == 'FILLED':
                    avg_price = float(order_info['avgPrice'])
                    logger.info(f"Order filled: {executed_qty} at {avg_price}")
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
        """Update position tracking"""
        if not order_result:
            return
            
        quantity = order_result['quantity']
        price = order_result['price']
        side = order_result['side']
        
        if side == 'BUY':
            self.position_size += quantity
            self.position_cost += quantity * price
            logger.info(f"Accumulated BUY: +{quantity:.6f} at {price:.6f}")
        elif side == 'SELL':
            self.position_size -= quantity
            self.position_cost -= quantity * price
            logger.info(f"Accumulated SELL: -{quantity:.6f} at {price:.6f}")
            
    def get_position_pnl(self, current_price):
        """Calculate position PnL"""
        if self.position_size == 0:
            return 0.0
            
        if self.position_side == 'LONG':
            pnl = (current_price * self.position_size) - self.position_cost
        else:  # SHORT
            pnl = self.position_cost - (current_price * self.position_size)
            
        return pnl
        
    def place_tp_sl_orders(self, current_price):
        """Place TP and SL orders after maturation"""
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
            tp_price = round(tp_price, self.price_precision)
            sl_price = round(sl_price, self.price_precision)
            quantity = round(abs(self.position_size), self.qty_precision)
            
            # Place TP order (limit)
            tp_order = self.client.futures_create_order(
                symbol=self.config.symbol,
                side=tp_side,
                type='LIMIT',
                timeInForce='GTC',
                quantity=quantity,
                price=tp_price
            )
            self.tp_order_id = tp_order['orderId']
            
            # Place SL order (stop market)
            sl_order = self.client.futures_create_order(
                symbol=self.config.symbol,
                side=sl_side,
                type='STOP_MARKET',
                quantity=quantity,
                stopPrice=sl_price
            )
            self.sl_order_id = sl_order['orderId']
            
            logger.info(f"TP/SL orders placed: TP@{tp_price} (ID:{self.tp_order_id}), SL@{sl_price} (ID:{self.sl_order_id})")
            
        except Exception as e:
            logger.error(f"Error placing TP/SL orders: {e}")
            
    def check_tp_sl_hit(self):
        """Check if TP or SL orders were filled"""
        try:
            if not self.tp_order_id and not self.sl_order_id:
                return False
                
            # Check TP order
            if self.tp_order_id:
                tp_order = self.client.futures_get_order(
                    symbol=self.config.symbol,
                    orderId=self.tp_order_id
                )
                if tp_order['status'] == 'FILLED':
                    logger.info(f"TP HIT! Order {self.tp_order_id} filled at {tp_order['avgPrice']}")
                    self.close_all_positions("TP Hit")
                    return True
                    
            # Check SL order
            if self.sl_order_id:
                sl_order = self.client.futures_get_order(
                    symbol=self.config.symbol,
                    orderId=self.sl_order_id
                )
                if sl_order['status'] == 'FILLED':
                    logger.info(f"SL HIT! Order {self.sl_order_id} filled at {sl_order['avgPrice']}")
                    self.close_all_positions("SL Hit")
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error checking TP/SL orders: {e}")
            return False
            
    def close_all_positions(self, reason):
        """Close all positions and reset"""
        try:
            if self.position_size != 0:
                # Cancel existing TP/SL orders
                if self.tp_order_id:
                    try:
                        self.client.futures_cancel_order(symbol=self.config.symbol, orderId=self.tp_order_id)
                    except:
                        pass
                        
                if self.sl_order_id:
                    try:
                        self.client.futures_cancel_order(symbol=self.config.symbol, orderId=self.sl_order_id)
                    except:
                        pass
                
                # Close position with market order
                current_price = self.get_current_price()
                if current_price:
                    pnl = self.get_position_pnl(current_price)
                    pnl_percent = (pnl / self.position_cost) * 100 if self.position_cost > 0 else 0
                    
                    side = 'SELL' if self.position_size > 0 else 'BUY'
                    quantity = round(abs(self.position_size), self.qty_precision)
                    
                    close_order = self.client.futures_create_order(
                        symbol=self.config.symbol,
                        side=side,
                        type='MARKET',
                        quantity=quantity
                    )
                    
                    logger.info(f"Position closed: {reason} - PnL: {pnl:.4f} USDT ({pnl_percent:.2f}%)")
                    
            # Reset all tracking
            self.position_side = None
            self.position_size = 0.0
            self.position_cost = 0.0
            self.series_count = 0
            self.matured = False
            self.tp_order_id = None
            self.sl_order_id = None
            self.consecutive_neutral_closes = 0
            
        except Exception as e:
            logger.error(f"Error closing positions: {e}")
            
    def start_new_series(self, side, current_price):
        """Start a new trading series"""
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
            
    def accumulate_position(self, current_price):
        """Accumulate existing position"""
        if self.series_count >= self.config.max_series:
            logger.info(f"Maximum series limit ({self.config.max_series}) reached")
            return
            
        order_size = self.calculate_order_size(current_price)
        api_side = 'BUY' if self.position_side == 'LONG' else 'SELL'
        
        order_result = self.place_market_order(api_side, order_size, current_price)
        if order_result:
            self.update_position(order_result)
            
    def run(self):
        """Main bot loop"""
        logger.info("DCM Bot started")
        
        try:
            while True:
                # Get current data
                current_price = self.get_current_price()
                if not current_price:
                    time.sleep(60)
                    continue
                    
                closes = self.get_klines()
                if not closes:
                    time.sleep(60)
                    continue
                    
                ma = self.calculate_ma(closes)
                if not ma:
                    time.sleep(60)
                    continue
                    
                upper_dcm, lower_dcm = self.calculate_dcm_levels(ma)
                divergence = ((current_price - ma) / ma) * 100
                
                # Check if TP/SL hit (after maturation)
                if self.matured and (self.tp_order_id or self.sl_order_id):
                    if self.check_tp_sl_hit():
                        continue  # Position was closed, continue monitoring
                        
                # Calculate PnL
                pnl = self.get_position_pnl(current_price)
                
                # Log current status
                logger.info(f"Monitor: Price: {current_price:.4f}, MA: {ma:.4f}, Div: {divergence:.2f}%, Series: {self.series_count}, PnL: {pnl:.4f}")
                
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
                        self.start_new_series('LONG', current_price)
                    elif current_price > upper_dcm:
                        self.start_new_series('SHORT', current_price)
                        
                time.sleep(60)  # 1 minute intervals for 5m timeframe
                
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
            if self.position_side:
                self.close_all_positions("Manual stop")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            if self.position_side:
                self.close_all_positions("Error stop")

if __name__ == "__main__":
    config = BotConfig()
    bot = DCMBot(config)
    bot.run()
