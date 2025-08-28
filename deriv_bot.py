import asyncio
import websockets
import json
import pandas as pd
import numpy as np
import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from flask import Flask
import threading
import os

app = Flask(__name__)

@app.route('/health')
def health():
    return {'status': 'running', 'bot_active': True}

class TradingStrategyAnalyzer:
    def __init__(self):
        self.strategies = {
            'ultra_aggressive': {'entry_offset': 0.00005, 'tp_ratio': 0.8, 'sl_ratio': 1.5},
        }
    
    def calculate_donkey_parameters(self) -> Dict:
        return {
            'entry_offset': 0.00005,
            'tp_distance': 0.0002,
            'sl_distance': 0.0008,
        }

class NotificationManager:
    def __init__(self, email_config: Dict = None, telegram_config: Dict = None):
        self.email_config = email_config
        self.telegram_config = telegram_config
        
    def send_telegram(self, message: str):
        if not self.telegram_config:
            return
        try:
            url = f"https://api.telegram.org/bot{self.telegram_config['bot_token']}/sendMessage"
            data = {'chat_id': self.telegram_config['chat_id'], 'text': message}
            requests.post(url, data=data, timeout=5)
        except Exception as e:
            logging.error(f"Telegram failed: {e}")
    
    def notify(self, subject: str, message: str):
        self.send_telegram(f"*{subject}*\n{message}")

class RiskManager:
    def __init__(self, max_daily_loss_percent: float = 50.0):
        self.max_daily_loss_percent = max_daily_loss_percent
        self.daily_start_balance = 0
        self.current_daily_loss = 0
        self.last_reset_date = datetime.now().date()
    
    def reset_daily_tracking(self, current_balance: float):
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_start_balance = current_balance
            self.current_daily_loss = 0
            self.last_reset_date = today
    
    def can_trade(self, current_balance: float) -> bool:
        self.reset_daily_tracking(current_balance)
        if current_balance < 0.35:
            return False
        if self.daily_start_balance == 0:
            return True
        daily_loss_percent = (self.current_daily_loss / self.daily_start_balance) * 100
        return daily_loss_percent < self.max_daily_loss_percent
    
    def calculate_position_size(self, account_balance: float, risk_per_trade: float = 8.0) -> float:
        calculated_risk = (account_balance * risk_per_trade / 100)
        min_trade = 0.35
        max_trade = min(account_balance * 0.8, 5.00)
        return max(min_trade, min(calculated_risk, max_trade))
    
    def update_daily_loss(self, loss_amount: float):
        if loss_amount > 0:
            self.current_daily_loss += loss_amount

class TechnicalAnalyzer:
    def __init__(self, deriv_api):
        self.deriv_api = deriv_api
        self.trade_counter = 0
    
    def detect_ultra_aggressive_patterns(self, df: pd.DataFrame) -> Dict:
        if len(df) < 3:
            return {'pattern': 'instant_trade', 'signal': 'buy'}
        
        close = df['close'].values
        current_price = close[-1]
        
        # ULTRA AGGRESSIVE - Trade on ANY price movement
        if len(close) >= 2:
            prev_price = close[-2]
            price_diff = current_price - prev_price
            
            # Contrarian on ANY movement > 0.001%
            if price_diff > 0.00001:
                return {'pattern': 'micro_up_fade', 'signal': 'sell'}
            elif price_diff < -0.00001:
                return {'pattern': 'micro_down_fade', 'signal': 'buy'}
        
        # Even if no movement, alternate between buy/sell
        self.trade_counter += 1
        signal = 'buy' if self.trade_counter % 2 == 0 else 'sell'
        return {'pattern': 'forced_alternating', 'signal': signal}
    
    async def generate_signal_async(self, symbol: str) -> Dict:
        signal = {'symbol': symbol, 'action': 'none', 'entry_price': 0, 
                 'take_profit': 0, 'stop_loss': 0, 'confidence': 0}
        
        try:
            df = await self.deriv_api._get_candles_async(symbol, granularity=60, count=10)
            if df.empty:
                # Even with no data, force a trade
                return {'symbol': symbol, 'action': 'buy', 'entry_price': 1.0,
                       'take_profit': 1.002, 'stop_loss': 0.998, 'confidence': 95}
            
            patterns = self.detect_ultra_aggressive_patterns(df)
            current_price = df['close'].iloc[-1]
            
            # Ultra-tight parameters for maximum trades
            entry_offset = 0.00002
            tp_distance = 0.0003
            sl_distance = 0.001
            
            if patterns['signal'] == 'buy':
                entry_price = current_price + entry_offset
                take_profit = entry_price + tp_distance
                stop_loss = entry_price - sl_distance
                signal.update({'action': 'buy', 'entry_price': round(entry_price, 5),
                              'take_profit': round(take_profit, 5), 'stop_loss': round(stop_loss, 5), 'confidence': 95})
                
            elif patterns['signal'] == 'sell':
                entry_price = current_price - entry_offset
                take_profit = entry_price - tp_distance
                stop_loss = entry_price + sl_distance
                signal.update({'action': 'sell', 'entry_price': round(entry_price, 5),
                              'take_profit': round(take_profit, 5), 'stop_loss': round(stop_loss, 5), 'confidence': 95})
                
            logging.info(f"Generated {patterns['signal']} signal for {symbol} - Pattern: {patterns['pattern']}")
        except Exception as e:
            logging.error(f"Signal generation error for {symbol}: {e}")
            # Force a signal even on error
            signal = {'symbol': symbol, 'action': 'buy', 'entry_price': 1.0,
                     'take_profit': 1.002, 'stop_loss': 0.998, 'confidence': 90}
        
        return signal

class DerivAPI:
    def __init__(self, app_id: str, api_token: str):
        self.app_id = app_id
        self.api_token = api_token
        self.ws_url = f"wss://ws.binaryws.com/websockets/v3?app_id={app_id}"
        self.websocket = None
        self.is_connected = False
        self.balance = 0
        self.req_id = 0
        self.current_prices = {}  # Store current prices for contracts
        self.connection_lock = asyncio.Lock()
        self.ping_task = None
        
    async def connect(self):
        try:
            if self.websocket and not self.websocket.closed:
                await self.websocket.close()
            
            # Add connection timeout and proper error handling
            self.websocket = await asyncio.wait_for(
                websockets.connect(
                    self.ws_url,
                    ping_interval=20,  # Send ping every 20 seconds
                    ping_timeout=10,   # Wait 10 seconds for pong
                    close_timeout=10,  # Wait 10 seconds for close
                    max_size=2**20     # 1MB message size limit
                ), 
                timeout=30
            )
            
            self.is_connected = True
            
            # Start keepalive ping task
            self.ping_task = asyncio.create_task(self._keepalive())
            
            # Authorize and get balance
            auth_success = await self.authorize()
            if auth_success:
                await self.get_balance()
                logging.info("✅ Connected and authorized with Deriv API")
                return True
            else:
                logging.error("❌ Authorization failed")
                return False
                
        except asyncio.TimeoutError:
            logging.error("❌ Connection timeout")
            return False
        except Exception as e:
            logging.error(f"❌ Connection failed: {e}")
            return False
    
    async def _keepalive(self):
        """Keep connection alive with periodic pings"""
        try:
            while self.is_connected and self.websocket and not self.websocket.closed:
                await asyncio.sleep(30)  # Ping every 30 seconds
                if self.websocket and not self.websocket.closed:
                    try:
                        await self.websocket.ping()
                        logging.debug("📡 Keepalive ping sent")
                    except Exception as e:
                        logging.warning(f"⚠️ Keepalive ping failed: {e}")
                        break
        except Exception as e:
            logging.error(f"❌ Keepalive task error: {e}")
    
    async def ensure_connected(self):
        """Ensure connection is active, reconnect if needed"""
        if not self.is_connected or not self.websocket or self.websocket.closed:
            logging.warning("🔄 Connection lost, attempting reconnect...")
            return await self.connect()
        return True
    
    async def send_request(self, request: Dict) -> Dict:
        async with self.connection_lock:
            # Ensure we're connected before sending
            if not await self.ensure_connected():
                return {"error": {"message": "Connection failed"}}
            
            try:
                self.req_id += 1
                request['req_id'] = self.req_id
                
                # Send request with timeout
                await asyncio.wait_for(
                    self.websocket.send(json.dumps(request)), 
                    timeout=10
                )
                
                # Receive response with timeout
                response_str = await asyncio.wait_for(
                    self.websocket.recv(), 
                    timeout=15
                )
                
                response = json.loads(response_str)
                logging.debug(f"📨 API Response: {response}")
                return response
                
            except asyncio.TimeoutError:
                logging.error("⏰ API request timeout")
                return {"error": {"message": "Request timeout"}}
            except websockets.exceptions.ConnectionClosed:
                logging.error("🔌 WebSocket connection closed")
                self.is_connected = False
                return {"error": {"message": "Connection closed"}}
            except Exception as e:
                logging.error(f"❌ API request error: {e}")
                return {"error": {"message": str(e)}}
    
    async def authorize(self):
        response = await self.send_request({"authorize": self.api_token})
        if response.get('error'):
            logging.error(f"Authorization failed: {response['error']}")
            return False
        logging.info("Authorized with Deriv")
        return True
    
    async def get_balance(self):
        response = await self.send_request({"balance": 1})
        if response.get('balance'):
            self.balance = response['balance']['balance']
            logging.info(f"Balance: {self.balance}")
        return self.balance
    
    async def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol with improved error handling"""
        try:
            # First try to get price from ticks_history (more reliable)
            request = {
                "ticks_history": symbol, 
                "adjust_start_time": 1, 
                "count": 1,
                "end": "latest", 
                "start": 1
            }
            
            response = await self.send_request(request)
            
            if response.get('history'):
                prices = response['history'].get('prices', [])
                if prices:
                    price = float(prices[-1])
                    self.current_prices[symbol] = price
                    logging.info(f"💰 Got price for {symbol}: {price}")
                    return price
            
            # Fallback: try live ticks subscription
            request = {"ticks": symbol}
            response = await self.send_request(request)
            
            if response.get('tick'):
                price = round(float(response['tick']['quote']), 2)  # Round to 2 decimal places
                self.current_prices[symbol] = price
                logging.info(f"💰 Got live price for {symbol}: {price}")
                return price
            
            # If we have a cached price, use it
            if symbol in self.current_prices:
                cached_price = self.current_prices[symbol]
                logging.warning(f"⚠️ Using cached price for {symbol}: {cached_price}")
                return cached_price
            
            logging.error(f"❌ Could not get any price for {symbol}")
            return 0.0
            
        except Exception as e:
            logging.error(f"❌ Error getting price for {symbol}: {e}")
            
            # Return cached price if available
            if symbol in self.current_prices:
                return self.current_prices[symbol]
            return 0.0
    
    async def _get_candles_async(self, symbol: str, granularity: int = 60, count: int = 10) -> pd.DataFrame:
        request = {"ticks_history": symbol, "adjust_start_time": 1, "count": count,
                  "end": "latest", "start": 1, "style": "candles", "granularity": granularity}
        response = await self.send_request(request)
        
        if response.get('candles'):
            df = pd.DataFrame(response['candles'])
            df['time'] = pd.to_datetime(df['epoch'], unit='s')
            for col in ['open', 'high', 'low', 'close']:
                df[col] = df[col].astype(float)
            return df
        return pd.DataFrame()
    
    async def buy_contract(self, symbol: str, contract_type: str, amount: float, 
                          barrier: float = None, duration: int = None, price: float = None) -> Dict:
        """
        Fixed buy_contract method with proper price parameter
        """
        # Get current price if not provided
        if price is None:
            price = await self.get_current_price(symbol)
            if price == 0.0:
                logging.error(f"Could not get price for {symbol}")
                return {"error": {"message": "Could not get current price"}}
        
        # Build the contract request with all required parameters
        request = {
            "buy": 1, 
            "parameters": {
                "contract_type": contract_type,
                "symbol": symbol, 
                "amount": round(amount, 2),  # Ensure amount is properly rounded
                "duration": duration or 1,
                "duration_unit": "m",
                "currency": "USD"
            },
            "price": round(price, 2)  # Price must have max 2 decimal places
        }
        
        # Add barrier for barrier contracts
        if barrier is not None:
            request["parameters"]["barrier"] = round(barrier, 2)  # Barrier also max 2 decimals
        
        logging.info(f"Sending buy request: {json.dumps(request, indent=2)}")
        
        response = await self.send_request(request)
        
        if response.get('buy'):
            logging.info(f"🚀 CONTRACT BOUGHT: {response['buy']['contract_id']} for ${amount}")
            return response
        elif response.get('error'):
            error_msg = response['error'].get('message', 'Unknown error')
            logging.error(f"Buy failed: {error_msg}")
            return response
        else:
            logging.error(f"Unexpected response: {response}")
            return {"error": {"message": "Unexpected response format"}}
    
    async def get_portfolio(self) -> Dict:
        response = await self.send_request({"portfolio": 1})
        return response.get('portfolio', {})
    
    async def disconnect(self):
        try:
            self.is_connected = False
            
            # Cancel ping task
            if self.ping_task and not self.ping_task.done():
                self.ping_task.cancel()
                try:
                    await self.ping_task
                except asyncio.CancelledError:
                    pass
            
            # Close websocket
            if self.websocket and not self.websocket.closed:
                await self.websocket.close()
                
            logging.info("🔌 Disconnected from Deriv API")
        except Exception as e:
            logging.error(f"❌ Disconnect error: {e}")

class DerivDonkeyBot:
    def __init__(self, config: Dict):
        self.config = config
        self.is_running = False
        self.deriv_api = DerivAPI(config['deriv_app_id'], config['deriv_api_token'])
        self.risk_manager = RiskManager(config.get('max_daily_loss_percent', 50.0))
        self.notification_manager = NotificationManager(config.get('email_config'), config.get('telegram_config'))
        self.technical_analyzer = TechnicalAnalyzer(self.deriv_api)
        self.symbols = config.get('symbols', ['R_10', 'R_25'])
        self.trade_count = 0
        
        self.init_database()
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    def init_database(self):
        try:
            conn = sqlite3.connect('deriv_donkey_trades.db')
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME, symbol TEXT,
                contract_type TEXT, amount REAL, buy_price REAL, sell_price REAL,
                profit_loss REAL, contract_id TEXT, status TEXT)''')
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"Database init error: {e}")
    
    def save_trade_to_db(self, trade_data: Dict):
        try:
            conn = sqlite3.connect('deriv_donkey_trades.db')
            cursor = conn.cursor()
            cursor.execute('''INSERT INTO trades (timestamp, symbol, contract_type, amount, 
                             buy_price, sell_price, profit_loss, contract_id, status)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                          (datetime.now(), trade_data.get('symbol', ''), trade_data.get('contract_type', ''),
                           trade_data.get('amount', 0), trade_data.get('buy_price', 0), trade_data.get('sell_price', 0),
                           trade_data.get('profit_loss', 0), trade_data.get('contract_id', ''), trade_data.get('status', 'open')))
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"Database save error: {e}")
    
    async def place_contract(self, signal: Dict) -> bool:
        try:
            current_balance = await self.deriv_api.get_balance()
            if not self.risk_manager.can_trade(current_balance):
                logging.warning("Cannot trade - insufficient balance or risk limit")
                return False
            
            risk_amount = self.risk_manager.calculate_position_size(current_balance)
            contract_type = "CALL" if signal['action'] == 'buy' else "PUT"
            
            logging.info(f"🎯 ATTEMPTING TRADE: {signal['action'].upper()} {signal['symbol']} ${risk_amount}")
            
            # Get current price first
            current_price = await self.deriv_api.get_current_price(signal['symbol'])
            if current_price == 0.0:
                logging.error(f"Cannot get price for {signal['symbol']}")
                return False
            
            # Use simple CALL/PUT contracts with duration and proper price
            response = await self.deriv_api.buy_contract(
                symbol=signal['symbol'], 
                contract_type=contract_type,
                amount=risk_amount, 
                duration=1,  # 1 minute contracts for ultra-fast trading
                price=current_price
            )
            
            if response.get('buy'):
                self.trade_count += 1
                trade_data = {
                    'symbol': signal['symbol'], 
                    'contract_type': contract_type,
                    'amount': response['buy']['buy_price'], 
                    'buy_price': response['buy']['buy_price'],
                    'contract_id': response['buy']['contract_id'], 
                    'status': 'open'
                }
                self.save_trade_to_db(trade_data)
                
                message = f"🚀 TRADE #{self.trade_count}\n{contract_type} {signal['symbol']}\n${response['buy']['buy_price']}\nID: {response['buy']['contract_id']}"
                self.notification_manager.notify("ULTRA TRADE", message)
                logging.info(f"✅ TRADE EXECUTED: #{self.trade_count}")
                return True
            else:
                error_msg = response.get('error', {}).get('message', 'Unknown error')
                logging.error(f"❌ TRADE FAILED: {error_msg}")
                return False
        except Exception as e:
            logging.error(f"Contract placement error: {e}")
        return False
    
    async def monitor_positions(self):
        try:
            portfolio = await self.deriv_api.get_portfolio()
            if portfolio.get('contracts'):
                for contract in portfolio['contracts']:
                    if contract.get('is_settled'):
                        profit_loss = contract.get('sell_price', 0) - contract.get('buy_price', 0)
                        
                        try:
                            conn = sqlite3.connect('deriv_donkey_trades.db')
                            cursor = conn.cursor()
                            cursor.execute('''UPDATE trades SET status = ?, sell_price = ?, profit_loss = ?
                                             WHERE contract_id = ?''',
                                          ('closed', contract.get('sell_price', 0), profit_loss, str(contract['contract_id'])))
                            conn.commit()
                            conn.close()
                        except:
                            pass
                        
                        if profit_loss < 0:
                            self.risk_manager.update_daily_loss(abs(profit_loss))
                        
                        status = "✅ WIN" if profit_loss > 0 else "❌ LOSS"
                        message = f"{status} ${profit_loss:.2f}\nID: {contract['contract_id']}"
                        self.notification_manager.notify("RESULT", message)
                        logging.info(f"📊 CONTRACT CLOSED: {status} ${profit_loss:.2f}")
        except Exception as e:
            logging.error(f"Position monitoring error: {e}")
    
    async def scan_for_signals_async(self):
        signals = []
        for symbol in self.symbols:
            try:
                signal = await self.technical_analyzer.generate_signal_async(symbol)
                if signal['action'] != 'none':
                    signals.append(signal)
                    logging.info(f"🔍 SIGNAL FOUND: {signal['action'].upper()} {symbol}")
            except Exception as e:
                logging.error(f"Signal scan error for {symbol}: {e}")
                # Force signal even on error for ultra-aggressive mode
                signals.append({
                    'symbol': symbol, 'action': 'buy', 'entry_price': 1.0,
                    'take_profit': 1.002, 'stop_loss': 0.998, 'confidence': 99
                })
        return signals
    
    async def run_trading_cycle(self):
        try:
            logging.info("🔄 RUNNING ULTRA AGGRESSIVE CYCLE...")
            
            # Ensure connection is stable before trading
            if not await self.deriv_api.ensure_connected():
                logging.error("❌ Connection check failed, skipping cycle")
                return
            
            await self.monitor_positions()
            signals = await self.scan_for_signals_async()
            
            for signal in signals:
                if signal['confidence'] >= 30:  # Very low threshold
                    success = await self.place_contract(signal)
                    if success:
                        await asyncio.sleep(2)  # Slight delay between trades
                    else:
                        logging.warning("⚠️ Trade placement failed")
                        # Small delay before next attempt
                        await asyncio.sleep(5)
                        
        except Exception as e:
            logging.error(f"Trading cycle error: {e}")
            # Add recovery delay
            await asyncio.sleep(10)
    
    async def start(self):
        logging.info("🚀 STARTING ULTRA AGGRESSIVE BOT...")
        if not await self.deriv_api.connect():
            logging.error("❌ CONNECTION FAILED")
            return False
        
        self.is_running = True
        self.notification_manager.notify("ULTRA BOT STARTED", "🔥 Ultra Aggressive Donkey Bot LIVE!")
        logging.info("✅ ULTRA AGGRESSIVE BOT STARTED")
        
        cycle_count = 0
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.is_running:
            try:
                cycle_count += 1
                logging.info(f"📈 CYCLE #{cycle_count} - HUNTING FOR TRADES...")
                await self.run_trading_cycle()
                consecutive_errors = 0  # Reset error count on success
                await asyncio.sleep(20)  # Slightly longer delay to prevent API overload
                
            except Exception as e:
                consecutive_errors += 1
                logging.error(f"❌ Main loop error #{consecutive_errors}: {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    logging.error(f"🚨 Too many consecutive errors ({consecutive_errors}), attempting reconnection...")
                    # Try to reconnect
                    if await self.deriv_api.connect():
                        consecutive_errors = 0
                        logging.info("✅ Reconnection successful")
                    else:
                        logging.error("❌ Reconnection failed, stopping bot")
                        break
                
                # Progressive backoff delay
                delay = min(30 + (consecutive_errors * 10), 120)
                await asyncio.sleep(delay)
        
        await self.deriv_api.disconnect()
    
    def stop(self):
        self.is_running = False
        self.notification_manager.notify("Bot Stopped", "🛑 Ultra Bot stopped")

def create_config():
    return {
        'deriv_app_id': os.getenv('DERIV_APP_ID', 'your_deriv_app_id'),
        'deriv_api_token': os.getenv('DERIV_API_TOKEN', 'your_deriv_api_token'),
        'max_daily_loss_percent': float(os.getenv('MAX_DAILY_LOSS', '50.0')),
        'symbols': os.getenv('TRADING_SYMBOLS', 'R_10,R_25').split(','),
        'telegram_config': {
            'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
            'chat_id': os.getenv('TELEGRAM_CHAT_ID')
        } if os.getenv('TELEGRAM_BOT_TOKEN') else None
    }

def run_flask():
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

async def main():
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    
    config = create_config()
    bot = DerivDonkeyBot(config)
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        bot.stop()

if __name__ == "__main__":
    asyncio.run(main())
