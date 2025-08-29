import asyncio
import websockets
import json
import pandas as pd
import numpy as np
import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import requests
from flask import Flask
import threading
import os
import time

app = Flask(__name__)

@app.route('/health')
def health():
    return {'status': 'running', 'bot_active': True}

class NotificationManager:
    def __init__(self, telegram_config: Dict = None):
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
        logging.info(f"{subject}: {message}")

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
    
    def calculate_position_size(self, account_balance: float, risk_per_trade: float = 2.0) -> float:
        calculated_risk = (account_balance * risk_per_trade / 100)
        min_trade = 0.35
        max_trade = min(account_balance * 0.1, 1.00)  # More conservative
        return max(min_trade, min(calculated_risk, max_trade))
    
    def update_daily_loss(self, loss_amount: float):
        if loss_amount > 0:
            self.current_daily_loss += loss_amount

class TechnicalAnalyzer:
    def __init__(self, deriv_api):
        self.deriv_api = deriv_api
        self.trade_counter = 0
    
    def detect_patterns(self, df: pd.DataFrame) -> Dict:
        if len(df) < 3:
            return {'pattern': 'instant_trade', 'signal': 'call'}
        
        close = df['close'].values
        current_price = close[-1]
        
        if len(close) >= 2:
            prev_price = close[-2]
            price_diff = current_price - prev_price
            
            if price_diff > 0.00005:
                return {'pattern': 'micro_up_fade', 'signal': 'put'}
            elif price_diff < -0.00005:
                return {'pattern': 'micro_down_fade', 'signal': 'call'}
        
        self.trade_counter += 1
        signal = 'call' if self.trade_counter % 2 == 0 else 'put'
        return {'pattern': 'alternating', 'signal': signal}
    
    async def generate_signal_async(self, symbol: str) -> Dict:
        signal = {'symbol': symbol, 'action': 'none', 'confidence': 0}
        
        try:
            df = await self.deriv_api._get_candles_async(symbol, granularity=60, count=5)
            if df.empty:
                return {'symbol': symbol, 'action': 'call', 'confidence': 90}
            
            patterns = self.detect_patterns(df)
            signal.update({
                'action': patterns['signal'], 
                'confidence': 85,
                'pattern': patterns['pattern']
            })
            
            logging.info(f"Generated {patterns['signal']} signal for {symbol}")
        except Exception as e:
            logging.error(f"Signal generation error for {symbol}: {e}")
            signal = {'symbol': symbol, 'action': 'call', 'confidence': 80}
        
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
        self.current_prices = {}
        self.connection_lock = asyncio.Lock()
        self.message_queue = asyncio.Queue()
        self.response_futures = {}
        self.is_listening = False
        
    async def connect(self):
        try:
            if self.websocket and not self.websocket.closed:
                await self.websocket.close()
            
            self.websocket = await asyncio.wait_for(
                websockets.connect(
                    self.ws_url,
                    ping_interval=30,
                    ping_timeout=10,
                    close_timeout=10
                ), 
                timeout=15
            )
            
            self.is_connected = True
            
            # Start message listener
            asyncio.create_task(self._message_listener())
            
            auth_success = await self.authorize()
            if auth_success:
                await self.get_balance()
                logging.info("✅ Connected and authorized with Deriv API")
                return True
            else:
                logging.error("❌ Authorization failed")
                return False
                
        except Exception as e:
            logging.error(f"❌ Connection failed: {e}")
            return False
    
    async def _message_listener(self):
        """Background task to handle incoming WebSocket messages"""
        self.is_listening = True
        try:
            while self.is_listening and self.websocket and not self.websocket.closed:
                try:
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    
                    req_id = data.get('req_id')
                    msg_type = data.get('msg_type', '')
                    
                    # Handle tick updates
                    if msg_type == 'tick' and 'tick' in data:
                        symbol = data['tick'].get('symbol', '')
                        price = data['tick'].get('quote', 0)
                        if symbol and price:
                            self.current_prices[symbol] = float(price)
                        continue
                    
                    # Handle responses to specific requests
                    if req_id and req_id in self.response_futures:
                        future = self.response_futures.pop(req_id)
                        if not future.done():
                            future.set_result(data)
                        continue
                    
                    # Log unhandled messages for debugging
                    if msg_type not in ['tick', 'ping', 'pong']:
                        logging.debug(f"Unhandled message: {msg_type}")
                        
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logging.error(f"Message listener error: {e}")
                    await asyncio.sleep(1)
                    
        except Exception as e:
            logging.error(f"Message listener crashed: {e}")
        finally:
            self.is_listening = False
    
    async def ensure_connected(self):
        if not self.is_connected or not self.websocket or self.websocket.closed:
            logging.warning("🔄 Connection lost, attempting reconnect...")
            return await self.connect()
        return True
    
    async def send_request(self, request: Dict, timeout: float = 15.0) -> Dict:
        async with self.connection_lock:
            if not await self.ensure_connected():
                return {"error": {"message": "Connection failed"}}
            
            try:
                self.req_id += 1
                request['req_id'] = self.req_id
                
                # Create future for response
                future = asyncio.Future()
                self.response_futures[self.req_id] = future
                
                # Send request
                await self.websocket.send(json.dumps(request))
                
                # Wait for response
                try:
                    response = await asyncio.wait_for(future, timeout=timeout)
                    return response
                except asyncio.TimeoutError:
                    # Clean up
                    if self.req_id in self.response_futures:
                        del self.response_futures[self.req_id]
                    return {"error": {"message": "Request timeout"}}
                
            except Exception as e:
                # Clean up on error
                if self.req_id in self.response_futures:
                    del self.response_futures[self.req_id]
                logging.error(f"❌ API request error: {e}")
                return {"error": {"message": str(e)}}
    
    async def authorize(self):
        response = await self.send_request({"authorize": self.api_token})
        if response.get('error'):
            logging.error(f"Authorization failed: {response['error']}")
            return False
        return True
    
    async def get_balance(self):
        response = await self.send_request({"balance": 1})
        if response.get('balance'):
            self.balance = response['balance']['balance']
            logging.info(f"Balance: ${self.balance}")
        return self.balance
    
    async def get_contracts_for(self, symbol: str) -> Dict:
        """Get available contracts for a symbol"""
        try:
            request = {"contracts_for": symbol}
            response = await self.send_request(request)
            return response
        except Exception as e:
            logging.error(f"Error getting contracts for {symbol}: {e}")
            return {}
    
    async def get_current_price(self, symbol: str) -> float:
        """Get current price using simple tick request (not subscription)"""
        try:
            # Use ticks without subscription for immediate price
            request = {"ticks": symbol}
            response = await self.send_request(request)
            
            if response.get('tick'):
                price = float(response['tick']['quote'])
                self.current_prices[symbol] = price
                logging.info(f"💰 Current price for {symbol}: {price}")
                return price
            
            if response.get('error'):
                logging.error(f"Error getting price for {symbol}: {response['error']}")
            
            return self.current_prices.get(symbol, 0.0)
            
        except Exception as e:
            logging.error(f"❌ Error getting price for {symbol}: {e}")
            return self.current_prices.get(symbol, 0.0)
    
    async def _get_candles_async(self, symbol: str, granularity: int = 60, count: int = 10) -> pd.DataFrame:
        request = {
            "ticks_history": symbol, 
            "adjust_start_time": 1, 
            "count": count,
            "end": "latest", 
            "start": 1, 
            "style": "candles", 
            "granularity": granularity
        }
        response = await self.send_request(request)
        
        if response.get('candles'):
            df = pd.DataFrame(response['candles'])
            df['time'] = pd.to_datetime(df['epoch'], unit='s')
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            return df
        return pd.DataFrame()
    
    async def buy_contract(self, symbol: str, contract_type: str, amount: float) -> Dict:
        """
        Enhanced buy_contract with proper validation
        """
        try:
            # Get current price first and validate
            current_price = await self.get_current_price(symbol)
            if current_price <= 0:
                return {"error": {"message": "Could not get valid current price"}}
            
            # Get contracts info to validate what's available
            contracts_info = await self.get_contracts_for(symbol)
            if contracts_info.get('error'):
                logging.error(f"Could not get contracts info: {contracts_info['error']}")
                return {"error": {"message": "Could not validate contracts"}}
            
            # Validate amount bounds
            amount = max(0.35, min(amount, 10.0))  # Ensure within reasonable bounds
            
            # Build contract request with all required parameters
            request = {
                "buy": 1, 
                "parameters": {
                    "contract_type": contract_type.upper(),
                    "symbol": symbol, 
                    "amount": round(amount, 2),
                    "duration": 1,
                    "duration_unit": "m",
                    "currency": "USD",
                    "basis": "stake"
                }
            }
            
            logging.info(f"🎯 Attempting to buy {contract_type} for {symbol}")
            logging.info(f"Current price: {current_price}, Stake: ${amount}")
            logging.debug(f"Buy request: {json.dumps(request, indent=2)}")
            
            response = await self.send_request(request, timeout=20.0)  # Longer timeout for buy
            
            if response.get('buy'):
                contract_id = response['buy']['contract_id']
                buy_price = response['buy']['buy_price']
                logging.info(f"✅ CONTRACT BOUGHT: {contract_id} for ${buy_price}")
                return response
            elif response.get('error'):
                error_msg = response['error'].get('message', 'Unknown error')
                error_code = response['error'].get('code', '')
                logging.error(f"❌ Buy failed: {error_msg} (Code: {error_code})")
                
                # Handle specific validation errors
                if 'price' in error_msg.lower():
                    logging.error(f"Price validation issue - Current price: {current_price}")
                
                return response
            else:
                logging.error(f"❌ Unexpected buy response format: {response}")
                return {"error": {"message": "Unexpected response format"}}
                
        except Exception as e:
            logging.error(f"❌ Contract purchase exception: {e}")
            return {"error": {"message": str(e)}}
    
    async def get_portfolio(self) -> Dict:
        response = await self.send_request({"portfolio": 1})
        return response.get('portfolio', {})
    
    async def disconnect(self):
        try:
            self.is_connected = False
            self.is_listening = False
            
            # Cancel all pending futures
            for req_id, future in self.response_futures.items():
                if not future.done():
                    future.cancel()
            self.response_futures.clear()
            
            if self.websocket and not self.websocket.closed:
                await self.websocket.close()
            logging.info("🔌 Disconnected from Deriv API")
        except Exception as e:
            logging.error(f"❌ Disconnect error: {e}")

class DerivTradingBot:
    def __init__(self, config: Dict):
        self.config = config
        self.is_running = False
        self.deriv_api = DerivAPI(config['deriv_app_id'], config['deriv_api_token'])
        self.risk_manager = RiskManager(config.get('max_daily_loss_percent', 30.0))
        self.notification_manager = NotificationManager(config.get('telegram_config'))
        self.technical_analyzer = TechnicalAnalyzer(self.deriv_api)
        self.symbols = config.get('symbols', ['R_25'])
        self.trade_count = 0
        self.last_trade_time = 0
        
        self.init_database()
        self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_bot.log'),
                logging.StreamHandler()
            ]
        )
    
    def init_database(self):
        try:
            conn = sqlite3.connect('deriv_trades.db')
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                timestamp DATETIME, 
                symbol TEXT,
                contract_type TEXT, 
                amount REAL, 
                buy_price REAL, 
                sell_price REAL,
                profit_loss REAL, 
                contract_id TEXT, 
                status TEXT
            )''')
            conn.commit()
            conn.close()
            logging.info("✅ Database initialized")
        except Exception as e:
            logging.error(f"❌ Database init error: {e}")
    
    def save_trade_to_db(self, trade_data: Dict):
        try:
            conn = sqlite3.connect('deriv_trades.db')
            cursor = conn.cursor()
            cursor.execute('''INSERT INTO trades (timestamp, symbol, contract_type, amount, 
                             buy_price, sell_price, profit_loss, contract_id, status)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                          (datetime.now(), trade_data.get('symbol', ''), 
                           trade_data.get('contract_type', ''),
                           trade_data.get('amount', 0), 
                           trade_data.get('buy_price', 0), 
                           trade_data.get('sell_price', 0),
                           trade_data.get('profit_loss', 0), 
                           trade_data.get('contract_id', ''), 
                           trade_data.get('status', 'open')))
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"❌ Database save error: {e}")
    
    async def place_contract(self, signal: Dict) -> bool:
        try:
            # Rate limiting - don't trade too frequently
            current_time = time.time()
            if current_time - self.last_trade_time < 60:  # Wait at least 1 minute between trades
                logging.info("⏰ Rate limiting - waiting between trades")
                return False
            
            current_balance = await self.deriv_api.get_balance()
            if not self.risk_manager.can_trade(current_balance):
                logging.warning("⚠️ Cannot trade - risk limit or low balance")
                return False
            
            stake_amount = self.risk_manager.calculate_position_size(current_balance)
            contract_type = signal['action'].upper()  # CALL or PUT
            
            # Additional validation
            if stake_amount < 0.35:
                logging.warning("⚠️ Calculated stake too low")
                return False
            
            logging.info(f"🎯 PLACING TRADE: {contract_type} {signal['symbol']} ${stake_amount}")
            logging.info(f"Current balance: ${current_balance}")
            
            response = await self.deriv_api.buy_contract(
                symbol=signal['symbol'], 
                contract_type=contract_type,
                amount=stake_amount
            )
            
            if response.get('buy'):
                self.trade_count += 1
                self.last_trade_time = current_time
                
                trade_data = {
                    'symbol': signal['symbol'], 
                    'contract_type': contract_type,
                    'amount': stake_amount, 
                    'buy_price': response['buy']['buy_price'],
                    'contract_id': response['buy']['contract_id'], 
                    'status': 'open'
                }
                self.save_trade_to_db(trade_data)
                
                message = f"🚀 TRADE #{self.trade_count}\n{contract_type} {signal['symbol']}\nStake: ${stake_amount}\nID: {response['buy']['contract_id']}"
                self.notification_manager.notify("NEW TRADE", message)
                return True
            else:
                error_msg = response.get('error', {}).get('message', 'Unknown error')
                logging.error(f"❌ TRADE FAILED: {error_msg}")
                
                # Handle specific errors
                if 'price' in error_msg.lower():
                    logging.error("Price validation failed - may need to wait for market conditions")
                elif 'balance' in error_msg.lower():
                    logging.error("Insufficient balance")
                
                return False
                
        except Exception as e:
            logging.error(f"❌ Contract placement error: {e}")
            return False
    
    async def monitor_positions(self):
        try:
            portfolio = await self.deriv_api.get_portfolio()
            if portfolio.get('contracts'):
                for contract in portfolio['contracts']:
                    if contract.get('is_settled') and contract.get('contract_id'):
                        profit_loss = contract.get('sell_price', 0) - contract.get('buy_price', 0)
                        
                        # Update database
                        try:
                            conn = sqlite3.connect('deriv_trades.db')
                            cursor = conn.cursor()
                            cursor.execute('''UPDATE trades SET status = ?, sell_price = ?, profit_loss = ?
                                             WHERE contract_id = ? AND status = 'open' ''',
                                          ('closed', contract.get('sell_price', 0), 
                                           profit_loss, str(contract['contract_id'])))
                            
                            if cursor.rowcount > 0:  # Only notify if we actually updated a record
                                conn.commit()
                                
                                if profit_loss < 0:
                                    self.risk_manager.update_daily_loss(abs(profit_loss))
                                
                                status = "✅ WIN" if profit_loss > 0 else "❌ LOSS"
                                message = f"{status} ${profit_loss:.2f}\nContract: {contract['contract_id']}"
                                self.notification_manager.notify("TRADE RESULT", message)
                            
                            conn.close()
                        except Exception as db_e:
                            logging.error(f"DB update error: {db_e}")
                        
        except Exception as e:
            logging.error(f"❌ Position monitoring error: {e}")
    
    async def scan_for_signals_async(self):
        signals = []
        for symbol in self.symbols:
            try:
                signal = await self.technical_analyzer.generate_signal_async(symbol)
                if signal['action'] != 'none':
                    signals.append(signal)
                    logging.info(f"🔍 SIGNAL: {signal['action'].upper()} {symbol} (confidence: {signal['confidence']}%)")
                    
                # Small delay between symbol processing
                await asyncio.sleep(1)
                
            except Exception as e:
                logging.error(f"❌ Signal scan error for {symbol}: {e}")
        return signals
    
    async def run_trading_cycle(self):
        try:
            logging.info("🔄 Running trading cycle...")
            
            if not await self.deriv_api.ensure_connected():
                logging.error("❌ Connection check failed")
                return
            
            # Monitor existing positions first
            await self.monitor_positions()
            
            # Get new signals
            signals = await self.scan_for_signals_async()
            
            # Execute trades with higher confidence threshold
            for signal in signals:
                if signal['confidence'] >= 80:  # Higher threshold
                    success = await self.place_contract(signal)
                    if success:
                        await asyncio.sleep(5)  # Delay between successful trades
                        break  # Only one trade per cycle
                    else:
                        await asyncio.sleep(15)  # Longer delay on failure
                        
        except Exception as e:
            logging.error(f"❌ Trading cycle error: {e}")
            await asyncio.sleep(30)
    
    async def start(self):
        logging.info("🚀 STARTING TRADING BOT...")
        
        if not await self.deriv_api.connect():
            logging.error("❌ CONNECTION FAILED")
            return False
        
        self.is_running = True
        self.notification_manager.notify("BOT STARTED", "🤖 Trading Bot is now LIVE!")
        
        cycle_count = 0
        error_count = 0
        max_errors = 5
        
        while self.is_running:
            try:
                cycle_count += 1
                logging.info(f"📈 CYCLE #{cycle_count}")
                
                await self.run_trading_cycle()
                error_count = 0  # Reset on success
                
                # Longer wait between cycles to be more conservative
                await asyncio.sleep(60)  # Wait 1 minute between cycles
                
            except Exception as e:
                error_count += 1
                logging.error(f"❌ Main loop error #{error_count}: {e}")
                
                if error_count >= max_errors:
                    logging.error("🚨 Too many errors, attempting reconnection...")
                    if await self.deriv_api.connect():
                        error_count = 0
                        logging.info("✅ Reconnection successful")
                    else:
                        logging.error("❌ Reconnection failed, stopping bot")
                        break
                
                await asyncio.sleep(120)  # Wait 2 minutes on error
        
        await self.deriv_api.disconnect()
        return True
    
    def stop(self):
        self.is_running = False
        self.notification_manager.notify("BOT STOPPED", "🛑 Trading bot stopped")
        logging.info("🛑 Bot stopped")

def create_config():
    return {
        'deriv_app_id': os.getenv('DERIV_APP_ID', '1089'),  # Default to 1089 if not set
        'deriv_api_token': os.getenv('DERIV_API_TOKEN', 'your_api_token_here'),
        'max_daily_loss_percent': float(os.getenv('MAX_DAILY_LOSS', '20.0')),  # More conservative
        'symbols': os.getenv('TRADING_SYMBOLS', 'R_25').split(','),
        'telegram_config': {
            'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
            'chat_id': os.getenv('TELEGRAM_CHAT_ID')
        } if os.getenv('TELEGRAM_BOT_TOKEN') else None
    }

def run_flask():
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

async def main():
    # Start Flask in background
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    # Create and start bot
    config = create_config()
    bot = DerivTradingBot(config)
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        logging.info("🛑 Bot interrupted by user")
        bot.stop()
    except Exception as e:
        logging.error(f"❌ Bot crashed: {e}")
        bot.stop()

if __name__ == "__main__":
    asyncio.run(main())
