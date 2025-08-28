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
    
    def calculate_position_size(self, account_balance: float, risk_per_trade: float = 5.0) -> float:
        calculated_risk = (account_balance * risk_per_trade / 100)
        min_trade = 0.35
        max_trade = min(account_balance * 0.5, 3.00)
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
            return {'pattern': 'instant_trade', 'signal': 'buy'}
        
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
    
    async def ensure_connected(self):
        if not self.is_connected or not self.websocket or self.websocket.closed:
            logging.warning("🔄 Connection lost, attempting reconnect...")
            return await self.connect()
        return True
    
    async def send_request(self, request: Dict) -> Dict:
        async with self.connection_lock:
            if not await self.ensure_connected():
                return {"error": {"message": "Connection failed"}}
            
            try:
                self.req_id += 1
                request['req_id'] = self.req_id
                
                await asyncio.wait_for(
                    self.websocket.send(json.dumps(request)), 
                    timeout=10
                )
                
                response_str = await asyncio.wait_for(
                    self.websocket.recv(), 
                    timeout=15
                )
                
                response = json.loads(response_str)
                return response
                
            except Exception as e:
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
            logging.info(f"Balance: {self.balance}")
        return self.balance
    
    async def get_current_price(self, symbol: str) -> float:
        try:
            request = {"ticks": symbol, "subscribe": 1}
            response = await self.send_request(request)
            
            if response.get('tick'):
                price = float(response['tick']['quote'])
                self.current_prices[symbol] = price
                logging.info(f"💰 Current price for {symbol}: {price}")
                return price
            
            if symbol in self.current_prices:
                return self.current_prices[symbol]
            
            return 0.0
            
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
        Fixed buy_contract with proper stake parameter
        """
        try:
            # Ensure we have current price
            current_price = await self.get_current_price(symbol)
            if current_price <= 0:
                return {"error": {"message": "Could not get current price"}}
            
            # Build proper contract request with stake (not payout)
            request = {
                "buy": 1, 
                "parameters": {
                    "contract_type": contract_type.upper(),  # CALL or PUT
                    "symbol": symbol, 
                    "amount": round(amount, 2),  # This is the stake amount
                    "duration": 1,
                    "duration_unit": "m",
                    "currency": "USD",
                    "basis": "stake"  # Important: specify this is stake-based
                }
            }
            
            logging.info(f"🎯 Buying {contract_type} for {symbol} with stake ${amount}")
            logging.debug(f"Buy request: {json.dumps(request, indent=2)}")
            
            response = await self.send_request(request)
            
            if response.get('buy'):
                contract_id = response['buy']['contract_id']
                buy_price = response['buy']['buy_price']
                logging.info(f"✅ CONTRACT BOUGHT: {contract_id} for ${buy_price}")
                return response
            elif response.get('error'):
                error_msg = response['error'].get('message', 'Unknown error')
                logging.error(f"❌ Buy failed: {error_msg}")
                return response
            else:
                logging.error(f"❌ Unexpected response: {response}")
                return {"error": {"message": "Unexpected response format"}}
                
        except Exception as e:
            logging.error(f"❌ Contract purchase error: {e}")
            return {"error": {"message": str(e)}}
    
    async def get_portfolio(self) -> Dict:
        response = await self.send_request({"portfolio": 1})
        return response.get('portfolio', {})
    
    async def disconnect(self):
        try:
            self.is_connected = False
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
        self.risk_manager = RiskManager(config.get('max_daily_loss_percent', 50.0))
        self.notification_manager = NotificationManager(config.get('telegram_config'))
        self.technical_analyzer = TechnicalAnalyzer(self.deriv_api)
        self.symbols = config.get('symbols', ['R_25'])
        self.trade_count = 0
        
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
            current_balance = await self.deriv_api.get_balance()
            if not self.risk_manager.can_trade(current_balance):
                logging.warning("⚠️ Cannot trade - risk limit or low balance")
                return False
            
            stake_amount = self.risk_manager.calculate_position_size(current_balance)
            contract_type = signal['action'].upper()  # CALL or PUT
            
            logging.info(f"🎯 PLACING TRADE: {contract_type} {signal['symbol']} ${stake_amount}")
            
            response = await self.deriv_api.buy_contract(
                symbol=signal['symbol'], 
                contract_type=contract_type,
                amount=stake_amount
            )
            
            if response.get('buy'):
                self.trade_count += 1
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
                                             WHERE contract_id = ?''',
                                          ('closed', contract.get('sell_price', 0), 
                                           profit_loss, str(contract['contract_id'])))
                            conn.commit()
                            conn.close()
                        except Exception as db_e:
                            logging.error(f"DB update error: {db_e}")
                        
                        if profit_loss < 0:
                            self.risk_manager.update_daily_loss(abs(profit_loss))
                        
                        status = "✅ WIN" if profit_loss > 0 else "❌ LOSS"
                        message = f"{status} ${profit_loss:.2f}\nContract: {contract['contract_id']}"
                        self.notification_manager.notify("TRADE RESULT", message)
                        
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
            except Exception as e:
                logging.error(f"❌ Signal scan error for {symbol}: {e}")
        return signals
    
    async def run_trading_cycle(self):
        try:
            logging.info("🔄 Running trading cycle...")
            
            if not await self.deriv_api.ensure_connected():
                logging.error("❌ Connection check failed")
                return
            
            await self.monitor_positions()
            signals = await self.scan_for_signals_async()
            
            for signal in signals:
                if signal['confidence'] >= 70:  # Only high confidence signals
                    success = await self.place_contract(signal)
                    if success:
                        await asyncio.sleep(3)  # Delay between trades
                    else:
                        await asyncio.sleep(10)  # Longer delay on failure
                        
        except Exception as e:
            logging.error(f"❌ Trading cycle error: {e}")
            await asyncio.sleep(15)
    
    async def start(self):
        logging.info("🚀 STARTING TRADING BOT...")
        
        if not await self.deriv_api.connect():
            logging.error("❌ CONNECTION FAILED")
            return False
        
        self.is_running = True
        self.notification_manager.notify("BOT STARTED", "🤖 Trading Bot is now LIVE!")
        
        cycle_count = 0
        error_count = 0
        max_errors = 3
        
        while self.is_running:
            try:
                cycle_count += 1
                logging.info(f"📈 CYCLE #{cycle_count}")
                await self.run_trading_cycle()
                error_count = 0  # Reset on success
                await asyncio.sleep(30)  # Wait 30 seconds between cycles
                
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
                
                await asyncio.sleep(60)  # Wait 1 minute on error
        
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
        'max_daily_loss_percent': float(os.getenv('MAX_DAILY_LOSS', '30.0')),
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
