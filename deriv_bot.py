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

# Flask health check
app = Flask(__name__)

@app.route('/health')
def health():
    return {'status': 'running', 'bot_active': True}

class TradingStrategyAnalyzer:
    def __init__(self):
        self.strategies = {
            'support_resistance': {'entry_offset': 0.0003, 'tp_ratio': 1.5, 'sl_ratio': 0.8},
            'chart_patterns': {'entry_offset': 0.0005, 'tp_ratio': 1.3, 'sl_ratio': 0.8},
            'indicators': {'entry_offset': 0.0002, 'tp_ratio': 1.2, 'sl_ratio': 0.8},
            'smart_money': {'entry_offset': 0.0008, 'tp_ratio': 2.0, 'sl_ratio': 0.8},
            'ict_concepts': {'entry_offset': 0.0006, 'tp_ratio': 1.8, 'sl_ratio': 0.8}
        }
    
    def calculate_donkey_parameters(self) -> Dict:
        total = len(self.strategies)
        avg_entry = sum(s['entry_offset'] for s in self.strategies.values()) / total
        avg_tp = sum(s['tp_ratio'] for s in self.strategies.values()) / total
        avg_sl = sum(s['sl_ratio'] for s in self.strategies.values()) / total
        
        return {
            'entry_offset': round(avg_entry, 6),
            'tp_distance': round(avg_sl * 0.6, 6),
            'sl_distance': round(avg_sl * 0.4, 6),
            'original_avg_tp': round(avg_tp, 2),
            'original_avg_sl': round(avg_sl, 2)
        }

class NotificationManager:
    def __init__(self, email_config: Dict = None, telegram_config: Dict = None):
        self.email_config = email_config
        self.telegram_config = telegram_config
        
    def send_email(self, subject: str, message: str):
        if not self.email_config:
            return
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = self.email_config['to_email']
            msg['Subject'] = subject
            msg.attach(MIMEText(message, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['from_email'], self.email_config['password'])
            server.send_message(msg)
            server.quit()
        except Exception as e:
            logging.error(f"Email failed: {e}")
    
    def send_telegram(self, message: str):
        if not self.telegram_config:
            return
        try:
            url = f"https://api.telegram.org/bot{self.telegram_config['bot_token']}/sendMessage"
            data = {'chat_id': self.telegram_config['chat_id'], 'text': message}
            requests.post(url, data=data)
        except Exception as e:
            logging.error(f"Telegram failed: {e}")
    
    def notify(self, subject: str, message: str):
        self.send_email(subject, message)
        self.send_telegram(f"*{subject}*\n{message}")

class RiskManager:
    def __init__(self, max_daily_loss_percent: float = 2.0):
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
        if self.daily_start_balance == 0:
            return True
        daily_loss_percent = (self.current_daily_loss / self.daily_start_balance) * 100
        return daily_loss_percent < self.max_daily_loss_percent
    
    def calculate_position_size(self, account_balance: float, risk_per_trade: float = 1.0) -> float:  # Increased to 1%
        calculated_risk = (account_balance * risk_per_trade / 100)
        min_trade = 0.35
        max_trade = 2.00  # Increased max trade size
        return max(min_trade, min(calculated_risk, max_trade))
    
    def update_daily_loss(self, loss_amount: float):
        if loss_amount > 0:
            self.current_daily_loss += loss_amount

class TechnicalAnalyzer:
    def __init__(self, deriv_api):
        self.deriv_api = deriv_api
        self.event_loop = None
    
    def detect_patterns(self, df: pd.DataFrame) -> Dict:
        if len(df) < 10:
            return {'pattern': 'none', 'signal': 'none'}
        
        # Super aggressive contrarian signals
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # Multiple aggressive signal types
        current_price = close[-1]
        prev_price = close[-2] if len(close) > 1 else current_price
        price_change = (current_price - prev_price) / prev_price * 100
        
        # 5-period moving average for quick signals
        if len(close) >= 5:
            ma5 = np.mean(close[-5:])
            ma3 = np.mean(close[-3:])
            
            # Contrarian: Price above MA = SELL, below MA = BUY
            if current_price > ma5 * 1.001:  # Even 0.1% above MA
                return {'pattern': 'overbought_contrarian', 'signal': 'sell'}
            if current_price < ma5 * 0.999:  # Even 0.1% below MA
                return {'pattern': 'oversold_contrarian', 'signal': 'buy'}
        
        # Momentum contrarian - fade strong moves
        if price_change > 0.05:  # Any upward move > 0.05%
            return {'pattern': 'momentum_fade', 'signal': 'sell'}
        if price_change < -0.05:  # Any downward move > 0.05%
            return {'pattern': 'momentum_fade', 'signal': 'buy'}
        
        # Range contrarian - trade against range extremes
        recent_high = np.max(high[-5:])
        recent_low = np.min(low[-5:])
        range_size = recent_high - recent_low
        
        if range_size > 0:
            if current_price > (recent_low + range_size * 0.6):  # Upper 40% of range
                return {'pattern': 'range_top', 'signal': 'sell'}
            if current_price < (recent_low + range_size * 0.4):  # Lower 40% of range  
                return {'pattern': 'range_bottom', 'signal': 'buy'}
        
        # Default contrarian on any price movement
        if len(close) >= 3:
            if close[-1] > close[-2]:
                return {'pattern': 'up_contrarian', 'signal': 'sell'}
            if close[-1] < close[-2]:
                return {'pattern': 'down_contrarian', 'signal': 'buy'}
        
        return {'pattern': 'sideways', 'signal': 'buy'}  # Default to buy when sideways
    
    async def generate_signal_async(self, symbol: str) -> Dict:
        signal = {'symbol': symbol, 'action': 'none', 'entry_price': 0, 
                 'take_profit': 0, 'stop_loss': 0, 'confidence': 0}
        
        try:
            df = await self.deriv_api._get_candles_async(symbol, granularity=60, count=50)
            if df.empty:
                return signal
            
            patterns = self.detect_patterns(df)
            current_price = df['close'].iloc[-1]
            donkey_params = TradingStrategyAnalyzer().calculate_donkey_parameters()
            
            # Super aggressive entry - reduced offsets
            entry_offset = donkey_params['entry_offset'] * 0.3  # Much smaller offset
            tp_distance = donkey_params['tp_distance'] * 0.8    # Smaller TP
            sl_distance = donkey_params['sl_distance'] * 1.2    # Bigger SL for more room
            
            if patterns['signal'] == 'buy':
                entry_price = current_price + entry_offset
                take_profit = entry_price + tp_distance
                stop_loss = entry_price - sl_distance
                signal.update({'action': 'buy', 'entry_price': round(entry_price, 5),
                              'take_profit': round(take_profit, 5), 'stop_loss': round(stop_loss, 5), 'confidence': 85})
                
            elif patterns['signal'] == 'sell':
                entry_price = current_price - entry_offset
                take_profit = entry_price - tp_distance
                stop_loss = entry_price + sl_distance
                signal.update({'action': 'sell', 'entry_price': round(entry_price, 5),
                              'take_profit': round(take_profit, 5), 'stop_loss': round(stop_loss, 5), 'confidence': 85})
        except Exception as e:
            logging.error(f"Signal generation error for {symbol}: {e}")
        
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
        
    async def connect(self):
        try:
            self.websocket = await websockets.connect(self.ws_url)
            self.is_connected = True
            await self.authorize()
            await self.get_balance()
            logging.info("Connected to Deriv API")
            return True
        except Exception as e:
            logging.error(f"Connection failed: {e}")
            return False
    
    async def send_request(self, request: Dict) -> Dict:
        if not self.is_connected:
            return {}
        try:
            self.req_id += 1
            request['req_id'] = self.req_id
            await self.websocket.send(json.dumps(request))
            response = await self.websocket.recv()
            return json.loads(response)
        except Exception as e:
            logging.error(f"API request error: {e}")
            return {}
    
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
    
    async def _get_candles_async(self, symbol: str, granularity: int = 60, count: int = 100) -> pd.DataFrame:
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
                          barrier: float = None, duration: int = None) -> Dict:
        request = {"buy": 1, "parameters": {"contract_type": contract_type, "symbol": symbol, "amount": amount}}
        if barrier:
            request["parameters"]["barrier"] = str(barrier)
        if duration:
            request["parameters"]["duration"] = duration
            request["parameters"]["duration_unit"] = "m"
        
        response = await self.send_request(request)
        if response.get('buy'):
            logging.info(f"Contract bought: {response['buy']['contract_id']}")
        return response
    
    async def get_portfolio(self) -> Dict:
        response = await self.send_request({"portfolio": 1})
        return response.get('portfolio', {})
    
    async def disconnect(self):
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False

class DerivDonkeyBot:
    def __init__(self, config: Dict):
        self.config = config
        self.is_running = False
        self.deriv_api = DerivAPI(config['deriv_app_id'], config['deriv_api_token'])
        self.risk_manager = RiskManager(config.get('max_daily_loss_percent', 2.0))
        self.notification_manager = NotificationManager(config.get('email_config'), config.get('telegram_config'))
        self.technical_analyzer = TechnicalAnalyzer(self.deriv_api)
        self.symbols = config.get('symbols', ['R_10', 'R_25'])
        
        self.init_database()
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    def init_database(self):
        conn = sqlite3.connect('deriv_donkey_trades.db')
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME, symbol TEXT,
            contract_type TEXT, amount REAL, buy_price REAL, sell_price REAL,
            profit_loss REAL, contract_id TEXT, status TEXT)''')
        conn.commit()
        conn.close()
    
    def save_trade_to_db(self, trade_data: Dict):
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
    
    async def place_contract(self, signal: Dict) -> bool:
        try:
            current_balance = await self.deriv_api.get_balance()
            if not self.risk_manager.can_trade(current_balance):
                logging.warning("Daily loss limit reached")
                return False
            
            risk_amount = self.risk_manager.calculate_position_size(current_balance)
            contract_type = "CALLE" if signal['action'] == 'buy' else "PUTE"
            
            response = await self.deriv_api.buy_contract(
                symbol=signal['symbol'], contract_type=contract_type,
                amount=risk_amount, barrier=signal['take_profit'], duration=5)
            
            if response.get('buy'):
                trade_data = {'symbol': signal['symbol'], 'contract_type': contract_type,
                             'amount': response['buy']['buy_price'], 'buy_price': response['buy']['buy_price'],
                             'contract_id': response['buy']['contract_id'], 'status': 'open'}
                self.save_trade_to_db(trade_data)
                
                message = f"🚀 CONTRACT BOUGHT\nSymbol: {signal['symbol']}\nType: {contract_type}\nAmount: ${response['buy']['buy_price']}\nID: {response['buy']['contract_id']}"
                self.notification_manager.notify("New Contract", message)
                return True
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
                        conn = sqlite3.connect('deriv_donkey_trades.db')
                        cursor = conn.cursor()
                        cursor.execute('''UPDATE trades SET status = ?, sell_price = ?, profit_loss = ?
                                         WHERE contract_id = ?''',
                                      ('closed', contract.get('sell_price', 0), profit_loss, str(contract['contract_id'])))
                        conn.commit()
                        conn.close()
                        
                        if profit_loss < 0:
                            self.risk_manager.update_daily_loss(abs(profit_loss))
                        
                        status = "✅ PROFIT" if profit_loss > 0 else "❌ LOSS"
                        message = f"{status}\nP/L: ${profit_loss:.2f}\nID: {contract['contract_id']}"
                        self.notification_manager.notify("Contract Finished", message)
        except Exception as e:
            logging.error(f"Position monitoring error: {e}")
    
    async def scan_for_signals_async(self):
        signals = []
        for symbol in self.symbols:
            try:
                signal = await self.technical_analyzer.generate_signal_async(symbol)
                if signal['action'] != 'none':
                    signals.append(signal)
            except Exception as e:
                logging.error(f"Signal generation error for {symbol}: {e}")
        return signals
    
    async def run_trading_cycle(self):
        try:
            await self.monitor_positions()
            signals = await self.scan_for_signals_async()
            for signal in signals:
                if signal['confidence'] >= 50:  # Much lower threshold
                    await self.place_contract(signal)
                    await asyncio.sleep(2)  # Slightly longer delay between trades
        except Exception as e:
            logging.error(f"Trading cycle error: {e}")
    
    async def start(self):
        if not await self.deriv_api.connect():
            return False
        
        self.is_running = True
        self.notification_manager.notify("Bot Started", "🐴 AGGRESSIVE Donkey Strategy Bot is running!")
        logging.info("Aggressive bot started successfully")
        
        while self.is_running:
            try:
                await self.run_trading_cycle()
                await asyncio.sleep(30)  # Check every 30 seconds instead of 60
            except Exception as e:
                logging.error(f"Main loop error: {e}")
                await asyncio.sleep(30)
        
        await self.deriv_api.disconnect()
    
    def stop(self):
        self.is_running = False
        self.notification_manager.notify("Bot Stopped", "🛑 Bot has been stopped")

def create_config():
    return {
        'deriv_app_id': os.getenv('DERIV_APP_ID', 'your_deriv_app_id'),
        'deriv_api_token': os.getenv('DERIV_API_TOKEN', 'your_deriv_api_token'),
        'max_daily_loss_percent': float(os.getenv('MAX_DAILY_LOSS', '2.0')),
        'symbols': os.getenv('TRADING_SYMBOLS', 'R_10,R_25').split(','),
        'email_config': {
            'from_email': os.getenv('EMAIL_FROM'),
            'to_email': os.getenv('EMAIL_TO'),
            'password': os.getenv('EMAIL_PASSWORD'),
            'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': int(os.getenv('SMTP_PORT', '587'))
        } if os.getenv('EMAIL_FROM') else None,
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
