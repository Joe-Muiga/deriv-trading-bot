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
    def __init__(self, max_daily_loss_percent: float = 30.0):
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
        if current_balance < 1.0:  # Higher minimum balance
            return False
        if self.daily_start_balance == 0:
            return True
        daily_loss_percent = (self.current_daily_loss / self.daily_start_balance) * 100
        return daily_loss_percent < self.max_daily_loss_percent
    
    def calculate_position_size(self, account_balance: float) -> float:
        # Very conservative position sizing
        min_trade = 0.35
        max_trade = min(account_balance * 0.05, 1.00)  # Only 5% of balance, max $1
        return max(min_trade, max_trade)
    
    def update_daily_loss(self, loss_amount: float):
        if loss_amount > 0:
            self.current_daily_loss += loss_amount

class TechnicalAnalyzer:
    def __init__(self, deriv_api):
        self.deriv_api = deriv_api
        self.trade_counter = 0
    
    async def generate_signal_async(self, symbol: str) -> Dict:
        """Simple alternating strategy to reduce complexity"""
        self.trade_counter += 1
        action = 'call' if self.trade_counter % 2 == 0 else 'put'
        
        return {
            'symbol': symbol, 
            'action': action, 
            'confidence': 85,
            'pattern': 'alternating'
        }

class DerivAPI:
    def __init__(self, app_id: str, api_token: str):
        self.app_id = app_id
        self.api_token = api_token
        self.ws_url = f"wss://ws.binaryws.com/websockets/v3?app_id={app_id}"
        self.websocket = None
        self.is_connected = False
        self.balance = 0
        self.req_id = 0
        self.connection_lock = asyncio.Lock()
        self.response_futures = {}
        self.is_listening = False
        self.contract_prices = {}  # Cache for contract pricing
        
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
        """Enhanced message listener with better error handling"""
        self.is_listening = True
        try:
            while self.is_listening and self.websocket and not self.websocket.closed:
                try:
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=2.0)
                    data = json.loads(message)
                    
                    req_id = data.get('req_id')
                    msg_type = data.get('msg_type', '')
                    
                    # Route response to correct future
                    if req_id and req_id in self.response_futures:
                        future = self.response_futures.pop(req_id)
                        if not future.done():
                            future.set_result(data)
                        continue
                    
                    # Handle unmatched messages
                    if msg_type not in ['tick', 'ping', 'pong']:
                        logging.debug(f"Unhandled message type: {msg_type}")
                        
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logging.warning("WebSocket connection closed in listener")
                    break
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
    
    async def send_request(self, request: Dict, timeout: float = 30.0) -> Dict:
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
                logging.debug(f"Sent request: {json.dumps(request)}")
                
                # Wait for response
                try:
                    response = await asyncio.wait_for(future, timeout=timeout)
                    logging.debug(f"Received response: {json.dumps(response)}")
                    return response
                except asyncio.TimeoutError:
                    if self.req_id in self.response_futures:
                        del self.response_futures[self.req_id]
                    return {"error": {"message": f"Request timeout after {timeout}s"}}
                
            except Exception as e:
                if self.req_id in self.response_futures:
                    del self.response_futures[self.req_id]
                logging.error(f"❌ API request error: {e}")
                return {"error": {"message": str(e)}}
    
    async def authorize(self):
        response = await self.send_request({"authorize": self.api_token})
        if response.get('error'):
            logging.error(f"Authorization failed: {response['error']}")
            return False
        logging.info("✅ Successfully authorized")
        return True
    
    async def get_balance(self):
        response = await self.send_request({"balance": 1})
        if response.get('balance'):
            self.balance = float(response['balance']['balance'])
            logging.info(f"💰 Balance: ${self.balance}")
        return self.balance
    
    async def get_proposal(self, symbol: str, contract_type: str, stake: float) -> Dict:
        """Get proposal with pricing before buying - this is crucial for price validation"""
        try:
            proposal_request = {
                "proposal": 1,
                "amount": stake,
                "basis": "stake",
                "contract_type": contract_type.upper(),
                "currency": "USD",
                "duration": 1,
                "duration_unit": "m",
                "symbol": symbol
            }
            
            logging.info(f"Getting proposal for {contract_type} {symbol} with stake ${stake}")
            response = await self.send_request(proposal_request, timeout=15.0)
            
            if response.get('proposal'):
                proposal = response['proposal']
                logging.info(f"✅ Proposal received: ask_price={proposal.get('ask_price', 'N/A')}")
                return response
            elif response.get('error'):
                logging.error(f"❌ Proposal failed: {response['error']}")
                return response
            else:
                logging.error(f"❌ Unexpected proposal response: {response}")
                return {"error": {"message": "Unexpected proposal response"}}
                
        except Exception as e:
            logging.error(f"❌ Proposal request error: {e}")
            return {"error": {"message": str(e)}}
    
    async def buy_contract_with_proposal(self, symbol: str, contract_type: str, stake: float) -> Dict:
        """Buy contract using proposal ID - this eliminates price validation issues"""
        try:
            # Step 1: Get proposal first
            proposal_response = await self.get_proposal(symbol, contract_type, stake)
            
            if proposal_response.get('error'):
                return proposal_response
            
            if not proposal_response.get('proposal'):
                return {"error": {"message": "No proposal received"}}
            
            proposal = proposal_response['proposal']
            proposal_id = proposal.get('id')
            ask_price = proposal.get('ask_price')
            
            if not proposal_id:
                return {"error": {"message": "No proposal ID received"}}
            
            logging.info(f"🎯 Buying contract with proposal ID: {proposal_id}")
            logging.info(f"Expected cost: ${ask_price}")
            
            # Step 2: Buy using proposal ID
            buy_request = {
                "buy": proposal_id,
                "price": ask_price  # Use the exact price from proposal
            }
            
            response = await self.send_request(buy_request, timeout=20.0)
            
            if response.get('buy'):
                contract_id = response['buy']['contract_id']
                buy_price = response['buy']['buy_price']
                logging.info(f"✅ CONTRACT BOUGHT: {contract_id} for ${buy_price}")
                return response
            elif response.get('error'):
                error_msg = response['error'].get('message', 'Unknown error')
                error_code = response['error'].get('code', '')
                logging.error(f"❌ Buy failed: {error_msg} (Code: {error_code})")
                return response
            else:
                logging.error(f"❌ Unexpected buy response: {response}")
                return {"error": {"message": "Unexpected buy response format"}}
                
        except Exception as e:
            logging.error(f"❌ Contract purchase exception: {e}")
            return {"error": {"message": str(e)}}
    
    async def get_portfolio(self) -> Dict:
        response = await self.send_request({"portfolio": 1}, timeout=10.0)
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
        self.risk_manager = RiskManager(config.get('max_daily_loss_percent', 20.0))
        self.notification_manager = NotificationManager(config.get('telegram_config'))
        self.technical_analyzer = TechnicalAnalyzer(self.deriv_api)
        self.symbols = config.get('symbols', ['R_25'])
        self.trade_count = 0
        self.last_trade_time = 0
        self.consecutive_failures = 0
        
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
            # Rate limiting with exponential backoff on failures
            current_time = time.time()
            min_wait = 120 + (self.consecutive_failures * 60)  # Increase wait time with failures
            
            if current_time - self.last_trade_time < min_wait:
                logging.info(f"⏰ Rate limiting - waiting {min_wait}s between trades")
                return False
            
            # Check balance
            current_balance = await self.deriv_api.get_balance()
            if not self.risk_manager.can_trade(current_balance):
                logging.warning(f"⚠️ Cannot trade - balance: ${current_balance}")
                return False
            
            # Calculate stake
            stake_amount = self.risk_manager.calculate_position_size(current_balance)
            contract_type = signal['action'].upper()
            
            if stake_amount < 0.35:
                logging.warning(f"⚠️ Calculated stake too low: ${stake_amount}")
                return False
            
            logging.info(f"🎯 PLACING TRADE: {contract_type} {signal['symbol']} ${stake_amount}")
            logging.info(f"Current balance: ${current_balance}")
            
            # Use proposal-based buying to eliminate price validation issues
            response = await self.deriv_api.buy_contract_with_proposal(
                symbol=signal['symbol'], 
                contract_type=contract_type,
                stake=stake_amount
            )
            
            if response.get('buy'):
                self.trade_count += 1
                self.last_trade_time = current_time
                self.consecutive_failures = 0  # Reset failure counter on success
                
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
                self.consecutive_failures += 1
                error_msg = response.get('error', {}).get('message', 'Unknown error')
                logging.error(f"❌ TRADE FAILED #{self.consecutive_failures}: {error_msg}")
                
                # Send failure notification
                failure_msg = f"Trade Failed: {error_msg}\nConsecutive failures: {self.consecutive_failures}"
                self.notification_manager.notify("TRADE FAILURE", failure_msg)
                
                return False
                
        except Exception as e:
            self.consecutive_failures += 1
            logging.error(f"❌ Contract placement exception #{self.consecutive_failures}: {e}")
            return False
    
    async def monitor_positions(self):
        try:
            portfolio = await self.deriv_api.get_portfolio()
            if portfolio.get('contracts'):
                for contract in portfolio['contracts']:
                    if contract.get('is_settled') and contract.get('contract_id'):
                        profit_loss = float(contract.get('sell_price', 0)) - float(contract.get('buy_price', 0))
                        
                        # Update database
                        try:
                            conn = sqlite3.connect('deriv_trades.db')
                            cursor = conn.cursor()
                            cursor.execute('''UPDATE trades SET status = ?, sell_price = ?, profit_loss = ?
                                             WHERE contract_id = ? AND status = 'open' ''',
                                          ('closed', contract.get('sell_price', 0), 
                                           profit_loss, str(contract['contract_id'])))
                            
                            if cursor.rowcount > 0:
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
    
    async def run_trading_cycle(self):
        try:
            logging.info("🔄 Running trading cycle...")
            
            if not await self.deriv_api.ensure_connected():
                logging.error("❌ Connection check failed")
                return
            
            # Monitor existing positions
            await self.monitor_positions()
            
            # Only trade if we haven't had too many consecutive failures
            if self.consecutive_failures >= 5:
                logging.warning(f"⚠️ Too many consecutive failures ({self.consecutive_failures}), skipping trading")
                return
            
            # Generate and execute signal
            for symbol in self.symbols:
                try:
                    signal = await self.technical_analyzer.generate_signal_async(symbol)
                    if signal['action'] != 'none':
                        logging.info(f"🔍 SIGNAL: {signal['action'].upper()} {symbol}")
                        
                        success = await self.place_contract(signal)
                        if success:
                            break  # Only one trade per cycle
                        
                        # Break after first attempt regardless of success
                        break
                        
                except Exception as e:
                    logging.error(f"❌ Signal processing error for {symbol}: {e}")
                        
        except Exception as e:
            logging.error(f"❌ Trading cycle error: {e}")
    
    async def start(self):
        logging.info("🚀 STARTING TRADING BOT...")
        
        # Test connection first
        if not await self.deriv_api.connect():
            logging.error("❌ INITIAL CONNECTION FAILED")
            return False
        
        # Test authorization and balance
        balance = await self.deriv_api.get_balance()
        if balance <= 0:
            logging.error(f"❌ INSUFFICIENT BALANCE: ${balance}")
            return False
        
        self.is_running = True
        self.notification_manager.notify("BOT STARTED", f"🤖 Trading Bot LIVE! Balance: ${balance}")
        
        cycle_count = 0
        error_count = 0
        max_errors = 3
        
        while self.is_running:
            try:
                cycle_count += 1
                logging.info(f"📈 CYCLE #{cycle_count} (Failures: {self.consecutive_failures})")
                
                await self.run_trading_cycle()
                error_count = 0  # Reset on success
                
                # Progressive wait times based on failure count
                wait_time = 180 + (self.consecutive_failures * 60)  # 3+ minutes base
                logging.info(f"⏱️ Waiting {wait_time}s until next cycle")
                await asyncio.sleep(wait_time)
                
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
                
                await asyncio.sleep(300)  # Wait 5 minutes on error
        
        await self.deriv_api.disconnect()
        return True
    
    def stop(self):
        self.is_running = False
        self.notification_manager.notify("BOT STOPPED", "🛑 Trading bot stopped")
        logging.info("🛑 Bot stopped")

def create_config():
    return {
        'deriv_app_id': os.getenv('DERIV_APP_ID', '1089'),
        'deriv_api_token': os.getenv('DERIV_API_TOKEN', 'your_api_token_here'),
        'max_daily_loss_percent': float(os.getenv('MAX_DAILY_LOSS', '15.0')),  # Very conservative
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
