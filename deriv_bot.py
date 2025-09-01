import asyncio
import websockets
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Optional
import requests
import os
import time
import signal
import sys
from threading import Thread
from flask import Flask, jsonify
import gc

class ContrariaTradingBot:
    def __init__(self):
        self.app_id = os.getenv('DERIV_APP_ID', '1089')
        self.api_token = os.getenv('DERIV_API_TOKEN', 'your_token_here')
        self.ws_url = f"wss://ws.binaryws.com/websockets/v3?app_id={self.app_id}"
        self.websocket = None
        self.is_connected = False
        self.balance = 0
        self.req_id = 0
        self.response_futures = {}
        self.is_running = False
        self.trade_counter = 0
        self.last_trade_time = 0
        self.restart_count = 0
        self.max_restarts = int(os.getenv('MAX_RESTARTS', '100'))  # Maximum restarts before stopping
        self.start_time = datetime.now()
        self.last_heartbeat = time.time()
        self.connection_retry_count = 0
        self.max_connection_retries = 10
        self.memory_cleanup_counter = 0
        self.setup_logging()
        self.init_db()
        self.setup_signal_handlers()
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('bot.log'),
                logging.StreamHandler()
            ]
        )
        # Limit log file size
        if os.path.exists('bot.log') and os.path.getsize('bot.log') > 10*1024*1024:  # 10MB
            os.rename('bot.log', 'bot_old.log')
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        def signal_handler(signum, frame):
            logging.info(f"Received signal {signum}, shutting down gracefully...")
            self.is_running = False
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def init_db(self):
        try:
            conn = sqlite3.connect('contrarian_trades.db', timeout=10)
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY, timestamp DATETIME, symbol TEXT,
                contract_type TEXT, amount REAL, contract_id TEXT, status TEXT
            )''')
            
            # Add bot_stats table for monitoring
            cursor.execute('''CREATE TABLE IF NOT EXISTS bot_stats (
                id INTEGER PRIMARY KEY, timestamp DATETIME, restart_count INTEGER,
                uptime_hours REAL, total_trades INTEGER, balance REAL
            )''')
            
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"Database init error: {e}")
    
    def notify(self, message: str, force: bool = False):
        """Enhanced notification with rate limiting"""
        current_time = time.time()
        if not force and hasattr(self, 'last_notification_time'):
            if current_time - self.last_notification_time < 30:  # Rate limit notifications
                return
        
        self.last_notification_time = current_time
        logging.info(f"📱 {message}")
        
        if os.getenv('TELEGRAM_BOT_TOKEN'):
            try:
                url = f"https://api.telegram.org/bot{os.getenv('TELEGRAM_BOT_TOKEN')}/sendMessage"
                data = {'chat_id': os.getenv('TELEGRAM_CHAT_ID'), 'text': message}
                requests.post(url, data=data, timeout=10)
            except Exception as e:
                logging.warning(f"Notification failed: {e}")
    
    async def connect_with_retry(self) -> bool:
        """Enhanced connection with exponential backoff"""
        for attempt in range(self.max_connection_retries):
            try:
                if self.websocket:
                    try:
                        await self.websocket.close()
                    except:
                        pass
                
                # Exponential backoff
                if attempt > 0:
                    wait_time = min(2 ** attempt, 300)  # Max 5 minutes
                    logging.info(f"Retrying connection in {wait_time}s (attempt {attempt + 1}/{self.max_connection_retries})")
                    await asyncio.sleep(wait_time)
                
                logging.info(f"Connecting to Deriv API (attempt {attempt + 1})...")
                self.websocket = await websockets.connect(
                    self.ws_url, 
                    ping_interval=30, 
                    ping_timeout=10,
                    close_timeout=10
                )
                self.is_connected = True
                self.connection_retry_count = 0
                
                # Start listening task
                asyncio.create_task(self._listen())
                
                # Authorize
                auth_resp = await self.send_request({"authorize": self.api_token})
                if auth_resp.get('error'):
                    logging.error(f"Authorization failed: {auth_resp['error']}")
                    continue
                
                # Get balance
                balance_resp = await self.send_request({"balance": 1})
                if balance_resp.get('balance'):
                    self.balance = float(balance_resp['balance']['balance'])
                    logging.info(f"💰 Connected! Balance: ${self.balance}")
                    return True
                
            except Exception as e:
                logging.error(f"Connection attempt {attempt + 1} failed: {e}")
                self.is_connected = False
        
        logging.error("All connection attempts failed")
        return False
    
    async def _listen(self):
        """Enhanced listener with connection monitoring"""
        try:
            while self.is_connected and self.websocket:
                try:
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=60)
                    data = json.loads(message)
                    self.last_heartbeat = time.time()
                    
                    req_id = data.get('req_id')
                    if req_id and req_id in self.response_futures:
                        future = self.response_futures.pop(req_id)
                        if not future.done():
                            future.set_result(data)
                            
                except asyncio.TimeoutError:
                    # Send ping to check connection
                    try:
                        await self.websocket.ping()
                    except:
                        logging.warning("Connection lost - ping failed")
                        break
                        
                except websockets.exceptions.ConnectionClosed:
                    logging.warning("WebSocket connection closed")
                    break
                except Exception as e:
                    logging.error(f"Listen error: {e}")
                    break
                    
        except Exception as e:
            logging.error(f"Listener crashed: {e}")
        finally:
            self.is_connected = False
    
    async def send_request(self, request: Dict, timeout: float = 20.0) -> Dict:
        """Enhanced request with better error handling"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if not self.is_connected or not self.websocket:
                    raise Exception("Not connected")
                
                self.req_id += 1
                request['req_id'] = self.req_id
                
                future = asyncio.Future()
                self.response_futures[self.req_id] = future
                
                await self.websocket.send(json.dumps(request))
                result = await asyncio.wait_for(future, timeout=timeout)
                
                return result
                
            except asyncio.TimeoutError:
                logging.warning(f"Request timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                return {"error": {"message": "Request timeout"}}
                
            except Exception as e:
                logging.error(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                return {"error": {"message": str(e)}}
        
        return {"error": {"message": "Max retries exceeded"}}
    
    def generate_contrarian_signal(self) -> str:
        """CONTRARIAN LOGIC: Opposite of original alternating pattern"""
        self.trade_counter += 1
        return 'put' if self.trade_counter % 2 == 0 else 'call'
    
    def calculate_stake(self) -> float:
        """CONTRARIAN: More aggressive position sizing than original"""
        min_stake = 0.35
        max_stake = min(self.balance * 0.12, 2.0)
        return max(min_stake, max_stake)
    
    async def place_trade(self, signal: str, symbol: str = 'R_25') -> bool:
        """Enhanced trade placement with better error handling"""
        current_time = time.time()
        
        if current_time - self.last_trade_time < 60:
            return False
        
        if self.balance < 1.0:
            logging.warning(f"Insufficient balance: ${self.balance}")
            return False
        
        stake = self.calculate_stake()
        
        try:
            # Get proposal
            proposal_req = {
                "proposal": 1, "amount": stake, "basis": "stake",
                "contract_type": signal.upper(), "currency": "USD",
                "duration": 1, "duration_unit": "m", "symbol": symbol
            }
            
            proposal_resp = await self.send_request(proposal_req)
            if proposal_resp.get('error'):
                logging.error(f"Proposal error: {proposal_resp['error']}")
                return False
            
            if not proposal_resp.get('proposal'):
                logging.error("No proposal received")
                return False
            
            proposal = proposal_resp['proposal']
            proposal_id = proposal['id']
            ask_price = proposal['ask_price']
            
            # Buy contract
            buy_resp = await self.send_request({"buy": proposal_id, "price": ask_price})
            
            if buy_resp.get('error'):
                logging.error(f"Buy error: {buy_resp['error']}")
                return False
            
            if buy_resp.get('buy'):
                self.last_trade_time = current_time
                contract_id = buy_resp['buy']['contract_id']
                
                # Save to DB with retry
                for db_attempt in range(3):
                    try:
                        conn = sqlite3.connect('contrarian_trades.db', timeout=10)
                        cursor = conn.cursor()
                        cursor.execute('''INSERT INTO trades (timestamp, symbol, contract_type, amount, contract_id, status)
                                         VALUES (?, ?, ?, ?, ?, ?)''',
                                      (datetime.now(), symbol, signal, stake, contract_id, 'open'))
                        conn.commit()
                        conn.close()
                        break
                    except Exception as db_e:
                        logging.error(f"DB error (attempt {db_attempt + 1}): {db_e}")
                        if db_attempt < 2:
                            await asyncio.sleep(1)
                
                self.notify(f"🚀 CONTRARIAN TRADE: {signal.upper()} {symbol} ${stake} - ID: {contract_id}")
                return True
            
        except Exception as e:
            logging.error(f"Trade placement failed: {e}")
        
        return False
    
    async def monitor_positions(self):
        """Enhanced position monitoring"""
        try:
            portfolio_resp = await self.send_request({"portfolio": 1})
            if portfolio_resp.get('error'):
                logging.error(f"Portfolio error: {portfolio_resp['error']}")
                return
            
            if portfolio_resp.get('portfolio', {}).get('contracts'):
                for contract in portfolio_resp['portfolio']['contracts']:
                    if contract.get('is_settled'):
                        contract_id = str(contract['contract_id'])
                        profit_loss = float(contract.get('sell_price', 0)) - float(contract.get('buy_price', 0))
                        
                        # Update DB
                        try:
                            conn = sqlite3.connect('contrarian_trades.db', timeout=10)
                            cursor = conn.cursor()
                            cursor.execute('''UPDATE trades SET status = ? WHERE contract_id = ? AND status = 'open' ''',
                                          ('closed', contract_id))
                            if cursor.rowcount > 0:
                                conn.commit()
                                status = "✅ WIN" if profit_loss > 0 else "❌ LOSS"
                                self.notify(f"{status} ${profit_loss:.2f} - Contract: {contract_id}")
                            conn.close()
                        except Exception as db_e:
                            logging.error(f"DB update error: {db_e}")
                            
        except Exception as e:
            logging.error(f"Monitor error: {e}")
    
    def cleanup_memory(self):
        """Periodic memory cleanup"""
        self.memory_cleanup_counter += 1
        if self.memory_cleanup_counter % 50 == 0:  # Every 50 cycles
            # Clear old futures
            current_time = time.time()
            old_futures = [req_id for req_id, future in self.response_futures.items() 
                          if future.done() or current_time - req_id > 300]  # 5 minutes old
            
            for req_id in old_futures:
                self.response_futures.pop(req_id, None)
            
            # Force garbage collection
            gc.collect()
            logging.info(f"Memory cleanup completed. Active futures: {len(self.response_futures)}")
    
    def save_bot_stats(self):
        """Save bot statistics"""
        try:
            uptime = (datetime.now() - self.start_time).total_seconds() / 3600
            conn = sqlite3.connect('contrarian_trades.db', timeout=10)
            cursor = conn.cursor()
            cursor.execute('''INSERT INTO bot_stats (timestamp, restart_count, uptime_hours, total_trades, balance)
                             VALUES (?, ?, ?, ?, ?)''',
                          (datetime.now(), self.restart_count, uptime, self.trade_counter, self.balance))
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"Stats save error: {e}")
    
    async def health_check(self) -> bool:
        """Check if bot is healthy"""
        try:
            # Check connection age
            if time.time() - self.last_heartbeat > 300:  # 5 minutes
                logging.warning("Connection seems stale, reconnecting...")
                return False
            
            # Check balance
            balance_resp = await self.send_request({"balance": 1})
            if balance_resp.get('balance'):
                self.balance = float(balance_resp['balance']['balance'])
                return True
            
            return False
        except Exception as e:
            logging.error(f"Health check failed: {e}")
            return False
    
    async def trading_cycle(self):
        """Enhanced trading cycle with health checks"""
        try:
            # Health check every few cycles
            if self.trade_counter % 5 == 0:
                if not await self.health_check():
                    logging.warning("Health check failed, reconnecting...")
                    self.is_connected = False
                    return
            
            # Monitor positions
            await self.monitor_positions()
            
            # Generate contrarian signal and trade
            signal = self.generate_contrarian_signal()
            success = await self.place_trade(signal)
            
            if success:
                logging.info(f"✅ Contrarian trade executed: {signal}")
            
            # Periodic cleanup
            self.cleanup_memory()
            
            # Save stats occasionally
            if self.trade_counter % 10 == 0:
                self.save_bot_stats()
            
        except Exception as e:
            logging.error(f"Trading cycle error: {e}")
    
    async def start_with_auto_restart(self):
        """Main loop with automatic restart capability"""
        logging.info("🚀 STARTING CONTRARIAN BOT WITH AUTO-RESTART...")
        
        while self.restart_count < self.max_restarts:
            try:
                # Connect with retry
                if not await self.connect_with_retry():
                    logging.error("❌ Could not establish connection")
                    self.restart_count += 1
                    await asyncio.sleep(60)
                    continue
                
                if self.balance <= 0:
                    logging.error(f"❌ Insufficient balance: ${self.balance}")
                    await asyncio.sleep(300)  # Wait 5 minutes before retry
                    continue
                
                self.is_running = True
                uptime = (datetime.now() - self.start_time).total_seconds() / 3600
                
                self.notify(f"🤖 CONTRARIAN Bot {'RESTARTED' if self.restart_count > 0 else 'STARTED'}! "
                          f"Balance: ${self.balance} | Uptime: {uptime:.1f}h | Restarts: {self.restart_count}", 
                          force=True)
                
                cycle = 0
                consecutive_errors = 0
                
                while self.is_running and self.restart_count < self.max_restarts:
                    try:
                        cycle += 1
                        
                        if cycle % 10 == 1:  # Log every 10 cycles
                            logging.info(f"📈 CONTRARIAN CYCLE #{cycle} (Restart #{self.restart_count})")
                        
                        if not self.is_connected:
                            logging.warning("Connection lost, attempting reconnection...")
                            break
                        
                        await self.trading_cycle()
                        consecutive_errors = 0
                        
                        # CONTRARIAN: Shorter wait time (90s vs original 180s+)
                        await asyncio.sleep(90)
                        
                    except asyncio.CancelledError:
                        logging.info("Bot cancelled")
                        break
                    except KeyboardInterrupt:
                        logging.info("Bot stopped by user")
                        self.is_running = False
                        break
                    except Exception as e:
                        consecutive_errors += 1
                        logging.error(f"Cycle error #{consecutive_errors}: {e}")
                        
                        if consecutive_errors >= 5:
                            logging.error("Too many consecutive errors, restarting...")
                            break
                        
                        await asyncio.sleep(min(60 * consecutive_errors, 300))  # Exponential backoff
                
            except Exception as e:
                logging.error(f"Main loop error: {e}")
            
            finally:
                # Cleanup
                self.is_running = False
                self.is_connected = False
                if self.websocket:
                    try:
                        await self.websocket.close()
                    except:
                        pass
                
                if self.restart_count < self.max_restarts:
                    self.restart_count += 1
                    restart_delay = min(30 + (self.restart_count * 10), 300)
                    logging.info(f"🔄 Restarting in {restart_delay}s (Restart #{self.restart_count}/{self.max_restarts})")
                    await asyncio.sleep(restart_delay)
        
        logging.error(f"❌ Maximum restarts ({self.max_restarts}) reached. Bot stopped.")
        self.notify(f"🛑 CONTRARIAN Bot stopped after {self.max_restarts} restarts", force=True)

# Health check server for Render
app = Flask(__name__)
bot_instance = None

@app.route('/health')
def health():
    global bot_instance
    if bot_instance and bot_instance.is_running:
        uptime = (datetime.now() - bot_instance.start_time).total_seconds()
        return jsonify({
            'status': 'healthy',
            'uptime_seconds': uptime,
            'restart_count': bot_instance.restart_count,
            'balance': bot_instance.balance,
            'total_trades': bot_instance.trade_counter,
            'connected': bot_instance.is_connected
        })
    return jsonify({'status': 'unhealthy'}), 503

@app.route('/')
def root():
    return jsonify({'message': 'Contrarian Trading Bot is running'})

def run_health_server():
    """Run health check server in separate thread"""
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

async def main():
    global bot_instance
    
    # Start health check server
    health_thread = Thread(target=run_health_server, daemon=True)
    health_thread.start()
    
    bot_instance = ContrariaTradingBot()
    
    try:
        await bot_instance.start_with_auto_restart()
    except KeyboardInterrupt:
        bot_instance.is_running = False
        logging.info("Bot stopped by user")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
    finally:
        if bot_instance.websocket:
            await bot_instance.websocket.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot stopped by user")
    except Exception as e:
        print(f"Fatal error: {e}")
