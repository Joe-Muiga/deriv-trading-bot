import asyncio
import websockets
import json
import logging
import sqlite3
from datetime import datetime
from typing import Dict
import requests
import os
import time

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
        self.setup_logging()
        self.init_db()
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    def init_db(self):
        conn = sqlite3.connect('contrarian_trades.db')
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY, timestamp DATETIME, symbol TEXT,
            contract_type TEXT, amount REAL, contract_id TEXT, status TEXT
        )''')
        conn.commit()
        conn.close()
    
    def notify(self, message: str):
        logging.info(f"📱 {message}")
        if os.getenv('TELEGRAM_BOT_TOKEN'):
            try:
                url = f"https://api.telegram.org/bot{os.getenv('TELEGRAM_BOT_TOKEN')}/sendMessage"
                data = {'chat_id': os.getenv('TELEGRAM_CHAT_ID'), 'text': message}
                requests.post(url, data=data, timeout=5)
            except:
                pass
    
    async def connect(self):
        try:
            if self.websocket:
                await self.websocket.close()
            
            self.websocket = await websockets.connect(self.ws_url, ping_interval=30)
            self.is_connected = True
            asyncio.create_task(self._listen())
            
            # Authorize
            auth_resp = await self.send_request({"authorize": self.api_token})
            if auth_resp.get('error'):
                return False
            
            # Get balance
            balance_resp = await self.send_request({"balance": 1})
            if balance_resp.get('balance'):
                self.balance = float(balance_resp['balance']['balance'])
                logging.info(f"💰 Balance: ${self.balance}")
            
            return True
        except Exception as e:
            logging.error(f"Connection failed: {e}")
            return False
    
    async def _listen(self):
        try:
            while self.is_connected and self.websocket:
                message = await self.websocket.recv()
                data = json.loads(message)
                req_id = data.get('req_id')
                if req_id and req_id in self.response_futures:
                    future = self.response_futures.pop(req_id)
                    if not future.done():
                        future.set_result(data)
        except:
            self.is_connected = False
    
    async def send_request(self, request: Dict, timeout: float = 15.0) -> Dict:
        try:
            self.req_id += 1
            request['req_id'] = self.req_id
            
            future = asyncio.Future()
            self.response_futures[self.req_id] = future
            
            await self.websocket.send(json.dumps(request))
            return await asyncio.wait_for(future, timeout=timeout)
        except Exception as e:
            logging.error(f"Request failed: {e}")
            return {"error": {"message": str(e)}}
    
    def generate_contrarian_signal(self) -> str:
        """CONTRARIAN LOGIC: Opposite of original alternating pattern"""
        self.trade_counter += 1
        # Original was: 'call' if even, 'put' if odd
        # Contrarian: 'put' if even, 'call' if odd
        return 'put' if self.trade_counter % 2 == 0 else 'call'
    
    def calculate_stake(self) -> float:
        """CONTRARIAN: More aggressive position sizing than original"""
        # Original: 5% max, very conservative
        # Contrarian: 8-12% of balance, more aggressive
        min_stake = 0.35
        max_stake = min(self.balance * 0.12, 2.0)  # Up to 12% or $2 max
        return max(min_stake, max_stake)
    
    async def place_trade(self, signal: str, symbol: str = 'R_25') -> bool:
        """CONTRARIAN: Faster trading frequency"""
        current_time = time.time()
        # Original: 120s + failures wait
        # Contrarian: Only 60s wait (more frequent)
        if current_time - self.last_trade_time < 60:
            return False
        
        if self.balance < 1.0:
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
            if proposal_resp.get('error') or not proposal_resp.get('proposal'):
                return False
            
            proposal = proposal_resp['proposal']
            proposal_id = proposal['id']
            ask_price = proposal['ask_price']
            
            # Buy contract
            buy_resp = await self.send_request({"buy": proposal_id, "price": ask_price})
            
            if buy_resp.get('buy'):
                self.last_trade_time = current_time
                contract_id = buy_resp['buy']['contract_id']
                
                # Save to DB
                conn = sqlite3.connect('contrarian_trades.db')
                cursor = conn.cursor()
                cursor.execute('''INSERT INTO trades (timestamp, symbol, contract_type, amount, contract_id, status)
                                 VALUES (?, ?, ?, ?, ?, ?)''',
                              (datetime.now(), symbol, signal, stake, contract_id, 'open'))
                conn.commit()
                conn.close()
                
                self.notify(f"🚀 CONTRARIAN TRADE: {signal.upper()} {symbol} ${stake} - ID: {contract_id}")
                return True
            
        except Exception as e:
            logging.error(f"Trade failed: {e}")
        
        return False
    
    async def monitor_positions(self):
        try:
            portfolio_resp = await self.send_request({"portfolio": 1})
            if portfolio_resp.get('portfolio', {}).get('contracts'):
                for contract in portfolio_resp['portfolio']['contracts']:
                    if contract.get('is_settled'):
                        contract_id = str(contract['contract_id'])
                        profit_loss = float(contract.get('sell_price', 0)) - float(contract.get('buy_price', 0))
                        
                        # Update DB
                        conn = sqlite3.connect('contrarian_trades.db')
                        cursor = conn.cursor()
                        cursor.execute('''UPDATE trades SET status = ? WHERE contract_id = ? AND status = 'open' ''',
                                      ('closed', contract_id))
                        if cursor.rowcount > 0:
                            conn.commit()
                            status = "✅ WIN" if profit_loss > 0 else "❌ LOSS"
                            self.notify(f"{status} ${profit_loss:.2f} - Contract: {contract_id}")
                        conn.close()
        except Exception as e:
            logging.error(f"Monitor error: {e}")
    
    async def trading_cycle(self):
        try:
            if not self.is_connected:
                if not await self.connect():
                    return
            
            # Update balance
            balance_resp = await self.send_request({"balance": 1})
            if balance_resp.get('balance'):
                self.balance = float(balance_resp['balance']['balance'])
            
            # Monitor positions
            await self.monitor_positions()
            
            # Generate contrarian signal and trade
            signal = self.generate_contrarian_signal()
            success = await self.place_trade(signal)
            
            if success:
                logging.info(f"✅ Contrarian trade executed: {signal}")
            
        except Exception as e:
            logging.error(f"Cycle error: {e}")
    
    async def start(self):
        logging.info("🚀 STARTING CONTRARIAN BOT...")
        
        if not await self.connect():
            logging.error("❌ Connection failed")
            return
        
        if self.balance <= 0:
            logging.error(f"❌ Insufficient balance: ${self.balance}")
            return
        
        self.is_running = True
        self.notify(f"🤖 CONTRARIAN Bot LIVE! Balance: ${self.balance}")
        
        cycle = 0
        while self.is_running:
            try:
                cycle += 1
                logging.info(f"📈 CONTRARIAN CYCLE #{cycle}")
                
                await self.trading_cycle()
                
                # CONTRARIAN: Shorter wait time (90s vs original 180s+)
                await asyncio.sleep(90)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logging.error(f"Main loop error: {e}")
                await asyncio.sleep(60)
        
        await self.websocket.close() if self.websocket else None
        self.notify("🛑 CONTRARIAN Bot Stopped")

async def main():
    bot = ContrariaTradingBot()
    try:
        await bot.start()
    except KeyboardInterrupt:
        bot.is_running = False
        print("Bot stopped by user")

if __name__ == "__main__":
    asyncio.run(main())
