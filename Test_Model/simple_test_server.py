# complete_simple_server.py - เวอร์ชันสมบูรณ์
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
import uvicorn
import json
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import MT5 Connector
try:
    from mt5_connector import MT5Connector
    MT5_AVAILABLE = True
    logger.info("✅ MT5 Connector imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import MT5 Connector: {e}")
    MT5_AVAILABLE = False

# FastAPI App
app = FastAPI(title="AI Recovery Trading System - Simple", version="1.0.0")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
mt5_connector = None
fake_ai_running = False
fake_positions = []

# Pydantic Models
class TradingSettingsModel(BaseModel):
    account_balance: float = 5000.0
    daily_target_pct: float = 2.0
    monthly_target_pct: float = 20.0
    initial_lot_size: float = 0.01
    enabled_pairs: List[str] = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'XAUUSD']

@app.get("/")
async def root():
    """หน้าแรก - Dashboard เต็มรูปแบบ"""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Recovery Trading System</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .gradient-bg {
            background: linear-gradient(135deg, #0f172a 0%, #4c1d95 50%, #0f172a 100%);
        }
        .card {
            background: rgba(30, 41, 59, 0.8);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(71, 85, 105, 0.3);
        }
        .btn-primary {
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
            transition: all 0.3s ease;
        }
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(59, 130, 246, 0.3);
        }
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }
        .status-online { background: #10b981; animation: pulse 2s infinite; }
        .status-offline { background: #ef4444; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
    </style>
</head>
<body class="gradient-bg min-h-screen">
    <div class="container mx-auto p-4 max-w-7xl">
        <!-- Header -->
        <div class="mb-6">
            <div class="flex items-center justify-between">
                <div>
                    <h1 class="text-3xl font-bold text-white flex items-center gap-3">
                        🧠 AI Recovery Trading Brain (Simple Mode)
                        <span id="connectionStatus" class="status-indicator status-online"></span>
                    </h1>
                    <p class="text-gray-300 mt-2">
                        ระบบ AI เทรดแบบแก้ไม้อัตโนมัติ - Simple Test Version
                        <span id="connectionText" class="ml-2 text-xs px-2 py-1 rounded bg-green-600">LIVE</span>
                    </p>
                </div>
                <div class="flex gap-3">
                    <button id="startAI" class="btn-primary px-6 py-3 rounded-lg text-white font-semibold">
                        เริ่ม AI (Demo)
                    </button>
                    <button id="stopAI" class="btn-primary px-6 py-3 rounded-lg text-white font-semibold" disabled>
                        หยุด AI
                    </button>
                </div>
            </div>
        </div>

        <!-- Status Cards -->
        <div class="grid grid-cols-1 md:grid-cols-5 gap-4 mb-6">
            <div class="card rounded-lg p-4">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-gray-400 text-sm">สถานะ AI</p>
                        <p id="aiStrategy" class="text-white font-semibold">Standby</p>
                    </div>
                    <div id="aiStatusIndicator" class="w-3 h-3 rounded-full bg-red-500"></div>
                </div>
            </div>
            <div class="card rounded-lg p-4">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-gray-400 text-sm">Balance</p>
                        <p id="accountBalance" class="text-white font-semibold">$5,000.00</p>
                    </div>
                    <div class="text-blue-400">💰</div>
                </div>
            </div>
            <div class="card rounded-lg p-4">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-gray-400 text-sm">Equity</p>
                        <p id="accountEquity" class="text-white font-semibold">$5,000.00</p>
                    </div>
                    <div class="text-green-400">📈</div>
                </div>
            </div>
            <div class="card rounded-lg p-4">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-gray-400 text-sm">กำไรวันนี้</p>
                        <p id="dailyPnL" class="text-white font-semibold">$0.00</p>
                    </div>
                    <div class="text-yellow-400">📊</div>
                </div>
            </div>
            <div class="card rounded-lg p-4">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-gray-400 text-sm">โพซิชั่นเปิด</p>
                        <p id="activePositions" class="text-white font-semibold">0</p>
                        <p class="text-gray-400 text-xs">Risk: <span id="riskLevel">LOW</span></p>
                    </div>
                    <div class="text-purple-400">🛡️</div>
                </div>
            </div>
        </div>

        <!-- Main Panel -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <!-- MT5 Connection Test -->
            <div class="card rounded-lg p-6">
                <h3 class="text-white text-lg font-semibold mb-4">🔗 MT5 Connection Test</h3>
                <button id="testMT5" class="btn-primary px-4 py-2 rounded-lg text-white font-semibold mb-4">
                    Test MT5 Connection
                </button>
                <div id="mt5Result" class="text-gray-400">
                    Click button to test MT5 connection...
                </div>
            </div>

            <!-- System Logs -->
            <div class="card rounded-lg p-6">
                <h3 class="text-white text-lg font-semibold mb-4">📋 System Logs</h3>
                <div id="logsContainer" class="space-y-2 max-h-64 overflow-y-auto">
                    <div class="p-2 bg-slate-700 rounded text-sm">
                        <span class="text-gray-400">${new Date().toLocaleTimeString()}</span>
                        <span class="text-green-400 ml-2">System ready - Simple test mode</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <div class="mt-6 card rounded-lg p-4">
            <div class="flex items-center gap-2 text-blue-400 mb-2">
                <span>ℹ️</span>
                <span class="font-semibold">Simple Test Mode</span>
            </div>
            <ul class="text-gray-400 text-sm space-y-1">
                <li>• นี่คือเวอร์ชันทดสอบง่ายๆ ไม่มี ML logging</li>
                <li>• ระบบสามารถเชื่อมต่อ MT5 และดึงข้อมูลบัญชีได้</li>
                <li>• AI ยังไม่ได้เทรดจริง เป็นเพียงการทดสอบระบบ</li>
            </ul>
        </div>
    </div>

    <script>
        let aiRunning = false;

        // Utility Functions
        function formatCurrency(amount) {
            return new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD'
            }).format(amount);
        }

        function addLog(message, type = 'info') {
            const timestamp = new Date().toLocaleTimeString();
            const logContainer = document.getElementById('logsContainer');
            
            const logEntry = document.createElement('div');
            logEntry.className = 'p-2 bg-slate-700 rounded text-sm';
            
            let textColor = 'text-white';
            if (type === 'error') textColor = 'text-red-400';
            else if (type === 'success') textColor = 'text-green-400';
            else if (type === 'warning') textColor = 'text-yellow-400';
            
            logEntry.innerHTML = `
                <span class="text-gray-400">${timestamp}</span>
                <span class="${textColor} ml-2">${message}</span>
            `;
            
            logContainer.insertBefore(logEntry, logContainer.firstChild);
            
            // Keep only last 20 logs
            while (logContainer.children.length > 20) {
                logContainer.removeChild(logContainer.lastChild);
            }
        }

        // API Functions
        async function apiCall(endpoint, method = 'GET', data = null) {
            try {
                const options = {
                    method,
                    headers: { 'Content-Type': 'application/json' }
                };

                if (data && method !== 'GET') {
                    options.body = JSON.stringify(data);
                }

                const response = await fetch(endpoint, options);
                return await response.json();
            } catch (error) {
                console.error(`API call failed:`, error);
                return null;
            }
        }

        // Event Listeners
        document.getElementById('startAI').addEventListener('click', async () => {
            addLog('Starting AI Engine (Demo Mode)...', 'info');
            const result = await apiCall('/api/start-ai', 'POST');
            if (result && result.status === 'success') {
                aiRunning = true;
                document.getElementById('startAI').disabled = true;
                document.getElementById('stopAI').disabled = false;
                document.getElementById('aiStrategy').textContent = 'Demo Mode Running';
                document.getElementById('aiStatusIndicator').className = 'w-3 h-3 rounded-full bg-green-500 animate-pulse';
                addLog('AI Engine started in demo mode', 'success');
            }
        });

        document.getElementById('stopAI').addEventListener('click', async () => {
            addLog('Stopping AI Engine...', 'info');
            const result = await apiCall('/api/stop-ai', 'POST');
            if (result && result.status === 'success') {
                aiRunning = false;
                document.getElementById('startAI').disabled = false;
                document.getElementById('stopAI').disabled = true;
                document.getElementById('aiStrategy').textContent = 'Standby';
                document.getElementById('aiStatusIndicator').className = 'w-3 h-3 rounded-full bg-red-500';
                addLog('AI Engine stopped', 'info');
            }
        });

        document.getElementById('testMT5').addEventListener('click', async () => {
            addLog('Testing MT5 connection...', 'info');
            document.getElementById('mt5Result').innerHTML = '<div class="text-yellow-400">Testing...</div>';
            
            const result = await apiCall('/api/mt5/account-info');
            if (result && !result.error) {
                document.getElementById('mt5Result').innerHTML = `
                    <div class="text-green-400">✅ MT5 Connected</div>
                    <div class="text-sm text-gray-400 mt-2">
                        Account: ${result.login}<br>
                        Balance: ${formatCurrency(result.balance)}<br>
                        Equity: ${formatCurrency(result.equity)}<br>
                        Broker: ${result.company}
                    </div>
                `;
                
                // Update header
                document.getElementById('accountBalance').textContent = formatCurrency(result.balance);
                document.getElementById('accountEquity').textContent = formatCurrency(result.equity);
                
                addLog('MT5 connection successful', 'success');
            } else {
                document.getElementById('mt5Result').innerHTML = `
                    <div class="text-red-400">❌ Connection Failed</div>
                    <div class="text-sm text-gray-400 mt-2">${result?.error || 'Please ensure MT5 is running and logged in'}</div>
                `;
                addLog('MT5 connection failed', 'error');
            }
        });

        // Auto-refresh status
        setInterval(async () => {
            const status = await apiCall('/api/status');
            if (status) {
                // Update UI with status
            }
        }, 5000);

        // Initial load
        addLog('AI Recovery Trading System initialized (Simple Mode)', 'success');
    </script>
</body>
</html>
    """, media_type="text/html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/status")
async def get_status():
    """สถานะระบบ"""
    return {
        "is_running": fake_ai_running,
        "market_regime": "ranging",
        "active_positions": len(fake_positions),
        "total_pnl": 0.0,
        "daily_pnl": 0.0,
        "monthly_pnl": 0.0,
        "risk_level": "LOW",
        "enabled_pairs": ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'XAUUSD'],
        "current_strategy": "Demo Mode" if fake_ai_running else "Standby",
        "mt5_connected": MT5_AVAILABLE,
        "account_balance": 5000.0,
        "positions": fake_positions
    }

@app.get("/api/positions")
async def get_positions():
    """รายการโพซิชั่น"""
    return {"positions": fake_positions}

@app.get("/api/market-data")
async def get_market_data():
    """ข้อมูลตลาด"""
    # Fake market data
    market_data = {
        "EURUSD": {
            "symbol": "EURUSD",
            "bid": 1.1050,
            "ask": 1.1052,
            "spread": 0.0002,
            "timestamp": datetime.now().isoformat()
        },
        "GBPUSD": {
            "symbol": "GBPUSD", 
            "bid": 1.2580,
            "ask": 1.2582,
            "spread": 0.0002,
            "timestamp": datetime.now().isoformat()
        }
    }
    return {"market_data": market_data}

@app.post("/api/start-ai")
async def start_ai():
    """เริ่ม AI (Demo)"""
    global fake_ai_running
    fake_ai_running = True
    return {"status": "success", "message": "AI started in demo mode"}

@app.post("/api/stop-ai")
async def stop_ai():
    """หยุด AI"""
    global fake_ai_running
    fake_ai_running = False
    return {"status": "success", "message": "AI stopped"}

@app.get("/api/mt5/account-info")
async def get_mt5_account_info():
    """ข้อมูลบัญชี MT5"""
    global mt5_connector
    
    if not MT5_AVAILABLE:
        return {"error": "MT5 Connector not available"}
    
    try:
        if not mt5_connector:
            mt5_connector = MT5Connector()
        
        if await mt5_connector.auto_connect():
            account_info = await mt5_connector.get_account_info()
            if account_info:
                return account_info
            else:
                return {"error": "Cannot retrieve account info"}
        else:
            return {"error": "MT5 connection failed", "message": "Please ensure MT5 is running and logged in"}
            
    except Exception as e:
        logger.error(f"MT5 error: {e}")
        return {"error": f"MT5 error: {str(e)}"}

if __name__ == "__main__":
    print("🚀 AI Recovery Trading System - Complete Simple Version")
    print("🔗 Open: http://localhost:8000")
    print("📝 Features: Full UI + MT5 Integration + Demo AI")
    print("-" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)