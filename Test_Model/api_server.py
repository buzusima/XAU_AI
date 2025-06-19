from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
import asyncio
import json
import logging
from datetime import datetime
import uvicorn

# Setup logging FIRST
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import AI Engine AFTER logger is defined
try:
    from ai_recovery_engine import RecoveryEngine, TradingSettings
    from mt5_connector import MT5Connector
    AI_ENGINE_AVAILABLE = True
    logger.info("✅ AI Engine modules imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import AI Engine modules: {e}")
    AI_ENGINE_AVAILABLE = False

# FastAPI App
app = FastAPI(title="AI Recovery Trading System", version="1.0.0")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
ai_engine: Optional[RecoveryEngine] = None
mt5_connector = None
connected_clients: List[WebSocket] = []

# Pydantic Models
class TradingSettingsModel(BaseModel):
    account_balance: float = 5000.0
    daily_target_pct: float = 2.0
    monthly_target_pct: float = 20.0
    max_recovery_levels: int = 5
    recovery_multiplier: float = 1.5
    max_portfolio_risk: float = 30.0
    initial_lot_size: float = 0.01
    enabled_pairs: List[str] = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'XAUUSD']
    use_correlation_recovery: bool = True
    use_grid_recovery: bool = True
    use_basket_recovery: bool = True
    use_arbitrage_recovery: bool = True

class PositionModel(BaseModel):
    symbol: str
    direction: str
    size: float
    entry_price: float
    current_price: float
    pnl: float
    pnl_pips: float
    recovery_level: int
    is_recovery: bool
    position_id: str

class AIStatusModel(BaseModel):
    is_running: bool
    market_regime: str
    active_positions: int
    total_pnl: float
    daily_pnl: float
    monthly_pnl: float
    risk_level: str
    enabled_pairs: List[str]
    current_strategy: str = "Standby"
    last_action: str = "System Ready"

# WebSocket Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, data: dict):
        """ส่งข้อมูลไปยัง clients ทั้งหมด"""
        if self.active_connections:
            message = json.dumps(data)
            for connection in self.active_connections[:]:  # Copy list to avoid modification during iteration
                try:
                    await connection.send_text(message)
                except:
                    # Remove disconnected clients
                    self.active_connections.remove(connection)

manager = ConnectionManager()

# Static Files (ไม่ต้องใช้แล้วเพราะ HTML embed ในตัว)
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Routes
# Routes
@app.get("/")
async def root():
    """หน้าแรก - ส่ง GUI Dashboard แบบ HTML"""
    return HTMLResponse(content=get_dashboard_html(), media_type="text/html")

def get_dashboard_html():
    """สร้างหน้าเว็บ Dashboard HTML"""
    return """
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Recovery Trading System</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    colors: {
                        slate: {
                            800: '#1e293b',
                            700: '#334155',
                            600: '#475569',
                            900: '#0f172a'
                        }
                    }
                }
            }
        }
    </script>
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
        .btn-success {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        }
        .btn-danger {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
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
        .status-connecting { background: #f59e0b; animation: pulse 1s infinite; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .tab-active {
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
            color: white;
        }
        .tab-inactive {
            background: rgba(71, 85, 105, 0.5);
            color: #94a3b8;
        }
        .tab-inactive:hover {
            background: rgba(71, 85, 105, 0.8);
            color: white;
        }
        .pair-card {
            transition: all 0.3s ease;
            cursor: pointer;
        }
        .pair-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        }
        .pair-selected {
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
            color: white;
        }
        .log-entry {
            animation: slideIn 0.3s ease;
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-20px); }
            to { opacity: 1; transform: translateX(0); }
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
                        🧠 AI Recovery Trading Brain
                        <span id="connectionStatus" class="status-indicator status-connecting"></span>
                    </h1>
                    <p class="text-gray-300 mt-2">
                        ระบบ AI เทรดแบบแก้ไม้อัตโนมัติ - Live MT5 Integration
                        <span id="connectionText" class="ml-2 text-xs px-2 py-1 rounded bg-yellow-600">CONNECTING...</span>
                    </p>
                </div>
                <div class="flex gap-3">
                    <button id="startAI" class="btn-primary px-6 py-3 rounded-lg text-white font-semibold disabled:opacity-50" disabled>
                        <span id="startText">เริ่ม AI</span>
                    </button>
                    <button id="stopAI" class="btn-primary px-6 py-3 rounded-lg text-white font-semibold disabled:opacity-50" disabled>
                        หยุด AI
                    </button>
                    <button id="emergencyStop" class="btn-danger px-6 py-3 rounded-lg text-white font-semibold disabled:opacity-50">
                        🚨 Emergency Stop
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

        <!-- Tabs -->
        <div class="mb-6">
            <div class="flex space-x-1 bg-slate-800 p-1 rounded-lg">
                <button class="tab-button tab-active px-4 py-2 rounded-md font-medium" data-tab="settings">การตั้งค่า AI</button>
                <button class="tab-button tab-inactive px-4 py-2 rounded-md font-medium" data-tab="pairs">เลือกคู่เงิน</button>
                <button class="tab-button tab-inactive px-4 py-2 rounded-md font-medium" data-tab="strategy">กลยุทธ์แก้ไม้</button>
                <button class="tab-button tab-inactive px-4 py-2 rounded-md font-medium" data-tab="monitor">ติดตามการเทรด</button>
            </div>
        </div>

        <!-- Tab Content -->
        <div id="tabContent">
            <!-- Settings Tab -->
            <div id="settings-tab" class="tab-content">
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <!-- MT5 Account Info -->
                    <div id="mt5AccountInfo" class="card rounded-lg p-6 hidden">
                        <h3 class="text-white text-lg font-semibold mb-4 flex items-center gap-2">
                            📊 MT5 Account Information (Live)
                        </h3>
                        <div class="grid grid-cols-2 gap-4 text-sm">
                            <div>
                                <p class="text-gray-400">Account</p>
                                <p id="mt5Login" class="text-white font-semibold">-</p>
                            </div>
                            <div>
                                <p class="text-gray-400">Balance</p>
                                <p id="mt5Balance" class="text-white font-semibold">-</p>
                            </div>
                            <div>
                                <p class="text-gray-400">Equity</p>
                                <p id="mt5Equity" class="text-white font-semibold">-</p>
                            </div>
                            <div>
                                <p class="text-gray-400">Free Margin</p>
                                <p id="mt5FreeMargin" class="text-white font-semibold">-</p>
                            </div>
                            <div>
                                <p class="text-gray-400">Profit</p>
                                <p id="mt5Profit" class="text-white font-semibold">-</p>
                            </div>
                            <div>
                                <p class="text-gray-400">Broker</p>
                                <p id="mt5Company" class="text-white font-semibold">-</p>
                            </div>
                        </div>
                    </div>

                    <!-- Trading Settings -->
                    <div class="card rounded-lg p-6">
                        <h3 class="text-white text-lg font-semibold mb-4 flex items-center gap-2">
                            ⚙️ การตั้งค่าการเทรด
                        </h3>
                        <div class="space-y-4">
                            <div>
                                <label class="text-gray-300 text-sm">เป้าหมายกำไรรายวัน (%)</label>
                                <input type="number" id="dailyTarget" value="2.0" step="0.1" 
                                       class="w-full mt-1 px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white">
                            </div>
                            <div>
                                <label class="text-gray-300 text-sm">เป้าหมายกำไรรายเดือน (%)</label>
                                <input type="number" id="monthlyTarget" value="20.0" step="0.1"
                                       class="w-full mt-1 px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white">
                            </div>
                            <div>
                                <label class="text-gray-300 text-sm">ขนาด Lot เริ่มต้น</label>
                                <input type="number" id="initialLotSize" value="0.01" step="0.01"
                                       class="w-full mt-1 px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white">
                            </div>
                        </div>
                    </div>

                    <!-- Risk Management -->
                    <div class="card rounded-lg p-6">
                        <h3 class="text-white text-lg font-semibold mb-4 flex items-center gap-2">
                            🛡️ การจัดการความเสี่ยง
                        </h3>
                        <div class="space-y-4">
                            <div>
                                <label class="text-gray-300 text-sm">จำนวนชั้นแก้ไม้สูงสุด</label>
                                <input type="number" id="maxRecoveryLevels" value="5"
                                       class="w-full mt-1 px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white">
                            </div>
                            <div>
                                <label class="text-gray-300 text-sm">ตัวคูณขนาดแก้ไม้</label>
                                <input type="number" id="recoveryMultiplier" value="1.5" step="0.1"
                                       class="w-full mt-1 px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white">
                            </div>
                            <div>
                                <label class="text-gray-300 text-sm">ความเสี่ยงสูงสุดของพอร์ต (%)</label>
                                <input type="number" id="maxPortfolioRisk" value="30.0" step="1"
                                       class="w-full mt-1 px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white">
                            </div>
                            <button id="saveSettings" class="w-full btn-primary px-4 py-2 rounded-lg text-white font-semibold mt-4">
                                บันทึกการตั้งค่า
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Currency Pairs Tab -->
            <div id="pairs-tab" class="tab-content hidden">
                <div class="card rounded-lg p-6">
                    <h3 class="text-white text-lg font-semibold mb-4">เลือกคู่สกุลเงินสำหรับ AI Trading</h3>
                    <p class="text-gray-400 mb-6">เลือกคู่สกุลเงินที่ต้องการให้ AI เทรด (จาก MT5 Markets)</p>
                    
                    <!-- Major Pairs -->
                    <div class="mb-6">
                        <h4 class="text-white font-semibold mb-3 flex items-center gap-2">
                            <span class="bg-blue-600 px-2 py-1 rounded text-xs">Major Pairs</span>
                            คู่เงินหลัก - สภาพคล่องสูง
                        </h4>
                        <div id="majorPairs" class="grid grid-cols-2 md:grid-cols-4 gap-3">
                            <!-- Will be populated by JavaScript -->
                        </div>
                    </div>

                    <!-- Cross Pairs -->
                    <div class="mb-6">
                        <h4 class="text-white font-semibold mb-3 flex items-center gap-2">
                            <span class="bg-green-600 px-2 py-1 rounded text-xs">Cross Pairs</span>
                            คู่เงินไขว้ - เหมาะสำหรับ Correlation
                        </h4>
                        <div id="crossPairs" class="grid grid-cols-2 md:grid-cols-4 gap-3">
                            <!-- Will be populated by JavaScript -->
                        </div>
                    </div>

                    <!-- Metals -->
                    <div class="mb-6">
                        <h4 class="text-white font-semibold mb-3 flex items-center gap-2">
                            <span class="bg-yellow-600 px-2 py-1 rounded text-xs">Metals</span>
                            โลหะมีค่า - Safe Haven
                        </h4>
                        <div id="metalPairs" class="grid grid-cols-2 md:grid-cols-4 gap-3">
                            <!-- Will be populated by JavaScript -->
                        </div>
                    </div>

                    <!-- Selected Pairs -->
                    <div class="card p-4 rounded-lg">
                        <h4 class="text-white font-semibold mb-2">คู่เงินที่เลือกแล้ว (<span id="selectedCount">0</span> คู่)</h4>
                        <div id="selectedPairs" class="flex flex-wrap gap-2">
                            <!-- Will be populated by JavaScript -->
                        </div>
                    </div>
                </div>
            </div>

            <!-- Strategy Tab -->
            <div id="strategy-tab" class="tab-content hidden">
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <!-- Recovery Strategies -->
                    <div class="card rounded-lg p-6">
                        <h3 class="text-white text-lg font-semibold mb-4">กลยุทธ์การแก้ไม้</h3>
                        <p class="text-gray-400 mb-6">เลือกวิธีการแก้ไม้ที่ AI จะใช้</p>
                        
                        <div class="space-y-4">
                            <div class="flex items-center justify-between">
                                <div>
                                    <p class="text-white font-semibold">Correlation Recovery</p>
                                    <p class="text-gray-400 text-sm">ใช้คู่เงินที่มี correlation ในการแก้ไม้</p>
                                </div>
                                <input type="checkbox" id="useCorrelationRecovery" class="w-4 h-4" checked>
                            </div>
                            <div class="flex items-center justify-between">
                                <div>
                                    <p class="text-white font-semibold">Grid Recovery</p>
                                    <p class="text-gray-400 text-sm">เพิ่มโพซิชั่นตามระยะห่าง Grid</p>
                                </div>
                                <input type="checkbox" id="useGridRecovery" class="w-4 h-4" checked>
                            </div>
                            <div class="flex items-center justify-between">
                                <div>
                                    <p class="text-white font-semibold">Basket Recovery</p>
                                    <p class="text-gray-400 text-sm">ใช้ตะกร้าสกุลเงินในการแก้ไม้</p>
                                </div>
                                <input type="checkbox" id="useBasketRecovery" class="w-4 h-4" checked>
                            </div>
                            <div class="flex items-center justify-between">
                                <div>
                                    <p class="text-white font-semibold">Arbitrage Recovery</p>
                                    <p class="text-gray-400 text-sm">หาโอกาส Arbitrage เพื่อแก้ไม้</p>
                                </div>
                                <input type="checkbox" id="useArbitrageRecovery" class="w-4 h-4" checked>
                            </div>
                        </div>
                    </div>

                    <!-- Market Data -->
                    <div class="card rounded-lg p-6">
                        <h3 class="text-white text-lg font-semibold mb-4">Market Data (Live)</h3>
                        <p class="text-gray-400 mb-4">ข้อมูลตลาดแบบ Real-time</p>
                        <div id="marketDataContainer" class="space-y-2 max-h-64 overflow-y-auto">
                            <div class="text-center py-4 text-gray-400">
                                No market data available
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Monitor Tab -->
            <div id="monitor-tab" class="tab-content hidden">
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <!-- Current Positions -->
                    <div class="card rounded-lg p-6">
                        <h3 class="text-white text-lg font-semibold mb-4">โพซิชั่นปัจจุบัน (<span id="positionCount">0</span>)</h3>
                        <div id="positionsContainer" class="space-y-2 max-h-64 overflow-y-auto">
                            <div class="text-center py-8">
                                <div class="text-gray-400 text-4xl mb-2">⚠️</div>
                                <p class="text-gray-400">ไม่มีโพซิชั่นเปิดขณะนี้</p>
                            </div>
                        </div>
                    </div>

                    <!-- AI Decision Log -->
                    <div class="card rounded-lg p-6">
                        <h3 class="text-white text-lg font-semibold mb-4">AI Decision Log</h3>
                        <div id="logsContainer" class="space-y-2 max-h-64 overflow-y-auto">
                            <div class="text-center py-4 text-gray-400">
                                No logs available
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Market Overview -->
                <div class="card rounded-lg p-6 mt-6">
                    <h3 class="text-white text-lg font-semibold mb-4">Market Overview</h3>
                    <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div class="text-center">
                            <p class="text-gray-400 text-sm">Market Regime</p>
                            <span id="marketRegime" class="inline-block mt-1 px-2 py-1 bg-slate-700 rounded text-white text-sm">RANGING</span>
                        </div>
                        <div class="text-center">
                            <p class="text-gray-400 text-sm">Total P&L</p>
                            <p id="totalPnL" class="text-white font-semibold mt-1">$0.00</p>
                        </div>
                        <div class="text-center">
                            <p class="text-gray-400 text-sm">Daily P&L</p>
                            <p id="dailyPnLOverview" class="text-white font-semibold mt-1">$0.00</p>
                        </div>
                        <div class="text-center">
                            <p class="text-gray-400 text-sm">Risk Level</p>
                            <span id="riskLevelOverview" class="inline-block mt-1 px-2 py-1 bg-green-600 rounded text-white text-sm">LOW</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <div class="mt-6 card rounded-lg p-4">
            <div class="flex items-center gap-2 text-yellow-400 mb-2">
                <span>⚠️</span>
                <span class="font-semibold">Live System Status</span>
            </div>
            <ul class="text-gray-400 text-sm space-y-1">
                <li>• ระบบเชื่อมต่อ MT5 แบบ Real-time และดึงข้อมูลบัญชีจริง</li>
                <li>• AI ตัดสินใจการเทรดอัตโนมัติและส่งคำสั่งไปยัง MT5</li>
                <li>• WebSocket connection สำหรับข้อมูล Live Updates</li>
                <li>• ระบบแก้ไม้อัตโนมัติทำงานแบบ Real-time</li>
                <li>• กดปุ่ม Emergency Stop เพื่อปิดโพซิชั่นทั้งหมดทันที</li>
            </ul>
        </div>
    </div>

    <script>
        // Global Variables
        let ws = null;
        let isConnected = false;
        let aiRunning = false;
        let settings = {
            account_balance: 5000,
            daily_target_pct: 2.0,
            monthly_target_pct: 20.0,
            max_recovery_levels: 5,
            recovery_multiplier: 1.5,
            max_portfolio_risk: 30.0,
            initial_lot_size: 0.01,
            enabled_pairs: ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'XAUUSD'],
            use_correlation_recovery: true,
            use_grid_recovery: true,
            use_basket_recovery: true,
            use_arbitrage_recovery: true
        };

        // Currency Pairs Data
        const mt5Pairs = {
            majors: {
                'EURUSD': { symbol: 'EURUSD', name: 'Euro vs US Dollar' },
                'GBPUSD': { symbol: 'GBPUSD', name: 'British Pound vs US Dollar' },
                'USDJPY': { symbol: 'USDJPY', name: 'US Dollar vs Japanese Yen' },
                'USDCHF': { symbol: 'USDCHF', name: 'US Dollar vs Swiss Franc' },
                'AUDUSD': { symbol: 'AUDUSD', name: 'Australian Dollar vs US Dollar' },
                'NZDUSD': { symbol: 'NZDUSD', name: 'New Zealand Dollar vs US Dollar' },
                'USDCAD': { symbol: 'USDCAD', name: 'US Dollar vs Canadian Dollar' }
            },
            crosses: {
                'EURGBP': { symbol: 'EURGBP', name: 'Euro vs British Pound' },
                'EURJPY': { symbol: 'EURJPY', name: 'Euro vs Japanese Yen' },
                'EURCHF': { symbol: 'EURCHF', name: 'Euro vs Swiss Franc' },
                'EURAUD': { symbol: 'EURAUD', name: 'Euro vs Australian Dollar' },
                'EURCAD': { symbol: 'EURCAD', name: 'Euro vs Canadian Dollar' },
                'GBPJPY': { symbol: 'GBPJPY', name: 'British Pound vs Japanese Yen' },
                'GBPCHF': { symbol: 'GBPCHF', name: 'British Pound vs Swiss Franc' },
                'GBPAUD': { symbol: 'GBPAUD', name: 'British Pound vs Australian Dollar' },
                'AUDJPY': { symbol: 'AUDJPY', name: 'Australian Dollar vs Japanese Yen' },
                'AUDCHF': { symbol: 'AUDCHF', name: 'Australian Dollar vs Swiss Franc' },
                'CHFJPY': { symbol: 'CHFJPY', name: 'Swiss Franc vs Japanese Yen' }
            },
            metals: {
                'XAUUSD': { symbol: 'XAUUSD', name: 'Gold vs US Dollar' },
                'XAGUSD': { symbol: 'XAGUSD', name: 'Silver vs US Dollar' }
            }
        };

        // Utility Functions
        function formatCurrency(amount) {
            return new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD'
            }).format(amount);
        }

        function formatNumber(num, decimals = 2) {
            return Number(num).toFixed(decimals);
        }

        function addLog(message, type = 'info') {
            const timestamp = new Date().toLocaleTimeString();
            const logContainer = document.getElementById('logsContainer');
            
            const logEntry = document.createElement('div');
            logEntry.className = 'log-entry p-2 bg-slate-700 rounded text-sm';
            
            let textColor = 'text-white';
            if (type === 'error') textColor = 'text-red-400';
            else if (type === 'success') textColor = 'text-green-400';
            else if (type === 'warning') textColor = 'text-yellow-400';
            
            logEntry.innerHTML = `
                <span class="text-gray-400">${timestamp}</span>
                <span class="${textColor} ml-2">${message}</span>
            `;
            
            if (logContainer.children.length === 1 && logContainer.children[0].textContent.includes('No logs available')) {
                logContainer.innerHTML = '';
            }
            
            logContainer.insertBefore(logEntry, logContainer.firstChild);
            
            // Keep only last 50 logs
            while (logContainer.children.length > 50) {
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
                
                if (!response.ok) {
                    throw new Error(`API Error: ${response.status} ${response.statusText}`);
                }

                return await response.json();
            } catch (error) {
                console.error(`API call failed for ${endpoint}:`, error);
                addLog(`API Error: ${error.message}`, 'error');
                return null;
            }
        }

        // WebSocket Functions
        function connectWebSocket() {
            const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                isConnected = true;
                updateConnectionStatus('connected');
                addLog('Connected to AI Backend', 'success');
            };

            ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                handleWebSocketMessage(message);
            };

            ws.onclose = () => {
                isConnected = false;
                updateConnectionStatus('disconnected');
                addLog('Disconnected from AI Backend', 'warning');
                
                // Auto-reconnect after 5 seconds
                setTimeout(connectWebSocket, 5000);
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                updateConnectionStatus('error');
                addLog('WebSocket connection error', 'error');
            };
        }

        function handleWebSocketMessage(message) {
            switch (message.type) {
                case 'ai_status_update':
                    updateAIStatus(message.data);
                    break;
                case 'initial_data':
                    updateAIStatus(message.data);
                    break;
                case 'ai_started':
                    aiRunning = true;
                    updateAIControls();
                    addLog('AI Engine Started', 'success');
                    break;
                case 'ai_stopped':
                    aiRunning = false;
                    updateAIControls();
                    addLog('AI Engine Stopped', 'info');
                    break;
                case 'position_update':
                    updatePositions(message.data.positions || []);
                    break;
                case 'market_data_update':
                    updateMarketData(message.data.market_data || {});
                    break;
                case 'emergency_stop':
                    addLog('Emergency Stop Executed!', 'warning');
                    break;
                default:
                    console.log('Unknown message type:', message.type);
            }
        }

        function updateConnectionStatus(status) {
            const statusIndicator = document.getElementById('connectionStatus');
            const statusText = document.getElementById('connectionText');
            
            statusIndicator.className = 'status-indicator';
            
            switch (status) {
                case 'connected':
                    statusIndicator.classList.add('status-online');
                    statusText.textContent = 'LIVE';
                    statusText.className = 'ml-2 text-xs px-2 py-1 rounded bg-green-600';
                    break;
                case 'disconnected':
                    statusIndicator.classList.add('status-offline');
                    statusText.textContent = 'OFFLINE';
                    statusText.className = 'ml-2 text-xs px-2 py-1 rounded bg-red-600';
                    break;
                case 'error':
                    statusIndicator.classList.add('status-connecting');
                    statusText.textContent = 'ERROR';
                    statusText.className = 'ml-2 text-xs px-2 py-1 rounded bg-yellow-600';
                    break;
                default:
                    statusIndicator.classList.add('status-connecting');
                    statusText.textContent = 'CONNECTING...';
                    statusText.className = 'ml-2 text-xs px-2 py-1 rounded bg-yellow-600';
            }
            
            updateAIControls();
        }

        function updateAIStatus(data) {
            document.getElementById('aiStrategy').textContent = data.current_strategy || 'Standby';
            document.getElementById('dailyPnL').textContent = formatCurrency(data.daily_pnl || 0);
            document.getElementById('activePositions').textContent = data.active_positions || 0;
            document.getElementById('riskLevel').textContent = data.risk_level || 'LOW';
            document.getElementById('marketRegime').textContent = data.market_regime || 'RANGING';
            document.getElementById('totalPnL').textContent = formatCurrency(data.total_pnl || 0);
            document.getElementById('dailyPnLOverview').textContent = formatCurrency(data.daily_pnl || 0);
            document.getElementById('riskLevelOverview').textContent = data.risk_level || 'LOW';
            document.getElementById('positionCount').textContent = data.active_positions || 0;
            
            // Update AI status indicator
            const indicator = document.getElementById('aiStatusIndicator');
            if (data.is_running) {
                indicator.className = 'w-3 h-3 rounded-full bg-green-500 animate-pulse';
                aiRunning = true;
            } else {
                indicator.className = 'w-3 h-3 rounded-full bg-red-500';
                aiRunning = false;
            }
            
            updateAIControls();
            
            // Update positions if available
            if (data.positions) {
                updatePositions(data.positions);
            }
        }

        function updateAIControls() {
            const startBtn = document.getElementById('startAI');
            const stopBtn = document.getElementById('stopAI');
            const emergencyBtn = document.getElementById('emergencyStop');
            
            startBtn.disabled = !isConnected || aiRunning;
            stopBtn.disabled = !isConnected || !aiRunning;
            emergencyBtn.disabled = !isConnected;
            
            document.getElementById('startText').textContent = aiRunning ? 'AI กำลังทำงาน...' : 'เริ่ม AI';
        }

        function updatePositions(positions) {
            const container = document.getElementById('positionsContainer');
            
            if (positions.length === 0) {
                container.innerHTML = `
                    <div class="text-center py-8">
                        <div class="text-gray-400 text-4xl mb-2">⚠️</div>
                        <p class="text-gray-400">ไม่มีโพซิชั่นเปิดขณะนี้</p>
                    </div>
                `;
                return;
            }
            
            container.innerHTML = positions.map(position => `
                <div class="p-3 bg-slate-700 rounded-lg">
                    <div class="flex justify-between items-center">
                        <div>
                            <span class="text-white font-semibold">${position.symbol}</span>
                            <span class="ml-2 px-2 py-1 rounded text-xs ${position.direction === 'BUY' ? 'bg-blue-600' : 'bg-red-600'} text-white">
                                ${position.direction}
                            </span>
                            ${position.is_recovery ? `<span class="ml-1 px-2 py-1 rounded text-xs border border-yellow-400 text-yellow-400">Recovery L${position.recovery_level}</span>` : ''}
                        </div>
                        <div class="text-right">
                            <div class="font-semibold ${(position.pnl || 0) >= 0 ? 'text-green-400' : 'text-red-400'}">
                                ${formatCurrency(position.pnl || 0)}
                            </div>
                            <div class="text-gray-400 text-xs">
                                ${formatNumber(position.pnl_pips || 0, 1)} pips
                            </div>
                        </div>
                    </div>
                    <div class="mt-2 text-xs text-gray-400">
                        Size: ${position.size} | Entry: ${formatNumber(position.entry_price, 5)} | Current: ${formatNumber(position.current_price, 5)}
                    </div>
                </div>
            `).join('');
        }

        function updateMarketData(marketData) {
            const container = document.getElementById('marketDataContainer');
            
            if (Object.keys(marketData).length === 0) {
                container.innerHTML = `
                    <div class="text-center py-4 text-gray-400">
                        No market data available
                    </div>
                `;
                return;
            }
            
            container.innerHTML = Object.entries(marketData).map(([symbol, data]) => `
                <div class="flex justify-between items-center p-2 bg-slate-700 rounded">
                    <span class="text-white font-semibold">${symbol}</span>
                    <div class="text-right">
                        <div class="text-white text-sm">
                            ${formatNumber(data.bid, symbol.includes('JPY') ? 3 : 5)} / ${formatNumber(data.ask, symbol.includes('JPY') ? 3 : 5)}
                        </div>
                        <div class="text-gray-400 text-xs">
                            Spread: ${formatNumber(data.spread, symbol.includes('JPY') ? 3 : 5)}
                        </div>
                    </div>
                </div>
            `).join('');
        }

        // Tab Management
        function initializeTabs() {
            const tabButtons = document.querySelectorAll('.tab-button');
            const tabContents = document.querySelectorAll('.tab-content');
            
            tabButtons.forEach(button => {
                button.addEventListener('click', () => {
                    const tabName = button.getAttribute('data-tab');
                    
                    // Update button states
                    tabButtons.forEach(btn => {
                        btn.className = btn === button ? 'tab-button tab-active px-4 py-2 rounded-md font-medium' : 'tab-button tab-inactive px-4 py-2 rounded-md font-medium';
                    });
                    
                    // Update content visibility
                    tabContents.forEach(content => {
                        content.className = content.id === `${tabName}-tab` ? 'tab-content' : 'tab-content hidden';
                    });
                });
            });
        }

        // Currency Pairs Management
        function initializePairs() {
            const majorContainer = document.getElementById('majorPairs');
            const crossContainer = document.getElementById('crossPairs');
            const metalContainer = document.getElementById('metalPairs');
            
            // Render major pairs
            majorContainer.innerHTML = Object.entries(mt5Pairs.majors).map(([symbol, data]) => 
                createPairCard(symbol, data, 'bg-blue-600')).join('');
            
            // Render cross pairs
            crossContainer.innerHTML = Object.entries(mt5Pairs.crosses).map(([symbol, data]) => 
                createPairCard(symbol, data, 'bg-green-600')).join('');
            
            // Render metal pairs
            metalContainer.innerHTML = Object.entries(mt5Pairs.metals).map(([symbol, data]) => 
                createPairCard(symbol, data, 'bg-yellow-600')).join('');
            
            updateSelectedPairs();
        }

        function createPairCard(symbol, data, selectedColor) {
            const isSelected = settings.enabled_pairs.includes(symbol);
            return `
                <div class="pair-card p-3 rounded-lg border cursor-pointer ${isSelected ? 'pair-selected' : 'bg-slate-700 border-slate-600 text-gray-300'}" 
                     data-symbol="${symbol}" onclick="togglePair('${symbol}')">
                    <div class="font-semibold">${symbol}</div>
                    <div class="text-xs opacity-75">${data.name}</div>
                </div>
            `;
        }

        function togglePair(symbol) {
            if (settings.enabled_pairs.includes(symbol)) {
                settings.enabled_pairs = settings.enabled_pairs.filter(pair => pair !== symbol);
            } else {
                settings.enabled_pairs.push(symbol);
            }
            
            // Update UI
            const pairCard = document.querySelector(`[data-symbol="${symbol}"]`);
            if (settings.enabled_pairs.includes(symbol)) {
                pairCard.className = 'pair-card p-3 rounded-lg border cursor-pointer pair-selected';
            } else {
                pairCard.className = 'pair-card p-3 rounded-lg border cursor-pointer bg-slate-700 border-slate-600 text-gray-300';
            }
            
            updateSelectedPairs();
        }

        function updateSelectedPairs() {
            const container = document.getElementById('selectedPairs');
            const count = document.getElementById('selectedCount');
            
            count.textContent = settings.enabled_pairs.length;
            
            container.innerHTML = settings.enabled_pairs.map(pair => 
                `<span class="px-2 py-1 bg-blue-600 rounded text-white text-sm">${pair}</span>`
            ).join('');
        }

        // Settings Management
        function loadSettings() {
            document.getElementById('dailyTarget').value = settings.daily_target_pct;
            document.getElementById('monthlyTarget').value = settings.monthly_target_pct;
            document.getElementById('initialLotSize').value = settings.initial_lot_size;
            document.getElementById('maxRecoveryLevels').value = settings.max_recovery_levels;
            document.getElementById('recoveryMultiplier').value = settings.recovery_multiplier;
            document.getElementById('maxPortfolioRisk').value = settings.max_portfolio_risk;
            document.getElementById('useCorrelationRecovery').checked = settings.use_correlation_recovery;
            document.getElementById('useGridRecovery').checked = settings.use_grid_recovery;
            document.getElementById('useBasketRecovery').checked = settings.use_basket_recovery;
            document.getElementById('useArbitrageRecovery').checked = settings.use_arbitrage_recovery;
        }

        function saveSettings() {
            settings.daily_target_pct = parseFloat(document.getElementById('dailyTarget').value);
            settings.monthly_target_pct = parseFloat(document.getElementById('monthlyTarget').value);
            settings.initial_lot_size = parseFloat(document.getElementById('initialLotSize').value);
            settings.max_recovery_levels = parseInt(document.getElementById('maxRecoveryLevels').value);
            settings.recovery_multiplier = parseFloat(document.getElementById('recoveryMultiplier').value);
            settings.max_portfolio_risk = parseFloat(document.getElementById('maxPortfolioRisk').value);
            settings.use_correlation_recovery = document.getElementById('useCorrelationRecovery').checked;
            settings.use_grid_recovery = document.getElementById('useGridRecovery').checked;
            settings.use_basket_recovery = document.getElementById('useBasketRecovery').checked;
            settings.use_arbitrage_recovery = document.getElementById('useArbitrageRecovery').checked;
        }

        // Event Listeners
        function initializeEventListeners() {
            // AI Controls
            document.getElementById('startAI').addEventListener('click', async () => {
                addLog('Starting AI Engine...', 'info');
                saveSettings();
                await apiCall('/api/settings', 'POST', settings);
                const result = await apiCall('/api/start-ai', 'POST');
                if (result) {
                    addLog(`AI Engine ${result.status === 'success' ? 'started successfully' : result.message}`, 
                           result.status === 'success' ? 'success' : 'warning');
                }
            });

            document.getElementById('stopAI').addEventListener('click', async () => {
                addLog('Stopping AI Engine...', 'info');
                const result = await apiCall('/api/stop-ai', 'POST');
                if (result) {
                    addLog(`AI Engine ${result.status === 'success' ? 'stopped successfully' : result.message}`, 
                           result.status === 'success' ? 'success' : 'warning');
                }
            });

            document.getElementById('emergencyStop').addEventListener('click', async () => {
                if (confirm('Are you sure you want to execute Emergency Stop? This will close all positions!')) {
                    addLog('Executing Emergency Stop...', 'warning');
                    const result = await apiCall('/api/emergency-stop', 'POST');
                    if (result) {
                        addLog('Emergency Stop completed', 'warning');
                    }
                }
            });

            // Save Settings
            document.getElementById('saveSettings').addEventListener('click', async () => {
                saveSettings();
                const result = await apiCall('/api/settings', 'POST', settings);
                if (result) {
                    addLog('Settings saved successfully', 'success');
                }
            });
        }

        // Load Initial Data
        async function loadInitialData() {
            // Load AI Status
            const status = await apiCall('/api/status');
            if (status) {
                updateAIStatus(status);
            }

            // Load Positions
            const positionsData = await apiCall('/api/positions');
            if (positionsData) {
                updatePositions(positionsData.positions || []);
            }

            // Load Market Data
            const marketDataResponse = await apiCall('/api/market-data');
            if (marketDataResponse) {
                updateMarketData(marketDataResponse.market_data || {});
            }

            // Load MT5 Account Info
            const accountInfo = await apiCall('/api/mt5/account-info');
            if (accountInfo) {
                updateMT5AccountInfo(accountInfo);
            }
        }

        function updateMT5AccountInfo(accountInfo) {
            const container = document.getElementById('mt5AccountInfo');
            if (accountInfo && accountInfo.login) {
                container.classList.remove('hidden');
                document.getElementById('mt5Login').textContent = accountInfo.login;
                document.getElementById('mt5Balance').textContent = formatCurrency(accountInfo.balance);
                document.getElementById('mt5Equity').textContent = formatCurrency(accountInfo.equity);
                document.getElementById('mt5FreeMargin').textContent = formatCurrency(accountInfo.free_margin);
                document.getElementById('mt5Profit').textContent = `${formatCurrency(accountInfo.profit)} (${formatNumber(accountInfo.profit_percentage)}%)`;
                document.getElementById('mt5Company').textContent = accountInfo.company;
                
                // Update header balance
                document.getElementById('accountBalance').textContent = formatCurrency(accountInfo.balance);
                document.getElementById('accountEquity').textContent = formatCurrency(accountInfo.equity);
                
                settings.account_balance = accountInfo.balance;
            }
        }

        // Initialize Application
        function initializeApp() {
            initializeTabs();
            initializePairs();
            loadSettings();
            initializeEventListeners();
            connectWebSocket();
            loadInitialData();
            
            addLog('AI Recovery Trading System initialized', 'success');
        }

        // Start the application when DOM is loaded
        document.addEventListener('DOMContentLoaded', initializeApp);
    </script>
</body>
</html>
    """

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/settings")
async def update_settings(settings: TradingSettingsModel):
    """อัพเดทการตั้งค่า AI"""
    global ai_engine
    
    try:
        # แปลง Pydantic model เป็น TradingSettings
        trading_settings = TradingSettings(
            account_balance=settings.account_balance,
            daily_target_pct=settings.daily_target_pct,
            monthly_target_pct=settings.monthly_target_pct,
            max_recovery_levels=settings.max_recovery_levels,
            recovery_multiplier=settings.recovery_multiplier,
            max_portfolio_risk=settings.max_portfolio_risk,
            initial_lot_size=settings.initial_lot_size,
            enabled_pairs=settings.enabled_pairs,
            use_correlation_recovery=settings.use_correlation_recovery,
            use_grid_recovery=settings.use_grid_recovery,
            use_basket_recovery=settings.use_basket_recovery,
            use_arbitrage_recovery=settings.use_arbitrage_recovery
        )
        
        # สร้าง AI Engine ใหม่หรืออัพเดทการตั้งค่า
        if ai_engine:
            ai_engine.settings = trading_settings
            logger.info("AI settings updated")
        else:
            ai_engine = RecoveryEngine(trading_settings)
            logger.info("AI engine created with new settings")
        
        # แจ้ง clients ทั้งหมด
        await manager.broadcast({
            "type": "settings_updated",
            "data": settings.dict(),
            "timestamp": datetime.now().isoformat()
        })
        
        return {"status": "success", "message": "Settings updated successfully"}
        
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/start-ai")
async def start_ai():
    """เริ่มต้น AI Engine"""
    global ai_engine
    
    if not AI_ENGINE_AVAILABLE:
        raise HTTPException(status_code=500, detail="AI Engine modules not available")
    
    try:
        if not ai_engine:
            # สร้าง AI Engine ด้วยการตั้งค่าเริ่มต้น
            default_settings = TradingSettings()
            ai_engine = RecoveryEngine(default_settings)
        
        if ai_engine.is_running:
            return {"status": "warning", "message": "AI is already running"}
        
        # เริ่ม AI Engine ในแบบ background task
        success = await ai_engine.start_engine()
        
        if success:
            # เริ่ม background task สำหรับส่งข้อมูล real-time
            asyncio.create_task(broadcast_ai_status())
            
            logger.info("✅ AI Engine started successfully with MT5 connection")
            
            await manager.broadcast({
                "type": "ai_started",
                "data": {"is_running": True, "mt5_connected": True},
                "timestamp": datetime.now().isoformat()
            })
            
            return {"status": "success", "message": "AI started successfully with MT5 connection"}
        else:
            return {"status": "error", "message": "Failed to start AI Engine - MT5 connection failed"}
        
    except Exception as e:
        logger.error(f"Error starting AI: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stop-ai")
async def stop_ai():
    """หยุด AI Engine"""
    global ai_engine
    
    try:
        if not ai_engine or not ai_engine.is_running:
            return {"status": "warning", "message": "AI is not running"}
        
        await ai_engine.stop_engine()
        
        logger.info("AI Engine stopped successfully")
        
        await manager.broadcast({
            "type": "ai_stopped",
            "data": {"is_running": False},
            "timestamp": datetime.now().isoformat()
        })
        
        return {"status": "success", "message": "AI stopped successfully"}
        
    except Exception as e:
        logger.error(f"Error stopping AI: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_ai_status():
    """ได้สถานะปัจจุบันของ AI"""
    global ai_engine
    
    try:
        if not ai_engine:
            return {
                "is_running": False,
                "market_regime": "unknown",
                "active_positions": 0,
                "total_pnl": 0.0,
                "daily_pnl": 0.0,
                "monthly_pnl": 0.0,
                "risk_level": "LOW",
                "enabled_pairs": [],
                "positions": []
            }
        
        status = ai_engine.get_status()
        return status
        
    except Exception as e:
        logger.error(f"Error getting AI status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/positions")
async def get_positions():
    """ได้รายการโพซิชั่นปัจจุบัน"""
    global ai_engine
    
    try:
        if not ai_engine:
            return {"positions": []}
        
        positions = []
        for pos in ai_engine.positions.values():
            positions.append({
                "position_id": pos.position_id,
                "symbol": pos.symbol,
                "direction": pos.direction,
                "size": pos.size,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "pnl": pos.pnl,
                "pnl_pips": pos.pnl_pips,
                "recovery_level": pos.recovery_level,
                "is_recovery": pos.is_recovery,
                "timestamp": pos.timestamp.isoformat()
            })
        
        return {"positions": positions}
        
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market-data")
async def get_market_data():
    """ได้ข้อมูลตลาดปัจจุบัน"""
    global ai_engine
    
    try:
        if not ai_engine:
            return {"market_data": {}}
        
        market_data = {}
        for symbol, data in ai_engine.market_data.items():
            market_data[symbol] = {
                "symbol": data.symbol,
                "bid": data.bid,
                "ask": data.ask,
                "spread": data.spread,
                "timestamp": data.timestamp.isoformat()
            }
        
        return {"market_data": market_data}
        
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/emergency-stop")
async def emergency_stop():
    """หยุดฉุกเฉินและปิดโพซิชั่นทั้งหมด"""
    global ai_engine
    
    try:
        if ai_engine:
            await ai_engine.emergency_stop()
            logger.warning("🚨 Emergency stop executed by user")
        
        await manager.broadcast({
            "type": "emergency_stop",
            "data": {"message": "Emergency stop executed - All positions closed"},
            "timestamp": datetime.now().isoformat()
        })
        
        return {"status": "success", "message": "Emergency stop executed successfully"}
        
    except Exception as e:
        logger.error(f"Error during emergency stop: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket Endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket connection สำหรับข้อมูล Real-time"""
    await manager.connect(websocket)
    
    try:
        # ส่งข้อมูลเริ่มต้น
        if ai_engine:
            initial_data = {
                "type": "initial_data",
                "data": ai_engine.get_status(),
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send_text(json.dumps(initial_data))
        
        # รอรับข้อมูลจาก client
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # ประมวลผลข้อมูลจาก client
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }))
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

async def broadcast_ai_status():
    """Background task สำหรับส่งสถานะ AI แบบ Real-time"""
    while True:
        try:
            if ai_engine and ai_engine.is_running and manager.active_connections:
                status = ai_engine.get_status()
                
                await manager.broadcast({
                    "type": "ai_status_update",
                    "data": status,
                    "timestamp": datetime.now().isoformat()
                })
            
            # ส่งทุก 2 วินาที
            await asyncio.sleep(2)
            
        except Exception as e:
            logger.error(f"Error broadcasting AI status: {e}")
            await asyncio.sleep(5)

# MT5 Integration Endpoints
@app.post("/api/mt5/connect")
async def connect_mt5(mt5_config: dict):
    """เชื่อมต่อ MT5 (สำหรับอนาคต)"""
    try:
        # TODO: Implement MT5 connection
        logger.info("MT5 connection requested")
        return {"status": "success", "message": "MT5 connection established"}
    except Exception as e:
        logger.error(f"Error connecting to MT5: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/mt5/account-info")
async def get_mt5_account_info():
    """ได้ข้อมูลบัญชี MT5"""
    try:
        if ai_engine and ai_engine.mt5_connector and ai_engine.mt5_connector.is_connected:
            account_info = ai_engine.mt5_connector.get_account_info()
            if account_info:
                return account_info
            else:
                return {"error": "Cannot retrieve MT5 account info"}
        else:
            # ลองเชื่อมต่อ MT5 ใหม่
            global mt5_connector
            if not mt5_connector:
                mt5_connector = MT5Connector()
                if not await mt5_connector.auto_connect():
                    mt5_connector = None
            
            if mt5_connector and mt5_connector.is_connected:
                account_info = mt5_connector.get_account_info()
                return account_info if account_info else {"error": "Cannot retrieve account info"}
            else:
                return {
                    "error": "MT5 not connected",
                    "message": "Please make sure MT5 is running and logged in"
                }
    except Exception as e:
        logger.error(f"Error getting MT5 account info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error Handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "status_code": 404}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "status_code": 500}

# Startup Event
@app.on_event("startup")
async def startup_event():
    global mt5_connector
    logger.info("🚀 AI Recovery Trading API Server starting up...")
    
    # ลองเชื่อมต่อ MT5 ตั้งแต่เริ่มต้น
    if AI_ENGINE_AVAILABLE:
        try:
            mt5_connector = MT5Connector()
            if await mt5_connector.auto_connect():
                logger.info("✅ MT5 Auto-Connected at startup")
            else:
                logger.warning("⚠️ MT5 connection failed at startup")
                mt5_connector = None
        except Exception as e:
            logger.error(f"❌ Error connecting MT5 at startup: {e}")
            mt5_connector = None
    
    logger.info("🌐 Server ready at http://localhost:8000")
    logger.info("🔌 WebSocket available at ws://localhost:8000/ws")

# Shutdown Event
@app.on_event("shutdown")
async def shutdown_event():
    global ai_engine
    if ai_engine and ai_engine.is_running:
        await ai_engine.stop_engine()
    logger.info("AI Recovery Trading API Server shutting down...")

# Development Server
if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )