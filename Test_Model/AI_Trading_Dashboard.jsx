import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { AlertCircle, Settings, TrendingUp, Target, Shield, Brain, Wifi, WifiOff, Activity } from 'lucide-react';

const AITradingDashboard = () => {
  // API Configuration
  const API_BASE_URL = 'http://localhost:8000';
  const WS_URL = 'ws://localhost:8000/ws';

  // คู่สกุลเงินใน MT5 Markets
  const mt5Pairs = {
    majors: {
      'EURUSD': { symbol: 'EURUSD', name: 'Euro vs US Dollar', type: 'Major' },
      'GBPUSD': { symbol: 'GBPUSD', name: 'British Pound vs US Dollar', type: 'Major' },
      'USDJPY': { symbol: 'USDJPY', name: 'US Dollar vs Japanese Yen', type: 'Major' },
      'USDCHF': { symbol: 'USDCHF', name: 'US Dollar vs Swiss Franc', type: 'Major' },
      'AUDUSD': { symbol: 'AUDUSD', name: 'Australian Dollar vs US Dollar', type: 'Major' },
      'NZDUSD': { symbol: 'NZDUSD', name: 'New Zealand Dollar vs US Dollar', type: 'Major' },
      'USDCAD': { symbol: 'USDCAD', name: 'US Dollar vs Canadian Dollar', type: 'Major' }
    },
    crosses: {
      'EURGBP': { symbol: 'EURGBP', name: 'Euro vs British Pound', type: 'Cross' },
      'EURJPY': { symbol: 'EURJPY', name: 'Euro vs Japanese Yen', type: 'Cross' },
      'EURCHF': { symbol: 'EURCHF', name: 'Euro vs Swiss Franc', type: 'Cross' },
      'EURAUD': { symbol: 'EURAUD', name: 'Euro vs Australian Dollar', type: 'Cross' },
      'EURCAD': { symbol: 'EURCAD', name: 'Euro vs Canadian Dollar', type: 'Cross' },
      'GBPJPY': { symbol: 'GBPJPY', name: 'British Pound vs Japanese Yen', type: 'Cross' },
      'GBPCHF': { symbol: 'GBPCHF', name: 'British Pound vs Swiss Franc', type: 'Cross' },
      'GBPAUD': { symbol: 'GBPAUD', name: 'British Pound vs Australian Dollar', type: 'Cross' },
      'AUDJPY': { symbol: 'AUDJPY', name: 'Australian Dollar vs Japanese Yen', type: 'Cross' },
      'AUDCHF': { symbol: 'AUDCHF', name: 'Australian Dollar vs Swiss Franc', type: 'Cross' },
      'CHFJPY': { symbol: 'CHFJPY', name: 'Swiss Franc vs Japanese Yen', type: 'Cross' }
    },
    metals: {
      'XAUUSD': { symbol: 'XAUUSD', name: 'Gold vs US Dollar', type: 'Metal' },
      'XAGUSD': { symbol: 'XAGUSD', name: 'Silver vs US Dollar', type: 'Metal' }
    }
  };

  // State สำหรับการตั้งค่า
  const [settings, setSettings] = useState({
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
  });

  const [aiStatus, setAiStatus] = useState({
    is_running: false,
    market_regime: 'RANGING',
    active_positions: 0,
    total_pnl: 0,
    daily_pnl: 0,
    monthly_pnl: 0,
    risk_level: 'LOW',
    current_strategy: 'Standby',
    last_action: 'System Ready...'
  });

  const [positions, setPositions] = useState([]);
  const [marketData, setMarketData] = useState({});
  const [mt5Account, setMt5Account] = useState(null);
  const [activeTab, setActiveTab] = useState('settings');
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [logs, setLogs] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [ws, setWs] = useState(null);

  // API Functions
  const apiCall = async (endpoint, method = 'GET', data = null) => {
    try {
      const options = {
        method,
        headers: {
          'Content-Type': 'application/json',
        },
      };

      if (data && method !== 'GET') {
        options.body = JSON.stringify(data);
      }

      const response = await fetch(`${API_BASE_URL}${endpoint}`, options);
      
      if (!response.ok) {
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API call failed for ${endpoint}:`, error);
      addLog(`API Error: ${error.message}`, 'error');
      return null;
    }
  };

  // WebSocket Connection
  const connectWebSocket = useCallback(() => {
    try {
      const websocket = new WebSocket(WS_URL);
      
      websocket.onopen = () => {
        setConnectionStatus('connected');
        addLog('Connected to AI Backend', 'success');
      };

      websocket.onmessage = (event) => {
        const message = JSON.parse(event.data);
        handleWebSocketMessage(message);
      };

      websocket.onclose = () => {
        setConnectionStatus('disconnected');
        addLog('Disconnected from AI Backend', 'warning');
        // Auto-reconnect after 5 seconds
        setTimeout(connectWebSocket, 5000);
      };

      websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('error');
        addLog('WebSocket connection error', 'error');
      };

      setWs(websocket);
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
      setConnectionStatus('error');
    }
  }, []);

  // Handle WebSocket Messages
  const handleWebSocketMessage = (message) => {
    switch (message.type) {
      case 'ai_status_update':
        setAiStatus(message.data);
        break;
      case 'initial_data':
        setAiStatus(message.data);
        break;
      case 'ai_started':
        setAiStatus(prev => ({ ...prev, is_running: true }));
        addLog('AI Engine Started', 'success');
        break;
      case 'ai_stopped':
        setAiStatus(prev => ({ ...prev, is_running: false }));
        addLog('AI Engine Stopped', 'info');
        break;
      case 'position_update':
        setPositions(message.data.positions || []);
        break;
      case 'market_data_update':
        setMarketData(message.data.market_data || {});
        break;
      case 'emergency_stop':
        addLog('Emergency Stop Executed!', 'warning');
        break;
      default:
        console.log('Unknown message type:', message.type);
    }
  };

  // Add Log Entry
  const addLog = (message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [
      { id: Date.now(), timestamp, message, type },
      ...prev.slice(0, 49) // Keep only last 50 logs
    ]);
  };

  // Load Initial Data
  const loadInitialData = async () => {
    setIsLoading(true);
    
    // Load AI Status
    const status = await apiCall('/api/status');
    if (status) {
      setAiStatus(status);
    }

    // Load Positions
    const positionsData = await apiCall('/api/positions');
    if (positionsData) {
      setPositions(positionsData.positions || []);
    }

    // Load Market Data
    const marketDataResponse = await apiCall('/api/market-data');
    if (marketDataResponse) {
      setMarketData(marketDataResponse.market_data || {});
    }

    // Load MT5 Account Info
    const accountInfo = await apiCall('/api/mt5/account-info');
    if (accountInfo) {
      setMt5Account(accountInfo);
      setSettings(prev => ({ ...prev, account_balance: accountInfo.balance || prev.account_balance }));
    }

    setIsLoading(false);
  };

  // Component Mount
  useEffect(() => {
    connectWebSocket();
    loadInitialData();

    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, [connectWebSocket]);

  // Settings Functions
  const handleSettingChange = (key, value) => {
    setSettings(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const togglePairSelection = (pair) => {
    setSettings(prev => ({
      ...prev,
      enabled_pairs: prev.enabled_pairs.includes(pair)
        ? prev.enabled_pairs.filter(p => p !== pair)
        : [...prev.enabled_pairs, pair]
    }));
  };

  const saveSettings = async () => {
    setIsLoading(true);
    const result = await apiCall('/api/settings', 'POST', settings);
    if (result) {
      addLog('Settings saved successfully', 'success');
    }
    setIsLoading(false);
  };

  // AI Control Functions
  const startAI = async () => {
    setIsLoading(true);
    addLog('Starting AI Engine...', 'info');
    
    // Save settings first
    await saveSettings();
    
    // Start AI
    const result = await apiCall('/api/start-ai', 'POST');
    if (result) {
      addLog(`AI Engine ${result.status === 'success' ? 'started successfully' : result.message}`, 
             result.status === 'success' ? 'success' : 'warning');
    }
    setIsLoading(false);
  };

  const stopAI = async () => {
    setIsLoading(true);
    addLog('Stopping AI Engine...', 'info');
    
    const result = await apiCall('/api/stop-ai', 'POST');
    if (result) {
      addLog(`AI Engine ${result.status === 'success' ? 'stopped successfully' : result.message}`, 
             result.status === 'success' ? 'success' : 'warning');
    }
    setIsLoading(false);
  };

  const emergencyStop = async () => {
    if (window.confirm('Are you sure you want to execute Emergency Stop? This will close all positions!')) {
      setIsLoading(true);
      addLog('Executing Emergency Stop...', 'warning');
      
      const result = await apiCall('/api/emergency-stop', 'POST');
      if (result) {
        addLog('Emergency Stop completed', 'warning');
      }
      setIsLoading(false);
    }
  };

  // Format Currency
  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(amount);
  };

  // Format Number
  const formatNumber = (num, decimals = 2) => {
    return Number(num).toFixed(decimals);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-white flex items-center gap-3">
                <Brain className="text-purple-400" />
                AI Recovery Trading Brain
                {connectionStatus === 'connected' && <Wifi className="text-green-400" size={20} />}
                {connectionStatus === 'disconnected' && <WifiOff className="text-red-400" size={20} />}
                {connectionStatus === 'error' && <AlertCircle className="text-yellow-400" size={20} />}
              </h1>
              <p className="text-gray-300 mt-2">
                ระบบ AI เทรดแบบแก้ไม้อัตโนมัติ - Live MT5 Integration
                <span className={`ml-2 text-xs px-2 py-1 rounded ${
                  connectionStatus === 'connected' ? 'bg-green-600' : 
                  connectionStatus === 'error' ? 'bg-red-600' : 'bg-yellow-600'
                }`}>
                  {connectionStatus === 'connected' ? 'LIVE' : 
                   connectionStatus === 'error' ? 'ERROR' : 'CONNECTING...'}
                </span>
              </p>
            </div>
            <div className="flex gap-3">
              <Button 
                onClick={startAI}
                disabled={aiStatus.is_running || isLoading || connectionStatus !== 'connected'}
                className="bg-green-600 hover:bg-green-700 disabled:opacity-50"
              >
                {isLoading ? <Activity className="animate-spin" size={16} /> : null}
                {aiStatus.is_running ? 'AI กำลังทำงาน...' : 'เริ่ม AI'}
              </Button>
              <Button 
                onClick={stopAI}
                disabled={!aiStatus.is_running || isLoading}
                className="bg-blue-600 hover:bg-blue-700"
              >
                หยุด AI
              </Button>
              <Button 
                onClick={emergencyStop}
                disabled={isLoading}
                className="bg-red-600 hover:bg-red-700"
              >
                🚨 Emergency Stop
              </Button>
            </div>
          </div>
        </div>

        {/* AI Status Dashboard */}
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-6">
          <Card className="bg-slate-800 border-slate-700">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">สถานะ AI</p>
                  <p className="text-white font-semibold">{aiStatus.current_strategy}</p>
                </div>
                <div className={`w-3 h-3 rounded-full ${aiStatus.is_running ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800 border-slate-700">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Balance</p>
                  <p className="text-white font-semibold">
                    {mt5Account ? formatCurrency(mt5Account.balance) : formatCurrency(settings.account_balance)}
                  </p>
                </div>
                <Target className="text-blue-400" size={20} />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800 border-slate-700">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Equity</p>
                  <p className="text-white font-semibold">
                    {mt5Account ? formatCurrency(mt5Account.equity) : formatCurrency(settings.account_balance)}
                  </p>
                </div>
                <TrendingUp className="text-green-400" size={20} />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800 border-slate-700">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">กำไรวันนี้</p>
                  <p className={`font-semibold ${aiStatus.daily_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {formatCurrency(aiStatus.daily_pnl)}
                  </p>
                </div>
                <TrendingUp className="text-yellow-400" size={20} />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800 border-slate-700">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">โพซิชั่นเปิด</p>
                  <p className="text-white font-semibold">{positions.length}</p>
                  <p className="text-gray-400 text-xs">Risk: {aiStatus.risk_level}</p>
                </div>
                <Shield className="text-purple-400" size={20} />
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Content */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
          <TabsList className="grid w-full grid-cols-4 bg-slate-800">
            <TabsTrigger value="settings" className="text-white">การตั้งค่า AI</TabsTrigger>
            <TabsTrigger value="pairs" className="text-white">เลือกคู่เงิน</TabsTrigger>
            <TabsTrigger value="strategy" className="text-white">กลยุทธ์แก้ไม้</TabsTrigger>
            <TabsTrigger value="monitor" className="text-white">ติดตามการเทรด</TabsTrigger>
          </TabsList>

          {/* การตั้งค่า AI */}
          <TabsContent value="settings" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* MT5 Account Info */}
              {mt5Account && (
                <Card className="bg-slate-800 border-slate-700 md:col-span-2">
                  <CardHeader>
                    <CardTitle className="text-white flex items-center gap-2">
                      <Activity className="text-green-400" size={20} />
                      MT5 Account Information (Live)
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div>
                        <p className="text-gray-400 text-sm">Account</p>
                        <p className="text-white font-semibold">{mt5Account.login}</p>
                      </div>
                      <div>
                        <p className="text-gray-400 text-sm">Balance</p>
                        <p className="text-white font-semibold">{formatCurrency(mt5Account.balance)}</p>
                      </div>
                      <div>
                        <p className="text-gray-400 text-sm">Equity</p>
                        <p className="text-white font-semibold">{formatCurrency(mt5Account.equity)}</p>
                      </div>
                      <div>
                        <p className="text-gray-400 text-sm">Free Margin</p>
                        <p className="text-white font-semibold">{formatCurrency(mt5Account.free_margin)}</p>
                      </div>
                      <div>
                        <p className="text-gray-400 text-sm">Profit</p>
                        <p className={`font-semibold ${mt5Account.profit >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {formatCurrency(mt5Account.profit)} ({formatNumber(mt5Account.profit_percentage)}%)
                        </p>
                      </div>
                      <div>
                        <p className="text-gray-400 text-sm">Margin Level</p>
                        <p className="text-white font-semibold">{formatNumber(mt5Account.margin_level)}%</p>
                      </div>
                      <div>
                        <p className="text-gray-400 text-sm">Broker</p>
                        <p className="text-white font-semibold">{mt5Account.company}</p>
                      </div>
                      <div>
                        <p className="text-gray-400 text-sm">Server</p>
                        <p className="text-white font-semibold">{mt5Account.server}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}

              <Card className="bg-slate-800 border-slate-700">
                <CardHeader>
                  <CardTitle className="text-white flex items-center gap-2">
                    <Settings className="text-blue-400" size={20} />
                    การตั้งค่าการเทรด
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <label className="text-gray-300 text-sm">เป้าหมายกำไรรายวัน (%)</label>
                    <Input
                      type="number"
                      step="0.1"
                      value={settings.daily_target_pct}
                      onChange={(e) => handleSettingChange('daily_target_pct', parseFloat(e.target.value))}
                      className="bg-slate-700 border-slate-600 text-white mt-1"
                    />
                  </div>
                  
                  <div>
                    <label className="text-gray-300 text-sm">เป้าหมายกำไรรายเดือน (%)</label>
                    <Input
                      type="number"
                      step="0.1"
                      value={settings.monthly_target_pct}
                      onChange={(e) => handleSettingChange('monthly_target_pct', parseFloat(e.target.value))}
                      className="bg-slate-700 border-slate-600 text-white mt-1"
                    />
                  </div>
                  
                  <div>
                    <label className="text-gray-300 text-sm">ขนาด Lot เริ่มต้น</label>
                    <Input
                      type="number"
                      step="0.01"
                      value={settings.initial_lot_size}
                      onChange={(e) => handleSettingChange('initial_lot_size', parseFloat(e.target.value))}
                      className="bg-slate-700 border-slate-600 text-white mt-1"
                    />
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-slate-800 border-slate-700">
                <CardHeader>
                  <CardTitle className="text-white flex items-center gap-2">
                    <Shield className="text-red-400" size={20} />
                    การจัดการความเสี่ยง
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <label className="text-gray-300 text-sm">จำนวนชั้นแก้ไม้สูงสุด</label>
                    <Input
                      type="number"
                      value={settings.max_recovery_levels}
                      onChange={(e) => handleSettingChange('max_recovery_levels', parseInt(e.target.value))}
                      className="bg-slate-700 border-slate-600 text-white mt-1"
                    />
                  </div>
                  
                  <div>
                    <label className="text-gray-300 text-sm">ตัวคูณขนาดแก้ไม้</label>
                    <Input
                      type="number"
                      step="0.1"
                      value={settings.recovery_multiplier}
                      onChange={(e) => handleSettingChange('recovery_multiplier', parseFloat(e.target.value))}
                      className="bg-slate-700 border-slate-600 text-white mt-1"
                    />
                  </div>
                  
                  <div>
                    <label className="text-gray-300 text-sm">ความเสี่ยงสูงสุดของพอร์ต (%)</label>
                    <Input
                      type="number"
                      step="1"
                      value={settings.max_portfolio_risk}
                      onChange={(e) => handleSettingChange('max_portfolio_risk', parseFloat(e.target.value))}
                      className="bg-slate-700 border-slate-600 text-white mt-1"
                    />
                  </div>

                  <div className="pt-4">
                    <Button 
                      onClick={saveSettings} 
                      disabled={isLoading}
                      className="w-full bg-blue-600 hover:bg-blue-700"
                    >
                      {isLoading ? 'กำลังบันทึก...' : 'บันทึกการตั้งค่า'}
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* เลือกคู่เงิน */}
          <TabsContent value="pairs" className="space-y-4">
            <Card className="bg-slate-800 border-slate-700">
              <CardHeader>
                <CardTitle className="text-white">เลือกคู่สกุลเงินสำหรับ AI Trading</CardTitle>
                <p className="text-gray-400">เลือกคู่สกุลเงินที่ต้องการให้ AI เทรด (จาก MT5 Markets)</p>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  {/* Major Pairs */}
                  <div>
                    <h3 className="text-white font-semibold mb-3 flex items-center gap-2">
                      <Badge variant="default">Major Pairs</Badge>
                      คู่เงินหลัก - สภาพคล่องสูง
                    </h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                      {Object.entries(mt5Pairs.majors).map(([symbol, data]) => (
                        <div
                          key={symbol}
                          onClick={() => togglePairSelection(symbol)}
                          className={`p-3 rounded-lg border cursor-pointer transition-all ${
                            settings.enabled_pairs.includes(symbol)
                              ? 'bg-blue-600 border-blue-500 text-white'
                              : 'bg-slate-700 border-slate-600 text-gray-300 hover:bg-slate-600'
                          }`}
                        >
                          <div className="font-semibold">{symbol}</div>
                          <div className="text-xs opacity-75">{data.name}</div>
                          {marketData[symbol] && (
                            <div className="text-xs mt-1">
                              Bid: {formatNumber(marketData[symbol].bid, 5)}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Cross Pairs */}
                  <div>
                    <h3 className="text-white font-semibold mb-3 flex items-center gap-2">
                      <Badge variant="secondary">Cross Pairs</Badge>
                      คู่เงินไขว้ - เหมาะสำหรับ Correlation
                    </h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                      {Object.entries(mt5Pairs.crosses).map(([symbol, data]) => (
                        <div
                          key={symbol}
                          onClick={() => togglePairSelection(symbol)}
                          className={`p-3 rounded-lg border cursor-pointer transition-all ${
                            settings.enabled_pairs.includes(symbol)
                              ? 'bg-green-600 border-green-500 text-white'
                              : 'bg-slate-700 border-slate-600 text-gray-300 hover:bg-slate-600'
                          }`}
                        >
                          <div className="font-semibold text-sm">{symbol}</div>
                          <div className="text-xs opacity-75">{data.name}</div>
                          {marketData[symbol] && (
                            <div className="text-xs mt-1">
                              Bid: {formatNumber(marketData[symbol].bid, 5)}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Metals */}
                  <div>
                    <h3 className="text-white font-semibold mb-3 flex items-center gap-2">
                      <Badge variant="outline" className="border-yellow-500 text-yellow-400">Metals</Badge>
                      โลหะมีค่า - Safe Haven
                    </h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                      {Object.entries(mt5Pairs.metals).map(([symbol, data]) => (
                        <div
                          key={symbol}
                          onClick={() => togglePairSelection(symbol)}
                          className={`p-3 rounded-lg border cursor-pointer transition-all ${
                            settings.enabled_pairs.includes(symbol)
                              ? 'bg-yellow-600 border-yellow-500 text-white'
                              : 'bg-slate-700 border-slate-600 text-gray-300 hover:bg-slate-600'
                          }`}
                        >
                          <div className="font-semibold">{symbol}</div>
                          <div className="text-xs opacity-75">{data.name}</div>
                          {marketData[symbol] && (
                            <div className="text-xs mt-1">
                              Bid: {formatNumber(marketData[symbol].bid, 2)}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* คู่เงินที่เลือก */}
                  <div className="mt-6 p-4 bg-slate-700 rounded-lg">
                    <h4 className="text-white font-semibold mb-2">คู่เงินที่เลือกแล้ว ({settings.enabled_pairs.length} คู่)</h4>
                    <div className="flex flex-wrap gap-2">
                      {settings.enabled_pairs.map(pair => (
                        <Badge key={pair} variant="default" className="bg-blue-600">
                          {pair}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* กลยุทธ์แก้ไม้ */}
          <TabsContent value="strategy" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card className="bg-slate-800 border-slate-700">
                <CardHeader>
                  <CardTitle className="text-white">กลยุทธ์การแก้ไม้</CardTitle>
                  <p className="text-gray-400">เลือกวิธีการแก้ไม้ที่ AI จะใช้</p>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-white font-semibold">Correlation Recovery</p>
                      <p className="text-gray-400 text-sm">ใช้คู่เงินที่มี correlation ในการแก้ไม้</p>
                    </div>
                    <Switch
                      checked={settings.use_correlation_recovery}
                      onCheckedChange={(checked) => handleSettingChange('use_correlation_recovery', checked)}
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-white font-semibold">Grid Recovery</p>
                      <p className="text-gray-400 text-sm">เพิ่มโพซิชั่นตามระยะห่าง Grid</p>
                    </div>
                    <Switch
                      checked={settings.use_grid_recovery}
                      onCheckedChange={(checked) => handleSettingChange('use_grid_recovery', checked)}
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-white font-semibold">Basket Recovery</p>
                      <p className="text-gray-400 text-sm">ใช้ตะกร้าสกุลเงินในการแก้ไม้</p>
                    </div>
                    <Switch
                      checked={settings.use_basket_recovery}
                      onCheckedChange={(checked) => handleSettingChange('use_basket_recovery', checked)}
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-white font-semibold">Arbitrage Recovery</p>
                      <p className="text-gray-400 text-sm">หาโอกาส Arbitrage เพื่อแก้ไม้</p>
                    </div>
                    <Switch
                      checked={settings.use_arbitrage_recovery}
                      onCheckedChange={(checked) => handleSettingChange('use_arbitrage_recovery', checked)}
                    />
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-slate-800 border-slate-700">
                <CardHeader>
                  <CardTitle className="text-white">Market Data (Live)</CardTitle>
                  <p className="text-gray-400">ข้อมูลตลาดแบบ Real-time</p>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2 max-h-64 overflow-y-auto">
                    {Object.entries(marketData).map(([symbol, data]) => (
                      <div key={symbol} className="flex justify-between items-center p-2 bg-slate-700 rounded">
                        <span className="text-white font-semibold">{symbol}</span>
                        <div className="text-right">
                          <div className="text-white text-sm">
                            {formatNumber(data.bid, symbol.includes('JPY') ? 3 : 5)} / {formatNumber(data.ask, symbol.includes('JPY') ? 3 : 5)}
                          </div>
                          <div className="text-gray-400 text-xs">
                            Spread: {formatNumber(data.spread, symbol.includes('JPY') ? 3 : 5)}
                          </div>
                        </div>
                      </div>
                    ))}
                    {Object.keys(marketData).length === 0 && (
                      <div className="text-center py-4 text-gray-400">
                        No market data available
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* ติดตามการเทรด */}
          <TabsContent value="monitor" className="space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="bg-slate-800 border-slate-700">
                <CardHeader>
                  <CardTitle className="text-white">โพซิชั่นปัจจุบัน ({positions.length})</CardTitle>
                </CardHeader>
                <CardContent>
                  {positions.length === 0 ? (
                    <div className="text-center py-8">
                      <AlertCircle className="mx-auto text-gray-400 mb-2" size={48} />
                      <p className="text-gray-400">ไม่มีโพซิชั่นเปิดขณะนี้</p>
                    </div>
                  ) : (
                    <div className="space-y-2 max-h-64 overflow-y-auto">
                      {positions.map((position, index) => (
                        <div key={position.position_id || index} className="p-3 bg-slate-700 rounded-lg">
                          <div className="flex justify-between items-center">
                            <div>
                              <span className="text-white font-semibold">{position.symbol}</span>
                              <Badge variant={position.direction === 'BUY' ? 'default' : 'destructive'} className="ml-2">
                                {position.direction}
                              </Badge>
                              {position.is_recovery && (
                                <Badge variant="outline" className="ml-1 text-yellow-400 border-yellow-400">
                                  Recovery L{position.recovery_level}
                                </Badge>
                              )}
                            </div>
                            <div className="text-right">
                              <div className={`font-semibold ${(position.pnl || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                {formatCurrency(position.pnl || 0)}
                              </div>
                              <div className="text-gray-400 text-xs">
                                {formatNumber(position.pnl_pips || 0, 1)} pips
                              </div>
                            </div>
                          </div>
                          <div className="mt-2 text-xs text-gray-400">
                            Size: {position.size} | Entry: {formatNumber(position.entry_price, 5)} | Current: {formatNumber(position.current_price, 5)}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>

              <Card className="bg-slate-800 border-slate-700">
                <CardHeader>
                  <CardTitle className="text-white">AI Decision Log</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2 max-h-64 overflow-y-auto">
                    {logs.map((log) => (
                      <div key={log.id} className="p-2 bg-slate-700 rounded text-sm">
                        <span className="text-gray-400">{log.timestamp}</span>
                        <span className={`ml-2 ${
                          log.type === 'error' ? 'text-red-400' :
                          log.type === 'success' ? 'text-green-400' :
                          log.type === 'warning' ? 'text-yellow-400' :
                          'text-white'
                        }`}>
                          {log.message}
                        </span>
                      </div>
                    ))}
                    {logs.length === 0 && (
                      <div className="text-center py-4 text-gray-400">
                        No logs available
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Market Overview */}
            <Card className="bg-slate-800 border-slate-700">
              <CardHeader>
                <CardTitle className="text-white">Market Overview</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center">
                    <p className="text-gray-400 text-sm">Market Regime</p>
                    <Badge variant="outline" className="mt-1">
                      {aiStatus.market_regime}
                    </Badge>
                  </div>
                  <div className="text-center">
                    <p className="text-gray-400 text-sm">Total P&L</p>
                    <p className={`font-semibold mt-1 ${aiStatus.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {formatCurrency(aiStatus.total_pnl)}
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-gray-400 text-sm">Daily P&L</p>
                    <p className={`font-semibold mt-1 ${aiStatus.daily_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {formatCurrency(aiStatus.daily_pnl)}
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-gray-400 text-sm">Risk Level</p>
                    <Badge variant={
                      aiStatus.risk_level === 'LOW' ? 'secondary' : 
                      aiStatus.risk_level === 'MEDIUM' ? 'default' : 'destructive'
                    } className="mt-1">
                      {aiStatus.risk_level}
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        {/* Footer Info */}
        <div className="mt-6 p-4 bg-slate-800 rounded-lg border border-slate-700">
          <div className="flex items-center gap-2 text-yellow-400 mb-2">
            <AlertCircle size={16} />
            <span className="font-semibold">Live System Status</span>
          </div>
          <ul className="text-gray-400 text-sm space-y-1">
            <li>• ระบบเชื่อมต่อ MT5 แบบ Real-time และดึงข้อมูลบัญชีจริง</li>
            <li>• AI ตัดสินใจการเทรดอัตโนมัติและส่งคำสั่งไปยัง MT5</li>
            <li>• WebSocket connection สำหรับข้อมูล Live Updates</li>
            <li>• ระบบแก้ไม้อัตโนมัติทำงานแบบ Real-time</li>
            <li>• กดปุ่ม Emergency Stop เพื่อปิดโพซิชั่นทั้งหมดทันที</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default AITradingDashboard;