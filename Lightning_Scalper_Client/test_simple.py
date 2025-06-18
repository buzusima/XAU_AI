#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightning Scalper - Simple Test Version
ทดสอบการทำงานขั้นพื้นฐาน
"""

from flask import Flask, render_template_string
from flask_socketio import SocketIO
import webbrowser
import threading
import time

# Simple HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Lightning Scalper Test</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 50px; }
        .container { max-width: 800px; margin: 0 auto; }
        .card { background: #f5f5f5; padding: 20px; border-radius: 10px; margin: 20px 0; }
        .btn { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
        .success { background: #d4edda; color: #155724; }
        .error { background: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Lightning Scalper Test</h1>
        
        <div class="card">
            <h3>System Status</h3>
            <div id="status" class="status success">System Running Successfully! ✅</div>
            <p>Current Time: <span id="time"></span></p>
        </div>
        
        <div class="card">
            <h3>Test Functions</h3>
            <button class="btn" onclick="testFunction()">Test Connection</button>
            <button class="btn" onclick="alert('Hello from Lightning Scalper!')">Test Alert</button>
        </div>
        
        <div class="card">
            <h3>Next Steps</h3>
            <p>✅ Basic system is working</p>
            <p>⏳ Ready to integrate with MT5</p>
            <p>⏳ Ready to connect to main server</p>
        </div>
    </div>
    
    <script>
        function updateTime() {
            document.getElementById('time').innerText = new Date().toLocaleString();
        }
        
        function testFunction() {
            document.getElementById('status').innerHTML = 'Test function executed! ✅';
        }
        
        // Update time every second
        setInterval(updateTime, 1000);
        updateTime();
    </script>
</body>
</html>
"""

class SimpleTestApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'test_key'
        
        # Initialize SocketIO with simple config
        self.socketio = SocketIO(
            self.app, 
            cors_allowed_origins="*",
            async_mode='threading'
        )
        
        self._setup_routes()
    
    def _setup_routes(self):
        @self.app.route('/')
        def dashboard():
            return render_template_string(HTML_TEMPLATE)
        
        @self.app.route('/test')
        def test():
            return {"status": "OK", "message": "Lightning Scalper Test Working!"}
    
    def run(self, host='127.0.0.1', port=5000):
        print("🚀 Lightning Scalper Test Starting...")
        print(f"   URL: http://{host}:{port}")
        print("   Opening browser automatically...")
        
        # Open browser after delay
        threading.Timer(1.5, lambda: webbrowser.open(f"http://{host}:{port}")).start()
        
        # Run the app
        try:
            self.socketio.run(self.app, host=host, port=port, debug=False)
        except Exception as e:
            print(f"Error: {e}")
            print("Press Enter to exit...")
            input()

def main():
    print("🧪 Lightning Scalper - Simple Test Version")
    print("=" * 50)
    
    app = SimpleTestApp()
    app.run()

if __name__ == "__main__":
    main()