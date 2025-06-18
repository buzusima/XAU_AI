@echo off
echo ทดสอบ MT5...

python -c "import MetaTrader5; print('MT5 OK')"
if errorlevel 1 (
    echo MT5 ไม่ได้
    pip install MetaTrader5 --force-reinstall --no-deps
    python -c "import MetaTrader5; print('MT5 OK หลังติดตั้ง')"
)

echo.
echo รัน Lightning Scalper โดยตรง:
if exist "lightning_scalper_fixed_mt5.py" (
    python lightning_scalper_fixed_mt5.py
) else (
    echo ไม่พบไฟล์ .py
)

pause