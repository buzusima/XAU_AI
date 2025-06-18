@echo off
echo Building Simple Lightning Scalper...
pip install flask cryptography
pyinstaller --onefile --windowed --name Lightning_Scalper_Simple lightning_scalper_simple.py
echo Done! Check dist folder.
pause