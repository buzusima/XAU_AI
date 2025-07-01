import os

# Quick fix for unicode characters
file_path = 'correlation_hedging_system.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('→', '->')
content = content.replace('←', '<-')
content = content.replace('🎯', '[TARGET]')
content = content.replace('💰', '[MONEY]')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Fixed unicode characters!")