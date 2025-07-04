# remove_emojis.py - Script เพื่อลบ emoji ทั้งหมดออกจากไฟล์ Python

import os
import re
import shutil
from datetime import datetime

def remove_emojis_from_files():
    """ลบ emoji ทั้งหมดออกจากไฟล์ Python"""
    
    # รายการไฟล์ที่ต้องการทำความสะอาด
    python_files = [
        'mt5_forex_connector.py',
        'advanced_features.py',
        'trading_integration.py',
        'correlation_hedging_system.py',
        'forex_dashboard.html'  # ถ้ามี emoji ใน HTML ด้วย
    ]
    
    # Emoji และ special character mappings
    emoji_replacements = {
        # Arrows
        '→': '->',
        '←': '<-',
        '↑': '^',
        '↓': 'v',
        '↔': '<->',
        
        # Status indicators
        '✅': '[OK]',
        '❌': '[ERR]',
        '⚠️': '[WARN]',
        '🚨': '[ALERT]',
        '🔔': '[BELL]',
        '🚀': '[GO]',
        '🎯': '[TARGET]',
        '🔥': '[HOT]',
        '💡': '[IDEA]',
        '⭐': '[STAR]',
        '🌟': '[STAR]',
        '🏆': '[WIN]',
        '💎': '[DIAMOND]',
        
        # Charts and data
        '📊': '[CHART]',
        '📈': '[UP]',
        '📉': '[DOWN]',
        '📱': '[MOBILE]',
        '💻': '[PC]',
        '🖥️': '[SCREEN]',
        '📺': '[MONITOR]',
        
        # Money and trading
        '💰': '[MONEY]',
        '💵': '[CASH]',
        '💳': '[CARD]',
        '🏦': '[BANK]',
        '📦': '[BOX]',
        '🎲': '[DICE]',
        '🎪': '[CIRCUS]',
        
        # Tools and tech
        '🔧': '[TOOL]',
        '⚙️': '[GEAR]',
        '🔨': '[HAMMER]',
        '🔩': '[BOLT]',
        '⚡': '[LIGHTNING]',
        '🔌': '[PLUG]',
        '🔋': '[BATTERY]',
        '💾': '[DISK]',
        '💿': '[CD]',
        '📀': '[DVD]',
        
        # Security
        '🔒': '[LOCK]',
        '🔓': '[UNLOCK]',
        '🔑': '[KEY]',
        '🛡️': '[SHIELD]',
        '🔐': '[SECURE]',
        
        # Time and clocks
        '⏰': '[TIME]',
        '⏱️': '[TIMER]',
        '⏲️': '[CLOCK]',
        '🕐': '[1PM]',
        '🕑': '[2PM]',
        '🕒': '[3PM]',
        
        # Other symbols
        '🔄': '[REFRESH]',
        '🔀': '[SHUFFLE]',
        '🔁': '[REPEAT]',
        '🔂': '[REPEAT1]',
        '▶️': '[PLAY]',
        '⏸️': '[PAUSE]',
        '⏹️': '[STOP]',
        '⏭️': '[NEXT]',
        '⏮️': '[PREV]',
        '⏩': '[FF]',
        '⏪': '[RW]',
        
        # Shapes and objects
        '🔍': '[SEARCH]',
        '🔎': '[ZOOM]',
        '🎨': '[ART]',
        '🎭': '[MASK]',
        '🎮': '[GAME]',
        '🎵': '[MUSIC]',
        '🎶': '[NOTES]',
        '📝': '[NOTE]',
        '📋': '[CLIPBOARD]',
        '📄': '[PAGE]',
        '📃': '[DOC]',
        '📑': '[PAGES]',
        
        # Network and web
        '🌐': '[WEB]',
        '🌍': '[WORLD]',
        '🌎': '[EARTH]',
        '🌏': '[GLOBE]',
        '📡': '[SATELLITE]',
        '📶': '[SIGNAL]',
        '📳': '[VIBRATE]',
        '📴': '[OFF]',
        
        # Miscellaneous
        '🔮': '[CRYSTAL]',
        '🎳': '[BOWLING]',
        '🎯': '[DART]',
        '🎪': '[TENT]',
        '🎡': '[WHEEL]',
        '🎢': '[COASTER]',
        '🎠': '[CAROUSEL]',
        '🎨': '[PAINT]',
        '🎬': '[MOVIE]',
        '🎤': '[MIC]',
        '🎧': '[HEADPHONES]',
        '🎸': '[GUITAR]',
        '🎹': '[PIANO]',
        '🎺': '[TRUMPET]',
        '🎻': '[VIOLIN]',
        '🥁': '[DRUMS]',
        '🎲': '[DICE]',
        '🧩': '[PUZZLE]',
        '🃏': '[JOKER]',
        '🎴': '[CARDS]',
        '🀄': '[MAHJONG]',
        '🎯': '[TARGET]'
    }
    
    # Unicode emoji pattern (covers most emojis)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"  # misc symbols
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA70-\U0001FAFF"  # extended symbols
        "]+", 
        flags=re.UNICODE
    )
    
    def clean_file(filepath):
        """ทำความสะอาดไฟล์หนึ่งไฟล์"""
        try:
            # # สำรองไฟล์เดิม
            # backup_path = f"{filepath}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            # shutil.copy2(filepath, backup_path)
            # print(f"📄 Backup created: {backup_path}")
            
            # อ่านไฟล์
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_length = len(content)
            emojis_found = []
            
            # แทนที่ emoji ที่กำหนดไว้ก่อน
            for emoji, replacement in emoji_replacements.items():
                if emoji in content:
                    emojis_found.append(f"{emoji} -> {replacement}")
                    content = content.replace(emoji, replacement)
            
            # ลบ emoji ที่เหลือด้วย regex
            remaining_emojis = emoji_pattern.findall(content)
            if remaining_emojis:
                emojis_found.extend([f"{emoji} -> [REMOVED]" for emoji in set(remaining_emojis)])
                content = emoji_pattern.sub('[EMOJI]', content)
            
            # เขียนไฟล์ใหม่
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            new_length = len(content)
            
            print(f"✅ Cleaned: {filepath}")
            print(f"   Size: {original_length} -> {new_length} characters")
            if emojis_found:
                print(f"   Emojis replaced: {len(emojis_found)}")
                for emoji_info in emojis_found[:10]:  # Show first 10
                    print(f"     {emoji_info}")
                if len(emojis_found) > 10:
                    print(f"     ... and {len(emojis_found) - 10} more")
            else:
                print("   No emojis found")
            print()
            
            return len(emojis_found)
            
        except Exception as e:
            print(f"❌ Error cleaning {filepath}: {str(e)}")
            return 0
    
    # เริ่มทำความสะอาด
    print("🧹 Starting emoji cleanup process...")
    print("=" * 50)
    
    total_emojis_removed = 0
    files_processed = 0
    
    for filename in python_files:
        if os.path.exists(filename):
            emojis_removed = clean_file(filename)
            total_emojis_removed += emojis_removed
            files_processed += 1
        else:
            print(f"⚠️ File not found: {filename}")
    
    print("=" * 50)
    print(f"🏁 Cleanup completed!")
    print(f"   Files processed: {files_processed}")
    print(f"   Total emojis removed/replaced: {total_emojis_removed}")
    print(f"   Backup files created with timestamp")
    print()
    print("✅ All Unicode logging errors should now be resolved!")
    print("💡 If you need to restore, use the .backup_ files")

if __name__ == "__main__":
    remove_emojis_from_files()