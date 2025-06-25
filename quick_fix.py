#!/usr/bin/env python3
"""
Quick Fix Script for Common Syntax Errors
=========================================
แก้ไขปัญหา syntax errors พื้นฐานที่พบบ่อย
"""

import re

def quick_fix_text(text):
    """แก้ไขปัญหาพื้นฐาน"""
    
    # แก้ไข indentation ที่ผิด
    lines = text.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # แก้ไข decorator ที่ไม่มี function
        if line.strip().startswith('@self.app.route') and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if not next_line.startswith('def '):
                fixed_lines.append(line)
                # เพิ่ม function placeholder
                indent = ' ' * (len(line) - len(line.lstrip()))
                fixed_lines.append(f'{indent}def route_handler():')
                fixed_lines.append(f'{indent}    """Auto-generated handler"""')
                fixed_lines.append(f'{indent}    pass')
                continue
        
        # แก้ไข unexpected indentation
        if line.strip() and not line.strip().startswith(('#', '@', 'def ', 'class ')):
            if len(line) - len(line.lstrip()) > 50:  # ถ้า indent มากเกินไป
                line = '        ' + line.lstrip()  # ใช้ 8 spaces
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

if __name__ == "__main__":
    # ใช้สำหรับแก้ไขข้อความด่วน
    print("Quick Fix Script Ready")
