"""
FRANZ - Procedural Desktop Agent
=================================

Simplified architecture with pre-defined task procedures.
No Python execution, explicit desktop application workflows.

Built by Wojciech Gabrys
Date: 2025-02-09
Version: Procedural (hardcoded workflows)
"""

from __future__ import annotations

import base64
import ctypes
import ctypes.wintypes as w
import json
import re
import struct
import urllib.request
import zlib
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Any

try:
    ULONG_PTR = w.ULONG_PTR
except AttributeError:
    ULONG_PTR = ctypes.c_uint64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_uint32

# ============================================================================
# CONFIGURATION
# ============================================================================

API_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "qwen3-vl-2b-instruct-1m"
RES_W, RES_H = 536, 364

CONFIG = {
    "max_tokens": 250,
    "cycle_delay_ms": 1000,
}

SAMPLING = {
    "temperature": 0.6,
    "top_p": 0.85,
    "max_tokens": CONFIG["max_tokens"],
}

# ============================================================================
# TASK PROCEDURES - Hardcoded Workflows
# ============================================================================

TASK_PROCEDURES = {
    "open_paint": {
        "steps": [
            {"action": "key", "args": ["win"], "verify": "Start menu visible"},
            {"action": "wait", "args": ["500"]},
            {"action": "type", "args": ["paint"], "verify": "Search field shows 'paint'"},
            {"action": "wait", "args": ["800"]},
            {"action": "key", "args": ["enter"], "verify": "Paint window appears"},
            {"action": "wait", "args": ["1500"]},
        ],
        "completion_check": "MS Paint window visible with toolbar and white canvas"
    },
    
    "draw_cat_paint": {
        "steps": [
            {"action": "click", "args": ["200", "150"], "note": "Click brush tool area"},
            {"action": "wait", "args": ["300"]},
            {"action": "drag", "args": ["400", "300", "600", "300"], "note": "Draw body horizontal oval"},
            {"action": "wait", "args": ["200"]},
            {"action": "drag", "args": ["450", "250", "550", "250"], "note": "Draw head horizontal line"},
            {"action": "wait", "args": ["200"]},
            {"action": "drag", "args": ["470", "220", "450", "250"], "note": "Left ear triangle"},
            {"action": "wait", "args": ["200"]},
            {"action": "drag", "args": ["530", "220", "550", "250"], "note": "Right ear triangle"},
            {"action": "wait", "args": ["200"]},
            {"action": "drag", "args": ["420", "270", "380", "270"], "note": "Left whisker"},
            {"action": "wait", "args": ["200"]},
            {"action": "drag", "args": ["580", "270", "620", "270"], "note": "Right whisker"},
            {"action": "wait", "args": ["200"]},
            {"action": "drag", "args": ["600", "300", "650", "350"], "note": "Tail curve"},
        ],
        "completion_check": "Cat-like drawing visible on canvas with body, head, ears, whiskers"
    },
}

# ============================================================================
# TASK MATCHING
# ============================================================================

def match_task_to_procedure(goal_text: str) -> list[str]:
    """Match user goal to available procedures."""
    goal_lower = goal_text.lower()
    procedures = []
    
    if "open" in goal_lower and "paint" in goal_lower:
        procedures.append("open_paint")
    
    if "draw" in goal_lower and "cat" in goal_lower:
        procedures.append("draw_cat_paint")
    
    return procedures

# ============================================================================
# SIMPLIFIED PROMPTS
# ============================================================================

VISION_PROMPT = """You are a vision system. Describe what you see on screen.

Focus on:
- Open applications (name them specifically: MS Paint, Notepad, etc.)
- UI elements (toolbars, menus, buttons)
- Canvas/document content (is it blank or has content?)
- Text visible on screen

Output format (50-80 words):
APPLICATIONS OPEN: [list]
MAIN WINDOW: [which app is in focus]
CONTENT: [what's visible in main window]
NOTES: [anything unusual or important]

Be factual. Do not infer goals or intentions."""

COMPLETION_PROMPT = """You verify task completion by comparing before/after screens.

Task: {task_name}
Expected result: {completion_check}

Look at AFTER screenshot and answer:
TASK_COMPLETE: [YES / NO / PARTIAL]
EVIDENCE: [what you see that confirms/denies completion]

Be strict. Only say YES if result clearly matches expected state."""

# ============================================================================
# SCREEN CAPTURE (same as before)
# ============================================================================

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)

try:
    shcore = ctypes.WinDLL("shcore", use_last_error=True)
except Exception:
    shcore = None

SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79
DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = ctypes.c_void_p(-4)

def _enable_dpi_awareness() -> None:
    try:
        if hasattr(user32, "SetProcessDpiAwarenessContext"):
            user32.SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
            return
    except Exception:
        pass
    try:
        if shcore is not None and hasattr(shcore, "SetProcessDpiAwareness"):
            shcore.SetProcessDpiAwareness(2)
            return
    except Exception:
        pass

def _virtual_screen() -> tuple[int, int, int, int]:
    x = int(user32.GetSystemMetrics(SM_XVIRTUALSCREEN))
    y = int(user32.GetSystemMetrics(SM_YVIRTUALSCREEN))
    wv = int(user32.GetSystemMetrics(SM_CXVIRTUALSCREEN))
    hv = int(user32.GetSystemMetrics(SM_CYVIRTUALSCREEN))
    return x, y, wv, hv

class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", w.DWORD),
        ("biWidth", w.LONG),
        ("biHeight", w.LONG),
        ("biPlanes", w.WORD),
        ("biBitCount", w.WORD),
        ("biCompression", w.DWORD),
        ("biSizeImage", w.DWORD),
        ("biXPelsPerMeter", w.LONG),
        ("biYPelsPerMeter", w.LONG),
        ("biClrUsed", w.DWORD),
        ("biClrImportant", w.DWORD),
    ]

class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", w.DWORD * 3)]

def capture_screen(dw: int, dh: int) -> tuple[bytes, tuple[int, int, int, int]]:
    vx, vy, vw, vh = _virtual_screen()
    sdc = user32.GetDC(0)
    src_dc = gdi32.CreateCompatibleDC(sdc)
    dst_dc = gdi32.CreateCompatibleDC(sdc)
    src_bmp = None
    dst_bmp = None
    try:
        bmi_src = BITMAPINFO()
        bmi_src.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi_src.bmiHeader.biWidth = vw
        bmi_src.bmiHeader.biHeight = -vh
        bmi_src.bmiHeader.biPlanes = 1
        bmi_src.bmiHeader.biBitCount = 32
        src_bits = ctypes.c_void_p()
        src_bmp = gdi32.CreateDIBSection(sdc, ctypes.byref(bmi_src), 0, ctypes.byref(src_bits), None, 0)
        gdi32.SelectObject(src_dc, src_bmp)
        gdi32.BitBlt(src_dc, 0, 0, vw, vh, sdc, vx, vy, 0x40CC0020)
        bmi_dst = BITMAPINFO()
        bmi_dst.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi_dst.bmiHeader.biWidth = dw
        bmi_dst.bmiHeader.biHeight = -dh
        bmi_dst.bmiHeader.biPlanes = 1
        bmi_dst.bmiHeader.biBitCount = 32
        dst_bits = ctypes.c_void_p()
        dst_bmp = gdi32.CreateDIBSection(sdc, ctypes.byref(bmi_dst), 0, ctypes.byref(dst_bits), None, 0)
        gdi32.SelectObject(dst_dc, dst_bmp)
        gdi32.SetStretchBltMode(dst_dc, 4)
        gdi32.StretchBlt(dst_dc, 0, 0, dw, dh, src_dc, 0, 0, vw, vh, 0x00CC0020)
        size = dw * dh * 4
        bgra = bytes((ctypes.c_ubyte * size).from_address(dst_bits.value))
    finally:
        if dst_bmp:
            gdi32.DeleteObject(dst_bmp)
        if src_bmp:
            gdi32.DeleteObject(src_bmp)
        gdi32.DeleteDC(dst_dc)
        gdi32.DeleteDC(src_dc)
        user32.ReleaseDC(0, sdc)
    rgba = bytearray(len(bgra))
    rgba[0::4] = bgra[2::4]
    rgba[1::4] = bgra[1::4]
    rgba[2::4] = bgra[0::4]
    rgba[3::4] = b"\xff" * (size // 4)
    return _rgba_to_png(bytes(rgba), dw, dh), (vx, vy, vw, vh)

def _rgba_to_png(rgba: bytes, wv: int, hv: int) -> bytes:
    raw = bytearray()
    stride = wv * 4
    for y in range(hv):
        raw.append(0)
        raw.extend(rgba[y * stride : (y + 1) * stride])
    ihdr = struct.pack(">IIBBBBB", wv, hv, 8, 6, 0, 0, 0)
    idat = zlib.compress(bytes(raw), 6)
    def _chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")

# ============================================================================
# VLM INTERFACE
# ============================================================================

def call_vlm(system_prompt: str, user_text: str, images: list[bytes]) -> str:
    content = [{"type": "text", "text": user_text}]
    for img in images:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(img).decode()}"}})
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        **SAMPLING,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(API_URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.load(resp)
    return data["choices"][0]["message"]["content"]

# ============================================================================
# INPUT EXECUTION
# ============================================================================

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", w.LONG),
        ("dy", w.LONG),
        ("mouseData", w.DWORD),
        ("dwFlags", w.DWORD),
        ("time", w.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", w.WORD),
        ("wScan", w.WORD),
        ("dwFlags", w.DWORD),
        ("time", w.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]

class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", w.DWORD), ("wParamL", w.WORD), ("wParamH", w.WORD)]

class INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]

class INPUT(ctypes.Structure):
    _fields_ = [("type", w.DWORD), ("union", INPUTUNION)]

def _send_inputs(items: list[INPUT]) -> None:
    if not items:
        return
    user32.SendInput(len(items), (INPUT * len(items))(*items), ctypes.sizeof(INPUT))

def _mouse_abs(px: int, py: int, vx: int, vy: int, vw: int, vh: int) -> tuple[int, int]:
    x = max(vx, min(vx + vw - 1, px)) - vx
    y = max(vy, min(vy + vh - 1, py)) - vy
    ax = int(round(x * 65535 / (vw - 1))) if vw > 1 else 0
    ay = int(round(y * 65535 / (vh - 1))) if vh > 1 else 0
    return max(0, min(65535, ax)), max(0, min(65535, ay))

def _mouse_move(px: int, py: int, vs: tuple[int, int, int, int]) -> None:
    vx, vy, vw, vh = vs
    ax, ay = _mouse_abs(px, py, vx, vy, vw, vh)
    inp = INPUT(type=INPUT_MOUSE, union=INPUTUNION(mi=MOUSEINPUT(dx=ax, dy=ay, mouseData=0, dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK, time=0, dwExtraInfo=0)))
    _send_inputs([inp])

def _mouse_click() -> None:
    _send_inputs([
        INPUT(type=INPUT_MOUSE, union=INPUTUNION(mi=MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=MOUSEEVENTF_LEFTDOWN, time=0, dwExtraInfo=0))),
        INPUT(type=INPUT_MOUSE, union=INPUTUNION(mi=MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=MOUSEEVENTF_LEFTUP, time=0, dwExtraInfo=0))),
    ])

def _key_unicode(ch: str, up: bool) -> INPUT:
    flags = KEYEVENTF_UNICODE | (KEYEVENTF_KEYUP if up else 0)
    return INPUT(type=INPUT_KEYBOARD, union=INPUTUNION(ki=KEYBDINPUT(wVk=0, wScan=ord(ch), dwFlags=flags, time=0, dwExtraInfo=0)))

def _key_vk(vk_code: int) -> None:
    _send_inputs([
        INPUT(type=INPUT_KEYBOARD, union=INPUTUNION(ki=KEYBDINPUT(wVk=vk_code, wScan=0, dwFlags=0, time=0, dwExtraInfo=0))),
        INPUT(type=INPUT_KEYBOARD, union=INPUTUNION(ki=KEYBDINPUT(wVk=vk_code, wScan=0, dwFlags=KEYEVENTF_KEYUP, time=0, dwExtraInfo=0))),
    ])

def _type_text(text: str) -> None:
    items: list[INPUT] = []
    for ch in text:
        items.append(_key_unicode(ch, False))
        items.append(_key_unicode(ch, True))
    if items:
        _send_inputs(items)

def execute_command(action: str, args: list[str], vs: tuple[int, int, int, int]) -> str:
    vx, vy, vw, vh = vs
    
    def to_px(val_str: str, axis: str) -> int:
        try:
            v = int(val_str)
            if v <= 1000:
                return (vx if axis == "x" else vy) + int(round(v * (vw if axis == "x" else vh) / 1000))
            return v
        except ValueError:
            return vx if axis == "x" else vy
    
    VK_MAP = {
        "enter": 0x0D, "escape": 0x1B, "esc": 0x1B, "tab": 0x09,
        "windows": 0x5B, "win": 0x5B, "backspace": 0x08,
        "delete": 0x2E, "del": 0x2E, "space": 0x20,
        "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    }
    
    try:
        if action == "click" and len(args) >= 2:
            x, y = to_px(args[0], "x"), to_px(args[1], "y")
            _mouse_move(x, y, vs)
            sleep(0.02)
            _mouse_click()
            sleep(0.05)
            return f"CLICK ({x},{y})"
        
        elif action == "drag" and len(args) >= 4:
            x1, y1 = to_px(args[0], "x"), to_px(args[1], "y")
            x2, y2 = to_px(args[2], "x"), to_px(args[3], "y")
            _mouse_move(x1, y1, vs)
            sleep(0.02)
            _send_inputs([INPUT(type=INPUT_MOUSE, union=INPUTUNION(mi=MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=MOUSEEVENTF_LEFTDOWN, time=0, dwExtraInfo=0)))])
            sleep(0.06)
            _mouse_move(x2, y2, vs)
            sleep(0.04)
            _send_inputs([INPUT(type=INPUT_MOUSE, union=INPUTUNION(mi=MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=MOUSEEVENTF_LEFTUP, time=0, dwExtraInfo=0)))])
            sleep(0.05)
            return f"DRAG ({x1},{y1}) to ({x2},{y2})"
        
        elif action == "type":
            _type_text(args[0])
            sleep(0.04)
            return f"TYPE {args[0]}"
        
        elif action == "key":
            key_name = args[0].lower()
            if key_name in VK_MAP:
                _key_vk(VK_MAP[key_name])
                sleep(0.05)
                return f"KEY {key_name}"
        
        elif action == "wait":
            sleep(min(10.0, float(args[0])) / 1000.0)
            return f"WAIT {args[0]}ms"
    
    except Exception as e:
        return f"ERROR {action}: {e}"
    
    return "NO ACTION"

# ============================================================================
# MAIN LOOP - Procedural Execution
# ============================================================================

def main() -> None:
    _enable_dpi_awareness()
    dump = Path("dump") / datetime.now().strftime("%Y%m%d_%H%M%S")
    dump.mkdir(parents=True, exist_ok=True)
    log = dump / "log.txt"
    
    print("=" * 80)
    print("FRANZ - PROCEDURAL DESKTOP AGENT")
    print("=" * 80)
    print("Using hardcoded workflows for desktop tasks")
    print(f"Logs: {dump}")
    print("=" * 80)
    
    with open(log, "w", encoding="utf-8") as f:
        f.write(f"FRANZ PROCEDURAL SESSION START\n")
        f.write(f"Timestamp: {datetime.now()}\n\n")
    
    # Initial screen analysis
    print("\n[INITIAL SCAN]")
    png, vs = capture_screen(RES_W, RES_H)
    (dump / "00_initial.png").write_bytes(png)
    
    vision_desc = call_vlm(VISION_PROMPT, "Describe what you see.", [png])
    print(f"Vision: {vision_desc}\n")
    
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"Initial screen:\n{vision_desc}\n\n")
    
    # Extract user goal from screen
    goal_text = input("\nWhat is your goal? (or press Enter to detect from screen): ").strip()
    
    if not goal_text:
        print("Attempting to detect goal from screen text...")
        # Try to extract goal from vision description
        if "open" in vision_desc.lower() and "paint" in vision_desc.lower():
            goal_text = "open ms paint and draw a cat"
        else:
            goal_text = input("Could not detect goal. Please specify: ").strip()
    
    if not goal_text:
        print("No goal specified. Exiting.")
        return
    
    print(f"\nGOAL: {goal_text}")
    
    # Match goal to procedures
    procedures = match_task_to_procedure(goal_text)
    
    if not procedures:
        print(f"No matching procedures found for goal: {goal_text}")
        print(f"Available procedures: {list(TASK_PROCEDURES.keys())}")
        return
    
    print(f"MATCHED PROCEDURES: {procedures}\n")
    
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"GOAL: {goal_text}\n")
        f.write(f"PROCEDURES: {procedures}\n\n")
    
    # Execute each procedure
    for proc_name in procedures:
        procedure = TASK_PROCEDURES[proc_name]
        print(f"\n{'='*80}")
        print(f"EXECUTING PROCEDURE: {proc_name}")
        print(f"{'='*80}\n")
        
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"PROCEDURE: {proc_name}\n")
            f.write(f"{'='*80}\n\n")
        
        step_num = 0
        for step in procedure["steps"]:
            step_num += 1
            action = step["action"]
            args = step["args"]
            note = step.get("note", "")
            
            print(f"[Step {step_num}] {action.upper()} {' '.join(args)}")
            if note:
                print(f"  Note: {note}")
            
            # Capture before
            png_before, vs = capture_screen(RES_W, RES_H)
            (dump / f"{proc_name}_{step_num:02d}_before.png").write_bytes(png_before)
            
            # Execute command
            result = execute_command(action, args, vs)
            print(f"  Result: {result}")
            
            # Capture after
            png_after, vs = capture_screen(RES_W, RES_H)
            (dump / f"{proc_name}_{step_num:02d}_after.png").write_bytes(png_after)
            
            with open(log, "a", encoding="utf-8") as f:
                f.write(f"Step {step_num}: {action.upper()} {' '.join(args)}\n")
                if note:
                    f.write(f"  Note: {note}\n")
                f.write(f"  Result: {result}\n\n")
            
            sleep(0.2)
        
        # Verify procedure completion
        print(f"\n[VERIFYING {proc_name}]")
        completion_check = procedure["completion_check"]
        
        verification_prompt = COMPLETION_PROMPT.format(
            task_name=proc_name,
            completion_check=completion_check
        )
        
        verification = call_vlm(verification_prompt, "Verify completion.", [png_after])
        print(f"Verification: {verification}\n")
        
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"\nVERIFICATION:\n{verification}\n\n")
        
        if "YES" in verification.upper() or "COMPLETE" in verification.upper():
            print(f"[SUCCESS] {proc_name} completed\n")
        else:
            print(f"[PARTIAL] {proc_name} may not be fully complete\n")
    
    print("=" * 80)
    print("ALL PROCEDURES EXECUTED")
    print("=" * 80)

if __name__ == "__main__":
    print("""
FRANZ - Procedural Desktop Agent
Built by: Wojciech Gabrys
Date: February 9, 2025
Version: Hardcoded Workflows

This version uses pre-defined procedures for common tasks.
More reliable for small models, but less flexible.
    """)
    
    input("Press ENTER to begin... ")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
