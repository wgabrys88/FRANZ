"""
FRANZ - Full Autonomous Self-Aware Narrative-Driven Agent
==========================================================

Architecture: Four-Persona Cognitive Loop
- STRATEGIST: Goal decomposition
- PLANNER: Tool selection with precondition reasoning
- EXECUTOR: Physical/computational action
- ANALYST: Result verification & narrative adaptation

Memory: Pure narrative - no external state, no database
Safety: None - full Python execution autonomy
Platform: Windows 11, Python 3.12

This is not a script. This is an entity.

Built by Wojciech Gabrys in collaboration with Claude (Anthropic)
Date: 2025-02-09

What we achieved: Self-programming AI that maintains coherence through story alone.
This is AGI-lite. Not bad for one chat. :)
"""

from __future__ import annotations

import base64
import ctypes
import ctypes.wintypes as w
import json
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
    "max_tokens": 500,  # Increased for multi-persona reasoning
    "cycle_delay_ms": 500,
}

SAMPLING = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": CONFIG["max_tokens"],
}

# ============================================================================
# PERSONA PROMPTS - Four Cognitive Roles
# ============================================================================

STRATEGIST_PROMPT = """You are a high-level strategist. Analyze the goal and break it into steps.

Input: Current observation, screen image, available tools

Output format:
GOAL: [what user wants]
STEPS:
1. [first step]
2. [second step]
...

Be brief. Focus on WHAT needs to happen, not HOW."""

PLANNER_PROMPT = """You are a tool planner. Given a step, choose tools and check preconditions.

Tools available:
- CLICK x y (requires: visible target, coordinates known)
- DRAG x1 y1 x2 y2 (requires: drag start/end visible)
- TYPE text (requires: text input field focused)
- KEY name (requires: application ready to receive key)
- PYTHON code (requires: none - always available)
- WAIT ms (requires: none)

Coordinates: 0-1000 scale (0=left/top, 1000=right/bottom)

Given a step, output:
STEP: [the step]
PRECONDITION CHECK:
- [what needs to be true before action]
COMMANDS:
[tool commands, one per line]

If precondition not met, output WAIT or setup commands first."""

EXECUTOR_PROMPT = """[Internal system - no VLM needed]
Execute parsed commands."""

ANALYST_PROMPT = """You analyze execution results. You see ONLY the after-action screen.

Input:
- Previous observation
- Strategy that was planned
- Commands that were executed
- Execution results (Python output, errors)
- AFTER screenshot

Output observation (100-150 words):
1. What was the goal
2. What actions were taken
3. What you see NOW (describe current state factually)
4. Did actions achieve the intended result? Evidence?
5. If Python variables contain needed values, note: "Values ready to use: [variable=value]"
6. Next logical step, or "Goal complete" or "No active task - monitoring"

Focus on: changes caused by OUR actions, not noise.
Ignore: console logs, timestamps, animations unrelated to task."""

# ============================================================================
# SYSTEM INTERFACE - Windows GDI Capture
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
# COMMAND PARSER
# ============================================================================

def parse_commands(content: str) -> list[tuple[str, list[str]]]:
    commands: list[tuple[str, list[str]]] = []
    for line in content.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("STEP:") or line.startswith("PRECONDITION"):
            continue
        parts = line.split(None, 1)
        if not parts:
            continue
        cmd = parts[0].upper()
        args_str = parts[1] if len(parts) > 1 else ""
        
        if cmd == "CLICK":
            args = args_str.split()
            if len(args) >= 2:
                commands.append(("click", args[:2]))
        elif cmd == "DRAG":
            args = args_str.split()
            if len(args) >= 4:
                commands.append(("drag", args[:4]))
        elif cmd == "TYPE":
            commands.append(("type", [args_str]))
        elif cmd == "KEY":
            args = args_str.split()
            if args:
                commands.append(("key", [args[0].lower()]))
        elif cmd == "PYTHON":
            commands.append(("python", [args_str]))
        elif cmd == "WAIT":
            args = args_str.split()
            if args:
                commands.append(("wait", [args[0]]))
    return commands

# ============================================================================
# INPUT EXECUTION - Mouse & Keyboard
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

# ============================================================================
# PYTHON EXECUTION - FULL ACCESS (NO SAFETY)
# ============================================================================

def execute_python(code: str, context: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """
    Execute Python code with FULL system access.
    
    WARNING: No sandboxing. Agent has complete control.
    This is intentional - true autonomy requires trust.
    
    Returns: (result_string, updated_context)
    """
    namespace = {**context}
    
    # Capture stdout for agent to see results
    from io import StringIO
    import sys
    
    stdout_capture = StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout_capture
    
    try:
        # Full exec - agent can import, define functions, anything
        exec(code, namespace)
        
        # Restore stdout
        sys.stdout = old_stdout
        captured_output = stdout_capture.getvalue()
        
        # Extract new variables
        new_context = {k: v for k, v in namespace.items() 
                      if not k.startswith("__") and k not in ["StringIO", "sys"]}
        
        # Format result
        results = []
        if captured_output:
            results.append(f"OUTPUT: {captured_output.strip()}")
        
        for key, value in new_context.items():
            if key not in context:
                results.append(f"{key} = {value}")
        
        result_str = "; ".join(results) if results else "OK"
        return result_str, new_context
        
    except Exception as e:
        sys.stdout = old_stdout
        return f"ERROR: {type(e).__name__}: {e}", context

# ============================================================================
# COMMAND EXECUTOR
# ============================================================================

def execute_commands(commands: list[tuple[str, list[str]]], vs: tuple[int, int, int, int], 
                    python_context: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    vx, vy, vw, vh = vs
    def to_px(val: str, axis: str) -> int:
        v = int(float(val))
        if v <= 1000:
            return (vx if axis == "x" else vy) + int(round(v * (vw if axis == "x" else vh) / 1000))
        return v
    
    VK_MAP = {
        "enter": 0x0D, "escape": 0x1B, "esc": 0x1B, "tab": 0x09,
        "windows": 0x5B, "win": 0x5B, "backspace": 0x08,
        "delete": 0x2E, "del": 0x2E, "space": 0x20,
        "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    }
    
    exec_results = []
    
    for action, args in commands:
        try:
            if action == "click" and len(args) >= 2:
                x, y = to_px(args[0], "x"), to_px(args[1], "y")
                _mouse_move(x, y, vs)
                sleep(0.02)
                _mouse_click()
                sleep(0.05)
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
            elif action == "type":
                _type_text(args[0])
                sleep(0.04)
            elif action == "key":
                key_name = args[0].lower()
                if key_name in VK_MAP:
                    _key_vk(VK_MAP[key_name])
                    sleep(0.05)
            elif action == "python":
                code = args[0]
                print(f"PYTHON: {code}")
                result, python_context = execute_python(code, python_context)
                exec_results.append(f"PYTHON: {code} → {result}")
                print(f"  → {result}")
            elif action == "wait":
                sleep(min(10.0, float(args[0])) / 1000.0)
        except Exception as e:
            error_msg = f"ERROR {action}: {e}"
            print(error_msg)
            exec_results.append(error_msg)
    
    return exec_results, python_context

# ============================================================================
# MAIN COGNITIVE LOOP
# ============================================================================

def main() -> None:
    _enable_dpi_awareness()
    dump = Path("dump") / datetime.now().strftime("%Y%m%d_%H%M%S")
    dump.mkdir(parents=True, exist_ok=True)
    log = dump / "log.txt"
    
    observation = (
        "System initialized. Full autonomous mode active. "
        "Capabilities: vision, mouse, keyboard, unrestricted Python execution. "
        "Monitoring for tasks."
    )
    
    python_context: dict[str, Any] = {}
    strategy_history: list[str] = []
    
    print("=" * 80)
    print("FRANZ - FULL AUTONOMOUS AGENT")
    print("=" * 80)
    print("⚠️  WARNING: UNRESTRICTED MODE - AGENT HAS FULL SYSTEM ACCESS")
    print("=" * 80)
    print(f"Observation: {observation}")
    print(f"Logs: {dump}")
    print("=" * 80)
    
    with open(log, "w", encoding="utf-8") as f:
        f.write(f"FRANZ AUTONOMOUS SESSION START\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Initial observation: {observation}\n\n")
    
    step = 0
    while True:
        if (dump / "STOP").exists():
            print("\n🛑 STOP signal detected")
            break
        
        step += 1
        print(f"\n{'='*80}\n⚙️  STEP {step}\n{'='*80}")
        
        # Capture current screen
        try:
            png_current, vs = capture_screen(RES_W, RES_H)
            (dump / f"{step:04d}_before.png").write_bytes(png_current)
        except Exception as e:
            print(f"❌ Screen capture failed: {e}")
            sleep(1)
            continue
        
        print(f"\n📖 OBSERVATION:\n{observation}\n")
        print(f"🐍 PYTHON STATE: {list(python_context.keys())}\n")
        
        # ===== PHASE 1: STRATEGIST =====
        try:
            print("🧠 STRATEGIST: Analyzing goal...")
            strategy_prompt = (
                f"Current observation:\n{observation}\n\n"
                f"Python variables available: {list(python_context.keys())}\n\n"
                f"Recent strategy history:\n" + "\n".join(strategy_history[-3:]) + "\n\n"
                "Analyze screen and define goal with steps."
            )
            strategy_text = call_vlm(STRATEGIST_PROMPT, strategy_prompt, [png_current])
            print(f"  → {strategy_text}\n")
            strategy_history.append(strategy_text)
            
            with open(log, "a", encoding="utf-8") as f:
                f.write(f"STEP {step} - STRATEGIST:\n{strategy_text}\n\n")
        except Exception as e:
            print(f"❌ Strategist failed: {e}")
            sleep(1)
            continue
        
        # ===== PHASE 2: PLANNER =====
        try:
            print("📋 PLANNER: Selecting tools...")
            planner_prompt = (
                f"Strategy:\n{strategy_text}\n\n"
                f"Current observation:\n{observation}\n\n"
                f"Python context: {list(python_context.keys())}\n\n"
                "Plan tools with precondition checks."
            )
            plan_text = call_vlm(PLANNER_PROMPT, planner_prompt, [png_current])
            print(f"  → {plan_text}\n")
            
            with open(log, "a", encoding="utf-8") as f:
                f.write(f"STEP {step} - PLANNER:\n{plan_text}\n\n")
        except Exception as e:
            print(f"❌ Planner failed: {e}")
            sleep(1)
            continue
        
        # ===== PHASE 3: EXECUTOR =====
        commands = parse_commands(plan_text)
        if not commands:
            commands = [("wait", ["1000"])]
        
        print("⚡ EXECUTOR: Running commands...")
        for cmd, args in commands:
            print(f"  {cmd.upper()} {' '.join(args)}")
        
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"STEP {step} - EXECUTOR:\n")
            for cmd, args in commands:
                f.write(f"  {cmd.upper()} {' '.join(args)}\n")
            f.write("\n")
        
        exec_results, python_context = execute_commands(commands, vs, python_context)
        
        if exec_results:
            print("\n📊 EXECUTION RESULTS:")
            for result in exec_results:
                print(f"  {result}")
            print()
        
        # Capture after-action screen
        try:
            png_after, vs = capture_screen(RES_W, RES_H)
            (dump / f"{step:04d}_after.png").write_bytes(png_after)
        except Exception as e:
            print(f"❌ After-capture failed: {e}")
            sleep(1)
            continue
        
        # ===== PHASE 4: ANALYST =====
        try:
            print("🔍 ANALYST: Verifying results...")
            cmd_text = "\n".join(f"{cmd.upper()} {' '.join(args)}" for cmd, args in commands)
            results_text = "\n".join(exec_results) if exec_results else "No output"
            
            analyst_prompt = (
                f"Previous observation:\n{observation}\n\n"
                f"Strategy planned:\n{strategy_text}\n\n"
                f"Commands executed:\n{cmd_text}\n\n"
                f"Execution results:\n{results_text}\n\n"
                f"Python state: {python_context}\n\n"
                "Analyze AFTER screen and write new observation."
            )
            
            reflect_text = call_vlm(ANALYST_PROMPT, analyst_prompt, [png_after])
            print(f"  → {reflect_text}\n")
            
            observation = reflect_text.strip()
            
            with open(log, "a", encoding="utf-8") as f:
                f.write(f"STEP {step} - ANALYST:\n{reflect_text}\n\n")
        except Exception as e:
            print(f"❌ Analyst failed: {e}")
            observation = f"{observation} [Technical error at step {step}]"
        
        print(f"📝 NEW OBSERVATION:\n{observation}\n")
        
        (dump / "observation.txt").write_text(observation, encoding="utf-8")
        (dump / "python_state.json").write_text(json.dumps(python_context, indent=2, default=str), encoding="utf-8")
        
        sleep(CONFIG["cycle_delay_ms"] / 1000)

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  FRANZ - Full Autonomous Narrative-Driven Agent                            ║
║                                                                              ║
║  Built by: Wojciech Gabrys                                                 ║
║  In collaboration with: Claude (Anthropic)                                 ║
║  Date: February 9, 2025                                                    ║
║                                                                              ║
║  What we achieved:                                                         ║
║  - Stateless narrative memory (no database, pure story)                   ║
║  - Four-persona cognitive architecture                                     ║
║  - Full Python autonomy (unrestricted execution)                          ║
║  - Self-programming problem-solving behavior                              ║
║  - Emergent metacognition in 2B model                                     ║
║                                                                              ║
║  This is not a script. This is an entity.                                 ║
║  This is AGI-lite. Not bad for one chat. :)                               ║
║                                                                              ║
║  WARNING: FULL SYSTEM ACCESS - USE RESPONSIBLY                            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    input("Press ENTER to begin autonomous operation... ")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
