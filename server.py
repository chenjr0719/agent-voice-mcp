"""
Local STT/TTS MCP Server

File-based speech-to-text and text-to-speech tools for Claude Code,
powered by Whisper.cpp (STT) and Kokoro (TTS).

Self-hosted, zero API cost, ARM Linux compatible.

Connects to existing OpenAI-compatible STT/TTS services via HTTP API.
Works with VoiceMode's service manager or any whisper.cpp / Kokoro instance.
"""

import asyncio
import json
import os
import subprocess
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# ── Configuration ──────────────────────────────────────────────
WHISPER_URL = os.environ.get("WHISPER_URL", "http://localhost:2022")
KOKORO_URL = os.environ.get("KOKORO_URL", "http://localhost:8880")
OUTPUT_DIR = os.environ.get(
    "STT_TTS_OUTPUT_DIR",
    os.path.join(os.path.expanduser("~"), ".stt-tts-mcp", "output"),
)
TRANSCRIBE_TIMEOUT = int(os.environ.get("TRANSCRIBE_TIMEOUT", "120"))
SPEAK_TIMEOUT = int(os.environ.get("SPEAK_TIMEOUT", "60"))
DEFAULT_VOICE = os.environ.get("DEFAULT_VOICE", "af_sky")
DEFAULT_SPEED = float(os.environ.get("DEFAULT_SPEED", "1.0"))

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── MCP Server ─────────────────────────────────────────────────
app = Server("stt-tts")


@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="transcribe",
            description=(
                "Transcribe an audio file to text using local Whisper. "
                "Supports wav, mp3, m4a, ogg, webm, flac, aac, and more. "
                "Pass the absolute file path to the audio file."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the audio file",
                    },
                    "language": {
                        "type": "string",
                        "description": (
                            "Language hint for better accuracy "
                            "(e.g. 'en', 'zh', 'ja', 'ko'). "
                            "Auto-detected if omitted."
                        ),
                    },
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="speak",
            description=(
                "Convert text to speech using local Kokoro TTS. "
                "Returns the path to the generated audio file (MP3)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to convert to speech",
                    },
                    "voice": {
                        "type": "string",
                        "description": (
                            f"Voice name (default: '{DEFAULT_VOICE}'). "
                            "Common voices: af_sky, af_bella, am_adam, am_michael"
                        ),
                    },
                    "speed": {
                        "type": "number",
                        "description": (
                            f"Speech speed 0.5-2.0 (default: {DEFAULT_SPEED})"
                        ),
                    },
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="health",
            description="Check the status of Whisper STT and Kokoro TTS services.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "transcribe":
        return await do_transcribe(arguments)
    elif name == "speak":
        return await do_speak(arguments)
    elif name == "health":
        return await do_health()
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


# ── Transcribe ─────────────────────────────────────────────────
async def do_transcribe(args: dict):
    file_path = args["file_path"]
    language = args.get("language", "")

    if not os.path.exists(file_path):
        return [TextContent(type="text", text=f"File not found: {file_path}")]

    file_size = os.path.getsize(file_path)
    if file_size == 0:
        return [TextContent(type="text", text="Audio file is empty (0 bytes)")]

    cmd = [
        "curl", "-s", "-X", "POST",
        f"{WHISPER_URL}/v1/audio/transcriptions",
        "-F", f"file=@{file_path}",
        "-F", "model=whisper-1",
        "-F", "response_format=json",
    ]
    if language:
        cmd.extend(["-F", f"language={language}"])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=TRANSCRIBE_TIMEOUT
        )
        if result.returncode != 0:
            return [TextContent(type="text", text=f"Whisper error: {result.stderr}")]

        data = json.loads(result.stdout)
        text = data.get("text", "").strip()
        if not text:
            return [TextContent(type="text", text="(No speech detected in audio)")]

        return [TextContent(type="text", text=text)]
    except json.JSONDecodeError:
        return [TextContent(
            type="text",
            text=f"Unexpected Whisper response: {result.stdout[:200]}",
        )]
    except subprocess.TimeoutExpired:
        return [TextContent(
            type="text",
            text=f"Transcription timed out (>{TRANSCRIBE_TIMEOUT}s)",
        )]
    except Exception as e:
        return [TextContent(type="text", text=f"Transcription error: {e}")]


# ── Speak ──────────────────────────────────────────────────────
async def do_speak(args: dict):
    text = args["text"]
    voice = args.get("voice", DEFAULT_VOICE)
    speed = args.get("speed", DEFAULT_SPEED)

    if not text.strip():
        return [TextContent(type="text", text="No text provided")]

    timestamp = int(asyncio.get_event_loop().time() * 1000)
    out_file = os.path.join(OUTPUT_DIR, f"tts-{timestamp}.mp3")

    cmd = [
        "curl", "-s", "-X", "POST",
        f"{KOKORO_URL}/v1/audio/speech",
        "-H", "Content-Type: application/json",
        "-d", json.dumps({
            "model": "kokoro",
            "input": text,
            "voice": voice,
            "speed": speed,
        }),
        "-o", out_file,
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=SPEAK_TIMEOUT
        )
        if result.returncode != 0:
            return [TextContent(type="text", text=f"TTS error: {result.stderr}")]

        if os.path.exists(out_file) and os.path.getsize(out_file) > 100:
            size_kb = os.path.getsize(out_file) / 1024
            return [TextContent(
                type="text",
                text=f"Audio saved: {out_file} ({size_kb:.1f} KB)",
            )]
        else:
            if os.path.exists(out_file):
                os.remove(out_file)
            return [TextContent(type="text", text="TTS failed — empty or invalid output")]
    except subprocess.TimeoutExpired:
        return [TextContent(
            type="text",
            text=f"TTS timed out (>{SPEAK_TIMEOUT}s)",
        )]
    except Exception as e:
        return [TextContent(type="text", text=f"TTS error: {e}")]


# ── Health Check ───────────────────────────────────────────────
async def do_health():
    results = []

    # Check Whisper
    try:
        r = subprocess.run(
            ["curl", "-s", f"{WHISPER_URL}/health"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and "ok" in r.stdout.lower():
            results.append(f"Whisper STT: ✅ healthy ({WHISPER_URL})")
        else:
            results.append(f"Whisper STT: ❌ not responding ({WHISPER_URL})")
    except Exception:
        results.append(f"Whisper STT: ❌ unreachable ({WHISPER_URL})")

    # Check Kokoro
    try:
        r = subprocess.run(
            ["curl", "-s", f"{KOKORO_URL}/health"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and "healthy" in r.stdout.lower():
            results.append(f"Kokoro TTS: ✅ healthy ({KOKORO_URL})")
        else:
            results.append(f"Kokoro TTS: ❌ not responding ({KOKORO_URL})")
    except Exception:
        results.append(f"Kokoro TTS: ❌ unreachable ({KOKORO_URL})")

    return [TextContent(type="text", text="\n".join(results))]


# ── Entry Point ────────────────────────────────────────────────
async def main():
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
