"""
Agent Voice MCP Server

File-based speech-to-text and text-to-speech tools for Claude Code,
powered by Whisper (STT) and OpenAI-compatible TTS (Kokoro/Piper).

Self-hosted, zero API cost. Supports primary + fallback endpoints.

Connects to existing OpenAI-compatible STT/TTS services via HTTP API.
"""

import asyncio
import json
import os
import re
import subprocess
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# ── Configuration ──────────────────────────────────────────────
WHISPER_URL = os.environ.get("WHISPER_URL", "http://localhost:2022")
WHISPER_URL_FALLBACK = os.environ.get("WHISPER_URL_FALLBACK", "")
KOKORO_URL = os.environ.get("KOKORO_URL", "http://localhost:8880")
KOKORO_URL_FALLBACK = os.environ.get("KOKORO_URL_FALLBACK", "")
OUTPUT_DIR = os.environ.get(
    "VOICE_OUTPUT_DIR",
    os.path.join(os.path.expanduser("~"), ".agent-voice-mcp", "output"),
)
TRANSCRIBE_TIMEOUT = int(os.environ.get("TRANSCRIBE_TIMEOUT", "120"))
SPEAK_TIMEOUT = int(os.environ.get("SPEAK_TIMEOUT", "60"))
DEFAULT_VOICE = os.environ.get("DEFAULT_VOICE", "af_sky")
DEFAULT_SPEED = float(os.environ.get("DEFAULT_SPEED", "1.0"))

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Circuit breaker ──────────────────────────────────────────
# Tracks endpoint health to avoid waiting for timeouts on dead endpoints.
import time as _time

HEALTH_CHECK_INTERVAL = 30  # seconds between periodic health checks
UNHEALTHY_COOLDOWN = 60     # seconds to skip a failed endpoint before retrying

_endpoint_health: dict[str, dict] = {}
# { url: { "healthy": bool, "last_check": float, "last_fail": float } }


def _check_health_sync(url: str, keyword: str) -> bool:
    """Quick health check with 3s timeout."""
    try:
        r = subprocess.run(
            ["curl", "-s", "--max-time", "3", f"{url}/health"],
            capture_output=True, text=True, timeout=5,
        )
        return r.returncode == 0 and keyword in r.stdout.lower()
    except Exception:
        return False


def _is_endpoint_healthy(url: str, keyword: str) -> bool:
    """Check if endpoint is healthy, using cached status when fresh enough."""
    if not url:
        return False
    now = _time.time()
    state = _endpoint_health.get(url)

    if state:
        # If recently marked unhealthy, skip without checking
        if not state["healthy"] and (now - state["last_fail"]) < UNHEALTHY_COOLDOWN:
            return False
        # If recently checked and healthy, trust the cache
        if state["healthy"] and (now - state["last_check"]) < HEALTH_CHECK_INTERVAL:
            return True

    # Actually check
    healthy = _check_health_sync(url, keyword)
    _endpoint_health[url] = {
        "healthy": healthy,
        "last_check": now,
        "last_fail": now if not healthy else (state["last_fail"] if state else 0),
    }
    return healthy


def _mark_unhealthy(url: str):
    """Mark an endpoint as unhealthy after a request failure."""
    now = _time.time()
    state = _endpoint_health.get(url, {"healthy": True, "last_check": now, "last_fail": 0})
    state["healthy"] = False
    state["last_fail"] = now
    _endpoint_health[url] = state


def _pick_endpoint(primary: str, fallback: str, keyword: str) -> str:
    """Pick the best endpoint: primary if healthy, fallback otherwise."""
    if _is_endpoint_healthy(primary, keyword):
        return primary
    if fallback and _is_endpoint_healthy(fallback, keyword):
        return fallback
    # Both unknown or unhealthy — try primary anyway
    return primary


# ── Voice auto-selection ──────────────────────────────────────
# Maps language prefixes to default voices per TTS backend
VOICE_MAP = {
    "zh": "zf_xiaobei",   # Chinese female (Kokoro) / huayan (Piper)
    "ja": "jf_alpha",     # Japanese female (Kokoro)
    "en": "af_sky",       # English female (Kokoro) / amy (Piper)
}

# CJK Unicode ranges for language detection
_CJK_RE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')
_JA_RE = re.compile(r'[\u3040-\u309f\u30a0-\u30ff]')


def _detect_language(text: str) -> str:
    """Simple language detection based on character ranges."""
    if _JA_RE.search(text):
        return "ja"
    if _CJK_RE.search(text):
        return "zh"
    return "en"


def _resolve_voice(voice: str | None, text: str) -> str:
    """Resolve voice name: use explicit voice, or auto-detect from text language."""
    if voice and voice != DEFAULT_VOICE:
        return voice
    lang = _detect_language(text)
    return VOICE_MAP.get(lang, DEFAULT_VOICE)


# ── MCP Server ─────────────────────────────────────────────────
app = Server("voice")


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
                "Convert text to speech using local TTS. "
                "Returns the path to the generated audio file (MP3). "
                "Language is auto-detected from text (zh, ja, en). "
                "You can override the voice with the voice parameter."
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
                            "Voice name (auto-selected by language if omitted). "
                            "EN: af_sky, af_bella, am_adam | "
                            "ZH: zf_xiaobei, zf_xiaoni, zm_yunxi | "
                            "JA: jf_alpha, jf_gongitsune"
                        ),
                    },
                    "speed": {
                        "type": "number",
                        "description": (
                            f"Speech speed 0.5-2.0 (default: {DEFAULT_SPEED})"
                        ),
                    },
                    "leading_pad_ms": {
                        "type": "integer",
                        "description": "Milliseconds of silence prepended to the audio (default: 0).",
                        "default": 0,
                    },
                    "trailing_pad_ms": {
                        "type": "integer",
                        "description": "Milliseconds of silence appended to the audio (default: 0).",
                        "default": 0,
                    },
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="health",
            description="Check the status of Whisper STT and TTS services.",
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


# ── Fallback helper ───────────────────────────────────────────
def _try_with_fallback(run_fn, primary_url: str, fallback_url: str, health_keyword: str):
    """Pick the best endpoint via circuit breaker, try it, fall back if needed."""
    chosen = _pick_endpoint(primary_url, fallback_url, health_keyword)
    other = fallback_url if chosen == primary_url else primary_url

    try:
        result = run_fn(chosen)
        return result, chosen
    except Exception as first_err:
        _mark_unhealthy(chosen)
        if not other:
            raise
        try:
            result = run_fn(other)
            return result, other
        except Exception:
            _mark_unhealthy(other)
            raise first_err


# ── Transcribe ─────────────────────────────────────────────────
MAX_CHUNK_SIZE = 24 * 1024 * 1024  # 24 MB (safe margin under Whisper's 25 MB limit)
CHUNK_DURATION = 600  # 10 minutes per chunk for long audio


def _split_audio(file_path: str) -> list[str]:
    """Split audio into chunks using ffmpeg. Returns list of chunk file paths."""
    import tempfile
    chunk_dir = tempfile.mkdtemp(prefix="stt-chunks-")

    # Get duration
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", file_path],
        capture_output=True, text=True, timeout=10,
    )
    try:
        duration = float(probe.stdout.strip())
    except (ValueError, AttributeError):
        duration = 0

    if duration <= 0:
        return [file_path]  # Can't determine duration, try as-is

    # If short enough, no need to split
    file_size = os.path.getsize(file_path)
    if file_size <= MAX_CHUNK_SIZE and duration <= CHUNK_DURATION:
        return [file_path]

    # Split into chunks
    chunks = []
    offset = 0
    idx = 0
    while offset < duration:
        chunk_path = os.path.join(chunk_dir, f"chunk-{idx:03d}.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", file_path, "-ss", str(offset),
             "-t", str(CHUNK_DURATION), "-ar", "16000", "-ac", "1",
             "-f", "wav", chunk_path],
            capture_output=True, timeout=120,
        )
        if os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 100:
            chunks.append(chunk_path)
        offset += CHUNK_DURATION
        idx += 1

    return chunks if chunks else [file_path]


def _transcribe_single_with_url(file_path: str, language: str, whisper_url: str) -> str:
    """Transcribe a single audio file via Whisper API at a specific URL."""
    cmd = [
        "curl", "-s", "--max-time", str(TRANSCRIBE_TIMEOUT),
        "-X", "POST",
        f"{whisper_url}/v1/audio/transcriptions",
        "-F", f"file=@{file_path}",
        "-F", "model=whisper-1",
        "-F", "response_format=json",
    ]
    if language:
        cmd.extend(["-F", f"language={language}"])

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=TRANSCRIBE_TIMEOUT + 10
    )
    if result.returncode != 0:
        raise RuntimeError(f"Whisper error: {result.stderr}")

    # Check for HTTP errors in response
    stdout = result.stdout.strip()
    if not stdout or stdout.startswith("Internal Server Error") or stdout.startswith("<!"):
        raise RuntimeError(f"Whisper returned error: {stdout[:200]}")

    data = json.loads(stdout)
    return data.get("text", "").strip()


async def do_transcribe(args: dict):
    file_path = args["file_path"]
    language = args.get("language", "")

    if not os.path.exists(file_path):
        return [TextContent(type="text", text=f"File not found: {file_path}")]

    file_size = os.path.getsize(file_path)
    if file_size == 0:
        return [TextContent(type="text", text="Audio file is empty (0 bytes)")]

    try:
        chunks = _split_audio(file_path)
        is_chunked = len(chunks) > 1

        transcripts = []
        used_url = WHISPER_URL
        for i, chunk in enumerate(chunks):
            def run_transcribe(url):
                return _transcribe_single_with_url(chunk, language, url)

            text, used_url = _try_with_fallback(
                run_transcribe, WHISPER_URL, WHISPER_URL_FALLBACK, "ok"
            )
            if text:
                transcripts.append(text)

        # Clean up temp chunks
        if is_chunked:
            import shutil
            chunk_dir = os.path.dirname(chunks[0])
            if chunk_dir.startswith("/tmp/"):
                shutil.rmtree(chunk_dir, ignore_errors=True)

        if not transcripts:
            return [TextContent(type="text", text="(No speech detected in audio)")]

        full_text = " ".join(transcripts)
        meta_parts = []
        if is_chunked:
            meta_parts.append(f"{len(chunks)} chunks")
        if used_url != WHISPER_URL:
            meta_parts.append("fallback")
        meta = f"[{', '.join(meta_parts)}] " if meta_parts else ""
        return [TextContent(type="text", text=f"{meta}{full_text}".strip())]

    except json.JSONDecodeError as e:
        return [TextContent(type="text", text=f"Unexpected Whisper response: {e}")]
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
    voice = _resolve_voice(args.get("voice"), text)
    speed = args.get("speed", DEFAULT_SPEED)
    leading_pad_ms = int(args.get("leading_pad_ms") or 0)
    trailing_pad_ms = int(args.get("trailing_pad_ms") or 0)

    if not text.strip():
        return [TextContent(type="text", text="No text provided")]

    timestamp = int(asyncio.get_event_loop().time() * 1000)
    out_file = os.path.join(OUTPUT_DIR, f"tts-{timestamp}.mp3")

    def run_speak(tts_url):
        cmd = [
            "curl", "-s", "--max-time", str(SPEAK_TIMEOUT),
            "-X", "POST",
            f"{tts_url}/v1/audio/speech",
            "-H", "Content-Type: application/json",
            "-d", json.dumps({
                "model": "kokoro",
                "input": text,
                "voice": voice,
                "speed": speed,
            }),
            "-o", out_file,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=SPEAK_TIMEOUT + 10
        )
        if result.returncode != 0:
            raise RuntimeError(f"TTS error: {result.stderr}")
        if not os.path.exists(out_file) or os.path.getsize(out_file) <= 100:
            if os.path.exists(out_file):
                os.remove(out_file)
            raise RuntimeError("TTS failed — empty or invalid output")
        return out_file

    try:
        _, used_url = _try_with_fallback(
            run_speak, KOKORO_URL, KOKORO_URL_FALLBACK, "healthy"
        )

        if leading_pad_ms > 0 or trailing_pad_ms > 0:
            filters = []
            if leading_pad_ms > 0:
                filters.append(f"adelay={leading_pad_ms}:all=1")
            if trailing_pad_ms > 0:
                filters.append(f"apad=pad_dur={trailing_pad_ms / 1000:.3f}")
            padded = out_file[:-4] + "-p.mp3"
            subprocess.run(
                ["ffmpeg", "-y", "-i", out_file, "-af", ",".join(filters), padded],
                capture_output=True, timeout=30, check=True,
            )
            os.replace(padded, out_file)

        size_kb = os.path.getsize(out_file) / 1024
        lang = _detect_language(text)
        fallback_note = " (fallback)" if used_url != KOKORO_URL else ""
        return [TextContent(
            type="text",
            text=f"Audio saved: {out_file} ({size_kb:.1f} KB, voice={voice}, lang={lang}{fallback_note})",
        )]
    except subprocess.TimeoutExpired:
        return [TextContent(
            type="text",
            text=f"TTS timed out (>{SPEAK_TIMEOUT}s)",
        )]
    except Exception as e:
        return [TextContent(type="text", text=f"TTS error: {e}")]


# ── Health Check ───────────────────────────────────────────────
def _check_endpoint(url: str, healthy_keyword: str, timeout: int = 5) -> str:
    """Check a single endpoint health. Returns status string."""
    try:
        r = subprocess.run(
            ["curl", "-s", "--max-time", str(timeout), f"{url}/health"],
            capture_output=True, text=True, timeout=timeout + 2,
        )
        if r.returncode == 0 and healthy_keyword in r.stdout.lower():
            return f"✅ healthy ({url})"
        return f"❌ not responding ({url})"
    except Exception:
        return f"❌ unreachable ({url})"


async def do_health():
    results = []

    # Check Whisper (primary + fallback)
    results.append(f"Whisper STT: {_check_endpoint(WHISPER_URL, 'ok')}")
    if WHISPER_URL_FALLBACK:
        results.append(f"Whisper STT fallback: {_check_endpoint(WHISPER_URL_FALLBACK, 'ok')}")

    # Check TTS (primary + fallback)
    results.append(f"TTS: {_check_endpoint(KOKORO_URL, 'healthy')}")
    if KOKORO_URL_FALLBACK:
        results.append(f"TTS fallback: {_check_endpoint(KOKORO_URL_FALLBACK, 'healthy')}")

    return [TextContent(type="text", text="\n".join(results))]


# ── Entry Point ────────────────────────────────────────────────
async def main():
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
