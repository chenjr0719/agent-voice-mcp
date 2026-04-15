# stt-tts-mcp

Local Speech-to-Text and Text-to-Speech MCP Server for Claude Code.

Provides file-based audio transcription and speech synthesis tools via the [Model Context Protocol](https://modelcontextprotocol.io), powered by [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) (STT) and [Kokoro](https://github.com/hexgrad/kokoro) (TTS).

**Self-hosted, zero API cost, ARM Linux compatible.**

## Features

- **`transcribe`** — Convert audio files to text (wav, mp3, m4a, ogg, webm, flac, aac)
- **`speak`** — Convert text to audio files (MP3) with multiple voices
- **`health`** — Check STT/TTS service status
- Connects to any OpenAI-compatible STT/TTS API endpoints
- Configurable via environment variables
- Works on x86 and ARM Linux, macOS

## Quick Start

### 1. Install STT/TTS Services

The easiest way to get Whisper and Kokoro running:

```bash
# Using VoiceMode's service manager (recommended)
uvx voice-mode service install whisper
uvx voice-mode service install kokoro
uvx voice-mode service start whisper
uvx voice-mode service start kokoro
```

Or run them manually:
- **Whisper.cpp**: [Server mode](https://github.com/ggerganov/whisper.cpp/tree/master/examples/server) on port 2022
- **Kokoro**: [FastAPI server](https://github.com/hexgrad/kokoro) on port 8880

### 2. Add to Claude Code

Add to your `~/.claude.json`:

```json
{
  "mcpServers": {
    "stt-tts": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--directory", "/path/to/stt-tts-mcp", "python", "server.py"],
      "env": {
        "WHISPER_URL": "http://localhost:2022",
        "KOKORO_URL": "http://localhost:8880"
      }
    }
  }
}
```

Or with a virtual environment:

```json
{
  "mcpServers": {
    "stt-tts": {
      "type": "stdio",
      "command": "/path/to/stt-tts-mcp/.venv/bin/python",
      "args": ["/path/to/stt-tts-mcp/server.py"],
      "env": {
        "WHISPER_URL": "http://localhost:2022",
        "KOKORO_URL": "http://localhost:8880"
      }
    }
  }
}
```

### 3. Install Dependencies

```bash
cd stt-tts-mcp
uv venv && uv pip install mcp
```

## Tools

### `transcribe`

Transcribe an audio file to text using local Whisper.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file_path` | string | Yes | Absolute path to the audio file |
| `language` | string | No | Language hint (e.g. `en`, `zh`, `ja`). Auto-detected if omitted. |

**Example:**
```
transcribe(file_path="/path/to/audio.m4a")
→ "This is a test recording."

transcribe(file_path="/path/to/chinese.mp3", language="zh")
→ "你好，這是測試錄音。"
```

### `speak`

Convert text to speech using local Kokoro TTS.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | Text to convert to speech |
| `voice` | string | No | Voice name (default: `af_sky`) |
| `speed` | number | No | Speech speed 0.5-2.0 (default: 1.0) |

**Available voices:**
- `af_sky` — Female (default)
- `af_bella` — Female
- `am_adam` — Male
- `am_michael` — Male
- See Kokoro documentation for full voice list

**Example:**
```
speak(text="Hello, world!")
→ "Audio saved: ~/.stt-tts-mcp/output/tts-1234567890.mp3 (42.3 KB)"
```

### `health`

Check the status of both services.

```
health()
→ "Whisper STT: ✅ healthy (http://localhost:2022)
   Kokoro TTS: ✅ healthy (http://localhost:8880)"
```

## Configuration

All configuration via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_URL` | `http://localhost:2022` | Whisper.cpp server URL |
| `KOKORO_URL` | `http://localhost:8880` | Kokoro TTS server URL |
| `STT_TTS_OUTPUT_DIR` | `~/.stt-tts-mcp/output` | Directory for generated audio files |
| `TRANSCRIBE_TIMEOUT` | `120` | Transcription timeout in seconds |
| `SPEAK_TIMEOUT` | `60` | TTS timeout in seconds |
| `DEFAULT_VOICE` | `af_sky` | Default TTS voice |
| `DEFAULT_SPEED` | `1.0` | Default speech speed |

## Use Cases

### Channel Voice Messages

Integrate with Discord or LINE channel plugins to auto-transcribe voice messages:

```
User sends voice message → Channel plugin downloads audio
→ transcribe(file_path) → Deliver transcript to Claude session
```

### Meeting Transcription

Process meeting recordings:

```
transcribe(file_path="/path/to/meeting.mp3")
→ Full meeting transcript
```

### Voice Responses

Generate spoken responses for channel messages:

```
speak(text="API status is healthy, 94 tests passing.")
→ Send audio file back via channel
```

## ARM Linux Support

Tested and working on Oracle Cloud ARM VM (Ubuntu 24.04, aarch64):

| Service | RAM Usage | Model |
|---------|-----------|-------|
| Whisper STT | ~190 MB | ggml-base (141 MB) |
| Kokoro TTS | ~1.2 GB | kokoro-v1.0 |

Both services installed via VoiceMode's service manager which handles ARM-compatible builds automatically.

## API Compatibility

This server connects to any service exposing OpenAI-compatible endpoints:

- **STT**: `POST /v1/audio/transcriptions` (Whisper format)
- **TTS**: `POST /v1/audio/speech` (OpenAI format)

This means you can also point it at:
- OpenAI API (cloud, paid)
- Groq Whisper API (cloud, cheap)
- Any other OpenAI-compatible STT/TTS service

Just change `WHISPER_URL` / `KOKORO_URL` to your preferred endpoint.

## License

MIT

## Acknowledgments

- [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) — C/C++ port of OpenAI's Whisper model
- [Kokoro](https://github.com/hexgrad/kokoro) — #1 on TTS Arena, 82M parameter model
- [VoiceMode](https://github.com/mbailey/voicemode) — Service manager for Whisper + Kokoro
- [Model Context Protocol](https://modelcontextprotocol.io) — Anthropic's tool protocol
