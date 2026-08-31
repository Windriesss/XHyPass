from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import BinaryIO, Pattern

import serial


class SerialTimeout(TimeoutError):
    pass


def write_console_bytes(stream, raw: bytes) -> None:
    """Write UTF-8 device output without crashing a GBK Windows console."""
    text = raw.decode("utf-8", errors="replace")
    try:
        stream.write(text)
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "utf-8"
        safe = text.encode(encoding, errors="replace").decode(encoding)
        stream.write(safe)
    stream.flush()


class SerialSession:
    """Small expect-like wrapper over pyserial with a complete binary transcript."""

    def __init__(self, settings: dict, transcript: Path):
        self.settings = settings
        self.transcript_path = transcript
        self.port: serial.Serial | None = None
        self.log: BinaryIO | None = None
        self.buffer = bytearray()

    def __enter__(self) -> "SerialSession":
        self._open_log(self.transcript_path)
        try:
            self.port = serial.Serial(
                port=self.settings["port"],
                baudrate=int(self.settings.get("baudrate", 1500000)),
                bytesize=int(self.settings.get("bytesize", 8)),
                parity=self.settings.get("parity", "N"),
                stopbits=int(self.settings.get("stopbits", 1)),
                timeout=float(self.settings.get("read_timeout", 0.1)),
                write_timeout=float(self.settings.get("write_timeout", 2.0)),
            )
        except Exception:
            # Opening a busy/missing port happens before an experiment attempt
            # exists. Do not leave an empty directory under `completed`.
            if self.log:
                self.log.close()
                self.log = None
            if self.transcript_path.exists() and self.transcript_path.stat().st_size == 0:
                self.transcript_path.unlink()
            try:
                self.transcript_path.parent.rmdir()
            except OSError:
                pass
            raise
        self.port.reset_input_buffer()
        return self

    def switch_transcript(self, transcript: Path) -> None:
        """Start logging subsequent serial bytes to another run-specific file."""
        if self.log:
            self.log.close()
        self.transcript_path = transcript
        self._open_log(transcript)

    def truncate_transcript(self) -> None:
        """Discard pre-run UART bytes and continue logging to the same file."""
        if self.log:
            self.log.close()
        self.transcript_path.write_bytes(b"")
        self._open_log(self.transcript_path)

    def _open_log(self, transcript: Path) -> None:
        transcript.parent.mkdir(parents=True, exist_ok=True)
        self.log = transcript.open("ab")

    def __exit__(self, *_: object) -> None:
        if self.port:
            self.port.close()
        if self.log:
            self.log.close()

    def send(self, data: str | bytes) -> None:
        assert self.port is not None
        raw = data.encode("utf-8") if isinstance(data, str) else data
        self.port.write(raw)
        self.port.flush()

    def sendline(self, line: str = "") -> None:
        self.send(line + "\n")

    def drain(self, seconds: float = 0.5) -> bytes:
        deadline = time.monotonic() + seconds
        chunks = bytearray()
        while time.monotonic() < deadline:
            chunks.extend(self._read_once())
        return bytes(chunks)

    def expect(self, patterns: list[str], timeout: float, *, clear: bool = False) -> tuple[int, bytes]:
        compiled: list[Pattern[bytes]] = [re.compile(p.encode("utf-8"), re.I | re.M) for p in patterns]
        if clear:
            self.buffer.clear()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for index, pattern in enumerate(compiled):
                if pattern.search(self.buffer):
                    return index, bytes(self.buffer)
            self._read_once()
        tail = bytes(self.buffer[-1000:]).decode("utf-8", errors="replace")
        raise SerialTimeout(f"Timed out after {timeout:.1f}s waiting for {patterns}. Tail:\n{tail}")

    def command(self, command: str, prompt: str, timeout: float) -> bytes:
        self.buffer.clear()
        self.sendline(command)
        _, output = self.expect([prompt], timeout)
        return output

    def _read_once(self) -> bytes:
        assert self.port is not None and self.log is not None
        count = self.port.in_waiting
        raw = self.port.read(count if count else 1)
        if raw:
            self.log.write(raw)
            self.log.flush()
            if self.settings.get("show_output", True):
                write_console_bytes(sys.stdout, raw)
            self.buffer.extend(raw)
            if len(self.buffer) > 2_000_000:
                del self.buffer[:-1_000_000]
        return raw
