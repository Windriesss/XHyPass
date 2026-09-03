from __future__ import annotations

import socket
import re
import sys
import time
from io import BytesIO
from pathlib import Path

import paramiko

from .serial_session import write_console_bytes


class SSHCommandError(RuntimeError):
    pass


class SSHSession:
    """Persistent SSH/SFTP connection to the Jailhouse root cell."""

    def __init__(self, settings: dict):
        self.settings = settings
        self.client: paramiko.SSHClient | None = None

    def connect(self) -> None:
        deadline = time.monotonic() + float(self.settings.get("connect_timeout", 60))
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            remaining = max(0.5, deadline - time.monotonic())
            probe_timeout = min(
                float(self.settings.get("tcp_probe_timeout", 2)), remaining
            )
            try:
                with socket.create_connection(
                    (self.settings["host"], int(self.settings.get("port", 22))),
                    timeout=probe_timeout,
                ):
                    pass
            except OSError as exc:
                last_error = exc
                retry = min(
                    float(self.settings.get("retry_interval", 2)),
                    max(0, deadline - time.monotonic()),
                )
                if retry:
                    print(f"[SSH] TCP/22 unavailable ({exc}); retry in {retry:g}s...")
                    time.sleep(retry)
                continue

            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                password = self.settings.get("password")
                key_file = self.settings.get("key_file") or None
                client.connect(
                    hostname=self.settings["host"],
                    port=int(self.settings.get("port", 22)),
                    username=self.settings.get("username", "root"),
                    password=password,
                    key_filename=key_file,
                    timeout=min(3, remaining),
                    banner_timeout=min(3, remaining),
                    auth_timeout=min(3, remaining),
                    allow_agent=bool(self.settings.get("allow_agent", True)),
                    look_for_keys=bool(self.settings.get("look_for_keys", True)),
                )
                self.client = client
                transport = client.get_transport()
                if transport:
                    transport.set_keepalive(10)
                print(
                    f"[SSH] Connected: {self.settings.get('username', 'root')}@"
                    f"{self.settings['host']}:{self.settings.get('port', 22)}"
                )
                return
            except (OSError, paramiko.SSHException, EOFError) as exc:
                last_error = exc
                client.close()
                time.sleep(float(self.settings.get("retry_interval", 2)))
        raise ConnectionError(
            f"SSH connection to {self.settings['host']} failed: {last_error}"
        )

    def close(self) -> None:
        if self.client:
            self.client.close()
            self.client = None

    def is_active(self) -> bool:
        """Return whether the persistent SSH transport can open new channels."""
        if not self.client:
            return False
        transport = self.client.get_transport()
        return bool(
            transport
            and transport.is_active()
            and transport.is_authenticated()
        )

    def reconnect(self) -> None:
        """Replace a stale persistent connection without retrying any command."""
        self.close()
        self.connect()

    def tcp_available(self, timeout: float = 2) -> bool:
        """Probe this session's configured SSH endpoint without using Paramiko."""
        try:
            with socket.create_connection(
                (self.settings["host"], int(self.settings.get("port", 22))),
                timeout=timeout,
            ):
                return True
        except OSError:
            return False

    def run(
        self,
        command: str,
        *,
        timeout: float,
        log_path: Path,
        check: bool = True,
        show_output: bool = True,
    ) -> tuple[int, bytes, bytes]:
        if not self.client:
            raise ConnectionError("SSH session is not connected")
        started = time.monotonic()
        print(f"\n[SSH CMD] {command}")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab") as log:
            header = f"\n$ {command}\n".encode("utf-8")
            log.write(header)
            log.flush()
            _, stdout, _ = self.client.exec_command(command, timeout=10)
            channel = stdout.channel
            out = bytearray()
            err = bytearray()
            deadline = time.monotonic() + timeout
            next_progress = time.monotonic() + float(
                self.settings.get("progress_interval", 10)
            )
            while True:
                progressed = False
                if channel.recv_ready():
                    chunk = channel.recv(65536)
                    out.extend(chunk)
                    log.write(chunk)
                    if show_output:
                        write_console_bytes(sys.stdout, chunk)
                    progressed = True
                if channel.recv_stderr_ready():
                    chunk = channel.recv_stderr(65536)
                    err.extend(chunk)
                    log.write(b"[stderr] " + chunk)
                    if show_output:
                        write_console_bytes(sys.stderr, chunk)
                    progressed = True
                if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
                    break
                if time.monotonic() >= deadline:
                    channel.close()
                    raise TimeoutError(f"SSH command timed out after {timeout}s: {command}")
                if show_output and time.monotonic() >= next_progress:
                    elapsed = time.monotonic() - started
                    remaining = max(0, deadline - time.monotonic())
                    print(
                        f"[SSH WAIT] running {elapsed:.0f}s, "
                        f"timeout remaining {remaining:.0f}s"
                    )
                    next_progress = time.monotonic() + float(
                        self.settings.get("progress_interval", 10)
                    )
                if not progressed:
                    time.sleep(0.05)
            status = channel.recv_exit_status()
            log.write(f"\n[exit={status}]\n".encode("ascii"))
        elapsed = time.monotonic() - started
        print(f"[SSH EXIT] code={status}, elapsed={elapsed:.1f}s")
        if check and status != 0:
            raise SSHCommandError(f"SSH command failed with exit {status}: {command}")
        return status, bytes(out), bytes(err)

    def get(self, remote: str, local: Path) -> None:
        if not self.client:
            raise ConnectionError("SSH session is not connected")
        local.parent.mkdir(parents=True, exist_ok=True)
        with self.client.open_sftp() as sftp:
            sftp.get(remote, str(local))

    def put(self, local: Path, remote: str) -> None:
        """Upload one experiment artifact through the persistent session."""
        if not self.client:
            raise ConnectionError("SSH session is not connected")
        with self.client.open_sftp() as sftp:
            sftp.put(str(local), remote)

    def path_exists(self, remote: str) -> bool:
        """Check a remote path without adding polling noise to experiment logs."""
        if not self.client:
            raise ConnectionError("SSH session is not connected")
        try:
            with self.client.open_sftp() as sftp:
                sftp.stat(remote)
        except FileNotFoundError:
            return False
        return True

    def put_bytes(self, remote: str, payload: bytes = b"") -> None:
        """Atomically publish a small coordinator marker through SFTP."""
        if not self.client:
            raise ConnectionError("SSH session is not connected")
        temporary = f"{remote}.tmp"
        with self.client.open_sftp() as sftp:
            sftp.putfo(BytesIO(payload), temporary)
            sftp.rename(temporary, remote)

    def wait_for_console_login(
        self,
        command: str,
        *,
        username: str,
        password: str,
        login_prompt: str,
        shell_prompts: list[str],
        timeout: float,
        log_path: Path,
    ) -> None:
        """Enter a Xen guest console, log in, then detach with Ctrl+]."""
        if not self.client:
            raise ConnectionError("SSH session is not connected")
        transport = self.client.get_transport()
        if not transport:
            raise ConnectionError("SSH transport is unavailable")
        channel = transport.open_session(timeout=10)
        channel.get_pty()
        channel.invoke_shell()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("ab") as log:
                log.write(f"\n[interactive] {command}\n".encode("utf-8"))
                channel.send("\n")
                self._interactive_expect(channel, shell_prompts, 15, log, "dom0 shell")
                channel.send(command + "\n")
                self._interactive_expect(channel, [login_prompt], timeout, log, "dom1 login")
                channel.send(username + "\n")
                matched, _ = self._interactive_expect(
                    channel,
                    [*shell_prompts, r"Password:\s*$"],
                    30,
                    log,
                    "dom1 shell",
                )
                if matched == len(shell_prompts):
                    channel.send(password + "\n")
                    self._interactive_expect(channel, shell_prompts, 30, log, "dom1 shell")
                print("[XEN/DOM1] Login succeeded; dom1 is ready.")
                log.write(b"\n[detach console: Ctrl+]]\n")
                channel.send(b"\x1d")
                time.sleep(.3)
        finally:
            channel.close()

    def run_xen_console_command(
        self,
        console_command: str,
        guest_command: str,
        *,
        username: str,
        password: str,
        login_prompt: str,
        guest_shell_prompts: list[str],
        completion_pattern: str,
        timeout: float,
        log_path: Path,
    ) -> None:
        """Run one guest command through `xl console` on a dom0 SSH channel."""
        if not self.client:
            raise ConnectionError("SSH session is not connected")
        transport = self.client.get_transport()
        if not transport or not transport.is_active():
            raise ConnectionError("SSH transport is unavailable")
        channel = transport.open_session(timeout=10)
        channel.get_pty()
        channel.invoke_shell()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("ab") as log:
                log.write(
                    f"\n[xl-console] {console_command}\n".encode("utf-8")
                )
                channel.send("\n")
                self._interactive_expect(
                    channel,
                    [r"root@[^\r\n]*#\s*$", r"#\s*$"],
                    15,
                    log,
                    "dom0 shell",
                )
                channel.send(console_command + "\n")
                # Attaching to an already-running xl console does not always
                # redraw its login or shell prompt.  Give xl a moment to
                # attach, then press Enter on the guest console before
                # waiting for output.  This is harmless while dom1 is still
                # booting and essential when it is idle at a silent prompt.
                time.sleep(.5)
                log.write(b"[xl-console] wake dom1 console with Enter\n")
                channel.send("\n")
                matched, _ = self._interactive_expect(
                    channel,
                    [login_prompt, *guest_shell_prompts],
                    timeout,
                    log,
                    "dom1 login or shell",
                    wake_input="\n",
                    wake_interval=5.0,
                )
                if matched == 0:
                    channel.send(username + "\n")
                    shell_match, _ = self._interactive_expect(
                        channel,
                        [*guest_shell_prompts, r"Password:\s*$"],
                        30,
                        log,
                        "dom1 shell",
                    )
                    if shell_match == len(guest_shell_prompts):
                        channel.send(password + "\n")
                        self._interactive_expect(
                            channel,
                            guest_shell_prompts,
                            30,
                            log,
                            "dom1 shell",
                        )
                channel.send(guest_command + "\n")
                self._interactive_expect(
                    channel,
                    [completion_pattern],
                    timeout,
                    log,
                    "dom1 command completion",
                )
                log.write(b"\n[detach xl console: Ctrl+]]\n")
                channel.send(b"\x1d")
                time.sleep(.3)
        finally:
            channel.close()

    def _interactive_expect(
        self,
        channel,
        patterns: list[str],
        timeout: float,
        log,
        description: str,
        *,
        wake_input: str | bytes | None = None,
        wake_interval: float = 5.0,
    ) -> tuple[int, bytes]:
        compiled = [re.compile(pattern.encode("utf-8"), re.I | re.M) for pattern in patterns]
        buffer = bytearray()
        deadline = time.monotonic() + timeout
        next_progress = time.monotonic() + float(self.settings.get("progress_interval", 10))
        next_wake = time.monotonic() + wake_interval
        while time.monotonic() < deadline:
            if channel.recv_ready():
                chunk = channel.recv(65536)
                if not chunk:
                    break
                buffer.extend(chunk)
                log.write(chunk)
                log.flush()
                write_console_bytes(sys.stdout, chunk)
                for index, pattern in enumerate(compiled):
                    if pattern.search(buffer):
                        return index, bytes(buffer)
            elif channel.closed or channel.exit_status_ready():
                break
            else:
                time.sleep(.05)
            if wake_input is not None and time.monotonic() >= next_wake:
                channel.send(wake_input)
                log.write(
                    f"\n[interactive] wake {description} with Enter\n".encode(
                        "utf-8"
                    )
                )
                log.flush()
                next_wake = time.monotonic() + wake_interval
            if time.monotonic() >= next_progress:
                print(f"[XEN/DOM1] Waiting for {description}...")
                next_progress = time.monotonic() + float(self.settings.get("progress_interval", 10))
        tail = bytes(buffer[-1000:]).decode("utf-8", errors="replace")
        raise TimeoutError(
            f"Timed out after {timeout:g}s waiting for {description} ({patterns}). "
            f"Tail:\n{tail}"
        )
