"""MediaMTX: RTMP ingest, internal RTSP bus, WebRTC previews."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from pathlib import Path

import httpx
import yaml

from .config import BIN, DATA, Store

log = logging.getLogger(__name__)

LOCAL = ["127.0.0.1", "::1"]
UNUSED_ADDRESSES = {
    "rtspsAddress": 8322, "rtpAddress": 8000, "rtcpAddress": 8001, "srtpAddress": 8004, "srtcpAddress": 8005,
    "rtmpsAddress": 1936, "hlsAddress": 8888, "srtAddress": 8890, "metricsAddress": 9998, "pprofAddress": 9999,
    "playbackAddress": 9996, "moqHTTP2Address": 8892, "moqHTTP3Address": 8892, "moqQUICAddress": 8893,
}
ALL_ACTIONS = [{"action": a} for a in ("publish", "read", "playback", "api", "metrics", "pprof")]


def _die_with_parent() -> None:
    """Make mediamtx exit if this process dies, so it never holds the ports as an orphan."""
    import ctypes
    import signal
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


def render_config(store: Store) -> dict:
    s = store.get("server")
    ip = s["bind_ip"]
    if not ip or ip in ("0.0.0.0", "::") or ":" in ip:
        raise SystemExit("server.bind_ip must be a specific IPv4 address (see data/config.json)")
    cfg = {
        "logLevel": "info",
        "api": True,
        "apiAddress": f"127.0.0.1:{s['api_port']}",
        "rtmp": True,
        "rtmpAddress": f"{ip}:{s['rtmp_port']}",
        "rtsp": True,
        "rtspTransports": ["tcp"],
        "rtspAddress": f"127.0.0.1:{s['rtsp_port']}",
        "hls": False,
        "srt": False,
        "moq": False,          # v1.21+ enables Media-over-QUIC by default on dual-stack :8892/:8893
        "metrics": False,
        "pprof": False,
        "playback": False,
        # Unused listeners pinned to localhost as well, so a changed upstream default can
        # never open a public (IPv6) port. The listener guard below enforces this at runtime.
        **{k: f"127.0.0.1:{port}" for k, port in UNUSED_ADDRESSES.items()},
        "webrtc": True,
        # signalling is only reachable via the authenticated web-UI proxy
        "webrtcAddress": f"127.0.0.1:{s['webrtc_http_port']}",
        "webrtcLocalUDPAddress": f"{ip}:{s['webrtc_udp_port']}",
        "webrtcLocalTCPAddress": f"{ip}:{s['webrtc_udp_port']}",
        # only advertise the bind address (+ public host), never the interfaces' IPv6 addresses
        "webrtcIPsFromInterfaces": False,
        "webrtcAdditionalHosts": [h for h in (ip, s["public_host"]) if h],
        "authMethod": "internal",
        "authInternalUsers": [
            # the local services (processor, forwarders, UI proxy) can do anything
            {"user": "any", "pass": "", "ips": LOCAL, "permissions": ALL_ACTIONS},
            # the glasses may only publish to the secret path
            {"user": "any", "pass": "", "ips": [],
             "permissions": [{"action": "publish", "path": store.raw_path}]},
        ],
        "pathDefaults": {"overridePublisher": True},
        "paths": {store.raw_path: record_conf(store, "raw"), "blurred": record_conf(store, "blurred")},
    }
    return cfg


def record_conf(store: Store, source: str) -> dict:
    """mediamtx native recording: starts with the first packet of every (re)connection."""
    rec = store.get("recording")
    return {
        "record": bool(rec[source]),
        # mediamtx insists on %path: files land in recordings/live/<key>/ and recordings/blurred/
        "recordPath": str(recordings_dir(store) / "%path" / "%Y-%m-%d_%H-%M-%S-%f"),
        "recordFormat": "fmp4",                 # fragmented: survives crashes / power loss
        "recordPartDuration": "1s",
        "recordSegmentDuration": f"{int(rec['segment_minutes'])}m",
        "recordDeleteAfter": "0s",              # never delete automatically
    }


class MediaMTX:
    def __init__(self, store: Store):
        self.store = store
        self.proc: subprocess.Popen | None = None
        self.cfg_path = DATA / "mediamtx.yml"
        self.log_path = DATA / "mediamtx.log"
        s = store.get("server")
        self.api = f"http://127.0.0.1:{s['api_port']}"
        self.whep_base = f"http://127.0.0.1:{s['webrtc_http_port']}"
        self.paths: dict[str, dict] = {}
        self.bitrates: dict[str, float] = {}    # path -> inbound kbit/s
        self._bytes: dict[str, tuple[float, int]] = {}
        self.blocked = ""       # set by the listener guard: refuse to run

    def start(self) -> None:
        self.cfg_path.write_text(yaml.safe_dump(render_config(self.store), sort_keys=False))
        binary = BIN / "mediamtx"
        if not binary.exists():
            raise SystemExit(f"{binary} not found - run scripts/setup.sh first")
        logf = open(self.log_path, "ab")
        self.proc = subprocess.Popen([str(binary), str(self.cfg_path)], stdout=logf, stderr=subprocess.STDOUT,
                                     cwd=DATA, preexec_fn=_die_with_parent)
        log.info("mediamtx started (pid %s)", self.proc.pid)

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def exposed_listeners(self) -> list[str]:
        """Sockets mediamtx listens on that are neither bind_ip nor loopback."""
        if not self.proc:
            return []
        allowed = {self.store.get("server")["bind_ip"], "127.0.0.1", "[::1]"}
        out = subprocess.run(["ss", "-Hltunp"], capture_output=True, text=True).stdout
        bad = []
        for line in out.splitlines():
            if f"pid={self.proc.pid}," not in line:
                continue
            local = line.split()[4]
            host = local.rsplit(":", 1)[0]
            if host not in allowed:
                bad.append(f"{line.split()[0]} {local}")
        return bad

    async def watch(self) -> None:
        """Restart if it dies; poll path status; kill it if it ever listens on a public address."""
        checks = 0
        async with httpx.AsyncClient(timeout=2) as c:
            while True:
                if self.blocked:
                    await asyncio.sleep(5)
                    continue
                checks += 1
                if checks % 10 == 4 and (bad := await asyncio.to_thread(self.exposed_listeners)):
                    self.blocked = f"mediamtx listened on non-allowed addresses {bad}; stopped it"
                    log.critical(self.blocked)
                    self.stop()
                    continue
                if self.proc and self.proc.poll() is not None:
                    log.error("mediamtx exited (%s), restarting - see %s", self.proc.returncode, self.log_path)
                    await asyncio.sleep(2)
                    self.start()
                try:
                    r = await c.get(f"{self.api}/v3/paths/list")
                    self.paths = {p["name"]: p for p in r.json().get("items", [])}
                    self._update_bitrates()
                except Exception:
                    self.paths = {}
                await asyncio.sleep(0.5)

    async def apply_recording(self) -> None:
        """Toggle recording at runtime (does not interrupt the publisher)."""
        async with httpx.AsyncClient(timeout=5) as c:
            for source, name in (("raw", self.store.raw_path), ("blurred", "blurred")):
                r = await c.patch(f"{self.api}/v3/config/paths/patch/{name}", json=record_conf(self.store, source))
                r.raise_for_status()

    def _update_bitrates(self) -> None:
        """Inbound bitrate per path, averaged over ~2 s windows."""
        now = time.monotonic()
        for name, p in self.paths.items():
            b = int(p.get("inboundBytes", p.get("bytesReceived", 0)) or 0)
            t0, b0 = self._bytes.get(name, (now, b))
            if not p.get("ready") or b < b0:
                self.bitrates[name], self._bytes[name] = 0.0, (now, b)
            elif now - t0 >= 2.0:
                self.bitrates[name], self._bytes[name] = (b - b0) * 8 / 1000 / (now - t0), (now, b)
            else:
                self._bytes.setdefault(name, (t0, b0))

    def ready(self, name: str) -> bool:
        p = self.paths.get(name)
        return bool(p and p.get("ready"))

    def rtsp_url(self, name: str) -> str:
        return f"rtsp://127.0.0.1:{self.store.get('server')['rtsp_port']}/{name}"


def recordings_dir(store: Store) -> Path:
    d = Path(store.get("recording")["path"] or DATA / "recordings")
    d.mkdir(parents=True, exist_ok=True)
    return d
