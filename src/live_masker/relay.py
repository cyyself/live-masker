"""Supervised ffmpeg jobs: forward raw/blurred to RTMP destinations."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque

from .config import Store
from .mediamtx import MediaMTX, die_with_parent

log = logging.getLogger(__name__)


class Job:
    def __init__(self, key: str, source: str, output_args: list[str], label: str):
        self.key, self.source, self.output_args, self.label = key, source, output_args, label
        self.proc: asyncio.subprocess.Process | None = None
        self.state = "idle"             # idle | waiting | running | backoff
        self.started_at = 0.0
        self.next_try = 0.0
        self.failures = 0
        self.log: deque[str] = deque(maxlen=30)
        self.progress: dict[str, str] = {}

    def info(self) -> dict:
        return {
            "state": self.state,
            "uptime": round(time.time() - self.started_at) if self.state == "running" else 0,
            "bitrate": self.progress.get("bitrate", ""),
            "speed": self.progress.get("speed", ""),
            "failures": self.failures,
            "log": list(self.log)[-8:],
        }

    async def start(self, input_url: str) -> None:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostats", "-progress", "pipe:1",
               "-rtsp_transport", "tcp", "-i", input_url, "-map", "0", "-c", "copy",
               # AAC arrives from mediamtx's RTSP without key flags; without this ffmpeg drops all audio
               "-copyinkf:a",
               *self.output_args]
        self.proc = await asyncio.create_subprocess_exec(
            *cmd, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            preexec_fn=die_with_parent)
        self.state, self.started_at = "running", time.time()
        self.log.append(f"{time.strftime('%H:%M:%S')} started")
        asyncio.create_task(self._pump_progress(self.proc))
        asyncio.create_task(self._pump_log(self.proc))
        asyncio.create_task(self._wait(self.proc))

    async def _pump_progress(self, proc) -> None:
        cur: dict[str, str] = {}
        async for line in proc.stdout:
            k, _, v = line.decode(errors="replace").strip().partition("=")
            cur[k] = v
            if k == "progress":
                self.progress, cur = cur, {}

    async def _pump_log(self, proc) -> None:
        async for line in proc.stderr:
            s = line.decode(errors="replace").rstrip()
            if s:
                self.log.append(f"{time.strftime('%H:%M:%S')} {s[:300]}")

    async def _wait(self, proc) -> None:
        rc = await proc.wait()
        if proc is not self.proc:
            return
        ran = time.time() - self.started_at
        self.failures = 0 if ran > 30 else self.failures + 1
        self.next_try = time.time() + min(30, 2 ** min(self.failures, 5))
        self.state = "backoff" if rc != 0 and rc != 255 else "idle"
        self.log.append(f"{time.strftime('%H:%M:%S')} exited rc={rc} after {ran:.0f}s")
        if rc not in (0, 255) and ran < 15 and any("Broken pipe" in l for l in list(self.log)[-6:]):
            self.log.append("hint: the platform accepted the connection, then closed it. Usually the stream key / "
                            "live session is not accepted (live room not started or already ended, key in use elsewhere).")
        self.proc = None

    async def stop(self) -> None:
        proc, self.proc = self.proc, None
        self.state = "idle"
        if proc and proc.returncode is None:
            proc.terminate()   # ffmpeg finalises the file / closes the RTMP session
            try:
                await asyncio.wait_for(proc.wait(), 5)
            except asyncio.TimeoutError:
                proc.kill()
            self.log.append(f"{time.strftime('%H:%M:%S')} stopped")


def dest_output_args(url: str) -> list[str]:
    fmt = "flv" if url.startswith(("rtmp://", "rtmps://")) else ("rtsp" if url.startswith("rtsp") else "flv")
    extra = ["-flvflags", "no_duration_filesize"] if fmt == "flv" else []
    return [*extra, "-f", fmt, url]


class RelayManager:
    def __init__(self, store: Store, mtx: MediaMTX):
        self.store, self.mtx = store, mtx
        self.jobs: dict[str, Job] = {}
        # reconcile() is called from the loop and from API handlers; without this lock two
        # concurrent calls could each start an ffmpeg for the same destination (one untracked).
        self._lock = asyncio.Lock()

    def _desired(self) -> dict[str, tuple[str, list[str], str]]:
        priv = self.store.get("privacy")
        want: dict[str, tuple[str, list[str], str]] = {}
        if not priv["on_air"]:          # "Stop live": no upstream connections at all
            return want
        for d in self.store.destinations():
            if not d.get("enabled"):
                continue
            if d["source"] == "raw" and priv["paused"] and priv["stop_raw_when_paused"]:
                continue
            want[f"dest:{d['id']}:{d['source']}:{d['url']}"] = (d["source"], dest_output_args(d["url"]), d["name"])
        return want

    def mtx_path(self, source: str) -> str:
        return self.store.raw_path if source == "raw" else "blurred"

    async def run(self) -> None:
        while True:
            try:
                await self.reconcile()
            except Exception:
                log.exception("reconcile failed")
            await asyncio.sleep(0.5)

    async def reconcile(self) -> None:
        async with self._lock:
            await self._reconcile()

    async def _reconcile(self) -> None:
        want = self._desired()
        for key in list(self.jobs):
            if key not in want:
                await self.jobs.pop(key).stop()
        now = time.time()
        for key, (source, args, label) in want.items():
            job = self.jobs.setdefault(key, Job(key, source, args, label))
            if job.proc is not None:
                continue
            if not self.mtx.ready(self.mtx_path(source)):
                job.state = "waiting"
                continue
            if job.state == "backoff" and now < job.next_try:
                continue
            await job.start(self.mtx.rtsp_url(self.mtx_path(source)))

    def dest_status(self, dest_id: str) -> dict | None:
        for key, job in self.jobs.items():
            if key.startswith(f"dest:{dest_id}:"):
                return job.info()
        return None

    async def stop_all(self) -> None:
        async with self._lock:
            jobs = list(self.jobs.values())
        for job in jobs:
            await job.stop()
