"""Persistent settings (data/config.json)."""

from __future__ import annotations

import copy
import json
import os
import secrets
import socket
import threading
from pathlib import Path

ROOT = Path(os.environ.get("LM_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("LM_DATA", ROOT / "data"))
MODELS = Path(os.environ.get("LM_MODELS", ROOT / "models"))
BIN = Path(os.environ.get("LM_BIN", ROOT / "bin"))

DEFAULTS: dict = {
    "server": {
        "web_port": 8056,
        "rtmp_port": 1935,
        "rtsp_port": 8554,          # localhost only
        "webrtc_http_port": 8889,   # localhost only (proxied through the web UI)
        "webrtc_udp_port": 8189,    # must be reachable by preview browsers
        "api_port": 9997,           # localhost only
        # Every externally reachable listener (web UI, RTMP, WebRTC media) binds ONLY to this
        # IPv4 address. Never a wildcard: Go (mediamtx) turns ":port" / "0.0.0.0:port" into a
        # dual-stack socket, which would expose the service on public IPv6.
        "bind_ip": "",              # auto-detected (IPv4 of the default route) on first run
        "public_host": "",          # hostname / IP shown in the UI + added as WebRTC ICE host
        "stream_key": "",           # generated on first run
        "ui_user": "admin",
        "ui_password": "",          # generated on first run
    },
    "output": {"width": 1920, "height": 1080, "fps": 30, "bitrate_kbps": 6000, "audio_kbps": 128},
    "privacy": {
        "on_air": True,             # master switch: False = every upstream RTMP connection is closed
        "delay_seconds": 10,        # broadcast delay for the masked stream (0-60 s)
        "paused": False,            # show slate + silence instead of the camera
        "full_blur": False,         # blur the entire frame (panic mode)
        "mute": False,              # replace audio with silence
        "faces": True,
        "screens": ["cell phone", "laptop"],
        "plates": True,             # vehicle license plates
        "face_conf": 0.2,
        "obj_conf": 0.2,
        "plate_conf": 0.2,
        "face_imgsz": 1280,
        "obj_imgsz": 960,
        "plate_imgsz": 1280,
        "blur_mode": "blur",        # blur | pixelate | solid
        "pad": 0.25,                # fraction of box size added on every side
        "hold_seconds": 1.0,        # keep blurring after an object is lost
        "grow_per_s": 0.6,          # extra padding per second while lost
        "min_size": 40,             # px
        "zones": [],                # [[x1,y1,x2,y2], ...] normalised, always blurred
        "paused_text": "Paused - back soon",
        "lost_text": "Reconnecting...",
        "stop_raw_when_paused": True,
    },
    # path: absolute folder for recordings ("" = data/recordings)
    "recording": {"raw": True, "blurred": False, "segment_minutes": 30, "path": ""},
    "models": {"face": "yolov11s-face.pt", "object": "yolo11s.pt",
               "plate": "license-plate-finetune-v1m.pt", "device": "cuda:0"},
    "destinations": [],             # {id, name, url, source: raw|blurred, enabled}
}


def detect_ipv4() -> str:
    """IPv4 address of the interface holding the default route (no packet is sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


class Store:
    def __init__(self, path: Path = DATA / "config.json"):
        self.path = path
        self._lock = threading.RLock()
        path.parent.mkdir(parents=True, exist_ok=True)
        saved = json.loads(path.read_text()) if path.exists() else {}
        self.data = _merge(DEFAULTS, saved)
        srv = self.data["server"]
        if not srv["stream_key"]:
            srv["stream_key"] = secrets.token_urlsafe(12).replace("-", "x").replace("_", "y")
        if not srv["bind_ip"]:
            srv["bind_ip"] = detect_ipv4()
        if not srv["ui_password"]:
            srv["ui_password"] = secrets.token_urlsafe(9)
        self.save()

    def save(self) -> None:
        with self._lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=2))
            os.chmod(tmp, 0o600)
            tmp.replace(self.path)

    def get(self, section: str) -> dict:
        with self._lock:
            return copy.deepcopy(self.data[section])

    def update(self, section: str, patch: dict) -> dict:
        with self._lock:
            cur = self.data[section]
            for k, v in patch.items():
                if k in cur:
                    cur[k] = v
            self.save()
            return copy.deepcopy(cur)

    def destinations(self) -> list[dict]:
        with self._lock:
            return copy.deepcopy(self.data["destinations"])

    def set_destinations(self, dests: list[dict]) -> None:
        with self._lock:
            self.data["destinations"] = dests
            self.save()

    @property
    def raw_path(self) -> str:
        return f"live/{self.data['server']['stream_key']}"
