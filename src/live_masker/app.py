"""Web UI + API."""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import shutil
import socket
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote, unquote

import cv2
import httpx
import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from .config import Store
from .detector import COCO_SCREEN_CLASSES
from .mediamtx import MediaMTX, recordings_dir
from .processor import Processor
from .relay import RelayManager

log = logging.getLogger("live_masker")
STATIC = Path(__file__).parent / "static"

store = Store()
mtx = MediaMTX(store)
proc = Processor(store, mtx)
relay = RelayManager(store, mtx)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    mtx.start()
    tasks = [asyncio.create_task(mtx.watch()), asyncio.create_task(relay.run())]
    proc.start()
    yield
    for t in tasks:
        t.cancel()
    await relay.stop_all()
    proc.stop()
    mtx.stop()


app = FastAPI(title="live-masker", lifespan=lifespan)
security = HTTPBasic()


def auth(cred: HTTPBasicCredentials = Depends(security)) -> None:
    s = store.get("server")
    ok = secrets.compare_digest(cred.username.encode(), s["ui_user"].encode()) and \
        secrets.compare_digest(cred.password.encode(), s["ui_password"].encode())
    if not ok:
        raise HTTPException(401, "bad credentials", headers={"WWW-Authenticate": "Basic"})


A = [Depends(auth)]


# ---------------------------------------------------------------- helpers
def host_guess(request: Request) -> str:
    s = store.get("server")
    if s["public_host"]:
        return s["public_host"]
    return (request.url.hostname or socket.gethostname())


def mask_url(url: str) -> str:
    # hide the stream key part of an RTMP url
    head, _, tail = url.rpartition("/")
    return f"{head}/{tail[:3]}…" if len(tail) > 4 else url


def jpeg(img: np.ndarray | None, width: int = 960, q: int = 70) -> bytes | None:
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes() if ok else None


# ---------------------------------------------------------------- pages
@app.get("/", dependencies=A)
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/state", dependencies=A)
def state(request: Request):
    s = store.get("server")
    host = host_guess(request)
    dests = []
    for d in store.destinations():
        dests.append({**{k: v for k, v in d.items() if k != "url"}, "url_masked": mask_url(d["url"]), "status": relay.dest_status(d["id"])})
    raw = mtx.paths.get(store.raw_path) or {}
    return {
        "ingest": {
            "online": mtx.ready(store.raw_path),
            "url": f"rtmp://{host}:{s['rtmp_port']}/{store.raw_path}",
            "server": f"rtmp://{host}:{s['rtmp_port']}/live",
            "key": s["stream_key"],
            "tracks": raw.get("tracks", []),
            "bytes": raw.get("inboundBytes", raw.get("bytesReceived", 0)),
            "info": proc.input_info,
            "bitrate_kbps": round(mtx.bitrates.get(store.raw_path, 0.0)),
            "errors": raw.get("inboundFramesInError", 0),
            "readers": len(raw.get("readers", [])),
        },
        "processor": proc.stats,
        "security_error": mtx.blocked,
        "blurred_online": mtx.ready("blurred"),
        "output_bitrate_kbps": round(mtx.bitrates.get("blurred", 0.0)),
        "privacy": store.get("privacy"),
        "output": store.get("output"),
        "recording": {**(rec := store.get("recording")),
                      "active": {"raw": rec["raw"] and mtx.ready(store.raw_path),
                                 "blurred": rec["blurred"] and mtx.ready("blurred")}},
        "destinations": dests,
        "public_host": s["public_host"],
        "screen_classes": list(COCO_SCREEN_CLASSES),
    }


# ---------------------------------------------------------------- settings
PRIVACY_TYPES = {
    "on_air": bool, "paused": bool, "full_blur": bool, "mute": bool, "faces": bool, "stop_raw_when_paused": bool,
    "face_conf": float, "obj_conf": float, "pad": float, "hold_seconds": float, "grow_per_s": float,
    "face_imgsz": int, "obj_imgsz": int, "min_size": int,
    "blur_mode": str, "paused_text": str, "lost_text": str, "screens": list, "zones": list,
}


@app.post("/api/privacy", dependencies=A)
async def set_privacy(patch: dict):
    clean = {}
    for k, v in patch.items():
        typ = PRIVACY_TYPES.get(k)
        if typ is None:
            raise HTTPException(400, f"unknown field {k}")
        clean[k] = v if typ is list else typ(v)
    if "blur_mode" in clean and clean["blur_mode"] not in ("blur", "pixelate", "solid"):
        raise HTTPException(400, "blur_mode")
    if "screens" in clean:
        clean["screens"] = [c for c in clean["screens"] if c in COCO_SCREEN_CLASSES]
    if "zones" in clean:
        clean["zones"] = [[min(1.0, max(0.0, float(v))) for v in z[:4]] for z in clean["zones"] if len(z) >= 4]
    for k in ("paused_text", "lost_text"):   # slate font is ASCII-only
        if k in clean:
            clean[k] = clean[k].encode("ascii", "replace").decode()[:60]
    res = store.update("privacy", clean)
    await relay.reconcile()   # pausing must stop raw forwards *now*
    return res


@app.post("/api/output", dependencies=A)
def set_output(patch: dict):
    clean = {k: int(v) for k, v in patch.items() if k in ("width", "height", "fps", "bitrate_kbps", "audio_kbps")}
    res = store.update("output", clean)
    proc.restart_output()
    return res


@app.post("/api/recording", dependencies=A)
async def set_recording(patch: dict):
    clean = {}
    for k in ("raw", "blurred"):
        if k in patch:
            clean[k] = bool(patch[k])
    if "segment_minutes" in patch:
        clean["segment_minutes"] = max(1, int(patch["segment_minutes"]))
    if "path" in patch:
        path = str(patch["path"]).strip()
        if path:
            d = Path(path).expanduser()
            if not d.is_absolute():
                raise HTTPException(400, "recording path must be absolute")
            try:
                d.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise HTTPException(400, f"cannot create {d}: {e}")
            if not os.access(d, os.W_OK):
                raise HTTPException(400, f"{d} is not writable")
            path = str(d.resolve())
        clean["path"] = path
    res = store.update("recording", clean)
    await mtx.apply_recording()
    return res


@app.post("/api/server", dependencies=A)
async def set_server(patch: dict):
    s = store.get("server")
    changed = False
    if "public_host" in patch and patch["public_host"] != s["public_host"]:
        store.update("server", {"public_host": str(patch["public_host"]).strip()})
        changed = True
    if patch.get("regenerate_key"):
        store.update("server", {"stream_key": secrets.token_urlsafe(12).replace("-", "x").replace("_", "y")})
        changed = True
    if changed:
        mtx.stop()
        mtx.start()
    return {"ok": True}


# ---------------------------------------------------------------- destinations
def _validate_dest(d: dict) -> dict:
    url = str(d.get("url", "")).strip()
    if not url.startswith(("rtmp://", "rtmps://")):
        raise HTTPException(400, "url must start with rtmp:// or rtmps://")
    src = d.get("source", "blurred")
    if src not in ("raw", "blurred"):
        raise HTTPException(400, "source must be raw or blurred")
    return {"name": str(d.get("name") or "destination")[:60], "url": url, "source": src,
            "enabled": bool(d.get("enabled", False))}


@app.post("/api/destinations", dependencies=A)
async def add_dest(d: dict):
    dests = store.destinations()
    new = {"id": uuid.uuid4().hex[:8], **_validate_dest(d)}
    dests.append(new)
    store.set_destinations(dests)
    await relay.reconcile()
    return new


@app.put("/api/destinations/{dest_id}", dependencies=A)
async def edit_dest(dest_id: str, d: dict):
    dests = store.destinations()
    for i, cur in enumerate(dests):
        if cur["id"] == dest_id:
            merged = {**cur, **{k: v for k, v in d.items() if k in ("name", "url", "source", "enabled")}}
            dests[i] = {"id": dest_id, **_validate_dest(merged)}
            store.set_destinations(dests)
            await relay.reconcile()
            return dests[i]
    raise HTTPException(404)


@app.delete("/api/destinations/{dest_id}", dependencies=A)
async def del_dest(dest_id: str):
    store.set_destinations([d for d in store.destinations() if d["id"] != dest_id])
    await relay.reconcile()
    return {"ok": True}


# ---------------------------------------------------------------- recordings
@app.get("/api/recordings", dependencies=A)
def list_recordings():
    root = recordings_dir(store)
    files = sorted((f for f in root.rglob("*") if f.suffix in (".mp4", ".mkv") and f.is_file()),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for f in files:
        rel = f.relative_to(root).as_posix()
        out.append({"name": rel, "source": "blurred" if rel.startswith("blurred/") else "raw",
                    "label": f.name, "size": f.stat().st_size, "mtime": f.stat().st_mtime})
    return {"files": out, "total": sum(f["size"] for f in out), "free": shutil.disk_usage(root).free,
            "dir": str(root)}


def _rec_file(name: str) -> Path:
    root = recordings_dir(store).resolve()
    p = (root / name).resolve()
    if not p.is_relative_to(root) or not p.is_file():
        raise HTTPException(404)
    return p


@app.get("/recordings/{name:path}", dependencies=A)
def get_recording(name: str):
    p = _rec_file(name)
    return FileResponse(p, filename=p.name)


@app.delete("/api/recordings/{name:path}", dependencies=A)
def del_recording(name: str):
    _rec_file(name).unlink()
    return {"ok": True}


# ---------------------------------------------------------------- previews
def _frame(kind: str):
    if kind == "raw":
        return proc.latest_raw, proc.raw_seq
    if kind == "blurred":
        return proc.latest_out, proc.out_seq
    raise HTTPException(404)


@app.get("/api/snapshot/{kind}.jpg", dependencies=A)
async def snapshot(kind: str, width: int = 1280):
    img, _ = _frame(kind)
    data = await asyncio.to_thread(jpeg, img, width, 85)
    if data is None:
        raise HTTPException(404, "no frame yet")
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/mjpeg/{kind}", dependencies=A)
async def mjpeg(kind: str, request: Request, fps: float = 8, width: int = 854):
    _frame(kind)

    async def gen():
        last = -1
        while not await request.is_disconnected():
            img, seq = _frame(kind)
            if seq != last and img is not None:
                last = seq
                data = await asyncio.to_thread(jpeg, img, width, 65)
                if data:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
            await asyncio.sleep(1 / max(1, min(fps, 30)))

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store"})


# WebRTC (WHEP) signalling proxy: mediamtx only listens on localhost, so every
# preview session has to go through this authenticated endpoint.
@app.post("/api/whep/{kind}", dependencies=A)
async def whep(kind: str, request: Request):
    path = store.raw_path if kind == "raw" else "blurred" if kind == "blurred" else None
    if path is None:
        raise HTTPException(404)
    body = await request.body()
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.post(f"{mtx.whep_base}/{path}/whep", content=body,
                         headers={"Content-Type": "application/sdp"})
    headers = {}
    if loc := r.headers.get("location"):
        headers["Location"] = f"/api/whep-session?loc={quote(loc, safe='')}"
    return Response(r.content, status_code=r.status_code, media_type="application/sdp", headers=headers)


@app.delete("/api/whep-session", dependencies=A)
async def whep_close(loc: str):
    loc = unquote(loc)
    if not loc.startswith("/") or "//" in loc:
        raise HTTPException(400)
    async with httpx.AsyncClient(timeout=5) as c:
        await c.delete(f"{mtx.whep_base}{loc}")
    return {"ok": True}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    s = store.get("server")
    print(
        "\n  live-masker\n"
        f"  Listening on: {s['bind_ip']} (IPv4 only)\n"
        f"  Web UI      : http://{s['bind_ip']}:{s['web_port']}/   (user: {s['ui_user']}  password: {s['ui_password']})\n"
        f"  Glasses URL : rtmp://{s['public_host'] or s['bind_ip']}:{s['rtmp_port']}/{store.raw_path}\n", flush=True)
    if not s["bind_ip"] or s["bind_ip"] in ("0.0.0.0", "::") or ":" in s["bind_ip"]:
        raise SystemExit("server.bind_ip must be a specific IPv4 address (see data/config.json)")
    uvicorn.run(app, host=s["bind_ip"], port=s["web_port"], log_level="warning", proxy_headers=False)


if __name__ == "__main__":
    main()
