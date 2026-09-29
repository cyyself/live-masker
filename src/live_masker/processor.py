"""The privacy processor: raw stream in, masked stream out.

Reader thread:  rtsp://.../live/<key>  ->  decoded BGR frames + resampled audio  ->  input queue
Masker thread:  input queue -> detect -> track -> blur -> delay buffer
Writer thread:  delay buffer -> NVENC/AAC -> rtsp://.../blurred

Masking and encoding run in separate threads so a frame's cost is max(mask, encode)
rather than their sum.

Broadcast delay: masked frames wait `delay_seconds` in a buffer before they are encoded.
Pause / Full blur / Mute apply to a frame if they were on at ANY moment between its capture
and its release: pressing Pause also removes the last seconds viewers haven't seen yet, and
Resume never releases anything that was captured or buffered while paused. (Local recording
of the original is not delayed.)

The output is *continuous*: while paused, while the glasses are disconnected or
while models are loading, a slate with silent audio is sent instead, so the
downstream RTMP connections (YouTube etc.) never drop.

Fail-closed rules (never leak an unmasked frame):
  * detector not ready / detector raised  -> whole frame blurred
  * processing falls behind               -> old frames are *dropped*, never passed through unprocessed
  * paused                                -> camera frames are never encoded, audio is zeroed
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from fractions import Fraction

import av
import cv2
import numpy as np

from .config import MODELS, Store
from .detector import Detector, Tracker, blur_full, blur_region
from .mediamtx import MediaMTX

log = logging.getLogger(__name__)

AUDIO_RATE = 48000
VIDEO_TB = Fraction(1, 90000)
MAX_VIDEO_BACKLOG = 2       # frames waiting before we start dropping the oldest
LOST_AFTER = 1.0            # s without camera frames leaving the delay buffer -> slate
MAX_DELAY = 60.0            # s; 60 s of 1080p BGR frames is ~11 GB of RAM
AUDIO_TOLERANCE = 2400      # samples (50 ms) of drift tolerated before trim / fill


def switches(p: dict) -> tuple[bool, bool, bool]:
    return bool(p["paused"]), bool(p["full_blur"]), bool(p["mute"])


def letterbox(img: np.ndarray, w: int, h: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    if (iw, ih) == (w, h):
        return img
    s = min(w / iw, h / ih)
    nw, nh = max(1, round(iw * s)), max(1, round(ih * s))
    out = np.zeros((h, w, 3), np.uint8)
    x, y = (w - nw) // 2, (h - nh) // 2
    out[y:y + nh, x:x + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    return out


def make_slate(w: int, h: int, text: str) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    grad = np.linspace(40, 15, h, dtype=np.float32)[:, None]
    img[:] = np.stack([grad * 1.3, grad * 0.9, grad * 0.7], axis=-1).astype(np.uint8)
    font, scale, th = cv2.FONT_HERSHEY_DUPLEX, h / 540, max(2, h // 270)
    (tw, tht), _ = cv2.getTextSize(text, font, scale, th)
    cv2.putText(img, text, ((w - tw) // 2, (h + tht) // 2), font, scale, (235, 235, 235), th, cv2.LINE_AA)
    return img


class Processor:
    def __init__(self, store: Store, mtx: MediaMTX):
        self.store, self.mtx = store, mtx
        self.detector: Detector | None = None
        self.tracker = Tracker()
        self._q: deque = deque()
        self._cv = threading.Condition()
        self._queued_video = 0
        self._stop = threading.Event()
        self._restart_output = threading.Event()
        self._slates: dict[str, np.ndarray] = {}
        # delay buffer shared by masker (producer) and writer (consumer). Entries:
        # [release_wall, kind, payload, t_in, sid, [paused, full_blur, mute]], arrival order.
        # The switch flags are sticky: once set while buffered they stay set.
        self._delayq: deque = deque()
        self._dq_cv = threading.Condition()

        self.session = 0
        self.last_input = 0.0          # monotonic time of last decoded video frame
        self.last_audio_in = 0.0
        self.ever_connected = False
        self.ever_connected_out = False
        self.input_info: dict = {}
        # previews (latest frames, read by the web app)
        self.latest_raw: np.ndarray | None = None
        self.latest_out: np.ndarray | None = None
        self.raw_seq = 0
        self.out_seq = 0
        self.stats = {
            "mode": "starting", "fps": 0.0, "det_ms": 0.0, "latency_ms": 0.0, "dropped": 0, "buffered_s": 0.0,
            "detections": {}, "regions": 0, "detector": "loading", "output": "connecting", "errors": 0,
        }

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        for fn in (self._load_models, self._reader, self._masker, self._writer):
            threading.Thread(target=fn, name=fn.__name__, daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def restart_output(self) -> None:
        self._restart_output.set()

    def _load_models(self) -> None:
        m = self.store.get("models")
        try:
            self.detector = Detector(MODELS, m["face"], m["object"], m["plate"], m["device"])
            self.stats["detector"] = "ready"
        except Exception as e:  # stays fail-closed (full blur) forever
            log.exception("model load failed")
            self.stats["detector"] = f"error: {e}"

    # ------------------------------------------------------------------ reader
    def _push(self, item) -> None:
        with self._cv:
            if item[0] == "v":
                if self._queued_video >= MAX_VIDEO_BACKLOG:
                    for i, it in enumerate(self._q):
                        if it[0] == "v":
                            del self._q[i]
                            self._queued_video -= 1
                            self.stats["dropped"] += 1
                            break
                self._queued_video += 1
            self._q.append(item)
            self._cv.notify()

    def _pop(self, timeout: float):
        with self._cv:
            if not self._q:
                self._cv.wait(timeout)
            if not self._q:
                return None
            item = self._q.popleft()
            if item[0] == "v":
                self._queued_video -= 1
            return item

    def _reader(self) -> None:
        while not self._stop.is_set():
            path = self.store.raw_path
            if not self.mtx.ready(path):
                time.sleep(0.2)
                continue
            try:
                self._read_session(path)
            except Exception as e:
                log.warning("input session ended: %s", e)
            self.input_info = {}
            self.latest_raw = None
            time.sleep(0.2)

    def _read_session(self, path: str) -> None:
        o = self.store.get("output")
        W, H = o["width"], o["height"]
        with av.open(self.mtx.rtsp_url(path), options={"rtsp_transport": "tcp", "timeout": "5000000"},
                     timeout=(5.0, 5.0)) as inp:
            self.session += 1
            sid = self.session
            vs = inp.streams.video[0] if inp.streams.video else None
            ast = inp.streams.audio[0] if inp.streams.audio else None
            if vs is None:
                raise RuntimeError("no video track")
            vs.thread_type = "AUTO"
            self.input_info = {
                "codec": vs.codec_context.name,
                "width": vs.codec_context.width, "height": vs.codec_context.height, "fps": 0.0,
                "audio": f"{ast.codec_context.name} {ast.codec_context.sample_rate}Hz" if ast else "none",
            }
            fps_n, fps_t0 = 0, time.monotonic()
            log.info("input session %d: %s", sid, self.input_info)
            resampler = av.AudioResampler(format="fltp", layout="stereo", rate=AUDIO_RATE)
            for pkt in inp.demux([s for s in (vs, ast) if s is not None]):
                if self._stop.is_set() or path != self.store.raw_path:
                    return
                try:
                    frames = pkt.decode()
                except av.error.InvalidDataError:
                    continue
                for fr in frames:
                    if fr.time is None:
                        continue
                    now = time.monotonic()
                    if isinstance(fr, av.VideoFrame):
                        img = letterbox(fr.to_ndarray(format="bgr24"), W, H)
                        self.last_input = now
                        # measured, not the advertised rate: shows drops on a weak uplink
                        fps_n += 1
                        if now - fps_t0 >= 2.0:
                            self.input_info.update(fps=round(fps_n / (now - fps_t0), 1),
                                                   width=fr.width, height=fr.height)
                            fps_n, fps_t0 = 0, now
                        self.ever_connected = True
                        self._push(("v", img, float(fr.time), sid, now))
                    else:
                        self.last_audio_in = now
                        for af in resampler.resample(fr):
                            t = float(af.time) if af.time is not None else float(fr.time)
                            self._push(("a", af.to_ndarray(), t, sid, now))

    # ------------------------------------------------------------------ masker
    def _masker(self) -> None:
        """Detect + mask each frame as soon as it arrives, then park it in the delay buffer."""
        mask_sid = -1
        while not self._stop.is_set():
            item = self._pop(0.2)
            if item is None:
                continue
            p = self.store.get("privacy")
            delay = min(MAX_DELAY, max(0.0, float(p["delay_seconds"])))
            if item[0] == "v":
                _, img, t_in, sid, t_arr = item
                if sid != mask_sid:
                    self.tracker.reset()
                    mask_sid = sid
                self.latest_raw, self.raw_seq = img, self.raw_seq + 1
                try:
                    masked = self._mask(img, t_in, p, img.shape[1], img.shape[0])
                except Exception:           # never let an unmasked frame through
                    log.exception("masking failed; blurring full frame")
                    self.stats["errors"] += 1
                    masked = img.copy()
                    blur_full(masked)
                self.stats["latency_ms"] = round((time.monotonic() - t_arr) * 1000, 1)
                entry = [t_arr + delay, "v", masked, t_in, sid, list(switches(p))]
            else:
                _, arr, t_in, sid, t_arr = item
                entry = [t_arr + delay, "a", arr, t_in, sid, list(switches(p))]
            with self._dq_cv:
                self._delayq.append(entry)
                self._dq_cv.notify()

    # ------------------------------------------------------------------ writer
    def _writer(self) -> None:
        while not self._stop.is_set():
            try:
                self._run_output()
            except Exception as e:
                self.stats["output"] = f"error: {e}"
                self.stats["errors"] += 1
                log.warning("output error: %s", e)
                time.sleep(1)

    def _slate(self, key: str, text: str, w: int, h: int) -> np.ndarray:
        k = f"{key}|{text}|{w}x{h}"
        if k not in self._slates:
            self._slates[k] = make_slate(w, h, text)
        return self._slates[k]

    def _run_output(self) -> None:
        self._restart_output.clear()
        o = self.store.get("output")
        W, H, fps = o["width"], o["height"], int(o["fps"])
        frame_dt = 1.0 / fps
        out = av.open(self.mtx.rtsp_url("blurred"), "w", format="rtsp",
                      options={"rtsp_transport": "tcp", "max_interleave_delta": "300000"})
        try:
            vs = out.add_stream("h264_nvenc", rate=fps)
            vs.width, vs.height, vs.pix_fmt = W, H, "yuv420p"
            vs.bit_rate = o["bitrate_kbps"] * 1000
            vs.codec_context.time_base = VIDEO_TB
            vs.codec_context.gop_size = fps * 2          # 2 s keyframes (YouTube wants <= 4 s)
            vs.codec_context.max_b_frames = 0            # WebRTC previews can't do B-frames
            vs.codec_context.options = {"preset": "p4", "tune": "ll", "rc": "cbr", "bf": "0",
                                        "maxrate": f"{o['bitrate_kbps']}k", "bufsize": f"{o['bitrate_kbps'] * 2}k"}
            ast = out.add_stream("aac", rate=AUDIO_RATE)
            ast.layout = "stereo"
            ast.bit_rate = o["audio_kbps"] * 1000
            fifo = av.AudioFifo()
            self.stats["output"] = "live"

            out_t = -frame_dt       # timeline position of last emitted video frame (s)
            last_vpts = -1
            a_next = 0              # timeline position (samples) of next audio sample to enqueue
            a_pts = 0               # pts of next encoded audio frame
            offset: float | None = None
            cur_sid = -1
            anchor_wall, anchor_t = time.monotonic(), 0.0   # wall clock <-> timeline mapping
            fps_n, fps_t0 = 0, time.monotonic()

            def emit_audio(arr: np.ndarray) -> None:
                nonlocal a_next, a_pts
                if arr.shape[1] == 0:
                    return
                f = av.AudioFrame.from_ndarray(np.ascontiguousarray(arr, dtype=np.float32), format="fltp", layout="stereo")
                f.sample_rate = AUDIO_RATE
                fifo.write(f)
                a_next += arr.shape[1]
                while fifo.samples >= 1024:
                    af = fifo.read(1024)
                    af.pts, af.time_base = a_pts, Fraction(1, AUDIO_RATE)
                    a_pts += 1024
                    out.mux(ast.encode(af))

            def silence_until(t: float) -> None:
                target = int(t * AUDIO_RATE)
                while a_next < target:
                    emit_audio(np.zeros((2, min(target - a_next, AUDIO_RATE)), np.float32))

            def emit_video(img: np.ndarray, t: float) -> None:
                nonlocal out_t, last_vpts, fps_n, fps_t0
                pts = max(int(round(t * 90000)), last_vpts + 1)
                f = av.VideoFrame.from_ndarray(img, format="bgr24")
                f.pts, f.time_base = pts, VIDEO_TB
                out.mux(vs.encode(f))
                last_vpts, out_t = pts, pts / 90000
                self.latest_out, self.out_seq = img, self.out_seq + 1
                fps_n += 1
                now = time.monotonic()
                if now - fps_t0 >= 1.0:
                    self.stats["fps"] = round(fps_n / (now - fps_t0), 1)
                    fps_n, fps_t0 = 0, now

            last_release = time.monotonic() - LOST_AFTER - 1   # wall time a camera frame last went out

            while not self._stop.is_set() and not self._restart_output.is_set():
                p = self.store.get("privacy")
                delay = min(MAX_DELAY, max(0.0, float(p["delay_seconds"])))
                with self._dq_cv:
                    dq = self._delayq
                    wait = frame_dt if not dq else max(0.0, min(frame_dt, dq[0][0] - time.monotonic()))
                    if wait > 0:
                        self._dq_cv.wait(wait)
                    # any switch that is on now also applies to everything still in the buffer
                    on = switches(p)
                    if any(on):
                        for e in dq:
                            e[5] = [a or b for a, b in zip(e[5], on)]
                    now = time.monotonic()
                    due = []
                    while dq and dq[0][0] <= now:
                        due.append(dq.popleft())
                    self.stats["buffered_s"] = round(max(0.0, dq[-1][0] - dq[0][0]), 1) if dq else 0.0

                # release what is due (encoding happens outside the lock)
                for rel, kind, payload, t_in, sid, (paused, full_blur, mute) in due:
                    if now - rel > 1.0:          # delay was shortened: skip what is overdue
                        continue
                    if kind == "v":
                        if sid != cur_sid or offset is None:
                            offset, cur_sid = out_t + frame_dt - t_in, sid
                        t = t_in + offset
                        if t <= out_t or t > out_t + 1.0:     # timestamp jump: re-anchor
                            offset += (out_t + frame_dt) - t
                            t = out_t + frame_dt
                        if paused:
                            self.stats["mode"] = "paused"
                            frame = self._slate("paused", p["paused_text"], W, H)
                        elif full_blur:
                            self.stats["mode"] = "live"
                            frame = payload.copy()
                            blur_full(frame)
                        else:
                            self.stats["mode"] = "live"
                            frame = payload
                        emit_video(frame, t)
                        anchor_wall, anchor_t = now, out_t
                        last_release = now
                        self.ever_connected_out = True
                        if now - self.last_audio_in > 0.5 + delay:   # input without audio: keep an audio track
                            silence_until(max(0.0, t - 0.2))
                    else:
                        if offset is None or sid != cur_sid:
                            continue
                        arr = payload
                        start = int((t_in + offset) * AUDIO_RATE)
                        if start > a_next + AUDIO_TOLERANCE:
                            silence_until(start / AUDIO_RATE)
                        elif start < a_next - AUDIO_TOLERANCE:
                            arr = arr[:, min(arr.shape[1], a_next - start):]
                        if paused or mute:
                            arr = np.zeros_like(arr)
                        emit_audio(arr)

                # no camera frames going out (disconnected, or the delay buffer is still filling):
                # keep the output alive with a slate. Its timeline follows the wall clock from the
                # last camera frame, so a dropout doesn't add viewer latency.
                if now - last_release > LOST_AFTER:
                    if p["paused"]:
                        mode, text = "paused", p["paused_text"]
                    elif now - self.last_input < LOST_AFTER:
                        mode, text = "buffering", ("Back in a few seconds" if self.ever_connected_out else "Stream starting soon")
                    elif self.ever_connected:
                        mode, text = "lost", p["lost_text"]
                    else:
                        mode, text = "waiting", "Stream starting soon"
                    self.stats["mode"] = mode
                    due = anchor_t + (now - anchor_wall)
                    if due - out_t > 0.5:          # first slate after a dropout: jump forward
                        out_t = due - frame_dt
                    while out_t + frame_dt <= due:
                        t = out_t + frame_dt
                        emit_video(self._slate(mode, text, W, H), t)
                        silence_until(max(0.0, t - 0.2))
        finally:
            self.stats["output"] = "restarting"
            try:
                out.close()
            except Exception:
                pass

    def _mask(self, img: np.ndarray, t: float, p: dict, W: int, H: int) -> np.ndarray:
        frame = img.copy()
        regions: list[tuple[int, int, int, int]] = []
        want_det = p["faces"] or p["screens"] or p["plates"]
        if want_det:
            if self.detector is None:
                blur_full(frame)                   # fail closed while models load
                self.stats["regions"] = -1
                return frame
            try:
                t0 = time.perf_counter()
                dets = self.detector.detect(
                    img, face_conf=p["face_conf"], obj_conf=p["obj_conf"], plate_conf=p["plate_conf"],
                    classes=p["screens"], face_imgsz=p["face_imgsz"], obj_imgsz=p["obj_imgsz"],
                    plate_imgsz=p["plate_imgsz"], faces=p["faces"], plates=p["plates"])
                self.stats["det_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            except Exception:
                log.exception("detection failed; blurring full frame")
                self.stats["errors"] += 1
                blur_full(frame)
                self.stats["regions"] = -1
                return frame
            counts: dict[str, int] = {}
            for d in dets:
                counts[d.label] = counts.get(d.label, 0) + 1
            self.stats["detections"] = counts
            self.tracker.update(dets, t, p["hold_seconds"])
            regions = self.tracker.regions(t, W, H, p["pad"], p["grow_per_s"], p["min_size"])
        for z in p["zones"]:
            x1, y1, x2, y2 = (int(round(v)) for v in (z[0] * W, z[1] * H, z[2] * W, z[3] * H))
            regions.append((max(0, x1), max(0, y1), min(W, x2), min(H, y2)))
        for r in regions:
            blur_region(frame, *r, p["blur_mode"])
        self.stats["regions"] = len(regions)
        return frame
