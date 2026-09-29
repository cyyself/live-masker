"""Face / screen / license-plate detection, temporal tracking and blurring.

Privacy first: every design choice here errs on the side of blurring too much.
  * Detections blur immediately (no "confirmation" frames).
  * A lost object keeps being blurred for `hold_seconds`, following its last
    known velocity, with a box that grows the longer it is unseen.
  * Boxes are padded and have a minimum size.
  * Any detector failure is reported to the caller, which blurs the full frame.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

# COCO class ids used by the generic object model
COCO_SCREEN_CLASSES = {"cell phone": 67, "laptop": 63, "tv": 62}


@dataclass
class Det:
    box: np.ndarray  # x1, y1, x2, y2 (float, pixels)
    label: str
    conf: float


class Detector:
    """Runs a face model, a COCO model (phones / screens) and a license-plate model on the GPU.

    The models run in parallel threads (one thread per model instance, so each model is
    only ever used by one thread): this overlaps their CPU pre/post-processing and keeps
    all three within a 30 fps frame budget.
    """

    def __init__(self, models_dir: Path, face_model: str, object_model: str, plate_model: str,
                 device: str = "cuda:0"):
        from ultralytics import YOLO  # imported lazily: heavy

        self.device = device
        self.quantize = 16 if device.startswith("cuda") else None
        self.face = YOLO(str(models_dir / face_model))
        self.obj = YOLO(str(models_dir / object_model))
        self.plate = YOLO(str(models_dir / plate_model))
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(3, thread_name_prefix="detect")
        warm = np.zeros((720, 1280, 3), np.uint8)
        for _ in range(2):
            self.detect(warm, face_conf=0.3, obj_conf=0.3, plate_conf=0.3, classes=["cell phone"],
                        face_imgsz=640, obj_imgsz=640, plate_imgsz=640)
        log.info("detector ready on %s", device)

    def _run(self, model, frame, conf, imgsz, classes=None) -> list[tuple[np.ndarray, int, float]]:
        res = model.predict(
            frame, conf=conf, imgsz=imgsz, classes=classes, device=self.device,
            quantize=self.quantize, verbose=False, max_det=100,
        )[0]
        b = res.boxes
        if b is None or len(b) == 0:
            return []
        xyxy = b.xyxy.float().cpu().numpy()
        cls = b.cls.int().cpu().numpy()
        cf = b.conf.float().cpu().numpy()
        return [(xyxy[i], int(cls[i]), float(cf[i])) for i in range(len(xyxy))]

    def detect(self, frame: np.ndarray, *, face_conf: float, obj_conf: float, plate_conf: float,
               classes: list[str], face_imgsz: int, obj_imgsz: int, plate_imgsz: int,
               faces: bool = True, plates: bool = True) -> list[Det]:
        names = {v: k for k, v in COCO_SCREEN_CLASSES.items()}
        ids = [COCO_SCREEN_CLASSES[c] for c in classes if c in COCO_SCREEN_CLASSES]
        with self._lock:
            futs = []
            if faces:
                futs.append(("face", self._pool.submit(self._run, self.face, frame, face_conf, face_imgsz)))
            if ids:
                futs.append(("obj", self._pool.submit(self._run, self.obj, frame, obj_conf, obj_imgsz, ids)))
            if plates:
                futs.append(("plate", self._pool.submit(self._run, self.plate, frame, plate_conf, plate_imgsz)))
            out: list[Det] = []
            for kind, fut in futs:          # .result() re-raises any model error -> caller fails closed
                for box, c, cf in fut.result():
                    label = names.get(c, str(c)) if kind == "obj" else kind
                    out.append(Det(box, label, cf))
        return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class Track:
    box: np.ndarray
    label: str
    last_seen: float
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))  # px / s of the centre

    def predicted(self, t: float) -> np.ndarray:
        dt = min(t - self.last_seen, 0.5)
        shift = np.tile(self.vel * dt, 2)
        return self.box + shift


class Tracker:
    """Very small IoU / centre-distance tracker whose only job is to keep blur stable."""

    def __init__(self):
        self.tracks: list[Track] = []

    def reset(self):
        self.tracks = []

    def update(self, dets: list[Det], t: float, hold: float) -> None:
        preds = [tr.predicted(t) for tr in self.tracks]
        used: set[int] = set()
        for d in sorted(dets, key=lambda d: -d.conf):
            best, best_score = -1, 0.0
            dc = (d.box[:2] + d.box[2:]) / 2
            dsz = max(d.box[2] - d.box[0], d.box[3] - d.box[1])
            for i, (tr, p) in enumerate(zip(self.tracks, preds)):
                if i in used or tr.label != d.label:
                    continue
                score = _iou(d.box, p)
                if score <= 0:
                    pc = (p[:2] + p[2:]) / 2
                    dist = float(np.linalg.norm(dc - pc))
                    if dist < 1.5 * dsz:
                        score = 0.01 * (1 - dist / (1.5 * dsz))
                if score > best_score:
                    best, best_score = i, score
            if best >= 0:
                tr = self.tracks[best]
                dt = t - tr.last_seen
                if dt > 1e-3:
                    oc = (tr.box[:2] + tr.box[2:]) / 2
                    v = (dc - oc) / dt
                    tr.vel = 0.6 * tr.vel + 0.4 * v
                tr.box = d.box.copy()
                tr.last_seen = t
                used.add(best)
            else:
                self.tracks.append(Track(d.box.copy(), d.label, t))
                used.add(len(self.tracks) - 1)
        self.tracks = [tr for tr in self.tracks if t - tr.last_seen <= hold]

    def regions(self, t: float, w: int, h: int, pad: float, grow_per_s: float, min_size: int) -> list[tuple[int, int, int, int]]:
        out = []
        for tr in self.tracks:
            age = t - tr.last_seen
            box = tr.box if age <= 0 else tr.predicted(t)
            bw, bh = box[2] - box[0], box[3] - box[1]
            p = pad + grow_per_s * max(0.0, age)
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            hw = max(bw * (1 + 2 * p), min_size) / 2
            hh = max(bh * (1 + 2 * p), min_size) / 2
            x1, y1 = int(max(0, cx - hw)), int(max(0, cy - hh))
            x2, y2 = int(min(w, cx + hw)), int(min(h, cy + hh))
            if x2 > x1 and y2 > y1:
                out.append((x1, y1, x2, y2))
        return out


def blur_region(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, mode: str) -> None:
    roi = frame[y1:y2, x1:x2]
    h, w = roi.shape[:2]
    if h < 2 or w < 2:
        return
    if mode == "solid":
        roi[:] = 0
        return
    # Downscale hard, then scale back: irreversible, and cheap even for big regions.
    f = 12 if mode == "pixelate" else 20
    sw, sh = max(1, w // f), max(1, h // f)
    small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA)
    if mode == "pixelate":
        roi[:] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        small = cv2.GaussianBlur(small, (0, 0), 1.2)
        roi[:] = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def blur_full(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (max(1, w // 48), max(1, h // 48)), interpolation=cv2.INTER_AREA)
    frame[:] = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
