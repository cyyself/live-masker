"""Unicode (CJK-capable) text rendering for slates and on-video captions.

OpenCV's built-in fonts are ASCII-only, so text is rendered with Pillow using a CJK font
found on the system, then alpha-blended onto frames. Rendered layers are cached per
(text, frame size), so the per-frame cost is only the blend of the caption region.
"""

from __future__ import annotations

import functools
import logging
import os
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger(__name__)

FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
]


@functools.cache
def find_font() -> str | None:
    """A font that covers Chinese and Japanese (kana + kanji), or None."""
    path = os.environ.get("LM_FONT")
    if path and os.path.exists(path):
        return path
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    try:
        p = subprocess.run(["fc-match", "-f", "%{file}", "sans-serif:lang=ja"],
                           capture_output=True, text=True, timeout=5).stdout.strip()
        if p and os.path.exists(p):
            return p
    except (OSError, subprocess.SubprocessError):
        pass
    log.warning("no CJK font found; non-ASCII caption text will not render")
    return None


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = find_font()
    return ImageFont.truetype(path, size) if path else ImageFont.load_default(size)


@functools.lru_cache(maxsize=16)
def caption_layer(text: str, w: int, h: int, box: bool) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    """Render `text` centred for a w x h frame.

    Returns (premultiplied_bgr, inverse_alpha_3ch, x, y) for the caption's bounding region,
    or None for empty text. Lines are split on newlines; the font shrinks until the widest line
    fits in 90 % of the frame width.
    """
    text = text.strip()
    if not text:
        return None
    lines = text.splitlines()[:6]
    size = max(12, h // 12)
    while True:
        font = _font(size)
        widths = [font.getlength(line) for line in lines]
        if max(widths) <= w * 0.9 or size <= 12:
            break
        size = int(size * 0.9)
    asc, desc = font.getmetrics() if hasattr(font, "getmetrics") else (size, size // 4)
    line_h = int((asc + desc) * 1.15)
    pad = size // 2
    bw = int(max(widths)) + 2 * pad
    bh = line_h * len(lines) + 2 * pad - (line_h - asc - desc)
    img = Image.new("RGBA", (bw, bh), (0, 0, 0, 150 if box else 0))
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        x = (bw - font.getlength(line)) / 2
        y = pad + i * line_h
        if not box:   # soft shadow keeps text readable on any background
            d.text((x + 2, y + 2), line, font=font, fill=(0, 0, 0, 200))
        d.text((x, y), line, font=font, fill=(240, 240, 240, 255))
    rgba = np.asarray(img).astype(np.float32)
    bw, bh = min(bw, w), min(bh, h)
    rgba = rgba[:bh, :bw]
    a = rgba[:, :, 3:4] / 255.0
    # precomputed for a fast per-frame blend: out = roi * inv / 255 + premultiplied colour
    premul = np.ascontiguousarray((rgba[:, :, 2::-1] * a).round().astype(np.uint8))
    inv = np.ascontiguousarray(np.repeat(255 - rgba[:, :, 3:4], 3, axis=2).astype(np.uint8))
    x0, y0 = (w - bw) // 2, (h - bh) // 2
    return premul, inv, x0, y0


def draw_caption(frame: np.ndarray, text: str, box: bool = True) -> None:
    """Alpha-blend a centred caption onto `frame` (BGR, in place)."""
    h, w = frame.shape[:2]
    layer = caption_layer(text, w, h, box)
    if layer is None:
        return
    premul, inv, x, y = layer
    roi = frame[y:y + premul.shape[0], x:x + premul.shape[1]]
    roi[:] = cv2.add(cv2.multiply(roi, inv, scale=1 / 255), premul)
