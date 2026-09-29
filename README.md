# live-masker

Privacy-masking relay for live streams from smart glasses (e.g. Rokid). The glasses push
RTMP to this server; an RTX-class GPU finds **faces**, **license plates** and **phone / laptop
screens**, blurs them, and the masked stream is forwarded to any number of RTMP destinations (YouTube,
Twitch, Bilibili, …). The original is recorded to disk. Everything is controlled from a
mobile-friendly web UI with live previews and a one-tap **Pause**.

```
Rokid glasses ──RTMP──► [HAProxy on cloud] ──VPN──► mediamtx (:1935, secret path live/<key>)
                                                        │   └── records original → data/recordings/live/<key>/*.mp4
                                                        ▼ rtsp (localhost)
                                              processor (GPU): decode → YOLO faces + plates + COCO screens
                                              → tracker → blur → delay buffer → NVENC  (audio re-encoded, mute-able)
                                                        ▼ rtsp (localhost)
                                                   mediamtx path "blurred"
                                                        ▼
                               ffmpeg forwarders (-c copy) ──► YouTube / Twitch / …
Web UI (:8056, basic auth) ── controls, WebRTC/MJPEG previews of raw + masked, recordings
```

## Privacy behaviour

The processor is built to fail closed:

| Situation | What viewers get |
|---|---|
| Face / license plate / phone / laptop detected | Blurred immediately (no confirmation frames), box padded 25 % |
| Object missed for a few frames | Still blurred for `hold_seconds` (1 s), following its last motion, box growing while lost |
| Always-blur zones | Rectangles you draw in the UI, blurred on every frame |
| Models still loading / detector error | Whole frame blurred |
| Processing falls behind | Old frames are **dropped**, never passed through unprocessed |
| **Broadcast delay** (default 10 s, 0-60 s adjustable) | Viewers see the masked stream this much later, which gives you time to react |
| **Pause** | "Paused" slate + silence; masked destinations stay connected; raw destinations are stopped |
| **Full blur** | Whole frame blurred, audio kept |
| **Mute** | Silence instead of the microphone |
| **Blur + Mute** | Whole frame blurred, silence, and a caption you type in the UI (any language, multi-line) |
| Glasses lose signal | "Reconnecting…" slate + silence; destinations stay connected, stream resumes automatically |

Pause, Full blur, Mute and Blur + Mute apply to every frame that was captured or still
buffered while they were on. Pressing Pause therefore also removes the last *delay* seconds that viewers
have not seen yet, and Resume never releases anything captured or buffered while paused.

Slate and caption texts are rendered with a CJK-capable system font (Noto Sans CJK,
WenQuanYi Zen Hei, ... or `LM_FONT=/path/to/font`), so Chinese and Japanese work.

These switches only affect what goes **upstream**. The original recording keeps
running from the first packet of every connection, regardless of these switches.

Raw destinations send the **unmasked**, **undelayed** camera feed; the UI marks them in red and asks for
confirmation before enabling them.

## Setup

Requirements: Linux, NVIDIA GPU + driver (NVENC), `ffmpeg`, [`uv`](https://docs.astral.sh/uv/).

```bash
scripts/setup.sh          # downloads mediamtx, YOLO weights, creates the venv
uv run live-masker        # prints the web UI password and the glasses URL
```

On first run `data/config.json` is created with a random stream key, a random UI password
(user `admin`) and `bind_ip` = the IPv4 address of the default route.

Run as a service: see [deploy/live-masker.service](deploy/live-masker.service).

### Network exposure

Every externally reachable listener binds **only** to `server.bind_ip` (IPv4). Wildcards are
refused on purpose: mediamtx is written in Go, where `:1935` and `0.0.0.0:1935` both
become dual-stack sockets and would be reachable on a public IPv6 address.

| Port | Bound to | Purpose |
|---|---|---|
| 1935/tcp | bind_ip | RTMP ingest (publish only, secret path) |
| 8056/tcp | bind_ip | Web UI + API (basic auth) |
| 8189/udp+tcp | bind_ip | WebRTC preview media |
| 8554, 8889, 9997 | 127.0.0.1 | internal RTSP bus, WebRTC signalling, mediamtx API |

Only the local services may read streams from mediamtx. Browsers get previews through the
authenticated `/api/whep/*` proxy, so nobody can watch the raw feed without the UI password.

### Exposing it through a cloud server (HAProxy)

If this host has no public IPv4, run HAProxy on a cloud VM that reaches this host over your
VPN. [deploy/haproxy.cfg](deploy/haproxy.cfg) forwards:

- `:1935` → RTMP ingest (TCP passthrough; optional RTMPS termination is included, commented out)
- `:443` → web UI with TLS (required, because the UI uses basic auth)
- `:8189` → WebRTC over ICE-TCP (if it doesn't connect, previews fall back to MJPEG over HTTPS)

Then set **Public host** in the UI (Glasses ingest card) to the cloud server's name. The UI
then shows `rtmp://<cloud>:1935/live/<key>` for the glasses, and WebRTC advertises that host.

Don't run the proxy on the live-masker host itself. mediamtx trusts connections from
127.0.0.1, so a local proxy would make every client look local.

## Using it

1. Paste the **Full RTMP URL** from the UI into the glasses' live-stream setting
   (or Server + Stream key if they ask for them separately).
2. Add destinations (preset for YouTube etc. + your stream key). They start **disabled**;
   switch them on when you're ready.
3. Watch both previews. Tune masking if needed:
   - lower confidence → blurs more (more false positives, fewer leaks)
   - larger *hold* / *grow* → steadier blur when objects are briefly lost
   - draw *always-blur zones*, e.g. where your handlebar phone mount appears
4. During the ride, use **Pause**, **Full blur**, **Mute** or **Blur + Mute** from your phone.

Recordings are fragmented MP4 (crash-safe), segmented every 30 min and never deleted
automatically. They go to `data/recordings/` unless you set another absolute folder in the
Recording card (`recording.path`). The change applies immediately, even mid-stream, without
dropping the glasses' connection. The UI warns when less than 20 GB is free (1080p ≈ 2.7 GB/h
per recorded stream). Download or delete recordings from the UI.

To test without the glasses: `scripts/test-publish.sh [video-or-image] [seconds]`.

## Settings

All settings are editable in the UI and saved to `data/config.json`
(`server`, `output`, `privacy`, `recording`, `models`, `destinations`).
Changing the stream key, ports or `bind_ip` restarts mediamtx.

Models (`models/`), downloaded by `scripts/setup.sh`:

| Model | Detects | Source |
|---|---|---|
| `yolov11s-face.pt` | faces | [akanametov/yolo-face](https://github.com/akanametov/yolo-face) |
| `license-plate-finetune-v1m.pt` | license plates | [morsetechlab/yolov11-license-plate-detection](https://huggingface.co/morsetechlab/yolov11-license-plate-detection) (AGPL-3.0) |
| `yolo11s.pt` (COCO) | `cell phone` / `laptop` / `tv` | [Ultralytics](https://github.com/ultralytics/ultralytics) |

The three models run in parallel threads, about 23 ms per 1080p frame on an RTX 3090.
The plate model was checked on Japanese street and parking-lot photos (white, yellow,
green, diplomatic and kei-car plates, including small distant ones). It also fires on some
shop signs; that only blurs a little extra.

## Limitations

- Phone detection uses the generic COCO class, so a phone at an unusual angle or mostly
  covered by a hand can be missed. Tune the confidence, keep a generous hold time, and use
  zones for fixed mounts.
- Name plates on houses ("hyosatsu"), mailboxes and other text with personal names are not
  detected. Pause (with the broadcast delay) is the tool for those.
- Very small, distant faces (a few pixels) may not be detected; raising *face model input*
  to 1600 helps at the cost of GPU time.
- WebRTC previews are video-only (AAC can't be carried over WebRTC); the forwarded streams have audio.
- The masked stream is re-encoded at the configured resolution / bitrate (default 1080p30, 6 Mbit/s CBR, 2 s GOP).
