#!/usr/bin/env bash
# Simulate the glasses: push a test video (or any file you pass) to the ingest URL.
#   scripts/test-publish.sh [video-or-image] [duration-seconds]
set -euo pipefail
cd "$(dirname "$0")/.."
KEY=$(python3 -c "import json;print(json.load(open('data/config.json'))['server']['stream_key'])")
PORT=$(python3 -c "import json;print(json.load(open('data/config.json'))['server']['rtmp_port'])")
IP=$(python3 -c "import json;print(json.load(open('data/config.json'))['server']['bind_ip'])")
SRC=${1:-}
DUR=${2:-60}
URL="rtmp://${IP}:${PORT}/live/${KEY}"
if [[ -z "$SRC" ]]; then
  SRC=data/test.jpg
  [[ -f $SRC ]] || curl -sL -o "$SRC" https://ultralytics.com/images/zidane.jpg
fi
if [[ "$SRC" =~ \.(jpg|jpeg|png)$ ]]; then
  IN=(-loop 1 -framerate 30 -i "$SRC")
  VF="scale=2112:1188,crop=1920:1080:96+96*sin(t):54+40*cos(t*0.7)"   # pan around like a head-mounted camera
else
  IN=(-stream_loop -1 -i "$SRC")
  VF="scale=1920:1080"
fi
echo "publishing $SRC -> $URL for ${DUR}s"
exec ffmpeg -hide_banner -loglevel error -re "${IN[@]}" -f lavfi -i "sine=frequency=440:sample_rate=44100" \
  -map 0:v -map 1:a -vf "$VF" -c:v libx264 -preset veryfast -tune zerolatency -g 60 -pix_fmt yuv420p \
  -c:a aac -b:a 128k -t "$DUR" -f flv "$URL"
