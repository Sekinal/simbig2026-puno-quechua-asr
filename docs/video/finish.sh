#!/usr/bin/env bash
# Final master: music bed + subtitle band.
#   bash finish.sh  ->  out/simbig2026_qxp_talk_final.mp4
# The animation is scaled to 87.5% (1680x945) at the top of the 1920x1080 frame; the subtitles
# sit in the 135 px band below it, on the same background colour, so they never cover content.
set -euo pipefail
cd "$(dirname "$0")"
bash mix_music.sh
uv run python captions.py
ffmpeg -loglevel error -stats -y -i out/simbig2026_qxp_talk_music.mp4 \
  -vf "scale=1680:945:flags=lanczos,pad=1920:1080:120:0:color=0xf7f6f2,ass=out/captions.ass" \
  -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p -c:a copy -movflags +faststart \
  out/simbig2026_qxp_talk_final.mp4
echo "done: out/simbig2026_qxp_talk_final.mp4 ($(du -h out/simbig2026_qxp_talk_final.mp4 | cut -f1))"
