#!/usr/bin/env bash
# Render every scene at 1080p60, mux each with its narration, and concatenate into the final video.
#   bash assemble.sh            # all scenes (renders up to JOBS in parallel)
#   bash assemble.sh S03 S08    # only these scenes (re-render), then reassemble everything
# Output: out/simbig2026_qxp_talk.mp4 (+ out/script.srt for subtitles).
set -euo pipefail
cd "$(dirname "$0")"
JOBS=${JOBS:-3}
SCENES=("$@")
[ ${#SCENES[@]} -eq 0 ] && SCENES=($(python3 -c "import json;print(' '.join(json.load(open('out/timing.json'))['duration']))"))
mkdir -p out/scenes

render() {  # $1 = scene id
  local sid=$1 low; low=$(echo "$sid" | tr 'A-Z' 'a-z')
  local md="media_final_$sid"
  uv run manim -qh --fps 60 --disable_caching --media_dir "$md" "scene_${low}.py" "$sid" > "out/scenes/$sid.log" 2>&1 \
    || { echo "RENDER FAILED $sid (see out/scenes/$sid.log)"; return 1; }
  local mp4; mp4=$(find "$md" -name "$sid.mp4" -path "*1080p60*" | head -1)
  # mux with this scene's narration: audio padded/trimmed to the video length
  ffmpeg -loglevel error -y -i "$mp4" -i "out/audio/$sid.wav" -map 0:v -map 1:a \
    -c:v copy -c:a aac -b:a 192k -ar 48000 -af apad -shortest "out/scenes/$sid.mp4"
  rm -rf "$md"
  local v a; v=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "out/scenes/$sid.mp4")
  a=$(python3 -c "import json;print(json.load(open('out/timing.json'))['duration']['$sid'])")
  echo "$sid rendered: video ${v}s, narration ${a}s"
}
export -f render

printf '%s\n' "${SCENES[@]}" | xargs -P "$JOBS" -I{} bash -c 'render {}'

ALL=($(python3 -c "import json;print(' '.join(json.load(open('out/timing.json'))['duration']))"))
: > out/concat.txt
for sid in "${ALL[@]}"; do
  [ -f "out/scenes/$sid.mp4" ] || { echo "missing out/scenes/$sid.mp4"; exit 1; }
  echo "file 'scenes/$sid.mp4'" >> out/concat.txt
done
ffmpeg -loglevel error -y -f concat -safe 0 -i out/concat.txt -c copy -movflags +faststart \
  out/simbig2026_qxp_talk.mp4
d=$(ffprobe -v error -show_entries format=duration -of csv=p=0 out/simbig2026_qxp_talk.mp4)
echo "final: out/simbig2026_qxp_talk.mp4 ($(python3 -c "print(f'{$d/60:.2f} min')"), $(du -h out/simbig2026_qxp_talk.mp4 | cut -f1))"
