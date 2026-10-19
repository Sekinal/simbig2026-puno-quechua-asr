#!/usr/bin/env bash
# Add a quiet, ducked music bed under the narration of the assembled talk video.
#   bash mix_music.sh   ->  out/simbig2026_qxp_talk_music.mp4  (video stream copied, audio remixed)
# Music: "Clean Soul" by Kevin MacLeod (incompetech.com), CC BY 4.0 (credited on the references card).
set -euo pipefail
cd "$(dirname "$0")"
IN=${IN:-out/simbig2026_qxp_talk.mp4}; OUT=${OUT:-out/simbig2026_qxp_talk_music.mp4}; M=music/clean_soul.mp3
MUSIC_DB=${MUSIC_DB:--18}   # music level before ducking, relative to its own loudness-normalized level
D=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$IN")
FADE_OUT_AT=$(python3 -c "print(max(0, $D - 7))")
# 1) music bed: loop with 4 s crossfades, normalize, set level, fade in/out, cut to the video length
ffmpeg -loglevel error -y -i "$M" -i "$M" -i "$M" -i "$M" -filter_complex "
  [0][1]acrossfade=d=4:c1=tri:c2=tri[a];[a][2]acrossfade=d=4:c1=tri:c2=tri[b];
  [b][3]acrossfade=d=4:c1=tri:c2=tri,loudnorm=I=-16:TP=-2:LRA=11,volume=${MUSIC_DB}dB,
  afade=t=in:st=0:d=3,afade=t=out:st=${FADE_OUT_AT}:d=7,atrim=0:${D},asetpts=N/SR/TB[m]" \
  -map "[m]" -ar 48000 -ac 2 /tmp/claude-1000/music_bed.wav
# 2) duck the music under the voice (sidechain), mix, final loudness to -16 LUFS / -1.5 dBTP
ffmpeg -loglevel error -y -i "$IN" -i /tmp/claude-1000/music_bed.wav -filter_complex "
  [0:a]asplit=2[v][sc];
  [1:a][sc]sidechaincompress=threshold=0.015:ratio=5:attack=40:release=600:makeup=1[md];
  [v][md]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.89[mix]" \
  -map 0:v -map "[mix]" -c:v copy -c:a pcm_s16le /tmp/claude-1000/mix.mkv
ffmpeg -hide_banner -i /tmp/claude-1000/mix.mkv -af loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json \
  -f null - 2> /tmp/claude-1000/ln.txt
read -r MI MTP MLRA MTH MOFF < <(python3 -c "
import json;s=open('/tmp/claude-1000/ln.txt').read();m=json.loads(s[s.rindex('{'):s.rindex('}')+1])
print(m['input_i'],m['input_tp'],m['input_lra'],m['input_thresh'],m['target_offset'])")
ffmpeg -loglevel error -y -i /tmp/claude-1000/mix.mkv -map 0:v -map 0:a -c:v copy \
  -af "loudnorm=I=-16:TP=-1.5:LRA=11:measured_I=$MI:measured_TP=$MTP:measured_LRA=$MLRA:measured_thresh=$MTH:offset=$MOFF:linear=true" \
  -c:a aac -b:a 192k -ar 48000 -movflags +faststart "$OUT"
rm -f /tmp/claude-1000/music_bed.wav /tmp/claude-1000/mix.mkv /tmp/claude-1000/ln.txt
echo "done: $OUT ($(du -h "$OUT" | cut -f1))"
