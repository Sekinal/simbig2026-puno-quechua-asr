"""Narration: parse script.md, make the text TTS-friendly, synthesize one WAV per scene (Piper).

Writes out/audio/<scene>.wav and out/timing.json: per scene its duration and the start time of
every narration sentence ("cues"), to which the animations are synced. With --recordings, scenes
recorded in recorder.py (your own voice) replace the TTS, and their cues come from your Space taps.

    uv run python narration.py                 # TTS for every scene
    uv run python narration.py --recordings    # your takes where recorded, TTS elsewhere
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "out"
LEAD = 1.2  # silence before the narration starts in each scene (s): visuals lead
TAIL = 2.6  # silence after it, so the last reveal can breathe (s)
SILENT = 9.0  # length of a scene without narration (the references card), s
REC_LEAD = 0.6  # silence before a recorded take (s)
REC_TAIL = 1.0  # silence after a recorded take (s)
PAUSE_CAP = 0.45  # longest pause kept inside a recorded take (s)

SAY = [  # written form -> spoken form (applied in order)
    (r"\bSIMBig\b", "SIM Big"),
    (r"\bLLM-ASR\b", "L L M A S R"),
    (r"\bomniASR\b", "Omni A S R"),
    (r"\bLoRA\b", "Laura"),
    (r"\bLLaMA\b", "Llama"),
    (r"\bCTC\b", "C T C"),
    (r"\bGSPO\b", "G S P O"),
    (r"\bMWER\b", "M W E R"),
    (r"\bDr GRPO\b", "Doctor G R P O"),
    (r"\bLLM\b", "L L M"),
    (r"\b7B\b", "seven B"),
    (r"\b1B\b", "one B"),
    (r"\b300M\b", "three hundred M"),
    (r"\benc\+dec\b", "encoder plus decoder"),
    (r"474,000", "four hundred seventy-four thousand"),
    (r"\bq\b", "Q"),
    (r"\bk\b", "K"),
]


def scenes() -> list[tuple[str, str, str]]:
    """(scene id, title, narration text) from script.md."""
    text = (HERE / "script.md").read_text()
    out = []
    for block in re.split(r"\n## ", text)[1:]:
        head, *body = block.split("\n")
        sid, title = re.match(r"(S\d+) — ([^(]+)", head).groups()
        lines = []
        for raw_line in body:
            line = raw_line.strip()
            if not line or line.startswith("**[") or line.startswith("---"):
                continue
            lines.append(re.sub(r"^- ", "", line).replace("**", ""))
        out.append((sid, title.strip(), " ".join(lines)))
    return out


def spoken(text: str) -> str:
    for pat, rep in SAY:
        text = re.sub(pat, rep, text)
    return text


def wav_seconds(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def sentences(text: str) -> list[str]:
    """Narration sentences: the units the animations are synced to (recorder and TTS alike)."""
    return [s for s in re.split(r"(?<=[.!?:])\s+", text) if s]


def ffmpeg(*args: str) -> str:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-y", *args], check=True, capture_output=True,
                       text=True)  # fmt: skip
    return r.stderr


def pad(src: Path, dst: Path, pre: str = "", tail: float = TAIL, lead: float = LEAD) -> None:
    """Optional filters, then lead-in and tail silence, 48 kHz stereo for the final mux."""
    chain = f"{pre}{',' if pre else ''}adelay={int(lead * 1000)}:all=1,apad=pad_dur={tail}"
    ffmpeg("-loglevel", "error", "-i", str(src), "-af", chain, "-ar", "48000", "-ac", "2", str(dst))


def take_speech(sid: str) -> float:
    """Seconds of a recorded take from just before the first tap to its end."""
    meta = json.loads((OUT / "voice" / f"{sid}.json").read_text())
    return meta["seconds"] - max(0.0, meta["taps"][0] - 0.3)


def tighten(x: np.ndarray, sr: int, cap: float = PAUSE_CAP) -> tuple[np.ndarray, list]:
    """Shorten pauses inside a take to at most ``cap`` seconds; the speech is not altered.

    Pauses are stretches 28 dB below the take's speech level (10 ms frames, 50 ms smoothing).
    Only the middle of a long pause is removed, with 10 ms fades on both sides of the cut, so
    breaths and word onsets stay intact. Returns the new signal and the cuts (t, removed s)."""
    hop = int(0.01 * sr)
    fr = x[: len(x) // hop * hop].reshape(-1, hop)
    e = np.convolve(20 * np.log10(np.sqrt((fr**2).mean(1)) + 1e-9), np.ones(5) / 5, "same")
    low = e < np.percentile(e, 80) - 28
    runs, s0 = [], None
    for i, flag in enumerate(low):
        if flag and s0 is None:
            s0 = i
        if (not flag or i == len(low) - 1) and s0 is not None:
            runs.append((s0, i))
            s0 = None
    runs = [r for r in runs if r[0] > 0 and r[1] < len(low) - 1]  # inner pauses only
    keep, cuts, pos, fade = [], [], 0, int(0.01 * sr)
    ramp = np.linspace(1, 0, fade, dtype=np.float32)
    for a, b in runs:
        dur = (b - a) * 0.01
        if dur <= cap:
            continue
        c0 = (a * hop) + int(cap / 2 * sr)  # keep cap/2 after the speech ...
        c1 = (b * hop) - int(cap / 2 * sr)  # ... and cap/2 before the next word
        seg = x[pos:c0].copy()
        seg[-fade:] *= ramp
        keep.append(seg)
        cuts.append((c0 / sr, (c1 - c0) / sr))
        pos = c1
        x = x.copy()
        x[pos : pos + fade] *= ramp[::-1]
    keep.append(x[pos:])
    return np.concatenate(keep), cuts


def from_recording(sid: str, n_sent: int) -> tuple[float, list[float]]:
    """Clean a recorded take (out/voice/<sid>.wav + taps in <sid>.json) into out/audio/<sid>.wav.

    Trim to just before the first tap and after the last word, shorten long pauses (``tighten``),
    high-pass at 75 Hz, gentle compression, two-pass loudness normalization to -16 LUFS (true
    peak -1.5 dB). The voice itself is not time-stretched or otherwise altered. Sentence cues are
    the Space taps, shifted by the trims and the lead-in."""
    meta = json.loads((OUT / "voice" / f"{sid}.json").read_text())
    taps = meta["taps"]
    if len(taps) != n_sent:
        print(f"  {sid}: {len(taps)} taps for {n_sent} sentences; missing cues are interpolated")
    sr = meta.get("sample_rate", 48000)
    x = np.frombuffer((OUT / "voice" / f"{sid}.wav").read_bytes()[44:], dtype="<f4")
    start = max(0.0, taps[0] - 0.3)
    x, cuts = tighten(x[int(start * sr) :], sr)
    trimmed = OUT / "audio" / f"{sid}_trim.wav"
    with wave.open(str(trimmed), "wb") as w:  # 16-bit PCM is plenty after this point
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    base = ("highpass=f=75,acompressor=threshold=-20dB:ratio=2.5:attack=10:release=160:makeup=1.5,"
            "areverse,silenceremove=start_periods=1:start_threshold=-48dB:start_silence=0.35,"
            "areverse")  # fmt: skip
    log = ffmpeg("-i", str(trimmed), "-af", f"{base},loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
                 "-f", "null", "-")  # fmt: skip
    m = json.loads(log[log.rindex("{") : log.rindex("}") + 1])
    norm = (f"loudnorm=I=-16:TP=-1.5:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
            f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:"
            f"offset={m['target_offset']}:linear=true")  # fmt: skip
    tmp = OUT / "audio" / f"{sid}_clean.wav"
    ffmpeg("-loglevel", "error", "-i", str(trimmed), "-af", f"{base},{norm}", "-ar", "48000",
           str(tmp))  # fmt: skip
    pad(tmp, OUT / "audio" / f"{sid}.wav", tail=REC_TAIL, lead=REC_LEAD)
    tmp.unlink()
    trimmed.unlink()

    def moved(t: float) -> float:  # tap time after the start trim and the pause cuts
        t -= start
        return t - sum(d for c, d in cuts if c < t)

    cues = [round(moved(t) + REC_LEAD, 2) for t in taps[:n_sent]]
    while len(cues) < n_sent:  # interpolate missing taps up to the end of speech
        end = wav_seconds(OUT / "audio" / f"{sid}.wav") - REC_TAIL
        cues.append(round(cues[-1] + (end - cues[-1]) / (n_sent - len(cues) + 1), 2))
    saved = sum(d for _, d in cuts)
    if saved > 0.05:
        print(f"  {sid}: {len(cuts)} long pauses shortened, {saved:.1f} s saved")
    return wav_seconds(OUT / "audio" / f"{sid}.wav"), cues


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default=str(HERE / "voices" / "en_US-ryan-high.onnx"))
    ap.add_argument("--speed", type=float, default=1.24, help="Piper length scale (>1 = slower)")
    ap.add_argument("--only", nargs="*", default=[], help="re-synthesize only these scenes")
    ap.add_argument(
        "--recordings",
        action="store_true",
        help="use out/voice/<scene>.wav takes from recorder.py where they exist",
    )
    args = ap.parse_args()
    (OUT / "audio").mkdir(parents=True, exist_ok=True)
    old = json.loads((OUT / "timing.json").read_text()) if (OUT / "timing.json").exists() else {}
    timing, cues = {}, {}
    for sid, _title, text in scenes():
        sents = sentences(text)
        dst = OUT / "audio" / f"{sid}.wav"
        if not sents:  # silent scene (references card)
            ffmpeg("-loglevel", "error", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t",
                   str(SILENT), str(dst))  # fmt: skip
            timing[sid], cues[sid] = SILENT, []
            continue
        if args.recordings and (OUT / "voice" / f"{sid}.wav").exists():
            dur, c = from_recording(sid, len(sents))
            timing[sid] = round(dur, 3)
            cues[sid] = [{"t": t, "text": s} for t, s in zip(c, sents, strict=True)]
            print(f"{sid}: {dur:6.1f} s  (your recording)")
            continue
        keep = (args.only and sid not in args.only) or (args.recordings and not args.only)
        if keep and sid in old.get("duration", {}) and dst.exists():
            timing[sid], cues[sid] = old["duration"][sid], old["cues"][sid]  # keep as is
            continue
        raw = OUT / "audio" / f"{sid}_raw.wav"
        subprocess.run(
            ["python", "-m", "piper", "-m", args.voice, "-f", str(raw),
             "--length-scale", str(args.speed), "--sentence-silence", "0.55"],
            input=spoken(text), text=True, check=True, capture_output=True,
        )  # fmt: skip
        pad(raw, dst)
        raw.unlink()
        dur = wav_seconds(dst)
        timing[sid] = round(dur, 3)
        # TTS has no taps: time each sentence by its share of characters
        speech, total, t = dur - LEAD - TAIL, sum(len(s) for s in sents), LEAD
        cues[sid] = []
        for s in sents:
            cues[sid].append({"t": round(t, 2), "text": s})
            t += speech * len(s) / total
        print(f"{sid}: {dur:6.1f} s  (TTS, {len(text.split())} words)")
    (OUT / "timing.json").write_text(json.dumps({"duration": timing, "cues": cues}, indent=1))
    print(f"total {sum(timing.values()) / 60:.2f} min")


if __name__ == "__main__":
    main()
