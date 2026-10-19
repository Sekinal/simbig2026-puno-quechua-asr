"""Voice-over studio: a local teleprompter that records your narration scene by scene.

    uv run python recorder.py          # then open http://127.0.0.1:8765 in Chrome or Firefox

Takes are saved to out/voice/<scene>.wav (32-bit float, mono, browser processing off) with the
sentence taps in out/voice/<scene>.json. Then:

    uv run python narration.py --recordings   # clean and level your takes, timing from your taps
    bash assemble.sh                          # re-render the video around your voice
"""

from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from narration import OUT, scenes, sentences

HERE = Path(__file__).parent
VOICE = OUT / "voice"
PORT = 8765


def script() -> list[dict]:
    timing = json.loads((OUT / "timing.json").read_text())["duration"]
    rows = []
    for sid, title, text in scenes():
        sents = sentences(text)
        if not sents:
            continue
        rec = VOICE / f"{sid}.json"
        rows.append({
            "sid": sid, "title": title, "sentences": sents, "tts": timing.get(sid, 0),
            "take": json.loads(rec.read_text()) if rec.exists() else None,
        })  # fmt: skip
    return rows


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, (HERE / "recorder.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/script":
            self._send(200, json.dumps(script()).encode(), "application/json")
        elif m := re.fullmatch(r"/api/take/(S\d\d)\.wav", self.path):
            f = VOICE / f"{m.group(1)}.wav"
            if f.exists():
                self._send(200, f.read_bytes(), "audio/wav")
            else:
                self._send(404, b"", "text/plain")
        else:
            self._send(404, b"", "text/plain")

    def do_POST(self) -> None:
        m = re.fullmatch(r"/api/save/(S\d\d)", self.path)
        if not m:
            self._send(404, b"", "text/plain")
            return
        sid = m.group(1)
        body = self.rfile.read(int(self.headers["Content-Length"]))
        meta = json.loads(self.headers["X-Take"])
        VOICE.mkdir(parents=True, exist_ok=True)
        (VOICE / f"{sid}.wav").write_bytes(body)
        (VOICE / f"{sid}.json").write_text(json.dumps(meta, indent=1))
        print(f"saved {sid}: {meta['seconds']:.1f} s, {len(meta['taps'])} sentence taps")
        self._send(200, b"ok", "text/plain")

    def log_message(self, *_args) -> None:  # quiet
        pass


if __name__ == "__main__":
    print(f"Voice-over studio: open http://127.0.0.1:{PORT}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
