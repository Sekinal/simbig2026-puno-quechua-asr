"""What does adapting the encoder change inside it? Per-layer CKA and phonetic-contrast probes.

States compared (omniASR-LLM-7B encoder, 128 layers): ``zs`` (pretrained; also the encoder of every
decoder-only LoRA), ``ed`` (enc+dec LoRA, exp34b) and ``ed_s1`` (its second seed, exp42: the
noise floor for CKA), by default; ``--card/--adapted`` select other encoders (300M, 1B). Hidden
states at every 1/16 of the encoder's depth are captured with forward hooks.

* **CKA** (linear, Kornblith et al. 2019) between ``zs`` and ``ed`` (and ``ed`` vs ``ed_s1``) per
  layer, on frames of in-domain speech (dev) and of the OOD radio drama: where does the
  representation move, and less so out of domain (Merchant et al. 2020)? Descriptive only.
* **Probes** (Martin et al. 2023; control: shuffled labels, Hewitt & Liang 2019): the reference is
  force-aligned to the CTC v3 posteriors (``torchaudio`` forced_align; 20 ms frames, as the
  encoder's); every stop p/t/k/q becomes a segment, the mean hidden state
  over the stop and its marker (``'`` ejective, ``h`` aspirated). Tasks: ejective vs plain,
  aspirated vs plain, q vs k (plain). Logistic regression on standardized features (balanced class
  weights), trained on the dev clips (read and spontaneous) and tested on the itest clips (other
  speakers), with the test classes balanced by subsampling; accuracy per layer for ``zs`` and
  ``ed`` on the same test items, compared by an exact McNemar test. No clip used here was seen in
  training.

Writes ``reports/probe_cka.csv`` and ``reports/probe_acc.csv``.

    uv run python scripts/encoder_probe.py
"""

from __future__ import annotations

import numpy as np
import polars as pl
import torch
import typer
from fairseq2.models.hub import load_model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torchaudio.functional import forced_align, merge_tokens

from qxp_asr.audio import load
from qxp_asr.decode import load_posteriors
from qxp_asr.lora import add_lora
from qxp_asr.nbest import make_batch
from qxp_asr.paths import ARTIFACTS, MANIFESTS
from qxp_asr.stats import mcnemar
from qxp_asr.text import normalize
from qxp_asr.vocab import INDEX, encode

ENC = r"encoder\.layers\.\d+\.(self_attn\.(q|k|v|output)_proj|ffn\.(gate|inner|output)_proj)"
STOPS = set("ptkq")


def segments(text: str, lp: np.ndarray) -> list[tuple[int, int, str, str]]:
    """(start frame, end frame, stop letter, class) for every stop of the aligned reference."""
    ids = encode(text)  # ``text`` holds alphabet characters only, so ids map 1:1 onto it
    em = torch.from_numpy(lp.astype(np.float32)).unsqueeze(0)
    path, scores = forced_align(em, torch.tensor([ids], dtype=torch.int32), blank=0)
    spans = merge_tokens(path[0], scores[0].exp(), blank=0)
    out = []
    for i, ch in enumerate(text):
        if ch not in STOPS:
            continue
        nxt = text[i + 1] if i + 1 < len(text) else " "
        kind = {"'": "ejective", "h": "aspirated"}.get(nxt, "plain")
        last = spans[i + 1] if kind != "plain" else spans[i]
        out.append((spans[i].start, last.end + 1, ch, kind))
    return out


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(0)
    y = y - y.mean(0)
    hsic = np.linalg.norm(y.T @ x) ** 2
    return float(hsic / (np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)))


def balanced(idx_a: np.ndarray, idx_b: np.ndarray, rng) -> np.ndarray:
    n = min(len(idx_a), len(idx_b))
    return np.concatenate(
        [rng.choice(idx_a, n, replace=False), rng.choice(idx_b, n, replace=False)]
    )


def main(
    frames_per_clip: int = 10,
    max_s: float = 20.0,
    seed: int = 0,
    limit: int = 0,
    card: str = "omniASR_LLM_7B_v2",
    rank: int = 16,
    adapted: str = "llm7b_lora_encdec",
    adapted_seed: str = typer.Option("llm7b_lora_encdec_s1", help="second seed ('' = none)"),
    tag: str = typer.Option("", help="suffix of the output CSVs"),
) -> None:
    runs = {"ed": adapted} | ({"ed_s1": adapted_seed} if adapted_seed else {})
    rng = np.random.default_rng(seed)
    man = pl.read_parquet(MANIFESTS / "all.parquet").filter(
        pl.col("text").is_not_null() & (pl.col("duration_s") <= max_s)
    )
    sets = {
        "dev": man.filter(pl.col("role").is_in(["dev_scripted", "dev_spont"])),
        "itest": man.filter(pl.col("role").is_in(["itest_scripted", "itest_spont"])),
        "ood": man.filter(pl.col("role") == "ood"),
    }
    if limit:  # smoke run
        sets = {k: v.head(limit) for k, v in sets.items()}
    model = load_model(card, device=torch.device("cuda"), dtype=torch.bfloat16)
    add_lora(model, ENC, rank=rank, alpha=2 * rank, dropout=0.0)
    n_layers = len(model.encoder.layers)
    layers = [*range(0, n_layers, max(1, n_layers // 16)), n_layers - 1]
    layers = sorted(set(layers))
    model.eval()
    pad = model.target_vocab_info.pad_idx
    captured: dict[int, torch.Tensor] = {}

    def hook(i):
        def f(_m, _inp, out):
            captured[i] = (out[0] if isinstance(out, tuple) else out).detach()

        return f

    for i in layers:
        model.encoder.layers[i].register_forward_hook(hook(i))

    def set_state(state: str) -> None:
        loaded = {}
        if state != "zs":
            loaded = torch.load(ARTIFACTS / "runs" / runs[state] / "best_trainable.pt",
                                map_location="cuda", weights_only=True)  # fmt: skip
        enc = {n for n in loaded if n.startswith("encoder.")}
        names = {n for n, _ in model.named_parameters()}
        if enc - names:
            raise KeyError(f"{len(enc - names)} encoder LoRA tensors not in the model")
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n.startswith("encoder.") and "lora_" in n:
                    if n in loaded:
                        p.copy_(loaded[n])
                    elif "lora_b" in n:
                        p.zero_()

    # alignment once (CTC v3 posteriors), then hidden states per state
    seg_meta, frame_pick = {}, {}
    for name, d in sets.items():
        ids = d["clip_id"].to_list()
        posts = load_posteriors(ARTIFACTS / "posteriors" / "ctc1b_v3_soup_wise0.85", ids)
        for cid, text, lp in zip(ids, d["text"].to_list(), posts, strict=True):
            t = " ".join("".join(c for c in normalize(text) if c in INDEX).split())
            try:
                seg_meta[cid] = segments(t, lp) if name != "ood" else []
            except RuntimeError:
                seg_meta[cid] = []
            frame_pick[cid] = rng.choice(len(lp), min(frames_per_clip, len(lp)), replace=False)
    feats: dict = {}  # (state, layer) -> list of segment means
    frames: dict = {}  # (state, set, layer) -> list of frame rows
    for state in ("zs", *runs):
        set_state(state)
        for name, d in sets.items():
            for cid, wav in d.select("clip_id", "wav").iter_rows():
                batch = make_batch([torch.from_numpy(load(wav))], [[0]], pad, "cuda")
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model.embed_audio(batch.source_seqs, batch.source_seq_lens)
                for i in layers:
                    h = captured[i][0].float().cpu().numpy()
                    pick = frame_pick[cid][frame_pick[cid] < len(h)]
                    if name != "itest":
                        frames.setdefault((state, name, i), []).append(h[pick])
                    if state != "ed_s1":
                        for s, e, _ch, _k in seg_meta[cid]:  # one row per segment, kept or not
                            lo = min(s, len(h) - 1)
                            end = max(min(e, len(h)), lo + 1)
                            feats.setdefault((state, i), []).append(h[lo:end].mean(0))
            print(f"[{state}] {name} done", flush=True)
    # CKA per layer
    cka = []
    for name in ("dev", "ood"):
        for i in layers:
            z = np.concatenate(frames[("zs", name, i)])
            e = np.concatenate(frames[("ed", name, i)])
            row = {"set": name, "layer": i, "cka_zs_ed": linear_cka(z, e)}
            if "ed_s1" in runs:
                row["cka_ed_seed"] = linear_cka(e, np.concatenate(frames[("ed_s1", name, i)]))
            cka.append(row)
    pl.DataFrame(cka).write_csv(f"reports/probe_cka{tag}.csv")
    # probes
    meta = [
        (sname, c, k)
        for sname in ("dev", "itest")
        for cid in sets[sname]["clip_id"].to_list()
        for _s, _e, c, k in seg_meta[cid]
    ]
    # segments were appended in the same order (dev then itest, clips, segments) for each state
    split = np.array([m[0] for m in meta])
    letter = np.array([m[1] for m in meta])
    kind = np.array([m[2] for m in meta])
    tasks = {
        "ejective vs plain": (kind == "ejective", kind == "plain"),
        "aspirated vs plain": (kind == "aspirated", kind == "plain"),
        "q vs k (plain)": (
            (letter == "q") & (kind == "plain"),
            (letter == "k") & (kind == "plain"),
        ),
    }
    rows = []
    for task, (pos, neg) in tasks.items():
        tr = np.where((pos | neg) & (split == "dev"))[0]
        te = balanced(np.where(pos & (split == "itest"))[0], np.where(neg & (split == "itest"))[0],
                      rng)  # fmt: skip
        y_tr, y_te = pos[tr].astype(int), pos[te].astype(int)
        if min(y_tr.sum(), (1 - y_tr).sum(), y_te.sum()) < 5:
            print(f"{task}: too few examples, skipped")
            continue
        y_ctrl = rng.permutation(y_tr)
        for i in layers:
            right, row = {}, {"task": task, "layer": i, "n_train": len(tr), "n_test": len(te)}
            for state in ("zs", "ed"):
                x = np.stack(feats[(state, i)])
                sc = StandardScaler().fit(x[tr])
                x_tr, x_te = sc.transform(x[tr]), sc.transform(x[te])
                clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000)
                right[state] = clf.fit(x_tr, y_tr).predict(x_te) == y_te
                ctrl = LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000)
                row[f"acc_{state}"] = float(right[state].mean())
                row[f"control_{state}"] = ctrl.fit(x_tr, y_ctrl).score(x_te, y_te)
            row["mcnemar_p"] = mcnemar(right["ed"], right["zs"])["p"]
            rows.append(row)
    df = pl.DataFrame(rows)
    df.write_csv(f"reports/probe_acc{tag}.csv")
    with pl.Config(tbl_rows=120, tbl_width_chars=200, float_precision=3):
        print(pl.DataFrame(cka))
        print(df)


if __name__ == "__main__":
    typer.run(main)
