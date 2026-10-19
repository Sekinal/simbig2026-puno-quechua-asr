"""Robust verification: score candidates by likelihood averaged over perturbed views of the audio.

exp25 showed the hybrid uses <5 % of the headroom its candidate pool offers (dev unseen: oracle 4.91
vs 9.73): a transcript found under perturbation usually has a low likelihood on the CLEAN audio, so
a clean-audio verifier is biased towards the model's own first guess. Hypothesis: the correct
transcript stays likely across perturbations, while a confident mistake is tied to the exact clean
signal. For a sample of clips per set we score every pooled candidate under each adapter on
``--views`` perturbed copies (``qxp_asr.augment.VIEWS``) and compare, per set,

* pairwise ranking accuracy and selection WER of: clean log-prob, view-averaged log-prob, sum;
* the tuned hybrid (``--weights``) with and without an added ``lambda * mean-view log-prob`` term,
  lambda chosen on the dev sets only (itest / OOD only report).

Run inside ``envs/omni``:
    uv run python scripts/robust_scores.py --cands pool_dev_aug \
        --weights hybrid_dev_aug.weights.json
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import torch
import typer
from fairseq2.data.tokenizers.hub import load_tokenizer
from fairseq2.models.hub import load_model

from joint_rescore2 import build_features
from llm_candidates import DECODER_LORA, set_adapter
from qxp_asr.audio import load
from qxp_asr.augment import VIEWS, view
from qxp_asr.lora import add_lora
from qxp_asr.metrics import wer_cer
from qxp_asr.paths import ARTIFACTS, MANIFESTS
from qxp_asr.text import normalize
from train_rl import reward
from train_verifier import score

SETS = {
    "dev_unseen": ((pl.col("role") == "dev_scripted") & pl.col("unseen_sentence"), "scripted"),
    "itest_unseen": ((pl.col("role") == "itest_scripted") & pl.col("unseen_sentence"), "scripted"),
    "dev_spont": (pl.col("role") == "dev_spont", "spont"),
    "itest_spont": (pl.col("role") == "itest_spont", "spont"),
    "ood": (pl.col("role") == "ood", "spont"),
}
LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def pair_acc(err: np.ndarray, s: np.ndarray) -> tuple[float, int]:
    better = err[:, None] < err[None, :]
    return float((s[:, None] > s[None, :])[better].sum()), int(better.sum())


def main(
    cands: str = typer.Option(...),
    weights: str = typer.Option(...),
    post: str = typer.Option("ctc1b_v3_soup_wise0.85"),
    lm: str = typer.Option("dev_v2"),
    lm_text: str = typer.Option("indomain"),
    adapter: list[str] = typer.Option(
        [
            "zs",
            "v2=/workspace/simbig/artifacts/runs/llm7b_lora_v2/best_trainable.pt",
        ]
    ),
    views: int = typer.Option(3),
    clips: int = typer.Option(150),
    out: str = typer.Option("robust_dev"),
    seed: int = typer.Option(0),
) -> None:
    table = pl.read_parquet(ARTIFACTS / "cands" / f"{cands}.parquet")
    man = pl.read_parquet(MANIFESTS / "all.parquet").filter(
        pl.col("text").is_not_null() & (pl.col("duration_s") <= 39)
    )
    picked = {}
    for name, (cond, _) in SETS.items():
        sub = man.filter(cond & pl.col("clip_id").is_in(table["clip_id"].unique()))
        picked[name] = sub.sample(n=min(clips, sub.height), seed=seed)
    allclips = pl.concat(picked.values())
    by_clip = table.filter(pl.col("clip_id").is_in(allclips["clip_id"])).partition_by(
        "clip_id", as_dict=True
    )
    tok = load_tokenizer("omniASR_LLM_7B_v2")
    enc = tok.create_encoder()
    model = load_model("omniASR_LLM_7B_v2", device=torch.device("cuda"), dtype=torch.bfloat16)
    add_lora(model, DECODER_LORA, rank=16, alpha=32, dropout=0.0)
    model.eval()
    pad = model.target_vocab_info.pad_idx
    specs = [(a, None) if a == "zs" else tuple(a.split("=", 1)) for a in adapter]
    rob: dict[str, dict[str, np.ndarray]] = {a: {} for a, _ in specs}
    wavs = {c: load(w) for c, w in allclips.select("clip_id", "wav").iter_rows()}
    for aname, path in specs:
        set_adapter(model, path)
        for cid in allclips["clip_id"]:
            texts = by_clip[(cid,)]["text"].to_list()
            per_view = []
            for j, (speed, snr) in enumerate(VIEWS[1 : 1 + views]):
                w = torch.from_numpy(view(wavs[cid], speed, snr, np.random.default_rng(j)))
                with torch.no_grad():
                    per_view.append(score(model, enc, w, texts, pad).float().cpu().numpy())
            rob[aname][cid] = np.mean(per_view, axis=0)
        print(f"[{aname}] robust scores done", flush=True)

    # analysis: clean vs robust log-probs, and the hybrid with a robust term
    lp_cols = sorted(c for c in table.columns if c.startswith("lp_"))
    saved = json.loads((ARTIFACTS / "cands" / weights).read_text())
    char_lm = str(ARTIFACTS / "lm" / lm / f"{lm_text}.char10.bin")
    view_feature = "naug" in saved["scripted"]
    feats = build_features(
        table, allclips["clip_id"].to_list(), post, char_lm, 8, False, view_feature
    )
    names = ["chr", "len", *lp_cols, *(["naug"] if view_feature else [])]
    rows, picks = [], {}
    for name, (_, domain) in SETS.items():
        w = np.array([saved[domain][k] for k in names])
        stats: dict[str, list] = {}
        for cid, ref in picked[name].select("clip_id", "text").iter_rows():
            texts, f, _ = feats[cid]
            order = {t: i for i, t in enumerate(by_clip[(cid,)]["text"].to_list())}
            idx = [order[t] for t in texts]
            err = np.array([-reward(normalize(ref), normalize(t)) for t in texts])
            hyb = f[:, 0] + f[:, 1:] @ w
            r_sum = sum(rob[a][cid][idx] for a, _ in specs)
            scores = {"hybrid": hyb, "robust_sum": r_sum}
            for a, _ in specs:
                scores[f"clean_{a}"] = f[:, 3 + lp_cols.index(f"lp_{a}")]
                scores[f"robust_{a}"] = rob[a][cid][idx]
            for lam in LAMBDAS:
                scores[f"hybrid+{lam:g}rob"] = hyb + lam * r_sum
            for k, s in scores.items():
                c, n = pair_acc(err, s)
                st = stats.setdefault(k, [0.0, 0, [], []])
                st[0] += c
                st[1] += n
                st[2].append(ref)
                st[3].append(texts[int(np.argmax(s))])
        for k, (c, n, refs, hyps) in stats.items():
            rows.append(
                {
                    "set": name,
                    "score": k,
                    "pair_acc": round(100 * c / max(1, n), 1),
                    "sel_wer": round(wer_cer(refs, hyps)["wer"], 2),
                }
            )
        picks[name] = stats
    res = pl.DataFrame(rows)
    res.write_csv(f"reports/{out}.csv")
    dev = res.filter(pl.col("set").is_in(["dev_unseen", "dev_spont"]))
    lam_rows = dev.filter(pl.col("score").str.starts_with("hybrid+"))
    best = lam_rows.group_by("score").agg(pl.col("sel_wer").mean()).sort("sel_wer")["score"][0]
    print(f"lambda chosen on dev: {best}")
    with pl.Config(tbl_rows=60, tbl_cols=-1, tbl_width_chars=200):
        print(res.pivot(on="set", index="score", values="pair_acc").sort("score"))
        print(res.pivot(on="set", index="score", values="sel_wer").sort("score"))


if __name__ == "__main__":
    typer.run(main)
