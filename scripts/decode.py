"""LM beam-search decoding of saved CTC posteriors, tuned per domain on dev, applied to itest/test.

Each domain (scripted / spontaneous) gets its own LM and its own (weight, bonus) pair, chosen by
grid search on ``dev_<domain>``; the chosen setting is then applied unchanged to ``itest_<domain>``
(unbiased estimate) and to the shared-task test clips of that ``test_type``.

Run inside ``envs/omni``:
    uv run python /workspace/simbig/repo/scripts/decode.py --post ctc1b_v1 --lm dev_v1 \
        --decoder word --scripted-lm scripted.word4 --spont-lm indomain.word4
"""

from __future__ import annotations

import itertools
import json
import time

import polars as pl
import typer

from qxp_asr.decode import CharLM, WordLM, greedy, load_posteriors
from qxp_asr.metrics import wer_cer
from qxp_asr.paths import ARTIFACTS, MANIFESTS

DOMAINS = {"scripted": ("dev_scripted", "itest_scripted"), "spont": ("dev_spont", "itest_spont")}
TEST_TYPE = {"scripted": "scripted", "spont": "spontaneous"}


def floats(spec: str) -> list[float]:
    return [float(x) for x in spec.split(",")]


def main(
    post: str = typer.Option(..., help="posteriors name under artifacts/posteriors"),
    lm: str = typer.Option(..., help="LM set name under artifacts/lm"),
    decoder: str = typer.Option("word", help="word | char"),
    scripted_lm: str = typer.Option("scripted.word4"),
    spont_lm: str = typer.Option("indomain.word4"),
    weights: str = typer.Option("0.3,0.6,0.9,1.2,1.6"),
    bonuses: str = typer.Option("-1,0,1,2,3"),
    tag: str = typer.Option(""),
    workers: int = typer.Option(10),
) -> None:
    man = pl.read_parquet(MANIFESTS / "all.parquet")
    post_dir = ARTIFACTS / "posteriors" / post
    lm_dir = ARTIFACTS / "lm" / lm
    results, outputs = [], []
    for domain, (dev_role, itest_role) in DOMAINS.items():
        lm_name = scripted_lm if domain == "scripted" else spont_lm
        lm_path = lm_dir / f"{lm_name}.bin"
        dec = (
            WordLM(lm_path, lm_dir / f"{lm_name.split('.')[0]}.unigrams.txt")
            if decoder == "word"
            else CharLM(lm_path)
        )
        dev = man.filter(pl.col("role") == dev_role)
        if domain == "scripted":  # seen dev sentences are in the LM text: tune on unseen only
            dev = dev.filter(pl.col("unseen_sentence"))
        dev_lp = load_posteriors(post_dir, dev["clip_id"].to_list())
        refs = dev["text"].to_list()
        best = (float("inf"), None)
        base = wer_cer(refs, greedy(dev_lp))
        print(f"[{domain}] greedy dev WER {base['wer']:.2f} CER {base['cer']:.2f}")
        for w, b in itertools.product(floats(weights), floats(bonuses)):
            t0 = time.time()
            dec.set_params(w, b)
            score = wer_cer(refs, dec(dev_lp, workers))
            print(
                f"  w={w} b={b}: WER {score['wer']:.2f} CER {score['cer']:.2f} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
            if score["wer"] < best[0]:
                best = (score["wer"], (w, b))
        w, b = best[1]
        dec.set_params(w, b)
        print(f"[{domain}] best w={w} b={b} dev WER {best[0]:.2f}")
        extra = ["ood"] if domain == "spont" else []  # unscripted broadcast speech: spont settings
        apply = man.filter(
            pl.col("role").is_in([dev_role, itest_role, *extra])
            | ((pl.col("role") == "test") & (pl.col("test_type") == TEST_TYPE[domain]))
        )
        lps = load_posteriors(post_dir, apply["clip_id"].to_list())
        hyps = dec(lps, workers)
        out = apply.select("clip_id", "role", "unseen_sentence", "text").with_columns(
            hyp=pl.Series(hyps), greedy=pl.Series(greedy(lps))
        )
        outputs.append(out)
        for role, g in out.filter(pl.col("text").is_not_null()).group_by("role"):
            results.extend(
                {"domain": domain, "role": role[0], "decode": kind, "w": w, "b": b}
                | wer_cer(g["text"].to_list(), g[kind].to_list())
                for kind in ("greedy", "hyp")
            )
            unseen = g.filter(pl.col("unseen_sentence"))
            if unseen.height:
                results.append(
                    {
                        "domain": domain,
                        "role": f"{role[0]}[unseen]",
                        "decode": "hyp",
                        "w": w,
                        "b": b,
                    }
                    | wer_cer(unseen["text"].to_list(), unseen["hyp"].to_list())
                )
    name = f"{post}__{lm}__{decoder}{tag}"
    pl.concat(outputs).write_parquet(ARTIFACTS / "preds" / f"{name}.parquet")
    table = pl.DataFrame(results).sort("domain", "role", "decode")
    (ARTIFACTS / "preds" / f"{name}.json").write_text(json.dumps(table.to_dicts(), indent=1))
    with pl.Config(tbl_rows=40):
        print(table)


if __name__ == "__main__":
    typer.run(main)
