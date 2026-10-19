"""Every interval and test reported in the paper, from per-clip predictions (one place, one method).

Paired bootstrap of corpus WER differences (``qxp_asr.stats``, 10,000 resamples, 95 % percentile
intervals, two-sided bootstrap p-values) under the resampling design that matches each set:

* read speech (seen / unseen sentences): two-way crossed speaker x sentence (pigeonhole) bootstrap;
* spontaneous speech: exact sign test over speakers (dev 4, itest 6: too few clusters for a cluster
  bootstrap; with 4 speakers no result can reach p < 0.05), clip-level interval descriptive;
* OOD radio drama: clips (no speaker labels).

The clip-level interval is always reported next to it. Families:

1. sentence reuse (Table 1), including the seen-vs-unseen interaction;
2. components, training seeds and adapters (Table 2);
3. public and organizer systems;
4. hybrid ablation, views-vs-samples control and LoRA soup as scorer (edit counts averaged over the
   three weight-search seeds of each variant);
5. RL arms vs their LoRA v2 start (re-decoded with the same pipeline), with LoRA v2 decoded twice
   as the decoding-noise floor;
6. proportions (Wilson intervals).

Primary (confirmatory) family, Holm-corrected: (H1) full system vs the best single model on dev
(LoRA v1, best on both dev domains) and (H2) perturbed views vs none, each on the two never-tuned
sets with enough resampling units (itest unseen: 19 speakers x 248 sentences; OOD: 109 clips).
itest spontaneous (6 speakers) is descriptive only. Everything else is secondary (unadjusted).
Writes ``reports/paper_stats.md`` and ``reports/paper_stats.csv``.

    uv run python scripts/paper_stats.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import polars as pl
import typer

from qxp_asr.metrics import official_normalize
from qxp_asr.paths import ARTIFACTS, MANIFESTS
from qxp_asr.stats import (
    bootstrap_diffs,
    edits,
    holm,
    paired_bootstrap,
    sign_test,
    summarize,
    wilson,
)
from qxp_asr.text import normalize

BENCH = Path(__file__).resolve().parents[1] / "reports" / "benchmarks"
READ = {"dev_unseen", "itest_unseen", "dev_seen", "itest_seen", "itest_all", "rl_unseen"}


def half_of(sentence_id: str) -> int:
    """Sentence half of the RL cross-fit (identical to scripts/train_rl.py)."""
    return int(hashlib.md5(sentence_id.encode()).hexdigest(), 16) % 2


def eval_sets(man: pl.DataFrame) -> dict[str, pl.DataFrame]:
    scr = pl.col("role").is_in(["dev_scripted", "itest_scripted"])
    man = man.with_columns(half=pl.col("sentence_id").map_elements(half_of, return_dtype=pl.Int64))
    return {
        "dev_unseen": man.filter((pl.col("role") == "dev_scripted") & pl.col("unseen_sentence")),
        "itest_unseen": man.filter(
            (pl.col("role") == "itest_scripted") & pl.col("unseen_sentence")
        ),
        "dev_seen": man.filter((pl.col("role") == "dev_scripted") & ~pl.col("unseen_sentence")),
        "itest_seen": man.filter((pl.col("role") == "itest_scripted") & ~pl.col("unseen_sentence")),
        "itest_all": man.filter(pl.col("role") == "itest_scripted"),
        "dev_spont": man.filter(pl.col("role") == "dev_spont"),
        "itest_spont": man.filter(pl.col("role") == "itest_spont"),
        "ood": man.filter(pl.col("role") == "ood"),
        # RL: dev + itest clips of the held-out sentences of half 1 (updates used half 0)
        "rl_unseen": man.filter(scr & pl.col("unseen_sentence") & (pl.col("half") == 1)),
        "rl_spont": man.filter((pl.col("role") == "dev_spont") & (pl.col("duration_s") <= 39)),
    }


class Stats:
    def __init__(self, resamples: int) -> None:
        man = pl.read_parquet(MANIFESTS / "all.parquet").filter(pl.col("text").is_not_null())
        self.sets = eval_sets(man)
        self.resamples = resamples
        self.rows: list[dict] = []
        self._cache: dict[str, dict[str, str]] = {}

    def hyps(self, system: str) -> dict[str, str]:
        if system not in self._cache:
            if system.startswith("bench:"):  # public / organizer predictions (reports/benchmarks)
                path = BENCH / system[len("bench:") :]
            else:
                path = ARTIFACTS / "preds" / f"{system}.parquet"
            df = pl.read_parquet(path)
            hyps = dict(df.select("clip_id", "hyp").iter_rows())
            if system.startswith("bench:"):  # raw outputs: same text normalization as the benchmark
                hyps = {c: normalize(h or "") for c, h in hyps.items()}
            self._cache[system] = hyps
        return self._cache[system]

    def design(self, name: str, d: pl.DataFrame) -> tuple[str, object, object]:
        if name in READ:
            sent = [official_normalize(t) for t in d["text"]]
            return "speaker x sentence", d["speaker"].to_list(), sent
        if name.endswith("spont"):
            return "speaker", d["speaker"].to_list(), None
        return "clip", None, None

    def _errors(self, system, d: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Edit counts of a system (or the mean over a list of systems) on the set's clips."""
        systems = system if isinstance(system, list) else [system]
        es = []
        for s in systems:
            h = self.hyps(s)
            e, n = edits(d["text"].to_list(), [h.get(c, "") for c in d["clip_id"]])
            es.append(e)
        return np.mean(es, axis=0), n

    def compare(self, family: str, label: str, a, b, sets: list[str]) -> None:
        for name in sets:
            d = self.sets[name]
            ha = self.hyps(a[0] if isinstance(a, list) else a)
            d = d.filter(pl.col("clip_id").is_in(list(ha)))
            if d.is_empty():
                continue
            ea, n = self._errors(a, d)
            eb, _ = self._errors(b, d)
            clip = paired_bootstrap(ea, eb, n, resamples=self.resamples)
            kind, g1, g2 = self.design(name, d)
            if kind == "speaker":  # 4-6 speakers: exact sign test, clip interval descriptive
                sg = sign_test(ea, eb, n, g1)
                clus = clip | {"p": sg["p"], "units": sg["units"]}
                kind = f"speaker sign test ({sg['better']}/{sg['units']} better)"
            elif g1 is not None:
                clus = paired_bootstrap(ea, eb, n, g1, g2, resamples=self.resamples)
            else:
                clus = clip
            self.rows.append(
                {
                    "family": family,
                    "comparison": label,
                    "set": name,
                    "clips": d.height,
                    "wer_a": round(clip["wer_a"], 2),
                    "wer_b": round(clip["wer_b"], 2),
                    "diff": round(clip["diff"], 2),
                    "clip_lo": round(clip["lo"], 2),
                    "clip_hi": round(clip["hi"], 2),
                    "clip_p": clip["p"],
                    "design": kind,
                    "units": clus["units"],
                    "lo": round(clus["lo"], 2),
                    "hi": round(clus["hi"], 2),
                    "p": clus["p"],
                }
            )

    def interaction(self, a: str, b: str, seen: str, unseen: str) -> None:
        """(a - b on seen sentences) - (a - b on unseen sentences); the subsets share no clips."""
        dist, point = [], []
        for k, name in enumerate((seen, unseen)):
            d = self.sets[name]
            ea, n = self._errors(a, d)
            eb, _ = self._errors(b, d)
            _, g1, g2 = self.design(name, d)
            dist.append(bootstrap_diffs(ea, eb, n, g1, g2, self.resamples, seed=k))
            point.append(100 * (ea - eb).sum() / n.sum())
        m = min(len(dist[0]), len(dist[1]))
        lo, hi, p = summarize(dist[0][:m] - dist[1][:m])
        self.rows.append(
            {
                "family": "1 sentence reuse",
                "comparison": f"interaction ({seen} - {unseen})",
                "set": f"{seen}/{unseen}",
                "diff": round(float(point[0] - point[1]), 2),
                "design": "speaker x sentence",
                "lo": round(lo, 2),
                "hi": round(hi, 2),
                "p": p,
            }
        )


def ablation_systems(stats: Stats, table: str, variant: str) -> list[str]:
    """Register the per-seed outputs of an ablation variant as pseudo-systems; return their keys."""
    df = pl.read_parquet(ARTIFACTS / "preds" / f"ablation_{table}.parquet").filter(
        pl.col("variant") == variant
    )
    keys = []
    for (seed,), g in df.group_by("seed", maintain_order=True):
        key = f"{table}:{variant}:{seed}"
        stats._cache[key] = dict(g.select("clip_id", "hyp").iter_rows())
        keys.append(key)
    return keys


def main(
    resamples: int = typer.Option(10_000),
    out: Path = typer.Option(Path("reports/paper_stats")),
) -> None:
    st = Stats(resamples)
    allsets = ["dev_unseen", "itest_unseen", "dev_spont", "itest_spont", "ood"]

    # 1. sentence reuse: CTC + word 4-gram (training prompts) vs the same CTC model greedy
    ctc = pl.read_parquet(ARTIFACTS / "cands" / "ctc_v3_in.parquet").sort("rank")
    st._cache["ctc_4gram"] = dict(
        ctc.group_by("clip_id").first().select("clip_id", "text").iter_rows()
    )
    greedy = "ctc1b_v3_soup_wise0.85_greedy"
    seen_sets = ["dev_seen", "dev_unseen", "itest_seen", "itest_unseen"]
    st.compare("1 sentence reuse", f"ctc_4gram - {greedy}", "ctc_4gram", greedy, seen_sets)
    st.compare(
        "1 sentence reuse", "hybrid_dev_aug - ctc_4gram", "hybrid_dev_aug", "ctc_4gram", seen_sets
    )
    for seen, unseen in (("dev_seen", "dev_unseen"), ("itest_seen", "itest_unseen")):
        st.interaction("ctc_4gram", greedy, seen, unseen)

    # 2. components (zs / v1 re-decoded with pause-based chunking: *_pc)
    fam = "2 components"
    for label, a, b in [
        ("LoRA v1 - zero-shot", "llm7b_lora_v1_pc", "zs_llm7b_pc"),
        ("LoRA v2 - LoRA v1", "llm7b_lora_v2", "llm7b_lora_v1_pc"),
        ("hybrid - LoRA v1", "hybrid_dev_in", "llm7b_lora_v1_pc"),
        ("hybrid - LoRA v2", "hybrid_dev_in", "llm7b_lora_v2"),
        ("views - hybrid", "hybrid_dev_aug", "hybrid_dev_in"),
        ("full - zero-shot", "hybrid_dev_aug", "zs_llm7b_pc"),
        ("full - CTC", "hybrid_dev_aug", greedy),
        ("full - LoRA v1", "hybrid_dev_aug", "llm7b_lora_v1_pc"),
        ("full - LoRA v2", "hybrid_dev_aug", "llm7b_lora_v2"),
        ("soup - LoRA v1", "llm7b_soup4", "llm7b_lora_v1_pc"),
        ("soup - LoRA v2", "llm7b_soup4", "llm7b_lora_v2"),
        ("LoRA v5 (cross-fit labels) - LoRA v2", "llm7b_lora_v5", "llm7b_lora_v2"),
        ("LoRA v2 seed 1 - seed 0", "llm7b_lora_v2_s1", "llm7b_lora_v2"),
        ("LoRA v2 seed 2 - seed 0", "llm7b_lora_v2_s2", "llm7b_lora_v2"),
        ("LoRA v2 seed 2 - seed 1", "llm7b_lora_v2_s2", "llm7b_lora_v2_s1"),
        ("enc+dec LoRA - LoRA v2", "llm7b_lora_encdec", "llm7b_lora_v2"),
        ("enc+dec LoRA seed 1 - LoRA v2", "llm7b_lora_encdec_s1", "llm7b_lora_v2"),
        ("enc+dec LoRA seed 1 - seed 0", "llm7b_lora_encdec_s1", "llm7b_lora_encdec"),
        ("full + enc+dec scorer - full", "hybrid_dev_ed", "hybrid_dev_aug"),
        ("encoder-only LoRA - LoRA v2", "llm7b_lora_enc", "llm7b_lora_v2"),
        ("encoder-only LoRA - enc+dec LoRA", "llm7b_lora_enc", "llm7b_lora_encdec"),
        ("enc+dec LoRA rank 64 - rank 16", "llm7b_lora_encdec_r64", "llm7b_lora_encdec"),
    ]:
        try:
            st.compare(fam, label, a, b, allsets)
        except FileNotFoundError as err:
            print(f"skipped {label}: {err}")
    st.compare(
        fam,
        "final: views - no views",
        "hybrid_final_aug",
        "hybrid_final_in_nr",
        ["dev_spont", "ood"],
    )

    # 3. public and organizer systems
    fam = "3 public systems"
    for label, b in [
        (
            "submission - omniASR-LLM-1B zero-shot",
            "bench:all_systems_runs/omniASR_LLM_1B_v2__ood.parquet",
        ),
        (
            "submission - omniASR-LLM-7B zero-shot",
            "bench:all_systems_runs/omniASR_LLM_7B_v2__ood.parquet",
        ),
        (
            "submission - organizer XLS-R CPT silver",
            "bench:all_systems_runs/xls-r-cpt-qxp-cv26-silver__ood.parquet",
        ),
    ]:
        st.compare(fam, label, "hybrid_final_aug", b, ["ood"])
    st.compare(
        fam,
        "dev-mode full - organizer XLS-R CPT validated",
        "hybrid_dev_aug",
        "bench:organizer_preds/xls-r-cpt-qxp-cv26-validated.preds.parquet",
        ["itest_all", "itest_unseen"],
    )

    # 4. ablation (means over 3 weight-search seeds) and controls
    try:
        fam = "4 ablation"
        full = ablation_systems(st, "hybrid_ablation", "full")
        variants = pl.read_parquet(ARTIFACTS / "preds" / "ablation_hybrid_ablation.parquet")[
            "variant"
        ].unique(maintain_order=True)
        for v in variants:
            if v != "full":
                st.compare(
                    fam, f"{v} (vs full)", ablation_systems(st, "hybrid_ablation", v), full, allsets
                )
        base = ablation_systems(st, "ctrl_base", "full")
        aug = ablation_systems(st, "ctrl_aug", "full")
        smp = ablation_systems(st, "ctrl_smp", "full")
        st.compare(fam, "control: views - none", aug, base, allsets)
        st.compare(fam, "control: samples - none", smp, base, allsets)
        st.compare(fam, "control: views - samples", aug, smp, allsets)
        for t, label in [
            ("abl_pool_dev_aug_soup", "soup added"),
            ("abl_pool_dev_aug_soup_only", "soup instead of v1+v2"),
        ]:
            st.compare(fam, f"{label} (vs full)", ablation_systems(st, t, "full"), full, allsets)
    except FileNotFoundError as err:  # scripts/runs/33 not finished yet
        print(f"skipped ablation family: {err}")

    # 5. RL arms vs their start, same decoding; LoRA v2 decoded twice = noise floor
    fam = "5 RL"
    rl_sets = ["rl_unseen", "rl_spont", "ood"]
    for label, a in [
        ("decoding noise (LoRA v2 decoded again)", "llm7b_lora_v2_redecode"),
        ("SFT 400", "rl20_sft_s400"),
        ("GSPO 400", "rl20_gspo_s400"),
        ("MWER 400", "rl20_mwer_s400"),
        ("Dr. GRPO 400", "rl20_drgrpo_s400"),
        ("MWER mixed rollouts 150", "rl24_mwer_beam_2_sample_3_aug_3_s150"),
    ]:
        try:
            st.compare(fam, f"{label} - LoRA v2", a, "llm7b_lora_v2", rl_sets)
        except FileNotFoundError as err:
            print(f"skipped {label}: {err}")
    for label, a in [  # RL from the encoder-adapted start (encoder LoRA frozen; exp46/46b)
        ("enc+dec start: SFT 400", "rl46_sft_s400"),
        ("enc+dec start: GSPO 400", "rl46_gspo_s400"),
    ]:
        try:
            st.compare(fam, f"{label} - enc+dec LoRA", a, "llm7b_lora_encdec", rl_sets)
        except FileNotFoundError as err:
            print(f"skipped {label}: {err}")

    # 6. where to adapt: encoder vs decoder across scale (exp54) and model family (exp55), which
    # encoder layers (exp53 trained halves; exp56 update kept in one block of the trained adapter)
    fam = "6 where to adapt"
    pairs = []
    for tag in ("1b", "300m"):
        m = f"llm{tag}"
        pairs += [
            (f"{m}: decoder LoRA - zero-shot", f"{m}_lora_dec", f"{m}_zs"),
            (f"{m}: encoder LoRA - decoder LoRA", f"{m}_lora_enc", f"{m}_lora_dec"),
            (f"{m}: enc+dec LoRA - decoder LoRA", f"{m}_lora_encdec", f"{m}_lora_dec"),
            (f"{m}: enc+dec LoRA - encoder LoRA", f"{m}_lora_encdec", f"{m}_lora_enc"),
        ]
    pairs += [
        ("Whisper: decoder LoRA - zero-shot", "whisper_lora_dec", "whisper_zs"),
        ("Whisper: encoder LoRA - decoder LoRA", "whisper_lora_enc", "whisper_lora_dec"),
        ("Whisper: enc+dec LoRA - decoder LoRA", "whisper_lora_encdec", "whisper_lora_dec"),
        ("Whisper: enc+dec LoRA - encoder LoRA", "whisper_lora_encdec", "whisper_lora_enc"),
        ("7B: encoder LoRA, bottom half - all layers", "llm7b_lora_enc_bottom", "llm7b_lora_enc"),
        ("7B: encoder LoRA, top half - all layers", "llm7b_lora_enc_top", "llm7b_lora_enc"),
        ("7B: encoder LoRA, top - bottom half", "llm7b_lora_enc_top", "llm7b_lora_enc_bottom"),
    ]
    for keep in ("0_32", "32_64", "64_96", "96_128", "0_64", "64_128", "32_128", "0_96"):
        lo, hi = keep.split("_")
        pairs.append(
            (f"7B: encoder update kept in layers {lo}-{int(hi) - 1} - all layers",
             f"llm7b_lora_enc_keep{keep}", "llm7b_lora_enc")
        )  # fmt: skip
    for label, a, b in pairs:
        try:
            st.compare(fam, label, a, b, allsets)
        except FileNotFoundError as err:
            print(f"skipped {label}: {err.filename}")

    df = pl.DataFrame(st.rows, infer_schema_length=None)
    # Holm over the primary family
    prim = pl.col("comparison").is_in(["full - LoRA v1", "views - hybrid"]) & pl.col("set").is_in(
        ["itest_unseen", "ood"]
    )
    idx = [i for i, m in enumerate(df.select(prim).to_series().to_list()) if m]
    adj = holm([df["p"][i] for i in idx])
    holm_col = [None] * df.height
    for i, a in zip(idx, adj, strict=True):
        holm_col[i] = a
    df = df.with_columns(p_holm=pl.Series(holm_col, dtype=pl.Float64))

    # 6. proportions (counts from reports/RESULTS.md)
    props = []
    for label, k, n in [
        ("beam contains a better transcript (LoRA v2, 150 held-out-sentence clips)", 9, 150),
        ("T=1 samples contain a better transcript", 21, 150),
        ("perturbed views contain a better transcript", 25, 150),
        ("union contains a better transcript", 36, 150),
        ("error events specific to one reading (146 multi-reading sentences)", 104, 145),
    ]:
        lo, hi = wilson(k, n)
        props.append(
            {
                "proportion": label,
                "k": k,
                "n": n,
                "share": round(k / n, 3),
                "lo": round(lo, 3),
                "hi": round(hi, 3),
            }
        )

    df.write_csv(out.with_suffix(".csv"))
    with out.with_suffix(".md").open("w") as f:
        f.write("# Paper statistics (scripts/paper_stats.py)\n\n")
        f.write(f"Paired bootstrap, {resamples} resamples; differences in WER points (a - b); ")
        f.write("`lo`/`hi`/`p` use the set's resampling design, `clip_*` resample clips.\n\n")
        for (fam,), g in df.group_by("family", maintain_order=True):
            f.write(f"## {fam}\n\n")
            cols = [
                c
                for c in [
                    "comparison",
                    "set",
                    "clips",
                    "wer_a",
                    "wer_b",
                    "diff",
                    "clip_lo",
                    "clip_hi",
                    "design",
                    "units",
                    "lo",
                    "hi",
                    "p",
                    "p_holm",
                ]
                if c in g.columns and g[c].null_count() < g.height
            ]
            f.write(_md(g.select(cols)))
            f.write("\n")
        f.write("## 6 proportions (Wilson 95 % intervals)\n\n")
        f.write(_md(pl.DataFrame(props)))
    print(f"{df.height} comparisons -> {out}.md / .csv")


def _md(df: pl.DataFrame) -> str:
    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.3g}" if abs(v) < 0.01 and v != 0 else f"{v:.2f}"
        return str(v)

    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.iter_rows()]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    typer.run(main)
