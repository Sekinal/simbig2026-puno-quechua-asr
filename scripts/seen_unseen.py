"""Does sentence reuse change which method looks best? WER on seen vs unseen sentences per system.

The official splits share every scripted sentence between training and test (reports/eda), while
the shared-task test reads new sentences. For each dev-mode system we score the scripted dev and
itest clips separately on sentences that occur in training ("seen") and on our held-out sentences
("unseen"), and rank the systems under both conditions.

``--ctc-cands`` adds the CTC + word-LM 1-best (rank 0 of a ``ctc_nbest.py`` table) as a system.

    uv run python scripts/seen_unseen.py --pred zs_llm7b_all --pred llm7b_lora_v2 \
        --pred hybrid_dev_aug --ctc-cands ctc_v3_in
"""

from __future__ import annotations

import polars as pl
import typer

from qxp_asr.metrics import wer_cer
from qxp_asr.paths import ARTIFACTS, MANIFESTS


def main(
    pred: list[str] = typer.Option(...),
    ctc_cands: str = typer.Option("", help="CTC N-best table: its rank-0 text is a system"),
    out: str = typer.Option("seen_unseen"),
) -> None:
    man = pl.read_parquet(MANIFESTS / "all.parquet").filter(
        pl.col("role").is_in(["dev_scripted", "itest_scripted"]) & pl.col("text").is_not_null()
    )
    systems = {p: pl.read_parquet(ARTIFACTS / "preds" / f"{p}.parquet") for p in pred}
    if ctc_cands:
        top = (
            pl.read_parquet(ARTIFACTS / "cands" / f"{ctc_cands}.parquet")
            .sort("rank")
            .group_by("clip_id", maintain_order=True)
            .first()
        )
        systems[f"{ctc_cands} (CTC + word LM)"] = top.select("clip_id", hyp="text")
    rows = []
    for name, p in systems.items():
        d = man.join(p.select("clip_id", "hyp"), on="clip_id")
        for (role, unseen), g in d.group_by("role", "unseen_sentence"):
            s = wer_cer(g["text"].to_list(), g["hyp"].to_list())
            rows.append(
                {
                    "system": name,
                    "set": f"{role.removesuffix('_scripted')} {'unseen' if unseen else 'seen'}",
                    "wer": round(s["wer"], 2),
                }
            )
    res = pl.DataFrame(rows).pivot(on="set", index="system", values="wer")
    res = res.select("system", *sorted(c for c in res.columns if c != "system"))
    res.write_csv(f"reports/{out}.csv")
    with pl.Config(tbl_cols=-1, tbl_width_chars=200):
        print(res.sort("dev seen"))
        print(res.sort("dev unseen"))


if __name__ == "__main__":
    typer.run(main)
