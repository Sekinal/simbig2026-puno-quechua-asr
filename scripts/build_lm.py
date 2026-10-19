"""Build KenLM n-gram language models for CTC beam-search decoding.

Inspirations (docs/LITERATURE.md §c): n-gram LM shallow fusion with CTC beam search (KenLM,
Heafield 2011; pyctcdecode; flashlight lexicon-free decoding, Likhomanenko et al. 2019 / 2021
"Rethinking evaluation in ASR"); character-level LMs for agglutinative / high-OOV languages
(22.5 % spontaneous OOV, reports/eda/FINDINGS.md §6).

Text domains (mode ``dev`` uses only role ``train`` so dev/itest sentences stay unseen):
* ``scripted``  — distinct curated scripted sentences;
* ``spont``     — spontaneous gold + silver transcripts (+ optional pseudo-label parquet files);
* ``external``  — optional respelled external Quechua text (``data/text/external_qxp.txt``).

Outputs in ``artifacts/lm/<name>/``: ``{domain}.word{N}.bin`` + ``{domain}.unigrams.txt`` for word
LMs and ``{domain}.char{N}.bin`` (tokens = characters, ``|`` = word boundary) for lexicon-free
decoding.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import polars as pl
import typer

from qxp_asr.paths import ARTIFACTS, DATA, MANIFESTS, ROOT
from qxp_asr.sources import roles_for_mode
from qxp_asr.text import normalize

KENLM = ROOT / "envs" / "kenlm" / "build" / "bin"


def kenlm(text_file: Path, out: Path, order: int, prune: str | None = None) -> None:
    arpa = out.with_suffix(".arpa")
    cmd = [str(KENLM / "lmplz"), "-o", str(order), "--discount_fallback", "-S", "20%"]
    if prune:
        cmd += ["--prune", *prune.split()]
    with text_file.open() as src, arpa.open("w") as dst:
        subprocess.run(cmd, stdin=src, stdout=dst, stderr=subprocess.DEVNULL, check=True)
    subprocess.run(
        [str(KENLM / "build_binary"), "trie", str(arpa), str(out)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    arpa.unlink()


def to_chars(line: str) -> str:
    return " ".join("|" if c == " " else c for c in line)


def domain_texts(man: pl.DataFrame, mode: str, pseudo: list[Path]) -> dict[str, list[str]]:
    base = man.filter(pl.col("role").is_in(roles_for_mode(mode)))
    scripted = (
        base.filter((pl.col("corpus") == "scripted") & pl.col("text").is_not_null())["text"]
        .unique()
        .sort()
        .to_list()
    )
    spont = base.filter(pl.col("corpus") == "spontaneous")
    spont_txt = spont["text"].drop_nulls().to_list() + spont["silver"].drop_nulls().to_list()
    for p in pseudo:
        ids = set(spont["clip_id"].to_list())
        tab = pl.read_parquet(p).filter(pl.col("clip_id").is_in(ids))
        spont_txt += tab["target"].to_list()
    texts = {"scripted": scripted, "spont": spont_txt}
    ext = DATA / "text" / "external_qxp.txt"
    if ext.exists():
        texts["external"] = ext.read_text().splitlines()
    return {k: [t for t in map(normalize, v) if t] for k, v in texts.items()}


def main(
    name: str = typer.Option(...),
    mode: str = typer.Option("dev"),
    word_order: int = typer.Option(4),
    char_order: int = typer.Option(6, help="flashlight-text bundles KenLM with max order 6"),
    pseudo: list[Path] | None = typer.Option(None),
) -> None:
    out = ARTIFACTS / "lm" / name
    out.mkdir(parents=True, exist_ok=True)
    texts = domain_texts(pl.read_parquet(MANIFESTS / "all.parquet"), mode, pseudo or [])
    texts["all"] = texts["scripted"] + texts["spont"] + texts.get("external", [])
    texts["indomain"] = texts["scripted"] + texts["spont"]
    for domain, lines in texts.items():
        txt = out / f"{domain}.txt"
        txt.write_text("\n".join(lines) + "\n")
        words = sorted({w for line in lines for w in line.split()})
        (out / f"{domain}.unigrams.txt").write_text("\n".join(words) + "\n")
        kenlm(txt, out / f"{domain}.word{word_order}.bin", word_order)
        chars = out / f"{domain}.chars.txt"
        chars.write_text("\n".join(to_chars(line) for line in lines) + "\n")
        kenlm(chars, out / f"{domain}.char{char_order}.bin", char_order)
        print(f"{domain}: {len(lines)} lines, {len(words)} word types")


if __name__ == "__main__":
    typer.run(main)
