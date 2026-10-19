"""Build an external Quechua text corpus respelled to Puno (qxp) orthography for n-gram LMs.

Sources (downloaded into ``data/text/raw``; JW300 and Bible corpora are deliberately excluded):

* **Llamacha/monolingual-quechua-iic** — QuBERT pre-training corpus (Zevallos et al., 2022,
  "Introducing QuBERT", DeepLo), Southern Quechua. Apache-2.0.
  https://huggingface.co/datasets/Llamacha/monolingual-quechua-iic
* **Quechua Wikipedia** — ``wikimedia/wikipedia`` config ``20231101.qu``. CC BY-SA 4.0 (and GFDL).
  https://huggingface.co/datasets/wikimedia/wikipedia
* **FineWeb-2** ``quz_Latn`` (Cusco) and ``quy_Latn`` (Ayacucho), train + test splits
  (Penedo et al., 2025, arXiv:2506.20920). ODC-By 1.0 (plus CommonCrawl terms of use). Documents
  from Bible / Jehovah's Witnesses sites are dropped.
  https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
* **FLORES-200** ``quy_Latn`` dev + devtest (NLLB Team, 2022, arXiv:2207.04672). CC BY-SA 4.0.
  https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz (the HF ``flores_plus`` copy is
  gated).

Pipeline (``build``): documents → lines → sentences; drop URL / markup / mostly-digit lines; drop
lines that look Spanish or English (stopword share vs. Quechua-token share); ``normalize()`` then
``respell()``; drop lines with < 3 words; exact dedupe (within and across sources). Outputs:
``data/text/external_qxp.txt`` (all sources), ``data/text/external_<source>.txt`` and
``data/text/external_qxp.parquet`` (source, normalized-only text, respelled text).

Other commands: ``learn`` regenerates the respelling resources shipped in ``qxp_asr/resources``
from the Common Voice original vs. curated sentences and reports held-out accuracy; ``download``
fetches the raw sources; ``stats`` reports per-source counts and spontaneous-dev OOV rates.
"""

from __future__ import annotations

import hashlib
import re
import tarfile
import urllib.request
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

import jiwer
import polars as pl
import typer

from qxp_asr.paths import DATA, EXTERNAL, MANIFESTS, SCRIPTED_DIR
from qxp_asr.respell import (
    Respeller,
    align_words,
    has_quechua_evidence,
    is_spanish_like,
    write_resources,
)
from qxp_asr.text import normalize

app = typer.Typer(add_completion=False, no_args_is_help=True)

TEXT = DATA / "text"
RAW_TEXT = TEXT / "raw"
CURATED_TSV = EXTERNAL / "shared-task" / "validated_sentences_curated.tsv"
ORIGINAL_TSV = SCRIPTED_DIR / "validated_sentences.tsv"

_HF = "https://huggingface.co/datasets"
_PQ = "resolve/refs%2Fconvert%2Fparquet"
FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
# source name -> list of (local file name, url)
DOWNLOADS: dict[str, list[tuple[str, str]]] = {
    "iic": [
        ("iic.parquet", f"{_HF}/Llamacha/monolingual-quechua-iic/{_PQ}/default/train/0000.parquet")
    ],
    "wiki": [
        ("wiki_qu.parquet", f"{_HF}/wikimedia/wikipedia/{_PQ}/20231101.qu/train/0000.parquet")
    ],
    **{
        f"fw2_{lang}": [
            (
                f"fw2_{lang}_{split}.parquet",
                f"{_HF}/HuggingFaceFW/fineweb-2/{_PQ}/{lang}_Latn/{split}/0000.parquet",
            )
            for split in ("train", "test")
        ]
        for lang in ("quz", "quy")
    },
    "flores_quy": [("flores200.tar.gz", FLORES_URL)],
}
SOURCES = tuple(DOWNLOADS)

# FineWeb-2 documents from these hosts are Bible / JW translations (excluded by design).
_RELIGIOUS_URL_RE = re.compile(
    r"jw\.org|jw-cdn|bible|biblia|ebible|scriptureearth|watchtower|wol\.jw|bibles\.org|"
    r"lds\.org|churchofjesuschrist",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"https?://|www\.|\.(?:com|org|net|pe|html?)\b|@", re.IGNORECASE)
_MARKUP_RE = re.compile(r"[|{}<>=#*_\\^~]|\[\[|\]\]|&[a-z]+;")
_CITATION_RE = re.compile(r"\[\d+\]")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+|\s*[\n\r••·]+\s*")
_TOKEN_DIGIT_RE = re.compile(r"\d")
# Central (Quechua I) spelling: affricate ts and long vowels (kankitsu, paawan) never occur in qxp.
_CENTRAL_RE = re.compile(r"ts|aa|ii|uu")

# Spanish + English function words (the bulk of any non-Quechua line). Quechua never uses these.
STOPWORDS = frozenset(
    [
        "a",
        "al",
        "algo",
        "algunas",
        "algunos",
        "ante",
        "antes",
        "aqui",
        "asi",
        "aun",
        "bien",
        "cada",
        "como",
        "con",
        "contra",
        "cual",
        "cuando",
        "de",
        "del",
        "desde",
        "donde",
        "durante",
        "e",
        "el",
        "ella",
        "ellas",
        "ellos",
        "en",
        "entre",
        "era",
        "es",
        "esa",
        "ese",
        "eso",
        "esta",
        "estaba",
        "estan",
        "este",
        "esto",
        "estos",
        "fue",
        "fueron",
        "ha",
        "hace",
        "hacia",
        "han",
        "hasta",
        "hay",
        "la",
        "las",
        "le",
        "les",
        "lo",
        "los",
        "mas",
        "me",
        "mi",
        "mis",
        "mucho",
        "muy",
        "ni",
        "no",
        "nos",
        "nosotros",
        "o",
        "otra",
        "otras",
        "otro",
        "otros",
        "para",
        "pero",
        "poco",
        "por",
        "porque",
        "que",
        "quien",
        "se",
        "sea",
        "segun",
        "ser",
        "si",
        "sin",
        "sino",
        "sobre",
        "son",
        "su",
        "sus",
        "tambien",
        "tiene",
        "todo",
        "todos",
        "tu",
        "un",
        "una",
        "uno",
        "unos",
        "y",
        "ya",
        "yo",
        "the",
        "of",
        "and",
        "to",
        "in",
        "is",
        "that",
        "for",
        "it",
        "with",
        "as",
        "was",
        "on",
        "are",
        "by",
        "this",
        "be",
        "from",
        "at",
        "or",
        "an",
        "not",
        "have",
        "has",
        "were",
        "which",
        "their",
        "they",
        "his",
        "her",
        "its",
        "but",
        "been",
        "also",
        "can",
        "will",
        "would",
        "one",
        "all",
        "more",
        "about",
    ]
)
# Frequent Quechua word endings, optionally followed by an evidential (-m/-mi/-n/-s/-si).
STRONG_SUFFIX_RE = re.compile(
    r"(?:kuna|manta|man|paq|pas|pis|sqa|sqan|spa|stin|ptin|chu|chis|chik|nku|nki|yki|wan|rqan|"
    r"rqa|shan|chkan|ykun|niku|yku|qa|taq|raq|puni|lla|ña|pi|ta)(?:m|mi|n|s|si)?$"
)


# --------------------------------------------------------------------------- helpers
def _sha_fold(s: str, k: int = 5) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % k


def load_cv_pairs(original: Path = ORIGINAL_TSV, curated: Path = CURATED_TSV) -> pl.DataFrame:
    orig = pl.read_csv(original, separator="\t", quote_char=None).select("sentence_id", "sentence")
    cur = pl.read_csv(curated, separator="\t", quote_char=None)
    return orig.join(cur, on="sentence_id").with_columns(
        fold=pl.col("sentence_id").map_elements(_sha_fold, return_dtype=pl.Int64)
    )


def _word_scores(
    r: Respeller | None, origs: list[str], refs: list[str]
) -> tuple[int, int, int, int]:
    """(fixed, needed, broken, kept) over 1:1-aligned word pairs (original vs curated)."""
    fixed = need = broken = kept = 0
    for a, b in zip(origs, refs, strict=True):
        for op, s, t in align_words(normalize(a).split(), b.split()):
            if op == "equal" or (op == "replace" and len(s) == len(t)):
                for x, y in zip(s, t, strict=True):
                    h = r.respell_word(x) if r else x
                    if x == y:
                        kept += 1
                        broken += h != y
                    else:
                        need += 1
                        fixed += h == y
    return fixed, need, broken, kept


def _report_line(name: str, refs: list[str], hyps: list[str], scores: tuple[int, ...]) -> str:
    fixed, need, broken, kept = scores
    return (
        f"| {name} | {100 * jiwer.wer(refs, hyps):.2f} | "
        f"{100 * sum(a == b for a, b in zip(refs, hyps, strict=True)) / len(refs):.1f} | "
        f"{fixed}/{need} ({100 * fixed / max(need, 1):.1f}) | "
        f"{broken}/{kept} ({100 * broken / max(kept, 1):.2f}) |"
    )


@app.command()
def learn(
    original: Path = ORIGINAL_TSV,
    curated: Path = CURATED_TSV,
    out_dir: Path = typer.Option(None, help="resource dir (default: qxp_asr/resources)"),
) -> None:
    """Learn the respelling resources from CV original vs curated sentences; report accuracy."""
    df = load_cv_pairs(original, curated)
    origs, refs = df["sentence"].to_list(), df["sentence_curated"].to_list()
    lines = [
        "| setting | WER % | sentence exact % | needed edits made (%) | correct words broken (%) |",
        "|---|---|---|---|---|",
    ]
    lines.append(
        _report_line(
            "normalize() only", refs, [normalize(x) for x in origs], _word_scores(None, origs, refs)
        )
    )
    for mode in ("rules", "dict+rules"):
        hyps, rr, oo, tot = [], [], [], [0, 0, 0, 0]
        for k in range(5):
            tr, te = df.filter(pl.col("fold") != k), df.filter(pl.col("fold") == k)
            full = Respeller.from_pairs(zip(tr["sentence"], tr["sentence_curated"], strict=True))
            r = full if mode == "dict+rules" else Respeller(lexicon=full.lexicon)
            te_o, te_r = te["sentence"].to_list(), te["sentence_curated"].to_list()
            hyps += [r(x) for x in te_o]
            rr += te_r
            oo += te_o
            tot = [a + b for a, b in zip(tot, _word_scores(r, te_o, te_r), strict=True)]
        lines.append(_report_line(f"{mode} (5-fold held-out)", rr, hyps, tuple(tot)))
    full = Respeller.from_pairs(zip(origs, refs, strict=True))
    lines.append(
        _report_line(
            "dict+rules (in-sample)",
            refs,
            [full(x) for x in origs],
            _word_scores(full, origs, refs),
        )
    )
    print("\n".join(lines))
    write_resources(full, *([out_dir] if out_dir else []))
    print(f"words={len(full.words)} phrases={len(full.phrases)} lexicon={len(full.lexicon)}")


# --------------------------------------------------------------------------- download
@app.command()
def download(raw_dir: Path = RAW_TEXT, force: bool = False) -> None:
    """Fetch raw source files (parquet via the HF parquet conversion, FLORES tarball)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, files in DOWNLOADS.items():
        for fname, url in files:
            dest = raw_dir / fname
            if dest.exists() and not force:
                continue
            print(f"[{name}] {url}")
            tmp = dest.with_suffix(dest.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(dest)
    flores = raw_dir / "flores200.tar.gz"
    with tarfile.open(flores) as tar:
        for member in tar.getmembers():
            if member.name.endswith(("quy_Latn.dev", "quy_Latn.devtest")):
                f = tar.extractfile(member)
                assert f is not None
                (raw_dir / Path(member.name).name).write_bytes(f.read())


# --------------------------------------------------------------------------- loading
def load_documents(source: str, raw_dir: Path = RAW_TEXT) -> tuple[list[str], dict[str, int]]:
    """Raw documents of one source, plus loader stats (e.g. religious docs dropped)."""
    info: dict[str, int] = {}
    if source == "flores_quy":
        docs = []
        for split in ("dev", "devtest"):
            docs += (raw_dir / f"quy_Latn.{split}").read_text(encoding="utf-8").splitlines()
        return docs, info
    frames = [pl.read_parquet(raw_dir / fname) for fname, _ in DOWNLOADS[source]]
    df = pl.concat([f.select(*(c for c in ("text", "url") if c in f.columns)) for f in frames])
    info["docs"] = df.height
    if "url" in df.columns:
        religious = pl.col("url").str.contains(_RELIGIOUS_URL_RE.pattern)
        info["docs_religious_dropped"] = df.filter(religious).height
        df = df.filter(~religious)
    return df["text"].drop_nulls().to_list(), info


def sentences(docs: Iterable[str]) -> Iterator[str]:
    for doc in docs:
        yield from filter(None, (p.strip() for p in _SENT_SPLIT_RE.split(doc)))


# --------------------------------------------------------------------------- filtering
class LineFilter:
    """Raw-line filters + a Quechua-vs-Spanish/English heuristic on normalized tokens."""

    def __init__(
        self,
        lexicon: frozenset[str],
        min_quechua: float = 0.6,
        max_stop: float = 0.2,
        max_digit: float = 0.3,
        max_central: float = 0.1,
    ):
        self.lexicon = lexicon
        self.min_quechua, self.max_stop, self.max_digit = min_quechua, max_stop, max_digit
        self.max_central = max_central
        self.counts: Counter[str] = Counter()

    def is_quechua_token(self, tok: str) -> bool:
        if tok in self.lexicon:
            return True
        if tok in STOPWORDS or len(tok) < 3 or is_spanish_like(tok):
            return False
        return has_quechua_evidence(tok) or bool(STRONG_SUFFIX_RE.search(tok))

    def raw_reason(self, raw: str) -> str | None:
        """Rejection reason based on the raw line (URLs, markup, numbers), else None."""
        if _URL_RE.search(raw):
            return "url"
        if _MARKUP_RE.search(raw):
            return "markup"
        raw_toks = raw.split()
        n_digit = sum(bool(_TOKEN_DIGIT_RE.search(t)) for t in raw_toks)
        if not raw_toks or n_digit / len(raw_toks) > self.max_digit:
            return "digits"
        return None

    def reject_reason(self, raw: str) -> tuple[str | None, str]:
        """(reason, normalized line); reason is None for an accepted line."""
        if (reason := self.raw_reason(raw)) is not None:
            return reason, ""
        norm = normalize(_CITATION_RE.sub(" ", raw))
        toks = norm.split()
        if len(toks) < 3:
            return "short", norm
        stop = sum(t in STOPWORDS for t in toks) / len(toks)
        que = sum(self.is_quechua_token(t) for t in toks) / len(toks)
        if stop > self.max_stop or que < self.min_quechua:
            return "not_quechua", norm
        if sum(bool(_CENTRAL_RE.search(t)) for t in toks) / len(toks) > self.max_central:
            return "central_quechua", norm
        return None, norm

    def __call__(self, raw: str) -> str | None:
        """Return the normalized line, or None (and count the reason) if rejected."""
        reason, norm = self.reject_reason(raw)
        if reason is not None:
            self.counts[reason] += 1
            return None
        return norm


def in_domain_texts(man: pl.DataFrame, roles: Iterable[str]) -> list[str]:
    rows = man.filter(pl.col("role").is_in(list(roles)))
    texts = rows["text"].drop_nulls().to_list() + rows["silver"].drop_nulls().to_list()
    return [normalize(t) for t in texts]


def _vocab(lines: Iterable[str]) -> Counter[str]:
    return Counter(w for line in lines for w in line.split())


# --------------------------------------------------------------------------- build
@app.command()
def build(
    sources: list[str] = typer.Option(list(SOURCES), "--source"),
    raw_dir: Path = RAW_TEXT,
    out_dir: Path = TEXT,
    min_words: int = 3,
    max_words: int = 120,
    min_quechua: float = 0.6,
    max_stop: float = 0.2,
) -> None:
    """Clean, normalize, respell and dedupe the external sources."""
    man = pl.read_parquet(MANIFESTS / "all.parquet")
    # Language-ID lexicon: curated scripted text + spontaneous train transcripts (gold + silver).
    lid_lexicon = frozenset(_vocab(in_domain_texts(man, ["train"]))) | Respeller.default().lexicon
    eval_texts = set(
        in_domain_texts(man, ["dev_scripted", "dev_spont", "itest_scripted", "itest_spont"])
    )
    respeller = Respeller.from_files(stems=lid_lexicon)
    frames = []
    for source in sources:
        docs, info = load_documents(source, raw_dir)
        filt = LineFilter(lid_lexicon, min_quechua=min_quechua, max_stop=max_stop)
        n_raw = 0
        rows = []
        for raw in sentences(docs):
            n_raw += 1
            norm = filt(raw)
            if norm is not None:
                rows.append((norm, respeller(norm)))
        df = pl.DataFrame(rows, schema=["norm", "text"], orient="row").with_columns(
            source=pl.lit(source), n_words=pl.col("text").str.count_matches(" ") + 1
        )
        df = df.filter(pl.col("n_words").is_between(min_words, max_words))
        print(
            f"[{source}] {info} raw_sentences={n_raw} rejected={dict(filt.counts)} kept={df.height}"
        )
        frames.append(df)
    all_df = pl.concat(frames)
    n_before = all_df.height
    all_df = all_df.unique("text", keep="first", maintain_order=True)
    overlap = all_df.filter(pl.col("text").is_in(list(eval_texts))).height
    all_df = all_df.filter(~pl.col("text").is_in(list(eval_texts)))
    print(
        f"dedupe: {n_before} -> {all_df.height + overlap}; "
        f"dropped {overlap} lines identical to dev/itest transcripts"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    all_df.select("source", "norm", "text").write_parquet(out_dir / "external_qxp.parquet")
    (out_dir / "external_qxp.txt").write_text("\n".join(all_df["text"]) + "\n", encoding="utf-8")
    for (source,), g in all_df.group_by("source"):
        (out_dir / f"external_{source}.txt").write_text(
            "\n".join(g["text"]) + "\n", encoding="utf-8"
        )
    stats(out_dir)


# --------------------------------------------------------------------------- stats
def _oov(dev: list[str], vocab: set[str]) -> tuple[float, float]:
    toks = [w for line in dev for w in line.split()]
    types = set(toks)
    return (
        100 * sum(w not in vocab for w in toks) / len(toks),
        100 * sum(w not in vocab for w in types) / len(types),
    )


@app.command()
def stats(out_dir: Path = TEXT) -> None:
    """Per-source line/token counts; OOV of spontaneous gold dev vs in-domain / external vocab."""
    ext = pl.read_parquet(out_dir / "external_qxp.parquet")
    per = (
        ext.with_columns(n_tok=pl.col("text").str.count_matches(" ") + 1)
        .group_by("source")
        .agg(
            lines=pl.len(),
            tokens=pl.col("n_tok").sum(),
            types=pl.col("text").str.split(" ").explode().n_unique(),
        )
        .sort("tokens", descending=True)
    )
    print("| source | lines | tokens | types |\n|---|---|---|---|")
    for r in per.iter_rows(named=True):
        print(f"| {r['source']} | {r['lines']:,} | {r['tokens']:,} | {r['types']:,} |")
    print(
        f"| **total** | {ext.height:,} | {per['tokens'].sum():,} | {len(_vocab(ext['text'])):,} |"
    )

    man = pl.read_parquet(MANIFESTS / "all.parquet")
    dev = [
        normalize(t)
        for t in man.filter(
            (pl.col("role") == "dev_spont") & pl.col("label").is_in(["gold", "pending"])
        )["text"].drop_nulls()
    ]
    train_vocab = set(_vocab(in_domain_texts(man, ["train"])))
    ext_vocab = set(_vocab(ext["text"]))
    ext_norm_vocab = set(_vocab(ext["norm"]))
    rows = [
        ("(a) in-domain train (scripted + spont gold + silver)", train_vocab),
        ("(b) (a) + external, respelled", train_vocab | ext_vocab),
        ("(c) (a) + external, normalize() only (no respell)", train_vocab | ext_norm_vocab),
        ("(d) external respelled only", ext_vocab),
    ]
    rows += [
        (
            f"(a) + {s} (respelled)",
            train_vocab | set(_vocab(ext.filter(pl.col("source") == s)["text"])),
        )
        for s in per["source"]
    ]
    n_tok = sum(len(x.split()) for x in dev)
    print(f"\nOOV of dev_spont gold ({len(dev)} utts, {n_tok} tokens)")
    print("| vocabulary | vocab size | token OOV % | type OOV % |\n|---|---|---|---|")
    for name, voc in rows:
        tok, typ = _oov(dev, voc)
        print(f"| {name} | {len(voc):,} | {tok:.2f} | {typ:.2f} |")


if __name__ == "__main__":
    app()
