"""Respell Southern Quechua text (Cusco, Ayacucho, mixed web orthographies) to Puno (qxp) spelling.

Target conventions are those of the organizers' curated scripted references
(``validated_sentences_curated.tsv``, see its README): three vowels, ``huq``, ``nuqa``, ``alqu``,
``ashkha``, ``ishkay``, ``chhaqay``, ``chaqra``, ``qulqi``, ``qhilqa``, ``llank'ay``, ``mishk'i``,
``ashwan``, ``chhiqaq``; ``n`` (not ``m``) before ``p`` at morpheme boundaries and as the
evidential; ``-hina`` joined after nouns, pronouns and ``-spa`` forms but written apart after
finite verbs and ``-chu``; ``rayku`` joined.

:func:`respell` works in three passes over :func:`qxp_asr.text.normalize`-d text:

1. **Phrase map** learned from multi-word edits in the curated file (``yachaywasipi`` →
   ``yachay wasipi``), plus ``hina`` / ``rayku`` joining rules.
2. **Word map** learned from 1:1 word edits between the original Common Voice sentences and the
   curated ones (``huk`` → ``huq``, ``chakrapi`` → ``chaqrapi``, ...).
3. **Rules** for unseen words. A word that already occurs in the curated Puno lexicon is never
   touched. Rules only fire on words without Spanish-only letters (``b c d f g j v x z``,
   ``qu`` + ``e/i``, ``rr``) and the vowel rules additionally need Quechua evidence (``q k w`` or
   an ejective) and stop at names / loans from the curated lexicon (``manuelqa``, ``pedroqa``).
   In order: Cusco 5-vowel spelling (``e/o`` next to ``q``) → ``i/u``; ``au/ai`` → ``aw/ay``;
   Puno stems (``huk`` → ``huq``, ``qillqa`` → ``qhilqa`` ...); Ayacucho ``-chka-`` → ``-sha-``
   and ``-chik`` → ``-chis``; ``m`` → ``n`` between a known word and a p-suffix; final
   evidential ``-m`` → ``-n``.

The learned maps and the lexicon ship in ``qxp_asr/resources`` and are regenerated with
``scripts/build_text_corpus.py learn``.
"""

from __future__ import annotations

import bisect
import difflib
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from qxp_asr.text import normalize

RESOURCES = Path(__file__).parent / "resources"
WORD_MAP_PATH = RESOURCES / "respell_words.tsv"
PHRASE_MAP_PATH = RESOURCES / "respell_phrases.tsv"
LEXICON_PATH = RESOURCES / "qxp_lexicon.txt"

# Letters (or digraphs) that only occur in Spanish / English spelling, never in Quechua words.
_SPANISH_RE = re.compile(r"[bcdfgjvxz]|qu[ei]|rr")
_SPANISH_C_OK = re.compile(r"ch")
_QUECHUA_EVIDENCE_RE = re.compile(r"[qkw]|[ptkq]'|ch'")
# Cusco/AMLQ orthography writes /i u/ as e o next to a uvular: qosqo, qhepa, sonqo, qelqa, ñoqa.
# A word with such a vowel is taken to be 5-vowel spelled throughout (mosoq -> musuq).
_CUSCO_E_O_RE = re.compile(r"(?<=q)[eo]|(?<=q[h'])[eo]|[eo](?=[lmnrsy]?q)")
_E_O = str.maketrans("eo", "iu")
_GLIDE_RE = re.compile(r"a([ui])(?=[^aeiou]|$)")
_GLIDE = {"u": "w", "i": "y"}
# Ayacucho progressive -chka- and 1pl.incl -chik.
_CHKA_RE = re.compile(r"(?<=[aiu])chka")
# -chik only after -n- (1pl.incl), -ki- (2pl) or imperative -y; agentive yachachik is left alone.
_CHIK_RE = re.compile(r"(?:(?<=n)|(?<=ki)|(?<=y))chik(?=$|[^aeiouy])")
# m before p (suffixes -pi -pa -paq -puni -pis -pas -pura -pu-, and stems like t'impu); stems that
# keep mp in the curated lexicon (pampa, hampi, llamphu, ayrampu) are protected by prefix lookup.
_MP_RE = re.compile(r"(?<=[aiu])m(?=p[aiu])")
_MP_STEMS = frozenset({"pampa", "tampu", "hampi", "kampu", "lampa", "wampu", "rumpu"})
_FINAL_M_RE = re.compile(r"(?<=[aeiou])m$")
_FINAL_M_EXCEPTIONS = frozenset({"album", "islam", "forum", "maximum", "minimum", "um", "uhm"})

# Puno stem forms (README list + recurrent edits in the curated file). Applied at word start.
STEM_RULES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(p), r)
    for p, r in (
        (r"^huk(?=$|[^aeiou'])", "huq"),
        (r"^wak(?=$|[^aeiouc'])", "waq"),
        (r"^ñ[uo]qa", "nuqa"),
        (r"^qam(?=$|[^aeiou])", "qan"),
        (r"^(?:ask|ashk|achk)h?a", "ashkha"),
        (r"^chaqay", "chhaqay"),
        (r"^allqu", "alqu"),
        (r"^allqa", "alqa"),
        (r"^aswa(?=[mn])", "ashwa"),
        (r"^iskay", "ishkay"),
        (r"^qullqi", "qulqi"),
        (r"^chakra", "chaqra"),
        (r"^sallqa", "salqa"),
        (r"^llamk'?a", "llank'a"),
        (r"^hamk'a", "hank'a"),
        (r"^usqay", "usqhay"),
        (r"^llimp'?i", "llinphi"),
        (r"^qh?ill?qa", "qhilqa"),
        (r"^misk'?i", "mishk'i"),
        (r"^maskh?a", "mashkha"),
        (r"^waskh?a", "washkha"),
        (r"^ñukñu", "ñuqñu"),
        (r"^qh?i(?:ch|s|sh)wa", "qhichwa"),
        (r"^ichu(?=$|[^aeiou])", "ichhu"),
        (r"^waqha", "waha"),
        (r"^chuqcha", "chukcha"),
        (r"^puqlla", "puklla"),
        (r"^chiqap(?=$|[^aeiou])", "chhiqaq"),
        (r"^chiqaq", "chhiqaq"),
        (r"^ch'ampa", "ch'anpa"),
        (r"^t'ampa", "t'anpa"),
        (r"^t'impu", "t'inpu"),
        (r"^chimpa", "chinpa"),
        (r"^chaupi", "chawpi"),
        (r"^kuska", "kushka"),
        (r"^musq'?u", "musqhu"),
        (r"^chusaq", "ch'usaq"),
        (r"^hapi", "hap'i"),
        (r"^wisk'?a", "wisq'a"),
        (r"^lliqlla", "lliklla"),
        (r"^chaqway", "ch'aqway"),
        (r"^ch'akchu", "ch'aqchu"),
        (r"^ch'ikta", "ch'iqta"),
        (r"^qinuwa", "kinuwa"),
        (r"^p'?uncha[wy]", "p'unchay"),
    )
)

# hina is joined after pronouns / demonstratives / -spa forms; written apart after -chu.
_HINA_HOSTS = frozenset(
    {
        "chay",
        "kay",
        "haqay",
        "chhaqay",
        "nuqa",
        "qan",
        "pay",
        "nuqanchis",
        "nuqayku",
        "qankuna",
        "paykuna",
        "imayna",
        "ima",
        "mana",
    }
)


def is_spanish_like(word: str) -> bool:
    """True if the spelling cannot be a Quechua word (Spanish-only letters or digraphs)."""
    return bool(_SPANISH_RE.search(_SPANISH_C_OK.sub("", word)))


def has_quechua_evidence(word: str) -> bool:
    return bool(_QUECHUA_EVIDENCE_RE.search(word))


@dataclass
class Respeller:
    """Dictionary + rule respeller. Build with :meth:`default` or :meth:`from_pairs`."""

    words: dict[str, str] = field(default_factory=dict)
    phrases: dict[tuple[str, ...], tuple[str, ...]] = field(default_factory=dict)
    lexicon: frozenset[str] = frozenset()
    hina_hosts: frozenset[str] = _HINA_HOSTS
    # Words accepted as the host of an ``m``+p-suffix (defaults to ``lexicon``); a larger in-domain
    # vocabulary can be passed here without widening the never-touch ``lexicon``.
    stems: frozenset[str] | None = None

    def __post_init__(self) -> None:
        self._sorted_lexicon = sorted(self.lexicon)
        self._stems = self.lexicon if self.stems is None else self.stems | self.lexicon
        # Curated words that keep e/o are names or loans: protect words that start with them.
        self._eo_prefixes = frozenset(
            w for w in self.lexicon if len(w) >= 4 and re.search("[eo]", w)
        )
        self._max_phrase = max((len(k) for k in self.phrases), default=1)
        self._cache: dict[str, str] = {}

    # ----------------------------------------------------------------- construction
    @classmethod
    def default(cls) -> Respeller:
        return _default_respeller()

    @classmethod
    def from_files(
        cls,
        words: Path = WORD_MAP_PATH,
        phrases: Path = PHRASE_MAP_PATH,
        lexicon: Path = LEXICON_PATH,
        stems: frozenset[str] | None = None,
    ) -> Respeller:
        word_map = dict(_read_tsv(words))
        phrase_map = {tuple(a.split()): tuple(b.split()) for a, b in _read_tsv(phrases)}
        lex = frozenset(lexicon.read_text(encoding="utf-8").split())
        return cls(words=word_map, phrases=phrase_map, lexicon=lex, stems=stems)

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[str, str]], min_ratio: float = 0.5) -> Respeller:
        """Learn word / phrase maps and the lexicon from (original, curated) sentence pairs."""
        pairs = [(normalize(a), normalize(b)) for a, b in pairs]
        words, phrases = learn_maps(pairs, min_ratio=min_ratio)
        lexicon = frozenset(w for _, b in pairs for w in b.split())
        return cls(words=words, phrases=phrases, lexicon=lexicon)

    # ----------------------------------------------------------------- word level
    def _prefix_in_lexicon(self, prefix: str) -> bool:
        i = bisect.bisect_left(self._sorted_lexicon, prefix)
        return i < len(self._sorted_lexicon) and self._sorted_lexicon[i].startswith(prefix)

    def _protected_name(self, word: str) -> bool:
        return any(word[:k] in self._eo_prefixes for k in range(4, len(word) + 1))

    def respell_word(self, word: str) -> str:
        """Respell a single normalized word (may return several space-separated words)."""
        if word in self.words:
            return self.words[word]
        if word in self.lexicon:
            return word
        if is_spanish_like(word):  # loans still take the Puno evidential: distritum -> distritun
            return word if word in _FINAL_M_EXCEPTIONS else _FINAL_M_RE.sub("n", word)
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        out = self._apply_rules(word)
        self._cache[word] = out
        return out

    def _apply_rules(self, word: str) -> str:
        w = word
        if has_quechua_evidence(w) and not self._protected_name(w):
            if _CUSCO_E_O_RE.search(w):
                w = w.translate(_E_O)
            w = _GLIDE_RE.sub(lambda m: "a" + _GLIDE[m.group(1)], w)
        for pat, rep in STEM_RULES:
            if pat.match(w):
                w = pat.sub(rep, w, count=1)
                break
        w = _CHKA_RE.sub("sha", w)
        w = self._sub_unprotected(_CHIK_RE, "chis", w)
        w = self._sub_mp(w)
        if w not in _FINAL_M_EXCEPTIONS:
            w = _FINAL_M_RE.sub("n", w)
        return w

    def _sub_mp(self, w: str) -> str:
        """m -> n before p only after a known word (wasi+m+pi, llaqta+m+pa), never inside stems
        or names (pampa, hampi, urupampa, qaqatampu)."""

        def repl(m: re.Match[str]) -> str:
            host = w[: m.start()]
            known = host in self._stems or (host.endswith("ni") and host[:-2] in self._stems)
            stem = w[max(0, m.start() - 2) : m.start() + 3]
            protected = stem in _MP_STEMS or self._prefix_in_lexicon(w[: m.end() + 1])
            return "n" if known and not protected else "m"

        return _MP_RE.sub(repl, w)

    def _sub_unprotected(self, pat: re.Pattern[str], rep: str, w: str) -> str:
        """Substitute unless the word prefix through the match is a curated lexicon stem."""

        def repl(m: re.Match[str]) -> str:
            stem = w[: m.end() + 1]
            return m.group() if self._prefix_in_lexicon(stem) else rep

        return pat.sub(repl, w)

    # ----------------------------------------------------------------- sentence level
    def _apply_phrases(self, toks: list[str]) -> list[str]:
        if not self.phrases:
            return toks
        out: list[str] = []
        i = 0
        while i < len(toks):
            for n in range(min(self._max_phrase, len(toks) - i), 1, -1):
                tgt = self.phrases.get(tuple(toks[i : i + n]))
                if tgt is not None:
                    out.extend(tgt)
                    i += n
                    break
            else:
                out.append(toks[i])
                i += 1
        return out

    def _join_clitics(self, toks: list[str]) -> list[str]:
        out: list[str] = []
        for tok in toks:
            prev = out[-1] if out else ""
            if (tok.startswith("rayku") and prev and not is_spanish_like(prev)) or (
                tok == "hina"
                and prev
                and (prev.endswith("spa") or prev in self.hina_hosts)
                and not prev.endswith("chu")
            ):
                out[-1] = prev + tok
            elif tok.endswith("chuhina") and len(tok) > len("chuhina"):
                out.extend([tok[: -len("hina")], "hina"])
            else:
                out.append(tok)
        return out

    def __call__(self, text: str) -> str:
        toks = normalize(text).split()
        toks = self._apply_phrases(toks)
        toks = " ".join(self.respell_word(t) for t in toks).split()
        return " ".join(self._join_clitics(toks))


# --------------------------------------------------------------------------- learning
def align_words(src: Sequence[str], tgt: Sequence[str]) -> list[tuple[str, tuple, tuple]]:
    """Word-level alignment as (op, src_words, tgt_words) chunks (``difflib`` opcodes)."""
    sm = difflib.SequenceMatcher(a=list(src), b=list(tgt), autojunk=False)
    return [(op, tuple(src[i1:i2]), tuple(tgt[j1:j2])) for op, i1, i2, j1, j2 in sm.get_opcodes()]


def learn_maps(
    pairs: Iterable[tuple[str, str]], min_ratio: float = 0.5
) -> tuple[dict[str, str], dict[tuple[str, ...], tuple[str, ...]]]:
    """Word map from 1:1 substitutions and phrase map from n:m edits of normalized pairs.

    A source word is mapped to its most frequent target only if that target accounts for at least
    ``min_ratio`` of the word's aligned occurrences (including those where it was kept).
    """
    targets: dict[str, Counter[str]] = defaultdict(Counter)
    phrases: dict[tuple[str, ...], Counter[tuple[str, ...]]] = defaultdict(Counter)
    for a, b in pairs:
        for op, s, t in align_words(a.split(), b.split()):
            if op == "equal":
                for w in s:
                    targets[w][w] += 1
            elif op == "replace" and len(s) == len(t):
                for x, y in zip(s, t, strict=True):
                    targets[x][y] += 1
            elif op == "replace" and len(s) == 1:  # split: yachaywasipi -> yachay wasipi
                targets[s[0]][" ".join(t)] += 1
            elif op == "replace":
                phrases[s][t] += 1
    words = {}
    for src, cnt in targets.items():
        tgt, n = cnt.most_common(1)[0]
        if tgt != src and n / cnt.total() >= min_ratio:
            words[src] = tgt
    phrase_map = {s: c.most_common(1)[0][0] for s, c in phrases.items() if s}
    return words, phrase_map


def write_resources(r: Respeller, out_dir: Path = RESOURCES) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / WORD_MAP_PATH.name).write_text(
        "".join(f"{a}\t{b}\n" for a, b in sorted(r.words.items())), encoding="utf-8"
    )
    (out_dir / PHRASE_MAP_PATH.name).write_text(
        "".join(f"{' '.join(a)}\t{' '.join(b)}\n" for a, b in sorted(r.phrases.items())),
        encoding="utf-8",
    )
    (out_dir / LEXICON_PATH.name).write_text("\n".join(sorted(r.lexicon)) + "\n", encoding="utf-8")


def _read_tsv(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            a, b = line.split("\t")
            rows.append((a, b))
    return rows


@cache
def _default_respeller() -> Respeller:
    return Respeller.from_files()


def respell(text: str) -> str:
    """Normalize ``text`` and respell it to Puno orthography with the shipped resources."""
    return _default_respeller()(text)
