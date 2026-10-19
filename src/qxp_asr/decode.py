"""CTC decoding of saved posteriors: greedy, word-LM beam search, char-LM lexicon-free beam search.

* ``WordLM`` — pyctcdecode (Kensho) beam search with KenLM shallow fusion; unigram list enables
  partial-word scoring; OOV words are allowed with ``unk_score_offset``.
* ``CharLM`` — flashlight-text lexicon-free decoder (Kahn et al. 2022, flashlight; Likhomanenko
  et al. 2019 "Who needs words? Lexicon-free speech recognition") with a character KenLM, which
  can emit unseen agglutinated word forms — important with 22.5 % spontaneous OOV.

All decoders consume float32 log-probabilities over ``qxp_asr.vocab.SYMBOLS``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from pyctcdecode import build_ctcdecoder
from torchaudio.models.decoder import ctc_decoder

from qxp_asr.vocab import SYMBOLS, ctc_collapse, decode, encode, pyctcdecode_labels


def load_posteriors(directory: Path, clip_ids: Sequence[str]) -> list[np.ndarray]:
    return [np.load(directory / f"{c}.npy").astype(np.float32) for c in clip_ids]


def greedy(log_probs: Sequence[np.ndarray]) -> list[str]:
    return [decode(ctc_collapse(lp.argmax(-1).tolist())) for lp in log_probs]


@dataclass
class WordLM:
    kenlm_path: Path
    unigrams_path: Path
    alpha: float = 0.5
    beta: float = 1.0
    beam_width: int = 64
    unk_score_offset: float = -10.0

    def __post_init__(self) -> None:
        unigrams = self.unigrams_path.read_text().split()
        self._decoder = build_ctcdecoder(
            pyctcdecode_labels(),
            kenlm_model_path=str(self.kenlm_path),
            unigrams=unigrams,
            alpha=self.alpha,
            beta=self.beta,
            unk_score_offset=self.unk_score_offset,
        )

    def set_params(self, alpha: float, beta: float) -> None:
        self._decoder.reset_params(alpha=alpha, beta=beta)
        self.alpha, self.beta = alpha, beta

    def __call__(self, log_probs: Sequence[np.ndarray], workers: int = 8) -> list[str]:
        with get_context("fork").Pool(workers) as pool:
            out = self._decoder.decode_batch(pool, list(log_probs), beam_width=self.beam_width)
        return [" ".join(t.split()) for t in out]


@dataclass
class CharLM:
    kenlm_path: Path
    lm_weight: float = 0.5
    word_score: float = 0.0
    beam_size: int = 100
    beam_threshold: float = 25.0

    def _build(self):
        tokens = ["<blank>", "|", *SYMBOLS[2:]]
        return ctc_decoder(
            lexicon=None,
            tokens=tokens,
            lm=str(self.kenlm_path),
            nbest=1,
            beam_size=self.beam_size,
            beam_threshold=self.beam_threshold,
            lm_weight=self.lm_weight,
            word_score=self.word_score,
            sil_score=0.0,
            blank_token="<blank>",
            sil_token="|",
            unk_word="<unk>",
        )

    def set_params(self, lm_weight: float, word_score: float) -> None:
        self.lm_weight, self.word_score = lm_weight, word_score

    def __call__(self, log_probs: Sequence[np.ndarray], workers: int = 8) -> list[str]:
        del workers  # flashlight is fast enough single-threaded for our set sizes
        decoder = self._build()
        out: list[str] = []
        for lp in log_probs:
            emissions = torch.from_numpy(lp)[None].contiguous()
            hyp = decoder(emissions)[0][0]
            chars = decoder.idxs_to_tokens(hyp.tokens)
            text = "".join(" " if c == "|" else c for c in chars if c != "<blank>")
            out.append(" ".join(text.split()))
        return out


def ctc_loglik(log_probs: np.ndarray, text: str) -> float:
    """Exact CTC log-likelihood of ``text`` (summed over all alignments); -1e9 if impossible."""
    target = encode(text)
    if not target or len(target) > log_probs.shape[0]:
        return -1e9
    loss = torch.nn.functional.ctc_loss(
        torch.from_numpy(log_probs).unsqueeze(1),
        torch.tensor([target]),
        torch.tensor([log_probs.shape[0]]),
        torch.tensor([len(target)]),
        blank=0,
        reduction="sum",
    )
    return -float(loss) if torch.isfinite(loss) else -1e9
