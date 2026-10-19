"""Training data pipeline: manifest → weighted, length-bucketed dynamic batches.

Design:
* every training example carries a ``source`` (scripted / spont_gold / spont_silver / pseudo …);
  each epoch draws examples with replacement according to per-source weights, so small but
  valuable sources (gold spontaneous, elderly speech) can be up-sampled without duplicating files;
* drawn examples are sorted by duration inside large shuffled pools and packed into batches whose
  *padded* length stays below ``max_batch_seconds`` (classic dynamic batching, cf. fairseq);
* waveforms are layer-normalized per utterance after augmentation, exactly like omniASR.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
from torch.nn.functional import layer_norm
from torch.utils.data import Dataset, Sampler

from qxp_asr.audio import load
from qxp_asr.augment import Augmenter
from qxp_asr.vocab import encode


@dataclass(slots=True)
class Batch:
    seqs: torch.Tensor  # (B, T) float32, layer-normalized, zero padded
    seq_lens: list[int]
    targets: torch.Tensor  # (B, L) int64, zero padded
    target_lens: list[int]
    clip_ids: list[str]


class AsrDataset(Dataset):
    """Rows need: clip_id, wav, duration_s, target (normalized text), source."""

    def __init__(
        self,
        rows: pl.DataFrame,
        augment: bool,
        seed: int = 0,
        target_ids: list[list[int]] | None = None,
        augment_profile: str = "mild",
    ) -> None:
        self.clip_ids = rows["clip_id"].to_list()
        self.wavs = rows["wav"].to_list()
        self.texts = rows["target"].to_list()
        self.durations = rows["duration_s"].to_numpy()
        self.sources = rows["source"].to_list()
        self.augment = augment
        self.augment_profile = augment_profile
        self.seed = seed
        self._aug: Augmenter | None = None
        # Optional pre-tokenized targets (e.g. omniASR tokenizer ids for the LLM decoder).
        self.target_ids = target_ids

    def __len__(self) -> int:
        return len(self.wavs)

    def __getitem__(self, i: int) -> tuple[str, torch.Tensor, torch.Tensor]:
        wav = load(self.wavs[i])
        if self.augment:
            if self._aug is None:  # one RNG per worker process
                info = torch.utils.data.get_worker_info()
                seed = self.seed * 1000 + (info.id if info else 0)
                self._aug = Augmenter(seed, profile=self.augment_profile)
            wav = self._aug(wav)
        x = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))
        ids = self.target_ids[i] if self.target_ids is not None else encode(self.texts[i])
        y = torch.tensor(ids, dtype=torch.long)
        return self.clip_ids[i], x, y


def collate(items: list[tuple[str, torch.Tensor, torch.Tensor]]) -> Batch:
    ids, xs, ys = zip(*items, strict=True)
    xs = [layer_norm(x, x.shape) for x in xs]
    seq_lens = [len(x) for x in xs]
    target_lens = [len(y) for y in ys]
    seqs = torch.zeros(len(xs), max(seq_lens))
    targets = torch.zeros(len(ys), max(*target_lens, 1), dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys, strict=True)):
        seqs[i, : len(x)] = x
        targets[i, : len(y)] = y
    return Batch(seqs, seq_lens, targets, target_lens, list(ids))


class WeightedDynamicBatchSampler(Sampler[list[int]]):
    """Yield index batches; per-epoch weighted resampling by source, duration-bucketed packing."""

    def __init__(
        self,
        durations: np.ndarray,
        sources: list[str],
        weights: Mapping[str, float],
        max_batch_seconds: float,
        epoch_seconds: float | None = None,
        pool_size: int = 2000,
        seed: int = 0,
    ) -> None:
        self.durations = durations
        src = np.asarray(sources)
        hours = {s: durations[src == s].sum() for s in set(sources)}
        # weight w means "each second of this source counts w times"
        per_item = np.array([weights.get(s, 0.0) for s in sources], dtype=np.float64)
        self.p = per_item * durations / max(1e-9, float((per_item * durations).sum()))
        self.mean_dur = float((self.p * durations).sum())
        total = sum(weights.get(s, 0.0) * h for s, h in hours.items())
        self.epoch_seconds = epoch_seconds or total
        self.max_batch_seconds = max_batch_seconds
        self.pool_size = pool_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batches(self) -> list[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        n = max(1, int(self.epoch_seconds / self.mean_dur))
        drawn = rng.choice(len(self.durations), size=n, replace=True, p=self.p)
        batches: list[list[int]] = []
        for start in range(0, n, self.pool_size):
            pool = drawn[start : start + self.pool_size]
            pool = pool[np.argsort(self.durations[pool])[::-1]]
            batch: list[int] = []
            longest = 0.0
            for idx in pool:
                d = float(self.durations[idx])
                longest = max(longest, d) if batch else d
                if batch and longest * (len(batch) + 1) > self.max_batch_seconds:
                    batches.append(batch)
                    batch, longest = [], d
                batch.append(int(idx))
            if batch:
                batches.append(batch)
        order = rng.permutation(len(batches))
        return [batches[i] for i in order]

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches()

    def __len__(self) -> int:
        return len(self._batches())
