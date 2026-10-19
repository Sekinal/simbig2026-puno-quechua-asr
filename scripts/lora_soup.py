"""Exact soup of LoRA adapters: the average of their weight updates as one higher-rank adapter.

Model soups (Wortsman et al., 2022) average fine-tuned weights to gain robustness. For LoRA the
update is B @ A, and averaging A and B separately is NOT the average update. Concatenating the
factors gives it exactly: A = [A_1; ...; A_n] (n*r x in), B = [B_1 ... B_n] / n (out x n*r), so
B @ A = mean_i(B_i @ A_i). With the scale alpha / rank kept equal (alpha = 2 * rank throughout),
the soup is a rank n*r adapter usable with ``infer_llm.py --decoder-lora <n*r>`` and
``llm_candidates.py --rank <n*r>``.

    uv run python scripts/lora_soup.py --out .../llm7b_soup/best_trainable.pt \
        --adapter .../llm7b_lora_v1/best_trainable.pt --adapter .../llm7b_lora_v2/best_trainable.pt
"""

from __future__ import annotations

from pathlib import Path

import torch
import typer


def main(adapter: list[Path] = typer.Option(...), out: Path = typer.Option(...)) -> None:
    states = [torch.load(a, map_location="cpu", weights_only=True) for a in adapter]
    n = len(states)
    soup = {}
    for name in states[0]:
        if name.endswith("lora_a"):
            soup[name] = torch.cat([s[name] for s in states], dim=0)
        elif name.endswith("lora_b"):
            soup[name] = torch.cat([s[name] for s in states], dim=1) / n
        else:
            soup[name] = torch.stack([s[name].float() for s in states]).mean(0).to(states[0][name])
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(soup, out)
    rank = next(v.shape[0] for k, v in soup.items() if k.endswith("lora_a"))
    print(f"{n} adapters -> rank {rank} soup at {out}")


if __name__ == "__main__":
    typer.run(main)
