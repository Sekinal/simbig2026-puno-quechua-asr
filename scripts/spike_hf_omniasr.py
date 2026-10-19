"""Feasibility spike: omniASR LLM-ASR under fairseq2 vs. Hugging Face transformers (PR #43265).

Subcommands run in two environments on the GPU box (both on CPU only):

* ``pick`` / ``fs2`` / ``fs2-lid``: ``envs/omni`` (fairseq2 + omnilingual-asr, read-only use of its
  venv python);
* ``convert`` / ``hf`` / ``lora-check`` / ``compare``: ``envs/hfspike`` (transformers PR head).

``convert`` is a pure-torch fairseq2 -> transformers converter (no fairseq2 import), so it also
works for the 7B checkpoint and for our ``qxp_asr.lora`` adapters, either merged into the base
weights (W + alpha/r * B A) or exported as a PEFT adapter. The decoder's Q/K rows are permuted from
fairseq2's interleaved RoPE layout to the half-split layout transformers uses, so LoRA ``B``
matrices of q/k projections get the same row permutation.

Example (300M parity)::

    python scripts/spike_hf_omniasr.py pick --out clips.json
    $OMNI/python scripts/spike_hf_omniasr.py fs2 --clips clips.json --out fs2.json
    $SPIKE/python scripts/spike_hf_omniasr.py hf --clips clips.json \
        --model bezzam/omniasr-llm-300m-v2 --out hf.json
    $SPIKE/python scripts/spike_hf_omniasr.py compare fs2.json hf.json
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

LANG = "qxp_Latn"
MANIFEST = "/workspace/simbig/data/manifests/all.parquet"
SR = 16_000

# fairseq2 -> transformers key mapping, from transformers PR #43265 (convert_omniasr_to_hf.py).
ENC = "model.audio_tower."
KEY_MAP = [
    (r"^encoder_frontend\.feature_extractor\.layers\.", f"{ENC}subsampling.conv_layers."),
    (r"^encoder_frontend\.post_extract_layer_norm\.", f"{ENC}subsampling.layer_norm."),
    (r"^encoder_frontend\.model_dim_proj\.", f"{ENC}subsampling.projection."),
    (r"^encoder_frontend\.pos_encoder\.conv\.", f"{ENC}encode_positions.conv."),
    (r"^encoder\.layer_norm\.", f"{ENC}layer_norm."),
    (r"^encoder\.layers\.(\d+)\.self_attn_layer_norm\.", rf"{ENC}layers.\1.layer_norm."),
    (r"^encoder\.layers\.(\d+)\.self_attn\.output_proj\.", rf"{ENC}layers.\1.attention.out_proj."),
    (r"^encoder\.layers\.(\d+)\.self_attn\.", rf"{ENC}layers.\1.attention."),
    (r"^encoder\.layers\.(\d+)\.ffn_layer_norm\.", rf"{ENC}layers.\1.final_layer_norm."),
    (
        r"^encoder\.layers\.(\d+)\.ffn\.inner_proj\.",
        rf"{ENC}layers.\1.feed_forward.intermediate_dense.",
    ),
    (r"^encoder\.layers\.(\d+)\.ffn\.output_proj\.", rf"{ENC}layers.\1.feed_forward.output_dense."),
    (r"^final_proj\.", "lm_head."),
    (r"^encoder_proj\.", "model.multi_modal_projector."),
    (r"^text_frontend\.", "model.language_model.embed_tokens."),
    (r"^llama_decoder\.layer_norm\.", "model.language_model.norm."),
    (
        r"^llama_decoder\.layers\.(\d+)\.self_attn_layer_norm\.",
        r"model.language_model.layers.\1.input_layernorm.",
    ),
    (
        r"^llama_decoder\.layers\.(\d+)\.self_attn\.output_proj\.",
        r"model.language_model.layers.\1.self_attn.o_proj.",
    ),
    (r"^llama_decoder\.layers\.(\d+)\.self_attn\.", r"model.language_model.layers.\1.self_attn."),
    (
        r"^llama_decoder\.layers\.(\d+)\.ffn_layer_norm\.",
        r"model.language_model.layers.\1.post_attention_layernorm.",
    ),
    (
        r"^llama_decoder\.layers\.(\d+)\.ffn\.gate_proj\.",
        r"model.language_model.layers.\1.mlp.gate_proj.",
    ),
    (
        r"^llama_decoder\.layers\.(\d+)\.ffn\.inner_proj\.",
        r"model.language_model.layers.\1.mlp.up_proj.",
    ),
    (
        r"^llama_decoder\.layers\.(\d+)\.ffn\.output_proj\.",
        r"model.language_model.layers.\1.mlp.down_proj.",
    ),
]
# PEFT target modules equivalent to qxp_asr's decoder LoRA (attention + FFN projections)
PEFT_TARGETS = (
    r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
)
DEC_HEADS = 8  # the LLaMA decoder is identical for every omniASR LLM size (12 x 4096, 8 heads)
ENC_HEADS = {1024: 16, 1280: 16, 2048: 16}  # 300M / 1B / 7B encoders (fairseq2 wav2vec2 archs)


def rename(key: str) -> str:
    for pattern, repl in KEY_MAP:
        new, n = re.subn(pattern, repl, key)
        if n:
            return new
    return key


def rope_permute(w, num_heads: int = DEC_HEADS):
    """Rows of a q/k projection: fairseq2 interleaved RoPE pairs -> transformers half-split."""
    out_dim, *rest = w.shape
    head_dim = out_dim // num_heads
    return w.reshape(num_heads, head_dim // 2, 2, *rest).transpose(1, 2).reshape(out_dim, *rest)


def _read_wav(path: str):
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32")
    if sr != SR or wav.ndim != 1:
        msg = f"{path}: expected 16 kHz mono, got {sr} Hz, shape {wav.shape}"
        raise ValueError(msg)
    return wav


@app.command()
def pick(
    out: Path = typer.Option(...),
    per_role: int = typer.Option(5),
    min_s: float = typer.Option(3.0),
    max_s: float = typer.Option(10.0),
    seed: int = typer.Option(0),
) -> None:
    """Pick short labelled dev clips (dev_spont + dev_scripted); never touches the test audio."""
    import polars as pl

    man = pl.read_parquet(MANIFEST)
    rows = []
    for role in ("dev_spont", "dev_scripted"):
        sub = man.filter(
            (pl.col("role") == role)
            & pl.col("text").is_not_null()
            & pl.col("duration_s").is_between(min_s, max_s)
        )
        sub = sub.sample(n=min(per_role, sub.height), seed=seed)
        rows += sub.select("clip_id", "role", "wav", "text", "duration_s").to_dicts()
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    print(f"{len(rows)} clips -> {out}")


def _fs2_lora(model, rank: int, weights: str | None) -> None:
    import torch

    from qxp_asr.lora import add_lora

    attn_ffn = r"\.layers\.\d+\.(self_attn\.(q|k|v|output)_proj|ffn\.(gate|inner|output)_proj)"
    add_lora(model, f"llama_decoder{attn_ffn}", rank=rank, alpha=2 * rank, dropout=0.0)
    if weights:
        model.load_state_dict(
            torch.load(weights, map_location="cpu", weights_only=True), strict=False
        )


@app.command()
def fs2(
    clips: Path = typer.Option(...),
    out: Path = typer.Option(...),
    card: str = typer.Option("omniASR_LLM_300M_v2"),
    lang: str = typer.Option(LANG),
    dtype: str = typer.Option("float32"),
    lora_rank: int = typer.Option(0),
    weights: str | None = typer.Option(None),
) -> None:
    """Reference: fairseq2 ASRInferencePipeline (nbest=1) + teacher-forced NLL of the reference."""
    import torch
    from fairseq2.data.tokenizers.hub import load_tokenizer
    from fairseq2.datasets.batch import Seq2SeqBatch
    from fairseq2.models.hub import load_model
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    from torch.nn.functional import layer_norm

    dt = getattr(torch, dtype)
    t0 = time.time()
    model = load_model(card, device=torch.device("cpu"), dtype=dt)
    if lora_rank:
        _fs2_lora(model, lora_rank, weights)
    tok = load_tokenizer(card)
    pipe = ASRInferencePipeline(model_card=None, model=model, tokenizer=tok, device="cpu", dtype=dt)
    print(f"loaded {card} in {time.time() - t0:.0f}s")
    enc = tok.create_encoder()
    rows = json.loads(clips.read_text())
    res = []
    for r in rows:
        wav = _read_wav(r["wav"])
        t0 = time.time()
        with torch.inference_mode():
            hyp = pipe.transcribe(
                [{"waveform": wav, "sample_rate": SR}], lang=[lang], batch_size=1
            )[0]
            target = enc(r["text"]).unsqueeze(0)
            src = layer_norm(torch.from_numpy(wav), wav.shape).unsqueeze(0).to(dt)
            batch = Seq2SeqBatch(
                source_seqs=src,
                source_seq_lens=[src.shape[1]],
                target_seqs=target,
                target_seq_lens=[target.shape[1]],
                example={"lang": [lang]},
            )
            nll = float(model(batch))
        res.append(r | {"hyp": hyp, "nll": nll, "n_tok": int(target.shape[1]) + 1})
        print(f"{time.time() - t0:5.1f}s nll={nll:.4f} | {hyp}")
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))


@app.command("fs2-lid")
def fs2_lid(out: Path = typer.Option(...)) -> None:
    """Dump fairseq2's language -> lang-embedding row mapping (to check the HF language tokens)."""
    from omnilingual_asr.models.wav2vec2_llama.lang_ids import supported_langs

    out.write_text(json.dumps(list(supported_langs)))
    print(len(supported_langs), "qxp_Latn" in supported_langs)


def _load_fs2_state(path: str):
    import torch

    ck = torch.load(path, mmap=True, map_location="cpu", weights_only=True)
    return ck.get("model", ck)


@app.command()
def convert(
    checkpoint: str = typer.Option(..., help="fairseq2 omniASR-LLM-*.pt (native fairseq2 keys)"),
    template: str = typer.Option(
        "bezzam/omniasr-llm-300m-v2", help="HF repo/dir giving the processor + language tokens"
    ),
    out: Path = typer.Option(...),
    lora: str | None = typer.Option(None, help="qxp_asr.lora trainable state (best_trainable.pt)"),
    lora_rank: int = typer.Option(16),
    mode: str = typer.Option("merge", help="merge (fold B A into W) or peft (adapter dir)"),
    dtype: str = typer.Option("bfloat16"),
    shard_gb: float = typer.Option(2.0),
    lid_row: int | None = typer.Option(
        None,
        help="text_frontend row fairseq2 uses as LID marker. omnilingual-asr's 300m/1b/3b_v2 archs "
        "reset target_vocab_size to 9812 (so marker = row 9812); 7b_v2 uses 10288 = the HF default",
    ),
) -> None:
    """fairseq2 checkpoint (+ optional LoRA) -> transformers OmniASRForConditionalGeneration dir."""
    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoProcessor

    dt = getattr(torch, dtype)
    sd = _load_fs2_state(checkpoint)
    out.mkdir(parents=True, exist_ok=True)

    # config: template config with the encoder dimensions read off the checkpoint
    cfg = AutoConfig.from_pretrained(template)
    dim = sd["encoder.layer_norm.weight"].shape[0]
    n_layers = len({k.split(".")[2] for k in sd if k.startswith("encoder.layers.")})
    ac = cfg.audio_config
    ac.hidden_size, ac.num_hidden_layers = dim, n_layers
    ac.intermediate_size = sd["encoder.layers.0.ffn.inner_proj.weight"].shape[0]
    ac.num_attention_heads = ENC_HEADS[dim]
    n_lang = sd["lang_embeddings.weight"].shape[0]
    n_vocab = sd["final_proj.weight"].shape[0]
    if cfg.text_config.vocab_size != n_vocab + 2 + n_lang:
        msg = f"template vocab {cfg.text_config.vocab_size} != {n_vocab} + 2 + {n_lang}"
        raise ValueError(msg)
    cfg.dtype = dtype
    cfg.save_pretrained(out)
    AutoProcessor.from_pretrained(template).save_pretrained(out)
    try:
        from transformers import GenerationConfig

        GenerationConfig.from_pretrained(template).save_pretrained(out)
    except OSError:
        print("template has no generation_config.json")

    lora_sd = {}
    if lora:
        lora_sd = torch.load(lora, map_location="cpu", weights_only=True)
        print(f"LoRA: {len(lora_sd)} tensors, rank {lora_rank}, mode {mode}")
    scale = 2.0  # qxp_asr: alpha = 2 * rank

    def lora_pair(key: str):
        base = key.removesuffix(".weight")
        a, b = lora_sd.get(f"{base}.lora_a"), lora_sd.get(f"{base}.lora_b")
        return (a, b) if a is not None else (None, None)

    shards, cur, cur_bytes, n_merged = [], {}, 0, 0
    emb = "model.language_model.embed_tokens.weight"
    for src_key in list(sd):
        if src_key.startswith("lang_embeddings.") or "num_updates" in src_key:
            continue
        if src_key.endswith("pos_encoder.conv.weight_g"):
            continue
        w, key = sd[src_key], src_key
        if key.endswith("pos_encoder.conv.weight_v"):  # fold weight norm (dim=2) into the kernel
            g = sd[key.replace("weight_v", "weight_g")]
            w = torch._weight_norm(w.float(), g.float(), 2)
            key = key.replace("weight_v", "weight")
        a, b = lora_pair(key)
        if a is not None and mode == "merge":
            w = w.float() + scale * (b.float() @ a.float())
            n_merged += 1
        name = rename(key)
        if ("language_model.layers" in name) and re.search(r"\.(q|k)_proj\.weight$", name):
            w = rope_permute(w)
        if name == emb:  # [text vocab + LID marker] + audio placeholder row + language rows
            if lid_row is not None and lid_row != n_vocab:
                w = w.clone()
                w[n_vocab] = w[lid_row]
                print(f"LID marker embedding <- text_frontend row {lid_row}")
            pad = torch.zeros(1, w.shape[1], dtype=w.dtype)
            w = torch.cat([w, pad, sd["lang_embeddings.weight"]])
        if name == "lm_head.weight":
            w = torch.cat([w, torch.zeros(2 + n_lang, w.shape[1], dtype=w.dtype)])
        t = w.to(dt).clone()  # detach from the mmap-ed checkpoint (safetensors cannot write it)
        cur[name] = t
        cur_bytes += t.numel() * t.element_size()
        if cur_bytes > shard_gb * 1e9:
            shards.append(cur)
            save_file(cur, out / f"part-{len(shards):05d}.safetensors", metadata={"format": "pt"})
            cur, cur_bytes = {}, 0
    if cur:
        shards.append(cur)
        save_file(cur, out / f"part-{len(shards):05d}.safetensors", metadata={"format": "pt"})
    index = {
        "metadata": {},
        "weight_map": {k: f"part-{i + 1:05d}.safetensors" for i, s in enumerate(shards) for k in s},
    }
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=1))
    print(f"wrote {sum(len(s) for s in shards)} tensors in {len(shards)} shards; merged {n_merged}")

    if lora_sd and mode == "peft":
        _write_peft(lora_sd, lora_rank, out / "adapter")


PEFT_MOD = {
    "self_attn.q_proj": "self_attn.q_proj",
    "self_attn.k_proj": "self_attn.k_proj",
    "self_attn.v_proj": "self_attn.v_proj",
    "self_attn.output_proj": "self_attn.o_proj",
    "ffn.gate_proj": "mlp.gate_proj",
    "ffn.inner_proj": "mlp.up_proj",
    "ffn.output_proj": "mlp.down_proj",
}


def _write_peft(lora_sd: dict, rank: int, out: Path) -> None:
    """qxp_asr LoRA (lora_a: r x in, lora_b: out x r) -> PEFT adapter on the HF decoder."""
    import torch
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    tensors = {}
    pat = re.compile(r"^llama_decoder\.layers\.(\d+)\.(.+)\.lora_(a|b)$")
    for key, val in lora_sd.items():
        m = pat.match(key)
        if not m:
            msg = f"unexpected LoRA key {key} (only decoder LoRA is supported)"
            raise KeyError(msg)
        layer, mod, ab = m.groups()
        hf_mod = PEFT_MOD[mod]
        t = val.float()
        if ab == "b" and hf_mod in {"self_attn.q_proj", "self_attn.k_proj"}:
            t = rope_permute(t)
        name = (
            f"base_model.model.model.language_model.layers.{layer}.{hf_mod}"
            f".lora_{ab.upper()}.weight"
        )
        tensors[name] = t.to(torch.float32).contiguous()
    save_file(tensors, out / "adapter_model.safetensors")
    cfg = {
        "peft_type": "LORA",
        "task_type": None,
        "r": rank,
        "lora_alpha": 2 * rank,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": PEFT_TARGETS,
        "base_model_name_or_path": str(out.parent),
        "inference_mode": True,
    }
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=1))
    print(f"PEFT adapter: {len(tensors)} tensors -> {out}")


def _hf_load(model_id: str, adapter: str | None, dtype: str):
    import torch
    from transformers import AutoProcessor, OmniASRForConditionalGeneration

    proc = AutoProcessor.from_pretrained(model_id)
    model = OmniASRForConditionalGeneration.from_pretrained(model_id, dtype=getattr(torch, dtype))
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    return proc, model.eval()


def _masked_nll(logits, labels, n_real: int) -> tuple[float, int]:
    """Mean NLL over the real text vocabulary only.

    The HF port appends 2 + 1694 zero rows to ``lm_head`` (LID marker, audio placeholder, language
    tokens). They score logit 0 and so steal softmax mass in ``forward``/loss (``generate`` hides
    this via ``suppress_tokens``); RL log-probs must be computed on ``logits[..., :n_real]``.
    """
    import torch

    lp = torch.log_softmax(logits[0, :-1, :n_real].float(), -1)
    tgt = labels[0, 1:]
    keep = tgt != -100
    nll = -lp[keep].gather(1, tgt[keep, None]).squeeze(1)
    return float(nll.mean()), int(keep.sum())


@app.command()
def hf(
    clips: Path = typer.Option(...),
    out: Path = typer.Option(...),
    model_id: str = typer.Option("bezzam/omniasr-llm-300m-v2", "--model"),
    adapter: str | None = typer.Option(None),
    lang: str | None = typer.Option(LANG),
    dtype: str = typer.Option("float32"),
    max_new_tokens: int = typer.Option(256),
) -> None:
    """transformers port: greedy generate + teacher-forced NLL of the reference."""
    import torch

    t0 = time.time()
    proc, model = _hf_load(model_id, adapter, dtype)
    print(f"loaded {model_id} in {time.time() - t0:.0f}s")
    dt = getattr(torch, dtype)
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    n_real = base.config.audio_token_id - 1  # = LID marker id = size of the real text vocabulary
    rows = json.loads(clips.read_text())
    res = []
    for r in rows:
        wav = _read_wav(r["wav"])
        t0 = time.time()
        with torch.inference_mode():
            inp = proc.apply_transcription_request(wav, language=lang)
            inp.pop("num_audio_tokens", None)
            inp = inp.to(dtype=dt)
            gen = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
            hyp = proc.decode(gen[:, inp["input_ids"].shape[1] :], skip_special_tokens=True)[0]
            tf = proc.apply_transcription_request(wav, language=lang, transcription=r["text"])
            tf.pop("num_audio_tokens", None)
            tf = tf.to(dtype=dt)
            o = model(**tf)
            nll, n_tok = _masked_nll(o.logits, tf["labels"], n_real)
        res.append(r | {"hyp": hyp, "nll": nll, "nll_hf_loss": float(o.loss), "n_tok": n_tok})
        print(f"{time.time() - t0:5.1f}s nll={nll:.5f} (HF loss {float(o.loss):.5f}) | {hyp}")
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))


@app.command()
def compare(a: Path, b: Path) -> None:
    """Exact-match rate and CER between two hypothesis files (plus each one's CER vs. reference)."""
    ra, rb = json.loads(a.read_text()), json.loads(b.read_text())
    if [x["clip_id"] for x in ra] != [x["clip_id"] for x in rb]:
        msg = "clip lists differ"
        raise ValueError(msg)
    exact = sum(x["hyp"] == y["hyp"] for x, y in zip(ra, rb, strict=True))
    import jiwer

    cer_ab = jiwer.cer([x["hyp"] or " " for x in ra], [y["hyp"] or " " for y in rb])
    refs = [x["text"] for x in ra]
    for x, y in zip(ra, rb, strict=True):
        flag = "==" if x["hyp"] == y["hyp"] else "!="
        print(f"{x['clip_id'][:24]:24} {flag} nll {x['nll']:.4f} vs {y['nll']:.4f}")
        if flag == "!=":
            print(f"   A: {x['hyp']}\n   B: {y['hyp']}")
    nll_diff = max(abs(x["nll"] - y["nll"]) for x, y in zip(ra, rb, strict=True))
    summary = {
        "n": len(ra),
        "exact_match": exact / len(ra),
        "cer_between": cer_ab,
        "max_abs_nll_diff": nll_diff,
        "cer_vs_ref_a": jiwer.cer(refs, [x["hyp"] for x in ra]),
        "cer_vs_ref_b": jiwer.cer(refs, [y["hyp"] for y in rb]),
        "wer_vs_ref_a": jiwer.wer(refs, [x["hyp"] for x in ra]),
        "wer_vs_ref_b": jiwer.wer(refs, [y["hyp"] for y in rb]),
    }
    print(json.dumps(summary, indent=1))


@app.command("check-convert")
def check_convert(mine: Path, reference: str = typer.Option("bezzam/omniasr-llm-300m-v2")) -> None:
    """Compare our converted tensors with the PR author's converted checkpoint, tensor by tensor."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    ref_dir = Path(snapshot_download(reference, allow_patterns=["*.safetensors", "*.json"]))

    def tensors(d: Path) -> dict:
        out = {}
        for f in sorted(d.glob("*.safetensors")):
            with safe_open(f, "pt") as h:
                out |= {k: h.get_tensor(k) for k in h.keys()}  # noqa: SIM118
        return out

    ta, tb = tensors(mine), tensors(ref_dir)
    print(f"keys: mine {len(ta)}, ref {len(tb)}; only mine {sorted(set(ta) - set(tb))[:5]}")
    print(f"only ref {sorted(set(tb) - set(ta))[:5]}")
    worst = max(
        ((float((ta[k].float() - tb[k].float()).abs().max()), k) for k in set(ta) & set(tb)),
        default=(0.0, ""),
    )
    shapes = [k for k in set(ta) & set(tb) if ta[k].shape != tb[k].shape]
    print(f"max abs diff {worst[0]:.3e} at {worst[1]}; shape mismatches {shapes[:5]}")


@app.command("lora-check")
def lora_check(
    model_id: str = typer.Option(...),
    clips: Path = typer.Option(...),
    rank: int = typer.Option(16),
) -> None:
    """Attach a random LoRA with plain PEFT; check backward works and merged == unmerged."""
    import torch
    from peft import LoraConfig, get_peft_model

    proc, model = _hf_load(model_id, None, "float32")
    cfg = LoraConfig(
        r=rank,
        lora_alpha=2 * rank,
        lora_dropout=0.0,
        target_modules=PEFT_TARGETS,
        init_lora_weights="gaussian",
    )
    pm = get_peft_model(model, cfg)
    pm.print_trainable_parameters()
    for n, p in pm.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=1e-3)
    r = json.loads(clips.read_text())[0]
    wav = _read_wav(r["wav"])
    tf = proc.apply_transcription_request(wav, language=LANG, transcription=r["text"])
    tf.pop("num_audio_tokens", None)
    pm.train()
    loss = pm(**tf).loss
    loss.backward()
    gn = sum(float(p.grad.norm()) for n, p in pm.named_parameters() if p.grad is not None)
    print(f"train-mode loss {float(loss):.4f}, LoRA grad norm sum {gn:.3e}")
    pm.eval()
    with torch.inference_mode():
        a = float(pm(**tf).loss)
        merged = pm.merge_and_unload()
        b = float(merged(**tf).loss)
    print(f"eval loss unmerged {a:.6f} merged {b:.6f}")


if __name__ == "__main__":
    app()
