# Spike: omniASR-LLM under transformers + Unsloth + vLLM for GSPO/GRPO (2026-10-09)

**Question.** Can our omniASR_LLM_7B_v2 + rank-16 decoder LoRA move to Hugging Face transformers
(+ Unsloth LoRA training) and vLLM (fast sampling), so RL fine-tuning (GSPO/GRPO) is cheap?

**Answer: no-go for this stack now; go for GSPO implemented directly in our fairseq2 code.** The
transformers port is numerically exact and our weights and LoRA convert cleanly. Unsloth and the vLLM
PR, however, add a lot of risk and do not speed up the expensive part (the frozen 7B encoder).

Everything ran on CPU (`CUDA_VISIBLE_DEVICES=""`, `nice 15`, 4 threads, `oom_score_adj=1000`) in
`/workspace/simbig/envs/hfspike` (uv). The training run, `envs/omni`, `repo`, `artifacts/runs` and
`logs` were not touched; `envs/omni` was only used read-only through its venv python. Script:
`scripts/spike_hf_omniasr.py` (pure-torch fairseq2→HF converter, parity runners, LoRA checks).

## 1. transformers PR #43265 (head `e3424489`, open, updated 2026-10-08)

- Installs with `uv add "transformers @ git+…@refs/pull/43265/head"` (5.19.0.dev0, torch 2.14.1).
- Architecture: Voxtral-style. `audio_tower` (wav2vec2) → `multi_modal_projector` → `LlamaModel`.
  Audio frames are scattered over `<extra_id_1>` placeholders. Prompt: `audio | <extra_id_0> (LID
  marker) | <|lang:xxx|> | <s>`. fairseq2's separate `lang_embeddings` table is folded into
  `embed_tokens` as 1694 language tokens (index = 10290 + fairseq2 row).
- **The language-ID token is supported.** `language="qxp_Latn"` maps to `<|lang:qxp_latn|>` (id 11535,
  which is fairseq2 lang row 1245). Training labels come from `apply_transcription_request(...,
  transcription=…)`.
- Decoder q/k rows are permuted from the interleaved RoPE layout to the half-split layout.
  `lm_head`/`embed_tokens` are padded with 2 + 1694 rows.
- Our pure-torch converter (no fairseq2 import, mmap) reproduces the author's
  `bezzam/omniasr-llm-300m-v2` exactly: 533/533 tensors, max abs diff 1.3e-7 (weight-norm conv).

**Parity on 300M_v2.** 10 dev clips (5 dev_spont, 5 dev_scripted, 3.4–9.9 s), fp32, CPU, greedy,
`qxp_Latn`. Reference is fairseq2 `ASRInferencePipeline` (nbest=1) plus teacher-forced NLL of the
reference text.

| HF model vs fairseq2 | exact match | CER between | max \|Δ mean NLL\| per clip |
|---|---|---|---|
| PR conversion as-is | 10/10 | 0.000 | 2.3e-2 |
| + fairseq2's LID-marker row (see bug 1) | 10/10 | 0.000 | **6.5e-6** |
| + our 7B LoRA v2 on the (identical) decoder, PEFT adapter | 10/10 | 0.000 | **6.2e-6** |
| + same LoRA merged into W | 10/10 | 0.000 | **6.2e-6** |
| **7B_v2 + our LoRA v2 merged, bf16, 4 clips** | 4/4 | 0.000 | 2.3e-3 (bf16 noise) |

Both systems score CER 1.96 % / WER 14.3 % against the references on the 10 clips (7B+LoRA: 1.8 % / 11.1 % on 4 clips).
The LoRA check is meaningful: the adapter changes fairseq2 NLL by up to 0.02, and HF tracks it to 6e-6.

**Gotchas found (relevant to anyone computing RL log-probs):**
1. *LID-marker row bug in omnilingual-asr.* `300m_v2`/`1b_v2`/`3b_v2` call `_7b_llama_v2()` and then
   replace `wav2vec2_asr_config`, which resets `target_vocab_size` to 9812. Their LID marker is
   therefore `text_frontend` row 9812, a real CJK token, not row 10288. `7b_v2` keeps 10288, so
   **our 7B is unaffected** and matches the HF port out of the box (confirmed above). The converter
   has `--lid-row 9812` for the smaller v2 models.
2. *Zero rows in `lm_head`.* The 1696 added rows score logit 0 and take softmax mass in
   `forward()`/`loss`. `generate()` hides this with `suppress_tokens`. Example: HF loss 0.00404 vs
   true NLL 0.00291. Log-probs must be computed on `logits[..., :10288]`
   (`_masked_nll` in the script).
3. Passing `sampling_rate=` to `apply_transcription_request` silently drops `labels` (kwargs override
   `processor_kwargs`).
4. The converted audio config carries `layerdrop=0.1` and dropout 0.1. Keep `audio_tower.eval()`
   when training only the decoder.

## 2. Unsloth

- `uv add unsloth` resolves only to **unsloth 2025.9.5** (a year old), because current **2026.10.3
  pins `transformers<=5.17.0`** while the PR is 5.19.0.dev0. 2026.10.3 installs only with a uv
  `override-dependencies` on transformers.
- Neither version imports without a GPU: `NotImplementedError: Unsloth currently only works on
  NVIDIA GPUs and Intel GPUs.` (raised in `unsloth/__init__.py` and `unsloth_zoo/__init__.py`).
  Nothing can be tested while the GPU is busy.
- Static reading of 2026.10.3: `omniasr` is not in `FastLanguageModel`'s dispatch list (llama,
  mistral, gemma, qwen…), so it is delegated to `FastModel`/`FastBaseModel`. That is the generic
  AutoModel/`AutoModelForMultimodalLM` + PEFT path with torch.compile and Unsloth gradient
  checkpointing; **none of Unsloth's hand-written LLaMA kernels apply**. Its GRPO
  `fast_inference` routes LoRA through vLLM `lora_request`, which the OmniASR vLLM model does not
  support (§3).
- **Plain transformers + PEFT works** (CPU): LoRA r=16 on `language_model.layers.*.{q,k,v,o,gate,up,down}_proj`
  gives 10,272,768 trainable parameters, identical to our fairseq2 LoRA. Backward runs, and merged ==
  unmerged to 1e-6. TRL 0.29.1 `GRPOTrainer` has GSPO (`importance_sampling_level="sequence"`)
  but **no audio input path** (images/pixel_values only), so rollouts and collation would be custom
  anyway.

## 3. vLLM PR #36864 (head `fd6f7fc5`, open, 161 commits behind main)

- **Installable without compiling CUDA.** In a separate project (`hfspike/vllm-env`), use
  `VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT=32fbfa15… uv add "vllm @ git+…@fd6f7fc5"`
  (merge-base wheel exists on wheels.vllm.ai). It took 3 min and pins torch 2.13.0 / transformers
  5.19.0. It imports on CPU and `OmniAsrForConditionalGeneration` is registered. GPU inference was
  not attempted.
- **7B: supported by config.** Encoder dims come from `config.json` (2048-d, 128 layers, 16 heads,
  ffn 8192); ziywang50 publishes a 7B (v1) conversion.
- **LID: supported.** `language=` goes through `lang_lookup.json` + `n_special_tokens`. If no
  language is given it defaults to `eng_Latn`, not the language-agnostic row.
- **LoRA: not supported** (`supports_lora(OmniAsr…) == False`; the class only implements
  Transcription/MultiModal). RL would need merged weights pushed into the engine at every step.
- It uses **its own checkpoint format**: fairseq2 key names + `weight_g/weight_v`, permuting q/k at
  load time, with its own `model_type: omniasr_llm` config. HF-port checkpoints cannot be loaded.
  Our v2 7B needs a hand-written config (`target_vocab_size` 10288, text vocab 10289,
  `lid_marker_token_id` 10288; the published 7B config is v1 with 9818), plus `lang_lookup.json`
  and `tokenizer.model`. Other limits: one audio per request, ≤40 s, no encoder padding mask.

## 4. Converting our 7B + LoRA

- **Done and validated:** `convert --checkpoint …/omniASR-LLM-7B-v2.pt --lora llm7b_lora_v2/best_trainable.pt
  --mode merge|peft`. The 31 GB fp32 checkpoint is mmap-read and written as 8 bf16 shards (15 GB).
  84 projections are merged. With `--mode peft`, `lora_B` of q/k gets the same RoPE row permutation.
- Key map: `llama_decoder.layers.N.self_attn.{q,k,v,output}_proj` →
  `model.language_model.layers.N.self_attn.{q,k,v,o}_proj`;
  `ffn.{gate,inner,output}_proj` → `mlp.{gate,up,down}_proj`. `lora_a`/`lora_b` become
  `lora_A`/`lora_B.weight`.
- RAM: conversion is shard-sized (a few GB). On CPU in bf16, fairseq2 7B inference peaked at
  29.4 GB RSS and HF 7B at 15.6 GB. vLLM format would be the fairseq2 state dict with LoRA merged,
  needing no permutation.

## 5. Effort and risk to get 7B + LoRA + GSPO running on this stack

| Piece | Status | Remaining effort | Risk |
|---|---|---|---|
| HF conversion of 7B + LoRA | done, exact | 0 | low |
| HF/PEFT trainer | works on CPU | 1–2 d: custom audio GSPO loop (TRL has no audio path), masked log-probs | medium |
| Unsloth | cannot import without GPU; version conflict; generic path only | ? (GPU needed to even test) | high, little benefit |
| vLLM sampler | installs; no LoRA; own format | 2–3 d: v2 config/lang table, weight sync per step, colocated memory (two 7B copies on one 48 GB L40S; our SFT already uses 19 GB) | high (unmerged PR, behind main, torch 2.13 vs 2.14 env split) |
| **Total** | | **≈4–6 days** | **high** |

**Compute argument.** Per 10 s clip, the frozen 7B encoder costs ≈ 2·6.5e9·500 ≈ 6.5 TFLOP.
Decoder prefill (1.3B params, ~560 positions) costs ≈ 1.4 TFLOP, and G=8 samples × ~60 tokens are
cheap, memory-bound KV-cache steps. vLLM only speeds up the small part. A vLLM + trainer split also
runs the encoder twice (once in vLLM, again in the trainer for log-probs).

## Recommendation

**No-go** for Unsloth + vLLM. **Go** for GSPO in our fairseq2 trainer (`train_llm.py` + `qxp_asr.lora`):

1. Run the encoder once per clip (frozen, `no_grad`).
2. Draw G temperature samples from the decoder with incremental decoding, reusing the
   `Wav2Vec2LlamaBeamSearchSeq2SeqGenerator` state-bag pattern.
3. Reward = −CER/WER (optionally the joint CTC/LM score).
4. Compute sequence-level GSPO ratios from teacher-forced log-probs over the cached encoder output.

Estimated effort is ~1–1.5 days at low–medium risk. It reuses the data pipeline, eval and checkpoint
format, so `infer_llm.py` and `joint_rescore.py` keep working unchanged.

Keep the validated converter for exporting the final model to HF once PR #43265 merges. If we ever
switch, apply gotchas 1–4.
