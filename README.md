# Puno Quechua ASR: When to Adapt the Speech Encoder, and Verification as the Bottleneck

**Irving Ernesto Quezada Ramírez** (Independent Researcher, research@irvingernesto.com)

This repository holds the code, experiment scripts, paper source and talk-video pipeline for our system in **Task 1 of the SIMBig 2026 shared task**: speech recognition for Puno Quechua (ISO 639-3 `qxp`), built only from the provided Common Voice data (Scripted Speech 27.0, Spontaneous Speech 5.0) and public pretrained models.

- Paper (LNCS, 8 pages): LaTeX source in [`docs/paper/`](docs/paper/main.tex)
- Trained weights (LoRA adapters, fine-tuned CTC model): [huggingface.co/Thermostatic/simbig2026-puno-quechua-asr](https://huggingface.co/Thermostatic/simbig2026-puno-quechua-asr)
- Talk video (15 min): pipeline in [`docs/video/`](docs/video/)
- Every number in the paper: [`reports/RESULTS.md`](reports/RESULTS.md) (full experiment log, including negative results) and [`reports/paper_stats.md`](reports/paper_stats.md) (all intervals)

## Findings

1. **The public splits reward memorization.** The official splits reuse every prompt sentence between training and test. A CTC model with a word 4-gram on the training prompts is best on seen sentences (dev 6.0% WER) and worse than no language model on held-out ones (18.7% vs 14.6%). The shared-task test reads new sentences, so we select systems on held-out sentences and speakers, and test on an out-of-domain (OOD) radio drama.
2. **A provided-data-only system that beats all 37 public checkpoints out of domain.** Hybrid rescoring of LoRA-adapted omniASR-LLM-7B, CTC and perturbed-audio candidates reaches 14.1% WER on the radio drama. The best public checkpoint gets 20.9%.
3. **Where to adapt an LLM-ASR model depends on the model.** LoRA on the speech encoder instead of the LLM decoder cuts held-out-sentence WER for omniASR-LLM-7B (20.2 to 16.4) and for Whisper large-v3, but not for the 300M or 1B omniASR encoders, even with as many adapter parameters. All three pretrained encoders already separate Quechua stop contrasts (ejective, aspirated, uvular vs velar) at 94-97% under linear probes; only the 7B encoder moves substantially under adaptation. In the 7B, the gain needs the bottom quarter of the encoder, survives truncation to 4 of 16 singular directions per projection, and leaves the N-best oracle unchanged while ranking candidates better.
4. **Verification is the bottleneck.** Sampling and perturbed audio find better transcripts than beam search, and our candidate pools already contain transcripts with half our WER. The reranker orders candidate pairs correctly only 71-78% of the time. RL on labelled data (GSPO, MWER, Dr. GRPO) did not beat supervised fine-tuning.

Main results (WER %, dev-mode systems unless noted; full table in the paper):

| System | dev unseen | itest unseen | dev spont | itest spont | OOD |
|---|---|---|---|---|---|
| omniASR-LLM-7B zero-shot (no fine-tuning by us) | 21.3 | 27.9 | 14.3 | 7.3 | 23.3 |
| LLM-7B + decoder LoRA (v2) | 13.1 | 20.2 | 14.9 | 6.6 | 15.1 |
| LLM-7B + enc+dec LoRA | 10.9 | 16.0 | 14.6 | 6.8 | 15.4 |
| Hybrid rescoring + perturbed views (full system) | 9.7 | 16.9 | 13.5 | 6.2 | 13.7 |
| + enc+dec LoRA as scorer | 9.2 | 15.5 | 13.5 | 6.2 | 13.4 |
| Submission (final mode) | - | - | 13.8 | - | 14.1 |

## Repository layout

- `src/qxp_asr/`: the library.
  - Text normalization and metrics; the speaker- and sentence-disjoint evaluation protocol (`splits.py`).
  - Data loading and augmentation; omniASR helpers; N-best, sampling and perturbed-view generation (`nbest.py`).
  - LoRA (`lora.py`); paired bootstrap, pigeonhole bootstrap, Holm and sign tests (`stats.py`).
- `scripts/`: command-line entry points. Each file documents its method and the papers it builds on.
  1. Data: `prepare_audio.py`, `build_manifest.py`, `eda.py`.
  2. CTC branch: `train_ctc.py`, `merge_weights.py` (checkpoint averaging, WiSE-FT).
  3. LLM-ASR branch: `train_llm.py` (decoder, encoder or enc+dec LoRA), `pseudo_label.py` (ROVER self-training), `infer_llm.py`.
  4. Candidates and rescoring: `build_lm.py`, `ctc_nbest.py`, `llm_candidates.py` (`--views`), `joint_rescore2.py`, `make_submission.py`.
  5. Analyses:
     - where to adapt: `train_whisper.py`, `encoder_probe.py` (probes, CKA), `lora_geometry.py`, `lora_surgery.py` (spectral truncation, seed averaging), `adaptation_mechanism.py` (coverage vs ranking);
     - verification and RL: `verifier_curve.py`, `view_scaling.py`, `train_rl.py` (GSPO / MWER / Dr. GRPO / SFT control);
     - statistics: `paper_stats.py`, `compare_runs.py`.
  6. Paper figures: `scripts/paper_figs/` (`bash scripts/paper_figs/make_all.sh`).
- `scripts/runs/`: the exact experiment scripts, numbered in the order they were run. Server addresses are replaced by placeholders (`<SERVER_A_HOST>` and so on).
- `docs/LITERATURE.md`: the annotated literature survey. Every technique used or tried is cited there.
- `docs/paper/`: paper source (`latexmk -pdf main.tex`).
- `docs/video/`: the talk video.
  - Script: `script.md`.
  - Manim scenes: `scene_s*.py`, `style.py`.
  - Local voice-over recorder: `recorder.py`, a browser teleprompter that records one take per scene with sentence taps.
  - Narration processing: `narration.py`.
  - Rendering and assembly: `assemble.sh`, `finish.sh` (music bed and subtitle band).
- `reports/`: results, figure data, EDA, and benchmark outputs of public checkpoints. The benchmark outputs contain hypotheses only, no reference transcripts.

## Reproducing

Environments are managed with [uv](https://docs.astral.sh/uv/).
- **Analysis code:** runs on CPU in the root project (`uv sync`).
- **Training and decoding:** need a separate environment with fairseq2 and `omnilingual-asr`, a CUDA GPU (about 48 GB for the 7B model), and KenLM built with `MAX_ORDER=12`.
- **Paths:** they default to `/workspace/simbig`; set `QXP_ROOT` to override.

**Data are not included.**
- Download the Common Voice Puno Quechua releases from the Mozilla Data Collective.
- Get the shared-task material (curated references, the out-of-domain audio) from the organizers.
- The shared-task test set is covered by a data-use agreement. Nothing from it is in this repository: no audio, no transcripts and no hypotheses.

```bash
uv sync                          # analysis / CPU environment
uv run ruff check . && uv run pytest
bash scripts/paper_figs/make_all.sh   # paper figures from reports/
```

## Talk video

The video is rendered from code. You need system packages: ffmpeg, Cairo and Pango (for Manim), and the Inter font. Then:

```bash
cd docs/video
uv sync
mkdir -p voices && (cd voices && uv run python -m piper.download_voices en_US-ryan-high)
uv run python narration.py                # narration (Piper TTS) and out/timing.json
uv run python recorder.py                 # optional: record your own voice, http://127.0.0.1:8765
uv run python narration.py --recordings   # ... then use your takes where recorded
bash assemble.sh                          # render all scenes at 1080p60 and mux them with the narration
curl -L -o music/clean_soul.mp3 --create-dirs \
  "https://incompetech.com/music/royalty-free/mp3-royaltyfree/Clean%20Soul.mp3"
bash finish.sh                            # music bed + subtitle band -> out/simbig2026_qxp_talk_final.mp4
```

Every animation is synchronized to the start of each narration sentence (`out/timing.json`), so the video re-times itself to any voice.

## Credits

- **Models:** Omnilingual ASR (Meta) and Whisper (OpenAI). `src/qxp_asr/nbest.py` adapts code from facebookresearch/omnilingual-asr (Apache-2.0).
- **Video:**
  - Animated with [Manim Community](https://www.manim.community/).
  - Map data: [Natural Earth](https://www.naturalearthdata.com/) (public domain).
  - Music: "Clean Soul" by Kevin MacLeod ([incompetech.com](https://incompetech.com)), licensed under [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/). The music file itself is not redistributed here.
- **Compute:** GPU credits from Lium.

## Use of AI assistance

Claude Opus 5.5 (Anthropic), via Claude Code, was used to:
- implement the code and run the experiments;
- search the literature;
- draft and edit the paper text;
- make the figures and the video's script and animations.

The narration is the author's own voice. All results were produced by the code in this repository and checked by the author, who takes full responsibility for the work.

## License

Apache License 2.0 (see `LICENSE`).
