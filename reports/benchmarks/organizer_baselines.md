# Organizer baseline checkpoints on our evaluation sets

Benchmarked 2026-10-09. These are apples-to-apples numbers for the SIMBig 2026 organizers' released Puno Quechua (qxp) ASR checkpoints, measured on our evaluation roles from `/workspace/simbig/data/manifests/all.parquet`. Script: `scripts/bench_organizer.py`.

**Bottom line:** the organizers' scripted results don't hold up on the speaker-disjoint cv26 scripted test when scored against the curated (official) references. Their best cv26 checkpoint, `xls-r-cpt-qxp-cv26-validated`, gets **17.9% WER / 3.2% CER on `itest_scripted`** (17.8% on unseen sentences). That is far from the roughly 1–2% the organizers report on their older `qxp_v2` test; they have published no cv26 numbers. `whisper-base` scores lower (14.4%), but most of our scripted eval clips are probably in its training data. Their spontaneous results can't be used as baselines: almost every clip in `dev_spont` and `itest_spont` is in the organizers' spontaneous **training** split (see "Leakage" below).

## Setup

| | |
|---|---|
| Checkpoints | `QuechuaBase/xls-r-cpt-qxp-cv26-{validated,silver}`, `QuechuaBase/xls-r-qxp-cv26-{validated,silver}` (fairseq `checkpoint_best.pt`), `QuechuaBase/wav2vec2-base-qxp-finetuned`, `QuechuaBase/whisper-base-qxp-finetuned` (transformers) |
| fairseq → HF | Converted in memory to `Wav2Vec2ForCTC`. No fairseq install is needed: the pickle is loaded with stub classes and the state-dict keys are remapped. The target dictionary comes from the checkpoint's own `task_state` and matches the repo's post-`d907f55` `dict.ltr.txt`: 50 symbols, `<s>`=0 is the CTC blank, `|`=5 is the word boundary. Each waveform is layer-normalised per utterance because the checkpoints set `normalize: true`. |
| Decoding | Greedy CTC (argmax, collapse repeats, drop blank) with no LM for every wav2vec2 model. `wav2vec2-base` (group-norm) decodes one utterance at a time so padding can't affect it. Whisper decodes greedily with the prompt it was trained with (`language=spanish, task=transcribe`); clips longer than 30 s (41 in `dev_spont`, 17 in `itest_spont`) are cut into consecutive 30 s windows. |
| Scoring | `qxp_asr.metrics.wer_cer`: NFC, lowercase, punctuation stripped except `'` (`ñ` kept), then **micro** WER/CER over the whole set with jiwer. |
| References | **ours** = manifest `text`, the curated orthography of the shared task (`validated_sentences_curated.tsv` conventions; this is the official scoring reference). As a diagnostic we also score against **organizers** = their own `transcript_for_asr` from `splits_cv26`, which is the text the cv26 models were trained on. |
| Hardware | `xls-r-cpt-qxp-cv26-validated` ran on the L40S in fp32 (peak 3.8 GiB). Once the concurrent training job grew to about 44 GB, all other models ran on the CPU in fp32 with 4 threads. |
| Subsets | `unseen` = `unseen_sentence == True` in the current manifest: 278 clips in `dev_scripted`, 297 in `itest_scripted`. The manifest was rebuilt during the runs, so `summarize` takes role, reference and unseen flag from the current manifest. Clip sets and reference text did not change. |

### Conversion check

Before the cv26 runs, the converter was tested against published output. The older checkpoint `QuechuaBase/xls-r-cpt-qxp-validated` was run on the first 120 clips of the organizers' `qxp_v2` test manifest. The hypotheses were compared with the organizers' published `results/ft_cpt_validated/test/hypo.word-checkpoint_best.pt-test.txt`, which came from fairseq `infer.py` with the viterbi decoder.

- **117 of 120 hypotheses were identical.**
- WER between our hypotheses and theirs was 0.42%. The 3 that differ are off by one character (`kaskan`/`kasqan`, `manachá`/`manacha`, `uwihakuna`/`uwihaykuna`), which fits small differences in how the mp3 audio was decoded and resampled.

The conversion and decoding are therefore faithful, and the WERs below reflect the models, not the port.

## Results — WER / CER (%) against our references (official curated orthography)

| model | dev_scripted | dev_scripted (unseen) | itest_scripted | itest_scripted (unseen) | dev_spont | itest_spont |
|---|---|---|---|---|---|---|
| `xls-r-cpt-qxp-cv26-validated` | 15.96 / 2.58 | 14.37 / 2.26 | 17.85 / 3.25 | 17.83 / 3.21 | 7.08 / 1.24 | 1.42 / 0.19 |
| `xls-r-cpt-qxp-cv26-silver` | 16.64 / 2.73 | 13.93 / 2.29 | 18.90 / 3.40 | 18.35 / 3.32 | 11.66 / 1.74 | 4.19 / 0.51 |
| `xls-r-qxp-cv26-validated` | 20.05 / 3.20 | 17.50 / 2.86 | 21.23 / 3.80 | 21.83 / 3.74 | 8.78 / 1.41 | 2.46 / 0.32 |
| `xls-r-qxp-cv26-silver` | 20.18 / 3.27 | 17.59 / 2.85 | 21.77 / 3.95 | 21.22 / 3.84 | 14.45 / 2.18 | 6.67 / 0.81 |
| `wav2vec2-base-qxp-finetuned` | 21.66 / 3.54 | 20.98 / 3.44 | 22.95 / 4.29 | 22.52 / 4.22 | 40.55 / 7.39 | 14.39 / 1.80 |
| `whisper-base-qxp-finetuned` | 15.11 / 2.48 | 14.46 / 2.39 | 14.36 / 2.58 | 14.17 / 2.54 | 35.16 / 19.08 | 12.29 / 5.85 |

Clips: dev_scripted 948 (unseen 278) · itest_scripted 909 (unseen 297) · dev_spont 310 · itest_spont 304. Each cell is WER / CER.

**The spontaneous columns for the four `xls-r-*-cv26-*` rows are training-set numbers** (see Leakage). **The whisper and wav2vec2-base rows are probably contaminated on every role.**

## Diagnostic — WER / CER (%) against the organizers' own `transcript_for_asr`

| model | dev_scripted | dev_scripted (unseen) | itest_scripted | itest_scripted (unseen) | dev_spont | itest_spont |
|---|---|---|---|---|---|---|
| `xls-r-cpt-qxp-cv26-validated` | 7.68 / 1.53 | 6.88 / 1.34 | 9.99 / 2.27 | 10.92 / 2.30 | 6.25 / 1.14 | 1.12 / 0.15 |
| `xls-r-cpt-qxp-cv26-silver` | 8.92 / 1.74 | 7.68 / 1.45 | 11.54 / 2.48 | 12.39 / 2.51 | 11.13 / 1.68 | 4.07 / 0.49 |
| `xls-r-qxp-cv26-validated` | 12.58 / 2.29 | 11.79 / 2.09 | 15.03 / 3.00 | 16.03 / 2.99 | 8.02 / 1.32 | 2.19 / 0.29 |
| `xls-r-qxp-cv26-silver` | 14.03 / 2.50 | 12.68 / 2.20 | 16.41 / 3.22 | 17.07 / 3.26 | 14.00 / 2.13 | 6.51 / 0.79 |
| `wav2vec2-base-qxp-finetuned` | 14.85 / 2.69 | 14.82 / 2.70 | 16.97 / 3.49 | 16.98 / 3.46 | 40.94 / 7.47 | 14.36 / 1.80 |
| `whisper-base-qxp-finetuned` | 6.57 / 1.47 | 6.88 / 1.53 | 6.05 / 1.55 | 6.33 / 1.56 | 34.88 / 19.05 | 12.25 / 5.85 |

The organizers' references cover every eval clip, so n is the same as above.

Notes:
- **Orthography accounts for about half of the scripted error.** Our curated references and the organizers' `transcript_for_asr` differ from each other by 9.7% WER on `dev_scripted` and 9.2% on `itest_scripted` (CER about 1.1%). The models write in Common Voice spelling, which differs from the curated Puno orthography. The most common substitutions are `huq→huk`, `qulqi→qullqi`, `chhaqay→chaqay`, `alqu→allqu`, `ashkha→askha`, `ishkay→iskay`, `chaqra→chakra`, `llank'a→llamk'a` and `ashwan→aswan`. Even scored against their own text, the cv26 XLS-R models only reach about 7–11% WER on the speaker-disjoint scripted sets.
- The roughly 1–2% scripted WER the organizers report is on the old `qxp_v2` test split, scored with their own references. That split has 2,299 clips from only 5 speakers. It is speaker-disjoint from its train split, but every one of its sentences also appears in training. The cv26 split has the same property: all 909 test sentences and all 948 dev sentences occur in the cv26 train split. Yet cv26 test covers 21 speakers. Our port reproduces the published old-split hypotheses (see "Conversion check"), so the much higher cv26 error does not come from the port. It seems to come from the broader speaker set and from the orthography mismatch. The organizers have not published cv26 test numbers that we could check this against.
- `dev_scripted` is exactly the organizers' cv26 scripted `dev` (948 of 948 clips), and `itest_scripted` is exactly their cv26 scripted `test` (909 of 909). The cv26 fine-tuning cfg has `valid_subset: valid,test` with `best_checkpoint_metric: wer`, so fairseq picks `checkpoint_best.pt` by WER on the first subset (`valid`). `dev_scripted` numbers carry mild selection bias, and the test set was at least monitored during training.
- **Whisper** `dev_spont` CER (19%) comes mostly from repetition-loop insertions: 8,355 inserted characters against 51,847 reference characters, concentrated in 14 clips whose hypothesis is more than 1.5 times the reference length.

## Leakage

### cv26 splits (the four `xls-r-*-cv26-*` checkpoints)

Command: `bench_organizer.py leakage` against `/workspace/simbig/data/external/splits_cv26/{scripted,spontaneous}/train.tsv` (`client_id`).

| role | clips | speakers | speakers in org **spontaneous** train | clips from those speakers | **same clip** in org spontaneous train | speakers / clips in org scripted train |
|---|---:|---:|---:|---:|---:|---:|
| dev_spont | 310 | 4 | **3** | 307 | **307** | 0 / 0 |
| itest_spont | 304 | 6 | **4** | 290 | **290** | 0 / 0 |
| dev_scripted | 948 | 21 | 0 | 0 | 0 | 0 / 0 |
| itest_scripted | 909 | 21 | 0 | 0 | 0 | 0 / 0 |

- The leakage goes beyond shared speakers: **307 of 310 `dev_spont` clips and 290 of 304 `itest_spont` clips are the same recordings as clips in the organizers' cv26 spontaneous training split.** The cv26 models' spontaneous scores on our sets are therefore training-set scores. For example, `xls-r-cpt-qxp-cv26-validated` gets 1.4% WER on `itest_spont` but about 18% on held-out scripted speech. These scores must not be compared with our spontaneous numbers.
- The scripted roles share no speakers or clips with the organizers' scripted train split. They do share sentences with it: every `dev_scripted` and `itest_scripted` sentence also appears in their cv26 scripted train, a property of the cv26 split design.
- Speaker IDs can't be compared across corpora: scripted `client_id`s are 128-hex and spontaneous ones are 32-hex. So we can't tell from the IDs whether a scripted test speaker also appears in spontaneous training.

### Older splits (`wav2vec2-base-qxp-finetuned`, `whisper-base-qxp-finetuned`)

These two models predate cv26. Their training scripts read `data/{scripted,spontaneous}/train.tsv` and the model card describes 65 h of scripted plus spontaneous audio. The exact split isn't published. The most likely candidates are the repo's `data/splits/{validated_scripted,validated_spontaneous,silver_spontaneous}`. Overlap with those (clips / speakers):

| role | in `validated_scripted/train` | in `validated_spontaneous/train` | in `silver_spontaneous/train` | in union of train splits |
|---|---:|---:|---:|---:|
| dev_scripted (948 clips / 21 spk) | 534 / 11 | – | – | 534 |
| itest_scripted (909 / 21) | 641 / 16 | – | – | 641 |
| dev_spont (310 / 4) | – | 152 / 2 | 161 / 3 | 161 |
| itest_spont (304 / 6) | – | 177 / 3 | 304 / 6 | 304 |

If those really were the training splits, about 60–70% of our scripted eval clips and all of `itest_spont` were seen in training. Their numbers above are therefore optimistic too.

## Commands

On the GPU box (`ssh -p <SERVER_A_PORT> root@<SERVER_A_HOST>`):

```bash
# env (fresh, uv only)
mkdir -p /workspace/simbig/envs/orgbench && cd /workspace/simbig/envs/orgbench
uv init --bare --python 3.12
uv add "torch==2.8.0" "transformers>=4.45" soundfile polars jiwer omegaconf huggingface_hub numpy typer
#   resolved: torch 2.8.0, transformers 5.19.0

# checkpoints -> /workspace/simbig/envs/orgbench/checkpoints/<repo>
uv run python -c "
from huggingface_hub import snapshot_download
for r in ['whisper-base-qxp-finetuned', 'wav2vec2-base-qxp-finetuned',
          'xls-r-cpt-qxp-cv26-validated', 'xls-r-cpt-qxp-cv26-silver',
          'xls-r-qxp-cv26-validated', 'xls-r-qxp-cv26-silver',
          'xls-r-cpt-qxp-validated']:  # last one only for the conversion check
    snapshot_download('QuechuaBase/' + r, local_dir='checkpoints/' + r)"

# scripts/bench_organizer.py copied to run/; qxp_asr imported read-only from the repo
export PYTHONPATH=/workspace/simbig/repo/src
uv run python run/bench_organizer.py run checkpoints/xls-r-cpt-qxp-cv26-validated \
    --name xls-r-cpt-qxp-cv26-validated --out-dir run/out            # GPU, fp32
for m in wav2vec2-base-qxp-finetuned whisper-base-qxp-finetuned xls-r-cpt-qxp-cv26-silver \
         xls-r-qxp-cv26-validated xls-r-qxp-cv26-silver; do
  uv run python run/bench_organizer.py run checkpoints/$m --name $m --out-dir run/out \
      --device cpu --threads 4 --batch-seconds 60
done
uv run python run/bench_organizer.py summarize --out-dir run/out    # -> run/out/summary.csv
uv run python run/bench_organizer.py leakage
```

The conversion check used a throwaway script, since deleted. It imported `run_fairseq` from `bench_organizer.py`, decoded the first 120 clips of the organizers' `data/manifests/finetune/qxp_v2/test.tsv` from our 16 kHz WAVs, and compared the output with their `hypo.word` file by utterance index.

Per-clip hypotheses (`*.preds.parquet`, dev/itest roles only) and `summary.csv` are in `reports/benchmarks/organizer_preds/`.
