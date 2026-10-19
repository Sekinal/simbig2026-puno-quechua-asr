#!/usr/bin/env bash
# Experiment 29 (server A, alongside exp27a/28): LoRA fine-tune OmniVoice (k2-fsa/OmniVoice) on the
# provided labelled training clips, so its text-conditioned masked-token likelihood
# (scripts/tts_score.py) can tell near-miss Quechua transcripts apart; zero-shot it is at chance
# (~50% pairwise on OOD). Recipe: OmniVoice examples/run_finetune_lora.sh (rank 16, all Qwen3 proj
# layers, audio embeddings/heads trained), one GPU, 3000 steps of 4096 tokens x 2 accumulation.
# Gate afterwards: pairwise accuracy on OOD and dev_spont candidates vs lp_v2 (77% OOD).
set -euo pipefail
OV=/workspace/simbig/ext/OmniVoice; D=$OV/data/qxp; E=$OV/exp/qxp_lora
export HF_HOME=/workspace/simbig/cache/hf PYTHONPATH=$OV
cd /workspace/simbig/repo
/workspace/simbig/envs/omni/.venv/bin/python scripts/tts_finetune_data.py --out $D
cd $OV
for split in train dev; do
    [ -s $D/tokens/$split/data.lst ] && continue
    .venv/bin/python -m omnivoice.scripts.extract_audio_tokens --input_jsonl $D/$split.jsonl \
        --tar_output_pattern "$D/tokens/$split/audios/shard-%06d.tar" \
        --jsonl_output_pattern "$D/tokens/$split/txts/shard-%06d.jsonl" \
        --nj_per_gpu 2 --shuffle True
done
cat > $D/data_config.json <<EOF
{"train": [{"manifest_path": ["$D/tokens/train/data.lst"]}],
 "dev": [{"manifest_path": ["$D/tokens/dev/data.lst"]}]}
EOF
.venv/bin/python - <<EOF
import json
c = json.load(open("examples/config/train_config_finetune_lora.json"))
c.update(steps=3000, batch_tokens=4096, gradient_accumulation_steps=2, eval_steps=500,
         save_steps=500, logging_steps=50, num_workers=2)
json.dump(c, open("$D/train_config.json", "w"), indent=1)
EOF
.venv/bin/accelerate launch --num_processes 1 --gpu_ids 0 -m omnivoice.cli.train \
    --train_config $D/train_config.json --data_config $D/data_config.json --output_dir $E
echo "EXP29 DONE"
