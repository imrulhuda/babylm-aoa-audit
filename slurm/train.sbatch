#!/bin/bash
#SBATCH --job-name=aoa_train
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=09:00:00
#SBATCH --array=0-5
set -euo pipefail

# CORPUS_DIR, CACHE_DIR, CKPT_DIR, TOKENIZER, MAX_WORDS, EPOCHS from the environment
CONDS=(curriculum curriculum curriculum flat_shuffle flat_shuffle flat_shuffle)
SEEDS=(42 43 44 42 43 44)
C=${CONDS[$SLURM_ARRAY_TASK_ID]}
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}

python -u scripts/05_train.py \
  --corpus     "$CORPUS_DIR/${C}_train.txt" \
  --cache_path "$CACHE_DIR/${C}.pt" \
  --out_dir    "$CKPT_DIR/${C}_seed${S}" \
  --run_name   "${C}_seed${S}" \
  --tokenizer_path "$TOKENIZER" \
  --max_words  "$MAX_WORDS" \
  --epochs     "$EPOCHS" \
  --seed       "$S"
