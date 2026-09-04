#!/bin/bash
#SBATCH --job-name=aoa_analyze
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-5
set -euo pipefail

# CKPT_DIR, RESULT_DIR, AOA_NORMS, SCORED from the environment
RUNS=(curriculum_seed42 curriculum_seed43 curriculum_seed44 \
      flat_shuffle_seed42 flat_shuffle_seed43 flat_shuffle_seed44)
RUN=${RUNS[$SLURM_ARRAY_TASK_ID]}

python -u scripts/06_trajectory_analysis.py \
  --checkpoints_dir  "$CKPT_DIR/$RUN" \
  --aoa_norms        "$AOA_NORMS" \
  --scored_sentences "$SCORED" \
  --out_dir          "$RESULT_DIR/$RUN"
