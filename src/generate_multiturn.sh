#!/bin/bash

# Ensure the script exits if any command fails
set -e

# List of student models
STUDENT_MODELS=(
  "Talking-Babies/opt-sam-orpo-mamba-2048-step448"
  "Talking-Babies/opt-sam-orpo-seqlen-2048-step559"
  "Talking-Babies/mamba-sam-seqlen-2048-original"
  "Talking-Babies/opt-sam-seqlen-2048-original"
  "Talking-Babies/cpo_opt_cosmos"
  "Talking-Babies/cpo_opt_base"
  "Talking-Babies/cpo_opt_100M_2048_preprocess"
  "Talking-Babies/orpo_opt_100M_2048_preprocess"
  "Talking-Babies/orpo_opt_cosmos"
)

# Loop over turns, max lengths, and student models
for turns in 4 6 8; do
  for len in 50 100 150 200 250; do
    for student in "${STUDENT_MODELS[@]}"; do
      echo "Generating dialogues for $student with $turns turns and max length $len..."
      CUDA_VISIBLE_DEVICES=0 python generate.py \
        --teacher_model meta-llama/Llama-3.2-3B-Instruct \
        --student_model "$student" \
        --num_turns "$turns" \
        --max_length "$len" \
        --num_samples 10 \
        --output_dir ./multiturn_dialogues
    done
  done
done
