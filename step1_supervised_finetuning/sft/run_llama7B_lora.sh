#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Note that usually LoRA needs to use larger learning rate
OUTPUT_PATH=./output
mkdir -p $OUTPUT_PATH

# lora_module_name="q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj"
lora_module_name="q_proj,k_proj"
echo ${lora_module_name}

deepspeed main.py \
   --data_input_path /data/Workspace/unicornPrivate/step1_supervised_finetuning/sft/MyData/data_demo.jsonl \
   --model_name_or_path ckpt/decapoda-research-llama-7b-hf \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --do_eval \
   --max_seq_len 512 \
   --learning_rate 5e-5 \
   --weight_decay 0.1 \
   --num_train_epochs 1 \
   --gradient_accumulation_steps 32 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage 2 \
   --lora_dim 8 \
   --lora_alpha 32 \
   --lora_dropout 0.05 \
   --lora_module_name ${lora_module_name} \
   --only_optimize_lora \
   --deepspeed \
   --output_dir $OUTPUT_PATH \
   &> $OUTPUT_PATH/training.log
