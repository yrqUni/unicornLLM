#!/bin/bash

OUTPUT_PATH=./output_p1
mkdir -p $OUTPUT_PATH

# deepspeed main.py \
# deepspeed --hostfile=host main.py \
deepspeed --hostfile=host main.py \
   --data_input_path /data/Workspace/TEMP/DATA/MyData/data_demo.jsonl \
   --model_name_or_path /data/Workspace/TEMP/DATA/vicuna-7b-p1 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --do_eval \
   --max_seq_len 1024 \
   --learning_rate 5e-5 \
   --weight_decay 0.1 \
   --num_train_epochs 1 \
   --gradient_accumulation_steps 4 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage 2 \
   --use_lora \
   --use_fp16 \
   --deepspeed \
   --output_dir $OUTPUT_PATH \
   &> $OUTPUT_PATH/training.log
