#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:
# v1.5 lora fintune 还没出
################## VICUNA ##################
PROMPT_VERSION=v1

CUDA_VISIBLE_DEVICES=0 deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --model_name_or_path ./llama/$MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path ./datasets/GeoLLaVA_Instruct1054.json \
    --image_folder ./datasets/train_images \
    --vision_tower ./openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./mm_projector/GeoLLaVA-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/GeoLLaVA-finetune_lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 4 \
    --report_to wandb
