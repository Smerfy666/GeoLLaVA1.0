#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:

########### DO NOT CHANGE ###########
python ./llava/eval/run_llava_semantic_gputest2.py \
    --model-path /workstation/LLaVA/LLaVA-v1.5-13b \
    --img_folder_path ./datasets \
    --query_file ./datasets/query.txt \
    --output_dir /path/to/result_file \
    --gpusid 0,4,5,6 \
    --write_size 4 \
    --batch_size_pergpu 200 \
    --result_gater 1 \
    --img_type streetview