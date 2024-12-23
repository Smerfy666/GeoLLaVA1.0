#!/bin/bash

# 街景全景图  切割变换

########### 单张图片 切割变换[成功] ###########
#CUDA_VISIBLE_DEVICES=0 python /home/gt/pky/llava/LLaVA-1.1.0/mydemo_v15/data_panorama_cut_gpu.py \
#    --data_path /workstation/pky_data/llava_test_data/streetview_test/image11189309119517025986.jpeg \
#    --output_path /workstation/pky_data/llava_test_data/streetview_test/streetview_cut \
#    --TEMP_DB_PATH 0 \
#    --output_db_filename None \
#    --db_num None

############# db数据 少量指定数量 切割变换 [成功]###########
CUDA_VISIBLE_DEVICES=0 python /home/gt/pky/llava/LLaVA-1.1.0/mydemo_v15/data_panorama_cut_gpu.py \
    --data_path /workstation/pky_data/llava_test_data/streetview_taiwan/TileDB_taiwan.db \
    --output_path /workstation/pky_data/llava_test_data/streetview_taiwan/test_result2024 \
    --TEMP_DB_PATH 0 \
    --output_db_filename TileDB_taiwan8_cut.db \
    --gpu_num 7 \
    --db_num 415,420,425,430,129,136,148,354

############# 大规模db数据 顺序 切割变换 调用GPU ###########
#python /home/gt/pky/llava/LLaVA-1.1.0/mydemo_v15/data_panorama_cut_gpu.py \
#    --data_path /workstation/pky_data/llava_test_data/streetview_taiwan10km/TileDB_taiwan10km2_test50.db \
#    --output_path /workstation/pky_data/llava_test_data/streetview_taiwan10km/cut_data_test \
#    --TEMP_DB_PATH 0 \
#    --output_db_filename taiwan10km2_cut_test.db \
#    --gpu_num 7

#nohup python /home/gt/pky/llava/LLaVA-1.1.0/mydemo_v15/data_panorama_cut_gpu.py \
#    --data_path /workstation/pky_data/llava_test_data/streetview_taiwan/TileDB_taiwan50.db \
#    --output_path /workstation/pky_data/llava_test_data/streetview_taiwan/test_result2024 \
#    --TEMP_DB_PATH 0 \
#    --output_db_filename TileDB_taiwan50_cut.db \
#    --batch_size 7000 \
#    --gpu_num 7 > /workstation/pky_data/llava_test_data/streetview_taiwan/test_result2024/log.txt 2>&1 &


############# 大规模db数据 顺序 切割变换 ###########
#CUDA_VISIBLE_DEVICES=0 python /home/gt/pky/llava/LLaVA-1.1.0/mydemo_v15/data_panorama_cut.py \
#    --data_path /workstation/pky_data/llava_test_data/streetview_taiwan/TileDB_taiwan.db \
#    --output_path /workstation/pky_data/llava_test_data/streetview_taiwan/cut_data \
#    --TEMP_DB_PATH 0 \
#    --output_db_filename TileDB_taiwan_cut4.db
##    --db_num
