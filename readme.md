# GeoLLaVA 1.0

GeoLLaVA 1.0 是一个基于 LLaVA（Large Language and Vision Assistant）的模型，通过结合地理信息和街景数据进行训练，专注于地理信息工程和地理空间任务。此模型可以处理和分析街景数据，为地理信息领域的应用提供智能支持。

## 主要特点

- **基于 LLaVA 的架构**：继承了 LLaVA 的强大功能，并针对街景数据和地理信息任务进行了微调。
- **街景数据集训练**：使用自收集的街景数据进行了模型训练，使其能够更好地理解和分析街景图像与地理信息之间的关系。
- **多模态能力**：结合了视觉信息与文本信息，支持多种地理信息任务，如位置识别、地标分类和路线规划等。

## 功能与应用

- **街景图像分析**：通过处理街景图像，为用户提供关于城市规划、交通流量和环境变化等方面的分析。
- **地理信息提取**：从街景图像中提取重要的地理特征，如道路标识、建筑物和地标等。
- **位置推理与问答**：用户可以通过文本问题与模型交互，获取关于地理位置、街道和地标的信息。
  
## 安装

### 环境要求

- Python 3.x
- PyTorch >= 1.9.0
- Transformers
- OpenCV
- 其他依赖项（见`requirements.txt`）

### 安装步骤
- **克隆本仓库**：
   ```bash
   git clone https://github.com/yourusername/GeoLLaVA.git
   cd GeoLLaVA
### 安装依赖
`   pip install -r requirements.txt`
### 训练数据准备
  `--data_dir /GeoLLaVA1.0/datasets `
- 街景视觉-指令跟随数据：GeoLLaVA_Instruct1054.json
- 街景图像（训练）：train_images.zip
- 街景图像（测试）：train_images1k.zip
- 指令（可自主修改）：query.txt
- 测试图像真实标注：test1k_gt.xlsx
- 转换全景图脚本：
`   python ./demo/data_panorama_cut_gpu.py`
### LoRA微调模型
- GeoLLaVA 在 1 个 A100 GPU 上进行训练，配备 80GB 内存。
- 启动训练脚本
    ```bash
    deepspeed llava/train/train_mem.py \
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

- ` --mm_projector_type mlp2x_gelu: MLP跨模态连接器`
- ` --vision_tower openai/clip-vit-large-patch14-336: 视觉编码器`
### 使用方法
- **加载预训练的GeoLLaVA1.0模型**：
   ```bash
   from llava.model.builder import load_pretrained_model
  from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
   tokenizer, model, image_processor, context_len = \
        load_pretrained_model(args.model_path, args.model_base, model_name)
- **使用模型进行单次推理**：
   ```bash
  python ./llava/serve/cli.py \
    --model-path ./LLaVA/GeoLLaVA-finetune_lora \
    --image-file "path/to/streetview_image.jpg" \
  
  input: 'Infer the elements present in the scene based on this image.'

- **使用模型进行单次推理**：
    
  ```bash
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

### 贡献
欢迎贡献代码，改进模型或提出新的功能建议。请确保在提交任何代码前运行测试，并遵循本项目的编码规范。
### 许可证
此项目采用 MIT 许可证。
### 联系信息
如有任何问题或建议，欢迎通过 GitHub Issues 或邮件联系我









