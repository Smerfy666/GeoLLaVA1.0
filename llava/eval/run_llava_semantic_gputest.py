# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
import torch
import argparse
# 检查 CUDA 是否可用
if torch.cuda.is_available():
    # 初始化 CUDA
    torch.cuda.init()
import torch.nn as nn
import torch.multiprocessing as t_mp
import os
import sys
import math
import time
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


import requests
from PIL import Image
from io import BytesIO
import io
import json
import random
import sqlite3


# QUERY文件读取后
# query_file_ = {
#     'elements': [
#         'What are the elements in the image',
#         'What type of scene is in the image'
#     ],
#     'type': [
#         'What type of scene is in the image',
#     ],
#   ...}

# 结果json文件示例
# json文件
# {
#     'img1': {
#         'elements': 'elements_result_1',
#         'type': 'type_result_1',
#         'texture': 'texture_result_1',
#         'building': 'building_result_1',
#         'spatial': 'spatial_result_1'
#     }
#}
#{
#     'img2': {
#         'elements': 'elements_result_1',
#         'type': 'type_result_1',
#         'texture': 'texture_result_1',
#         'building': 'building_result_1',
#         'spatial': 'spatial_result_1'
#     }
# }
# ...


# 获取gpu设置信息
def get_gpu(gpusid):
    gpu_ids = [int(gpu_id) for gpu_id in gpusid.split(",")]  # 解析 GPU ID 列表[0, 1, 2, 3]
    num_gpus = len(gpu_ids)
    return gpu_ids, num_gpus

# 获取正在运行的gpu信息
def get_current_gpu_info():
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        return current_device, device_name
    else:
        return None, "No GPU available."

# 数据库连接和游标管理的上下文管理器，确保资源的正确释放
class DBContextManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self.cursor = None

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        return self.cursor

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()


# 获取数据类型
def get_path_info(path):
    # 给定的路径是一个文件夹
    extensions = set()  # 空集合储存扩展名
    for root, dirs, files in os.walk(path):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()
            extensions.add(ext)
    if '.db' in extensions:
        data_type = 'db' # 优先检测
    elif any(ext in {'.jpeg', '.jpg', '.png'} for ext in extensions):
        data_type = 'img'
    else:
        print("没有符合条件的 '.db', '.jpeg', '.jpg', '.png' 图像")
        sys.exit()  # 终止程序执行

    return data_type


# 加载图像文件，将图像文件转换为 RGB 模式的 PIL 图像并返回
def load_image(image_data):
    if isinstance(image_data, str):  # 如果输入是文件路径
        if image_data.startswith('http') or image_data.startswith('https'):
            response = requests.get(image_data)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_data).convert('RGB')
    elif isinstance(image_data, bytes):  # 如果输入是图像数据的字节流
        image = Image.open(BytesIO(image_data)).convert('RGB')

    return image

#  ****query处理类函数*****

# 加载预设的query txt文件
def load_query_file(query_file):
    if query_file is not None:
        with open(query_file, 'r') as f:
            lines = f.readlines()
        # 从文本文件中解析查询
        query_file_ = {}  # 创建字典
        category = None
        for line in lines:
            line = line.strip()
            if line.endswith(":"):
                category = line[:-1]  # 去掉冒号，提取内容 elements...
                query_file_[category] = []  # 创建键
            elif line:
                query_file_[category].append(line)
        print('query_file_：{}'.format(query_file_))
    else:
        query_file_ = {
            'elements': ['Analyze the elements and their types present in this image scene.'],
            'type': ['Infer the type of scene in the image based on this image.'],
            'texture': ['Analyze the ground texture in detail based on this image.'],
            'building': ['Provide a detailed description of the building details in the given image.'],
            'spatial': ['Provide a detailed description of the spatial relationships between various elements in the image scene.'],
        }
        print('query_file is None, default query_file_：{}'.format(query_file_))
    return query_file_

# 根据数据类型，返回随机的query集合
def select_queries(queries_file, selected_categories):
    selected_queries = {category: [] for category in selected_categories}
    for category in selected_categories:
        category_queries = queries_file.get(category, [])
        if category_queries:
            selected_query = random.choice(category_queries)
            selected_queries[category] = [selected_query]

    return selected_queries

# 构建 输入query集合
def get_queries(query_file_,img_type):
    if img_type is not None:
        if img_type in ["geoimg", "streetview"]:
            query_categories = ["elements", "type", "building", "spatial"]
            queries = select_queries(query_file_, query_categories)
        elif img_type == "remote":
            query_categories = ["elements", "type", "texture", "building", "spatial"]
            queries = select_queries(query_file_, query_categories)
    else:
        queries = query_file_
        print(f'query_file or img_type is None')
    # 随机选择queru 覆盖默认查询
    # print(f'queries = {queries}')
    return queries

# 一个 图片-query 数据对 推理函数
def img_query_inferece(args, qs, image_tensor, model, tokenizer):
    # 启用了图像的起始和结束标记，将这些标记添加到查询文本的开头，以便模型能够理解图像和文本的连接
    # 没有启用图像的起始和结束标记，只将默认图像标记添加到查询文本前面
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    # 根据确定的对话模式创建对话模板 conv 的副本，以便在其上附加查询和生成答案。
    conv = conv_templates[args.conv_mode].copy()
    # 将查询文本添加到对话模板的第一个角色0（通常是用户），第二个角色1（通常是系统）
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    # 获取整个对话模板的提示，将用户和系统消息合并成一个输入提示
    prompt = conv.get_prompt()
    # 使用分词器将输入提示转换为模型所需的输入ID。然后，它创建一个包含这些输入ID的张量，并将其移至 GPU
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    # 根据对话模板的分隔符样式获取停止条件字符串
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    # 创建一个包含停止条件字符串的关键字列表
    keywords = [stop_str]
    # 创建一个用于停止生成的条件，根据关键字列表和分词器
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    # PyTorch 上下文管理器，用于在推理模式下执行下面的代码块。推理模式通常用于减少内存占用
    with torch.inference_mode():
        # 模型生成答案 传递参数来执行生成过程。
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])
    # 获取输入 ID 的长度；较输入和输出 ID 是否相同，并计算不同之处的数量；如果存在不同之处，则打印警告，表明生成的输出与输入不一致
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    # 将生成的输出 ID 解码为文本，并跳过特殊标记
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()  # 去除输出文本的首尾空白字符
    # 如果输出以停止条件字符串结尾，去除它
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()  # 再次去除可能在字符串处理中添加的额外空白字符
    return outputs


# 写入
def write_file(output_json, file_lock, results_dict_batch):
    file_lock.acquire()  # 获取锁
    try:
        with open(output_json, "a") as j_:
            for imgid, result in results_dict_batch.items():
                # print(f'write {imgid} ')
                j_.write(json.dumps({imgid: result}, ensure_ascii=False) + "\n")
            print(f'into[{output_json}] \n')
    finally:
        file_lock.release()  # 释放锁



# 分配一个batch的任务到每个GPU
def gpu_tasks(rows, gpu_ids):
    num_gpus = len(gpu_ids)
    num_rows = len(rows)  # 设置要使用的 GPU 数量args.gpu_num=7
    if num_rows < num_gpus:
        num_gpus = num_rows
    rows_per_gpu = num_rows // num_gpus
    gpu_tasks_ = []
    for i in range(num_gpus):
        start = i * rows_per_gpu
        end = (i + 1) * rows_per_gpu if i < num_gpus - 1 else num_rows  # 计算结束任务的索引,最后一个 GPU任务总数结束
        gpu_rows = rows[start:end]
        gpu_tasks_.append(gpu_rows)
    return gpu_tasks_, num_gpus

# GPU推理 一批图像
def process_batch(rows, gpu_id, args, file_lock, query_file_, tokenizer_, model_, image_processor_, model_name, output_json):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # 设置
    disable_torch_init()  # 禁用 PyTorch 加载模型初始化
    num_gpus = get_gpu(args.gpusid)[1]

    if num_gpus > 1:
        # 加载预训练模型及其相关配置 ：tokenizer用于处理文本数据的分词器、model预训练模型、image_processor用于处理图像数据的处理器、context_len上下文长度）
        s1 = time.time()
        tokenizer, model, image_processor, context_len = \
            load_pretrained_model(args.model_path, args.model_base, model_name)
        e1 = time.time()
        print(f'GPU {gpu_id} load model time: {e1-s1:.2f} seconds')
    else:
        tokenizer = tokenizer_
        model = model_
        image_processor = image_processor_

    results_dict_batch = {}  # 创建空字典，用于储存一个batch(文件路径)下的图片的推理结果
    dict_size = 0  # 跟踪字典的大小
    s2 = time.time()
    for row in rows:
        index, imgid, image_path = row
        if imgid not in results_dict_batch:
            results_dict_batch[imgid] = {}
        results_dict_batch[imgid]['index'] = index
        # image = Image.open(BytesIO(image_blob)).convert('RGB')
        image = load_image(image_path)  # 图像文件或者BLOB格式都行
        print(f'--Infer_Image--{index}[{imgid}]')
        # 对加载的图像进行预处理，包括像素值的标准化，然后将图像转换为 PyTorch 张量，并将其移至 GPU 上以进行计算。
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
        queries = get_queries(query_file_, args.img_type)  # 按图像类型 随机生成queries查询集合
        # 便利字典，使用字典中的键作为 key 值，query_list在这里只有一条qs
        for key, query_list in queries.items():
            qs = query_list[0]
            outputs = img_query_inferece(args, qs, image_tensor, model, tokenizer)
            # 将图片名和生成的答案保存到字典results
            results_dict_batch[imgid][key] = outputs.strip()
        dict_size += 1
        # print(results_dict_batch)
        if dict_size >= args.write_size:
            write_file(output_json, file_lock, results_dict_batch)
            results_dict_batch = {}  # 重置字典
            dict_size = 0  # 重置计数字典大小
    if results_dict_batch:
        write_file(output_json, file_lock, results_dict_batch) # 处理剩下的/小于batch的缓存
    e2 = time.time()
    print(f'GPU {gpu_id} infer {len(rows)} images time: {e2 - s2:.2f} seconds')



# 按比例分配db任务,返回分配到比例列表gpu_tasks_ratios_和任务字典gpu_tasks_
def gpu_tasks_ratios(gpu_ids,image_count,files,db_files):
    # GPU对应的任务分配比例
    # task_ratios = {'0': int(5),
    #                '1': int(3),
    #                '2': int(3),
    #                '3': int(3),
    #                '4': int(3),
    #                '5': int(3),
    #                '6': int(3), }
    task_ratios = {'0': int(1),
                   '1': int(1),
                   '2': int(1),
                   '3': int(1),
                   '4': int(1),
                   '5': int(1),
                   '6': int(1), }
    num_gpus = len(gpu_ids)  #4   [0,1,2,3]
    task_ratios_ = [task_ratios[str(gpu_id)] for gpu_id in gpu_ids]  # num_gpus=4 task_ratios_ =[5,3,3,3]
    gpu_tasks_ratios_=[]
    gpu_tasks_ = {}
    start = 0
    for i, ratio in enumerate(task_ratios_):
        gpu_tasks_[gpu_ids[i]] = []  # gpu_tasks{'0':值[列表(元组),()]}
        end = start + math.ceil(ratio / sum(
            task_ratios_) * image_count) if i < num_gpus - 1 else image_count  # 向上取证4.1-5，5-5 [(0,72),(72,115),(115,158),(158,200)]
        gpu_tasks_ratios_.append((start, end))
        k0 = 0  # 上一个计数
        for db_file in db_files:
            k = files[db_file][2]  # 下一个计数
            if start >= k0 and end <= k:
                gpu_tasks_[gpu_ids[i]].append((start - k0, end - k0, files[db_file]))  # 添加一个元组
                break
            elif start < k0 and end <= k:
                gpu_tasks_[gpu_ids[i]].append((0, end - k0, files[db_file]))
                break
            elif start >= k0 and start < k and end > k:
                gpu_tasks_[gpu_ids[i]].append((start - k0, k, files[db_file]))
            elif start < k0 and end > k:
                gpu_tasks_[gpu_ids[i]].append((0, k, files[db_file]))
            elif start > k:
                pass
            k0 = k
        start = end
    print(gpu_tasks_ratios_)
    print(gpu_tasks_)
    return gpu_tasks_ratios_,gpu_tasks_



# GPU推理 一批db图像
def process_batch_db(tasks, gpu_id, args, file_lock, query_file_, tokenizer_, model_, image_processor_, model_name, output_json):
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # 设置
    torch.cuda.set_device(gpu_id)
    disable_torch_init()  # 禁用 PyTorch 加载模型初始化
    num_gpus = get_gpu(args.gpusid)[1]
    current_gpu=get_current_gpu_info()[0]
    if num_gpus > 1:
        # 加载预训练模型及其相关配置 ：tokenizer用于处理文本数据的分词器、model预训练模型、image_processor用于处理图像数据的处理器、context_len上下文长度）
        s1 = time.time()
        tokenizer, model, image_processor, context_len = \
            load_pretrained_model(args.model_path, args.model_base, model_name)
        e1 = time.time()
        print(f'GPU {gpu_id} load model time: {e1-s1:.2f} seconds')
    else:
        tokenizer = tokenizer_
        model = model_
        image_processor = image_processor_

    # tasks[(0, 156, ('/workstation/pky_data/1.db', 156, 156, '/workstation/pky_data/result_1.json')),
    #       (0, 68, ('/workstation/pky_data/2.db', 235, 391, '/workstation/pky_data/result_2.json'))],

    for task in tasks:
        start = task[0] #
        end = task[1]
        db_path = task[2][0]
        dn_name = os.path.basename(db_path)
        # 创建数据库链接
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 分批处理数据
        results_dict_batch = {}  # 创建空字典，用于储存一个batch(文件路径)下的图片的推理结果
        dict_size = 0  # 跟踪字典的大小
        start_index = start
        while start_index < end:
            end_index = start_index + args.batch_size_pergpu
            if end_index > end:
                end_index = end
            sql = "SELECT rowid, * FROM street_image WHERE rowid > ? AND rowid <= ?"
            cursor.execute(sql, (start_index, end_index))   # (id, lon, lat, image_data)
            # 获取数据
            rows = cursor.fetchall()  # (id, lon, lat, image_data)
            print(f'GPU {current_gpu} : process image [{start_index}-{end_index}] in file {db_path}')

            s2 = time.time()
            for row in rows:
                rowid, id, lon, lat, time2, image_blob = row
                if id not in results_dict_batch:
                    results_dict_batch[id] = {}
                results_dict_batch[id]['rowid'] = rowid
                results_dict_batch[id]['datapath'] = dn_name
                results_dict_batch[id]['lon'] = lon
                results_dict_batch[id]['lat'] = lat
                results_dict_batch[id]['imgtype'] = args.img_type
                image = load_image(image_blob)  # 图像文件或者BLOB格式都行
                # print(f'--GPU{current_gpu}  Infer_Image--{rowid}[{id}]')
                image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
                queries = get_queries(query_file_, args.img_type)
                for key, query_list in queries.items():
                    qs = query_list[0]
                    outputs = img_query_inferece(args, qs, image_tensor, model, tokenizer)
                    print(f'{key},{outputs}')
                    # 将图片名和生成的答案保存到字典results
                    results_dict_batch[id][key] = outputs.strip()
                dict_size += 1
                if dict_size >= args.write_size:
                    write_file(output_json, file_lock, results_dict_batch)
                    results_dict_batch = {}  # 重置字典
                    dict_size = 0  # 重置计数字典大小
            e2 = time.time()
            print(f'GPU {current_gpu} infer {len(rows)} images [{start_index}-{end_index}] time: {e2 - s2:.2f} seconds')
            start_index = end_index

        if results_dict_batch:
            write_file(output_json, file_lock, results_dict_batch)  # 处理剩下的/小于batch的缓存



def eval_model(args):
    print(f'start time: {time.time()}')
    # 基础变量
    gpu_ids, num_gpus = get_gpu(args.gpusid)
    # # 在主进程中设置 GPU 设备
    for gpu_id in gpu_ids:
        torch.cuda.set_device(gpu_id)
    torch.cuda.set_device(0)
    print(get_current_gpu_info())
    data_type = get_path_info(args.img_folder_path)  # 数据格式判断
    query_file_ = load_query_file(args.query_file)  # query文件加载
    # 多进程设置
    t_mp.set_start_method('spawn', force=True)
    start_method = t_mp.get_start_method()
    file_lock = t_mp.Lock()  # 创建互斥锁
    print(f"Current start method: {start_method}")
    # 模型准备
    disable_torch_init() # 禁用 PyTorch 加载模型初始化
    model_name = get_model_name_from_path(args.model_path)  # 获取模型名称

    # 是否包含特定标识来确定对话模式（conv_mode）
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            '[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(
                conv_mode,
                args.conv_mode,
                args.conv_mode))
    else:
        args.conv_mode = conv_mode


    if num_gpus == 1:
        # 加载预训练模型及其相关配置 ：tokenizer用于处理文本数据的分词器、model预训练模型、image_processor用于处理图像数据的处理器、context_len上下文长度）
        print(f"Current GPU Model: {get_current_gpu_info()}")
        s1 = time.time()
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
        e1 = time.time()
        print(f'GPU {get_current_gpu_info()[0]} load model time: {e1-s1:.2f} seconds')

    if data_type=='img':
        # 输出文件名
        output_files = {}
        output_files['default'] = os.path.join(args.output_dir, 'result_all.json') # 默认单个输出
        # 获取所有图像路径
        batch_images_all = []  # 存储所有图像路径的列表
        subdirectories = [d for d in os.listdir(args.img_folder_path) if
                          os.path.isdir(os.path.join(args.img_folder_path, d))] # 数据文件夹路径集合,加载次级目录文件夹名
        for directory in subdirectories:
            output_json = os.path.join(args.output_dir, f'result_{directory}.json')
            output_files[directory] = output_json # 将子目录映射到对应的output_json和output_txt

            img_folder_path2 = os.path.join(args.img_folder_path, directory)  #次级目录完整路径
            imgs_names = [name for name in os.listdir(img_folder_path2) if
                          name.lower().endswith(('.jpg', '.png', '.jpeg'))]  # # 加载文件夹中的所有图片名，，过滤后缀['1.jpg', '2.jpg',...]
            imgs_names = sorted(imgs_names)
            batch_images = []  # 创建一个列表来存储 一个路径下的 所有图像路径
            for i, imgid in enumerate(imgs_names, start=1):
                index = '{}_{}'.format(directory, i)
                img_path = os.path.join(img_folder_path2, imgid)  # 图片绝对路径"/home/gt/pky/llava/LLaVA2/images/1.jpg"
                batch_images.append((index, imgid, img_path))  # 将图像数据添加到批次列表

            # 合并输出
            if args.result_gater == 1:
                batch_images_all.extend(batch_images)  # 将图像数据添加到总列表
            # 不合并输出
            else:
                print(f'\n===Infer images in folder [{directory}]\nbatch_images: {batch_images}\n')
                with open(output_json, "w"):
                    pass
                # 单GPU
                if num_gpus == 1:
                    process_batch(batch_images, gpu_ids[0], args, file_lock, query_file_,
                                  tokenizer, model, image_processor, None,
                                  output_json)
                # 多GPU
                elif num_gpus > 1:
                    gpu_tasks_, num_gpus = gpu_tasks(batch_images, gpu_ids)
                    processes = []
                    for i in range(num_gpus):
                        gpu_id = gpu_ids[i]  # 通过索引获取 gpu_id
                        p = t_mp.Process(target=process_batch,
                                         args=(gpu_tasks_[i], gpu_id, args, file_lock, query_file_,
                                               None, None, None, model_name,
                                               output_json))
                        p.start()  # 启动当前进程
                        print(f'{p.name}:   PID-{p.ident}   GPU-{gpu_id}')  # 获取子进程PID
                        processes.append(p)
                    for p in processes:
                        p.join()  # 等待每个进程完成其任务

        # 多gpu  输出合并
        if args.result_gater==1:
            print(f'===Infer all images === \n  batch_images_all: {batch_images_all}\n')
            output_json = output_files['default']
            with open(output_json, "w"):
                pass

            if num_gpus == 1:
                process_batch(batch_images_all, gpu_ids[0], args, file_lock, query_file_,
                              tokenizer, model, image_processor, None,
                              output_json)

            if num_gpus > 1:
                gpu_tasks_, num_gpus = gpu_tasks(batch_images_all, gpu_ids)
                processes = []
                for i in range(num_gpus):
                    gpu_id = gpu_ids[i]  # 通过索引获取 gpu_id
                    p = t_mp.Process(target=process_batch,
                                     args=(gpu_tasks_[i], gpu_id, args, file_lock, query_file_,
                                           None, None, None, model_name,
                                           output_json))
                    p.start()  # 启动当前进程
                    print(f'{p.name}:   PID-{p.ident}   GPU-{gpu_id}')  # 获取子进程PID
                    processes.append(p)
                for p in processes:
                    p.join()  # 等待每个进程完成其任务

    elif data_type == 'db':
        # 依次读取db文件的名称
        db_folder = args.img_folder_path
        db_files = [file for file in os.listdir(db_folder) if file.endswith('.db')]
        # 输出文件名
        image_count = 0
        files = {}
        for db_file in db_files:
            db_name = os.path.splitext(db_file)[0]  # 获取文件名（去掉扩展名）
            db_path = os.path.join(db_folder, db_file)  # 构建数据库文件的完整路径

            output_json = os.path.join(args.output_dir, f'result_{db_name}.json')

            with DBContextManager(db_path) as cursor:
                # 执行查询以获取数据条目数
                cursor.execute("SELECT COUNT(*) FROM street_image")
                image_nums = cursor.fetchone()[0]  # 获取数据条目数
                image_count = image_count + image_nums # 计算总的数据条目数
            files[db_file]=(db_path, image_nums, image_count, output_json)   # 将子目录映射到对应的output_json

        files['all'] = (None,None,image_count,
                        os.path.join(args.output_dir, 'resultdb_all.json'))  # 默认单个输出
        ss = time.time()
        if args.result_gater == 1:
            output_json = files['all'][3]
            with open(output_json, "w"):
                pass
            if num_gpus > 1:
                gpu_tasks_ratios_, gpu_tasks_ =gpu_tasks_ratios(gpu_ids,image_count,files,db_files) # 返回按比例分配的索引列表、对应的数据文件索引和路径
                processes = []
                for i, gpu_id in enumerate(gpu_ids):
                    p = t_mp.Process(target=process_batch_db,
                                     args=(gpu_tasks_[gpu_id], gpu_id, args, file_lock, query_file_,
                                           None, None, None, model_name,
                                           output_json))
                    p.start()  # 启动当前进程
                    print(f'{p.name}:   PID-{p.ident}   GPU-{gpu_id}')  # 获取子进程PID
                    processes.append(p)
                for p in processes:
                    p.join()  # 等待每个进程完成其任务
            if num_gpus==1:
                gpu_tasks_ratios_, gpu_tasks_ = gpu_tasks_ratios(gpu_ids, image_count, files, db_files)  # 返回按比例分配的索引列表、对应的数据文件索引和路径
                process_batch_db(gpu_tasks_[gpu_ids[0]], gpu_ids[0], args, file_lock, query_file_,
                              tokenizer, model, image_processor, None,
                              output_json)
        ee = time.time()
        print(f'GPU {gpu_ids} infer {image_count} images time: {ee - ss:.2f} seconds')

        if args.result_gater != 1:
            with open(output_json, "w"):
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/workstation/LLaVA/LLaVA-v1-5-13b")  # LMM模型文件夹
    parser.add_argument("--img_folder_path", type=str, default="/workstation/CX_data/test_data0804")  # 批量处理图片库，支持二级目录
    parser.add_argument("--query_file", type=str, default=None) # query需要txt配置文件
    parser.add_argument("--img_type", type=str, default=None) # streetview/geoimg/remote  构建不同的query集合
    parser.add_argument("--output_dir", type=str, default="/workstation/CX_data/result_cx0804")
    parser.add_argument("--model-base", type=str, default=None) # 需要基础模型 如果模型不完全则需要
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--batch_size_pergpu", type=int, default=20) # 80-90G，别超了
    parser.add_argument("--gpusid", type=str, default="0") # 0,1,2
    parser.add_argument("--write_size", type=int, default=100)
    parser.add_argument("--result_gater", type=int, default=1)  #1
    args = parser.parse_args()
    eval_model(args)