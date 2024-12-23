# data_panorama_cut.py
import argparse
import cv2
import numpy as np
import os
import sqlite3
import sys
import multiprocessing
from multiprocessing import Pool, Lock
from functools import partial
# 预设参数
THREAD_NUMS = 4  # 定义线程数量
width = 1024 # 输出图像的宽度和高度
height = 1024
# 定义全景视角变换参数与命名
PI=np.pi
face_transform = [
    [0.0, 0.0],  # front
    [PI/2, 0.0],  # right
    [PI, 0.0],  # back
    [-PI/2, 0.0]  # left
]
output_filenames = ["f", "r", "b", "l"]
db_lock = Lock()

def set_gpu_device(gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

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
    _, extension = os.path.splitext(path)
    if extension == '.db':
        path_type = 'db'
    elif extension in ['.jpeg', '.jpg', '.png', '.gif', '.bmp', '.tiff']:
        path_type = 'img'
    else:
        print("没有符合条件的 '.jpeg', '.jpg', '.png', '.gif', '.bmp', '.tiff' 图像")
        sys.exit()  # 终止程序执行

    return path_type


# 执行透视变换,创建一个视角的图像(视角分为1f2r3b4l)
def create_transformation(param, face_id):
    in_width = param['in'].shape[1]
    in_height = param['in'].shape[0]
    mapx = np.zeros((param['height'], param['width']), np.float32)
    mapy = np.zeros((param['height'], param['width']), np.float32)
    an = np.sin(np.pi / 4)
    ak = np.cos(np.pi / 4)
    ftu = face_transform[face_id][0]
    ftv = face_transform[face_id][1]
    # 对于每个图像计算相应的源坐标
    for y in range(param['height']):
        for x in range(param['width']):
            nx = (float(y) / float(param['height']) - 0.5) * 2
            ny = (float(x) / float(param['width']) - 0.5) * 2

            nx *= 2 * an
            ny *= 2 * an

            #Center faces
            if ftv == 0:
                u = np.arctan2(nx, ak)
                v = np.arctan2(ny * np.cos(u), ak)
                u += ftu
            # Bottom face
            elif ftv > 0:
                d = np.sqrt(nx * nx + ny * ny)
                v = np.pi / 2 - np.arctan2(d, ak)
                u = np.arctan2(ny, nx)
            # Top face    cout << "aaa"
            else:
                d = np.sqrt(nx * nx + ny * ny)
                v = -np.pi / 2 + np.arctan2(d, ak)
                u = np.arctan2(-ny, nx)

            # Map from angular coordinates to [-1, 1], respectively.
            u = u / np.pi
            v = v / (np.pi / 2)
            # Warp around, if our coordinates are out of bounds.
            while v < -1:
                v += 2
                u += 1
            while v > 1:
                v -= 2
                u += 1
            while u < -1:
                u += 2
            while u > 1:
                u -= 2

            # Map from [-1, 1] to in texture space
            u = u / 2.0 + 0.5
            v = v / 2.0 + 0.5
            u = u * (in_width - 1)
            v = v * (in_height - 1)
            mapx[x, y] = u
            mapy[x, y] = v

    if param['face'].shape[0] != param['width'] or \
            param['face'].shape[1] != param['height'] \
            or param['face'].dtype != param['in'].dtype:
        # 如果任一条件不满足，创建新的零矩阵，确保输出图像具有正确的尺寸和数据类型
        param['face'] = np.zeros((param['width'], param['height']), param['in'].dtype)
    # 执行透视变换
    param['face'] = cv2.remap(param['in'], mapx, mapy,
                              interpolation=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    # 返回切割后的图像
    return param['face']


# 处理一行BLOB数据,并储存到新数据库中 原始图像数据为BRG格式
def process_image(image_blob, panid, lon, lat, output_path):
    image = np.frombuffer(image_blob, dtype=np.uint8)
    src_image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    if args.TEMP_DB_PATH == 1:
        cv2.imwrite(f"{output_path}/{panid}.jpg", src_image)

    image_data_cache = []
    for i in range(THREAD_NUMS):
        result_image = np.zeros((width, height), src_image.dtype)  # 创建一个输出图像的初始空白画布
        param = {'in': src_image, 'face': result_image, 'width': width, 'height': height} # 构建全景图字典
        image_cut = create_transformation(param, i) #进行单视角转换

        id = f"{panid}_{output_filenames[i]}"  # 构建对应的id值和文件名

        if args.TEMP_DB_PATH == 1:
            # imgpath = f"{output_path}/{id}.jpg"
            cv2.imwrite(f"{output_path}/{id}.jpg", image_cut)
        image_cut_binary = cv2.imencode('.jpg', image_cut)[1].tobytes()  # 将图像数据编码为二进制数据

        image_data_cache.append((id, lon, lat, sqlite3.Binary(image_cut_binary)))
    return image_data_cache

# 处理一批数据rows 并存入数据库
def process_db_record(rows, output_db, output_path):
    connection = sqlite3.connect(output_db)  # 为不同线程创建数据库链接
    cursor_cut = connection.cursor()
    for record in rows:
        rowid, panid, lon, lat, image_blob = record
        print(f"Processed rowid: {rowid}, panid: {panid}")
        image_data_cache = process_image(image_blob, panid, lon, lat, output_path)

        with db_lock:
            for data_ in image_data_cache:
                cursor_cut.execute('INSERT INTO street_4_images (id, lon, lat, image_data) VALUES (?, ?, ?, ?)', data_)
            connection.commit()
    try:

        connection.commit()
    except Exception as e:
        print(f"An error occurred: {e}")
        connection.rollback()  # 回滚事务以撤消未提交的更改
    finally:
        connection.close()  # 确保关闭数据库连接

# 将任务平均分配给每个进程 并执行
def poolmap(rows, gpu_num,output_db,output_path,pool):
    num_rows = len(rows)
    if num_rows<gpu_num:
        gpu_num = num_rows  # 如果行数少于GPU数量，将GPU数量设置为行数
    num_gpus = gpu_num  # 设置要使用的 GPU 数量args.gpu_num=7
    rows_per_gpu = num_rows // num_gpus
    gpu_tasks = []
    for i in range(num_gpus):
        start = i * rows_per_gpu
        end = (i + 1) * rows_per_gpu if i < num_gpus - 1 else num_rows
        gpu_rows = rows[start:end]
        gpu_tasks.append(gpu_rows)

    process_db_record_partial = partial(process_db_record, output_db=output_db, output_path=output_path)
    pool.map(process_db_record_partial, gpu_tasks)

# *****************主函数**************
def cut(args):
    data_type = get_path_info(args.data_path)
    output_path = args.output_path
    gpu_num = args.gpu_num

    if data_type=='img':
        src_image = cv2.imread(args.data_path)
        imgname, img_extension = os.path.splitext(os.path.basename(args.data_path))  # 得到图像名称和后缀
        for i in range(THREAD_NUMS):
            id = f"{imgname}_{output_filenames[i]}{img_extension}"
            print(f'{id}')
            # 创建一个输出图像的初始空白画布
            result_image = np.zeros((width, height), src_image.dtype)
            param = {'in': src_image, 'face': result_image, 'width': width, 'height': height}
            create_transformation(param, i)
            # 保存
            cv2.imwrite(os.path.join(args.output_path, id), param['face'])

    elif data_type == 'db':
        if args.output_db_filename:
            output_db = os.path.join(args.output_path, args.output_db_filename)
        else:
            output_db = f'{output_path}/TileDB_cut0.db'
        if os.path.exists(output_db):
            os.remove(output_db)
        # 链接数据库
        with DBContextManager(args.data_path) as cursor:
            connection = sqlite3.connect(output_db)
            cursor_cut = connection.cursor()
            cursor_cut.execute(
                '''CREATE TABLE IF NOT EXISTS street_4_images (id STRING, lon REAL, lat REAL, image_data BLOB)''')

            if args.db_num:  # 如果指定图片索引
                pool = multiprocessing.Pool(processes=gpu_num)
                for gpu_id in range(gpu_num):
                    pool.apply_async(set_gpu_device, args=(gpu_id,))

                db_nums = args.db_num.split(",")  # 将输入的逗号分隔的数字字符串分割为列表
                placeholders = ",".join(["?"] * len(db_nums))  # 创建与数字数量相等的占位符
                sql = f"SELECT rowid, panid, lon, lat, image_data FROM street_image WHERE rowid IN ({placeholders})"  # ,43,44,43
                cursor.execute(sql, db_nums)  # 给sql命令填如查询的图片名
                rows = cursor.fetchall()
                poolmap(rows, gpu_num, output_db, output_path, pool)
                pool.close()
                pool.join()
            else:
                pool = multiprocessing.Pool(processes=args.gpu_num)
                for gpu_id in range(args.gpu_num):
                    pool.apply_async(set_gpu_device, args=(gpu_id,))

                sql = "SELECT rowid, panid, lon, lat, image_data FROM street_image"
                cursor.execute(sql)
                if args.batch_size:
                    batch_size = args.batch_size  # 每次处理100行7000
                    while True:
                        rows = cursor.fetchmany(batch_size)
                        if not rows:
                            break
                        poolmap(rows, gpu_num, output_db, output_path, pool)
                    remaining_rows = cursor.fetchall()
                    poolmap(remaining_rows, gpu_num, output_db, output_path, pool)
                else:
                    rows = cursor.fetchall()
                    poolmap(rows, gpu_num, output_db, output_path, pool)
                # 关闭进程池
                pool.close()
                pool.join()
                print(f'output to {output_db}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        default="./llava_test_data/streetview_test/Street View 360.jpg")  # 批量处理街景库
    parser.add_argument("--output_path", type=str,
                        default="./llava_test_data/streetview_test/output")
    parser.add_argument("--TEMP_DB_PATH", type=int, default=0)  # 保存中间图片为1 不保存为0
    parser.add_argument("--output_db_filename", type=str, default=None)  # 数据出书名称db数据专属
    parser.add_argument("--db_num", type=str, default=None) # 43,44,43 db数据的指定图像索引
    parser.add_argument("--gpu_num", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()
    cut(args)
