# 将微软公开数据集中的二进制序列可视化为图像需要用到的函数
import os
import py7zr
import numpy as np
import glob
from PIL import Image


# 将微软公开数据集的压缩包中的文件提取出来
def extract_file():
    base_dir = ""
    file_path = os.path.join(base_dir, "train.7z")  # 要解压的压缩包文件
    dest_path = os.path.join(base_dir, "train")  # 解压出来的文件保存的位置
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    with py7zr.SevenZipFile(file_path, mode="r") as archive:
        file_names = archive.getnames()
        byte_file = []
        for file_name in file_names:
            if file_name[-1] == 'm':  # m解压的是.asm文件，s解压的是.bytes文件
                byte_file.append(file_name)
        # print(byte_file)
        archive.extract(dest_path, byte_file)
    print("Extract file Ok")


# 将数组进行计算得出马尔可夫图像
def get_markov_image(array_byte):
    byte_len = len(array_byte) - 1
    byte_frequency_map = np.zeros([256, 256])
    byte_frequency_sum = np.zeros(256)

    for i in range(byte_len):
        m = array_byte[i]
        n = array_byte[i + 1]
        byte_frequency_map[m][n] = byte_frequency_map[m][n] + 1
        byte_frequency_sum[m] = byte_frequency_sum[m] + 1

    for i in range(256):
        byte_sum = byte_frequency_sum[i]
        if byte_sum == 0:
            continue
        for j in range(256):
            byte_frequency_map[i][j] = byte_frequency_map[i][j] / byte_sum

    byte_max = np.max(byte_frequency_map)

    for i in range(256):
        for j in range(256):
            p = ((byte_frequency_map[i][j] * 255) / byte_max) % 256
            byte_frequency_map[i][j] = p

    return np.round(byte_frequency_map)


# 将数组转为灰度图像
def get_gray_image(array_byte):
    byte_len = len(array_byte)
    file_size = int(byte_len / 1024)
    file_size_array = [10, 30, 60, 100, 200, 500, 1000]
    image_width_array = [32, 64, 128, 256, 384, 512, 768]
    image_width = 0

    for i in range(7):
        if file_size < file_size_array[i]:
            image_width = image_width_array[i]
            break
    if image_width == 0:
        image_width = 1024

    image_height = int(byte_len / image_width)
    image_bytes = np.zeros([image_height, image_width])
    k = 0

    for i in range(image_height):
        for j in range(image_width):
            image_bytes[i][j] = array_byte[k]
            k += 1
    return image_bytes


# 将.bytes中的文本16进制读出来
def read_bytes(file_path):
    res_bytes = []
    with open(file_path, mode='r') as fp:
        for byte_line in fp.readlines():    # 循环读出每一行
            str_bytes = byte_line.split(" ")  # 根据空格划分为数组
            for str_byte in str_bytes:
                if str_byte[0] == '?':  # 舍弃无用字符
                    continue
                byte = int(str_byte, 16)    # 将字符按照16进制转成数字
                if byte <= 0xFF:            # 每行前面的地址都要舍弃
                    res_bytes.append(byte)
    return res_bytes


# 根据文件路径得到不带后缀的文件名
def get_file_name(file_path):
    file_name_begin = file_path.rfind("/")
    file_name_end = file_path.rfind(".")
    return file_path[file_name_begin + 1: file_name_end]


# 将二进制序列转换为图片
def get_images(base_dir):
    count = 0
    # 保存生成的灰度图像的路径
    dest_path = os.path.join(base_dir, "..", "train_images")
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    # 获取保存.bytes文件的路径
    bytes_file_path = glob.glob(os.path.join(base_dir, "*.bytes"))
    for file_path in bytes_file_path:
        # 保存的图片名称
        img_path = os.path.join(dest_path, get_file_name(file_path)) + ".png"
        # 获取.bytes文件中的16进制序列
        array_byte = read_bytes(file_path)

        # 转换为灰度图
        img = get_gray_image(array_byte)
        # 转换为马尔可夫图
        # img = get_markov_image(array_byte)

        # 生成图片
        img = np.uint8(img)
        img = Image.fromarray(img)
        img.save(img_path)

        count += 1
        if count % 100 == 99:
            print(count)
    print("Get image ok! get %d images" % count)


if __name__ == '__main__':
    pass
