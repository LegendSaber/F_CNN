# 将CCF公开数据集中的二进制序列可视化为图像需要用到的函数
import os
import numpy as np
from PIL import Image


# 将PE中的文本16进制读出来
def read_bytes(path_byte):
    res_bytes = []
    fp_byte = open(path_byte, mode='rb')
    byte_lines = fp_byte.readlines()
    for byte_line in byte_lines:
        for byte in byte_line:
            if byte != 0 and byte != 0x90 and byte != 0xCC:
                res_bytes.append(byte)
    fp_byte.close()
    return res_bytes


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


def get_images(base_dir):
    count = 0
    dest_path = os.path.join(base_dir, "..", "train_images")
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    for file_name in os.listdir(base_dir):
        file_path = os.path.join(base_dir, file_name)
        img_path = os.path.join(dest_path, file_name) + ".png"
        array_byte = read_bytes(file_path)
        # 转换为灰度图
        img = get_gray_image(array_byte)
        # 转换为马尔可夫图
        # img = get_markov_image(array_byte)
        img = np.uint8(img)
        img = Image.fromarray(img)
        img.save(img_path)
        if count % 100 == 99:
            print(count)
        count = count + 1
    print("Get Images Ok, Get %d images" % count)


if __name__ == '__main__':
    pass