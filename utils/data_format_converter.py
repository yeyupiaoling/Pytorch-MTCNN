import os
import struct
import uuid
from tqdm import tqdm
import cv2


class DataSetWriter(object):
    def __init__(self, prefix):
        # 创建对应的数据文件
        self.data_file = open(prefix + '.data', 'wb')
        self.header_file = open(prefix + '.header', 'wb')
        self.label_file = open(prefix + '.label', 'wb')
        self.offset = 0
        self.header = ''

    def add_img(self, key, img):
        # 写入图像数据
        self.data_file.write(struct.pack('I', len(key)))
        self.data_file.write(key.encode('ascii'))
        self.data_file.write(struct.pack('I', len(img)))
        self.data_file.write(img)
        self.offset += 4 + len(key) + 4
        self.header = key + '\t' + str(self.offset) + '\t' + str(len(img)) + '\n'
        self.header_file.write(self.header.encode('ascii'))
        self.offset += len(img)

    def add_label(self, label):
        # 写入标签数据
        self.label_file.write(label.encode('ascii') + '\n'.encode('ascii'))

    def close(self):
        self.data_file.close()
        self.header_file.close()
        self.label_file.close()


def _parse_train_sample(item):
    sample = item.strip().split()
    # 获取图片路径
    image = sample[0]
    # 获取图片标签
    label = int(sample[1])
    # 做补0预操作
    bbox = [0, 0, 0, 0]
    landmark = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # 如果只有box，关键点就补0
    if len(sample) == 6:
        bbox = [float(i) for i in sample[2:]]
    # 如果只有关键点，那么box就补0
    if len(sample) == 12:
        landmark = [float(i) for i in sample[2:]]
    return image, label, bbox, landmark


def _write_data_entries(data_entries, output_prefix):
    writer = DataSetWriter(output_prefix)
    try:
        for img_bytes, label, bbox, landmark in tqdm(data_entries):
            try:
                key = str(uuid.uuid1())
                # 关键代码：这里直接写入已经编码好的图像字节，不再从磁盘读取中间图片文件。
                writer.add_img(key, img_bytes)
                label_str = str(label)
                bbox_str = ' '.join([str(x) for x in bbox])
                landmark_str = ' '.join([str(x) for x in landmark])
                writer.add_label('\t'.join([key, bbox_str, landmark_str, label_str]))
            except Exception:
                continue
    finally:
        writer.close()


def convert_data_list(train_list, output_prefix):
    train_image_list = []
    for item in train_list:
        if not item.strip():
            continue
        train_image_list.append(_parse_train_sample(item))
    print("训练数据大小：", len(train_image_list))

    data_entries = []
    for image, label, bbox, landmark in train_image_list:
        try:
            img = cv2.imread(image)
            _, img = cv2.imencode('.bmp', img)
            data_entries.append((img.tobytes(), label, bbox, landmark))
        except Exception:
            continue
    _write_data_entries(data_entries, output_prefix)


def convert_encoded_data_list(train_list, output_prefix):
    print("训练数据大小：", len(train_list))
    _write_data_entries(train_list, output_prefix)


# 人脸识别训练数据的格式转换
def convert_data(data_folder, output_prefix):
    # 读取全部的数据类别获取数据
    data_list_path = os.path.join(data_folder, 'all_data_list.txt')
    with open(data_list_path, "r") as f:
        train_list = f.readlines()
    convert_data_list(train_list, output_prefix)
