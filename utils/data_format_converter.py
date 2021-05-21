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


# 人脸识别训练数据的格式转换
def convert_data(data_folder, output_prefix):
    # 读取全部的数据类别获取数据
    data_list_path = os.path.join(data_folder, 'all_data_list.txt')
    train_list = open(data_list_path, "r").readlines()
    train_image_list = []
    for i, item in enumerate(train_list):
        sample = item.split(' ')
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
        # 加入到数据列表中
        train_image_list.append((image, label, bbox, landmark))
    print("训练数据大小：", len(train_image_list))

    # 开始写入数据
    writer = DataSetWriter(output_prefix)
    for image, label, bbox, landmark in tqdm(train_image_list):
        try:
            key = str(uuid.uuid1())
            img = cv2.imread(image)
            _, img = cv2.imencode('.bmp', img)
            # 写入对应的数据
            writer.add_img(key, img.tostring())
            label_str = str(label)
            bbox_str = ' '.join([str(x) for x in bbox])
            landmark_str = ' '.join([str(x) for x in landmark])
            writer.add_label('\t'.join([key, bbox_str, landmark_str, label_str]))
        except:
            continue
