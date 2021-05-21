import os
import pickle
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.append("../")

from utils.data_format_converter import convert_data
from utils.utils import py_nms, combine_data_list, crop_landmark_image, delete_old_img
from utils.utils import save_hard_example, generate_bbox, read_annotation, processed_image
from utils.utils import get_landmark_from_lfw_neg, get_landmark_from_celeba


# 模型路径
model_path = '../infer_models'

# 获取P模型
pnet = torch.load(os.path.join(model_path, 'PNet.pth'))
pnet.eval()

softmax = torch.nn.Softmax()


# 使用RNet模型预测
def predict(infer_data):
    # 添加待预测的图片
    infer_data = torch.tensor(infer_data, dtype=torch.float32)
    infer_data = torch.unsqueeze(infer_data, dim=0)
    # 执行预测
    cls_prob, bbox_pred, _ = pnet(infer_data)
    cls_prob = torch.squeeze(cls_prob)
    cls_prob = softmax(cls_prob)
    bbox_pred = torch.squeeze(bbox_pred)
    return cls_prob.numpy(), bbox_pred.numpy()


def detect_pnet(im, min_face_size, scale_factor, thresh):
    """通过pnet筛选box和landmark
    参数：
      im:输入图像[h,2,3]
    """
    net_size = 12
    # 人脸和输入图像的比率
    current_scale = float(net_size) / min_face_size
    im_resized = processed_image(im, current_scale)
    _, current_height, current_width = im_resized.shape
    all_boxes = list()
    # 图像金字塔
    while min(current_height, current_width) > net_size:
        # 类别和box
        cls_cls_map, reg = predict(im_resized)
        boxes = generate_bbox(cls_cls_map[1, :, :], reg, current_scale, thresh)
        current_scale *= scale_factor  # 继续缩小图像做金字塔
        im_resized = processed_image(im, current_scale)
        _, current_height, current_width = im_resized.shape

        if boxes.size == 0:
            continue
        # 非极大值抑制留下重复低的box
        keep = py_nms(boxes[:, :5], 0.7, mode='Union')
        boxes = boxes[keep]
        all_boxes.append(boxes)
    if len(all_boxes) == 0:
        return None
    all_boxes = np.vstack(all_boxes)
    # 将金字塔之后的box也进行非极大值抑制
    keep = py_nms(all_boxes[:, 0:5], 0.7)
    all_boxes = all_boxes[keep]
    # box的长宽
    bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
    bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1
    # 对应原图的box坐标和分数
    boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                         all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                         all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                         all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                         all_boxes[:, 4]])
    boxes_c = boxes_c.T

    return boxes_c


# 截取pos,neg,part三种类型图片并resize成24x24大小作为RNet的输入
def crop_24_box_image(data_path, filename, min_face_size, scale_factor, thresh):
    # pos，part,neg裁剪图片放置位置
    pos_save_dir = os.path.join(data_path, '24/positive')
    part_save_dir = os.path.join(data_path, '24/part')
    neg_save_dir = os.path.join(data_path, '24/negative')
    # RNet数据地址
    save_dir = os.path.join(data_path, '24/')

    # 创建文件夹
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(pos_save_dir):
        os.mkdir(pos_save_dir)
    if not os.path.exists(part_save_dir):
        os.mkdir(part_save_dir)
    if not os.path.exists(neg_save_dir):
        os.mkdir(neg_save_dir)

    # 读取标注数据
    data = read_annotation(data_path, filename)
    all_boxes = []
    landmarks = []
    empty_array = np.array([])

    # 使用PNet模型识别图片
    for image_path in tqdm(data['images']):
        assert os.path.exists(image_path), 'image not exists'
        im = cv2.imread(image_path)
        boxes_c = detect_pnet(im, min_face_size, scale_factor, thresh)
        if boxes_c is None:
            all_boxes.append(empty_array)
            landmarks.append(empty_array)
            continue
        all_boxes.append(boxes_c)

    # 把识别结果存放在文件中
    save_file = os.path.join(save_dir, 'detections.pkl')
    with open(save_file, 'wb') as f:
        pickle.dump(all_boxes, f, 1)

    save_hard_example(data_path, 24)


if __name__ == '__main__':
    data_path = '../dataset/'
    base_dir = '../dataset/WIDER_train/'
    filename = '../dataset/wider_face_train.txt'
    min_face_size = 20
    scale_factor = 0.79
    thresh = 0.6
    # 获取人脸的box图片数据
    print('开始生成bbox图像数据')
    crop_24_box_image(data_path, filename, min_face_size, scale_factor, thresh)
    # 获取人脸关键点的数据
    print('开始生成landmark图像数据')
    # 获取lfw negbox，关键点
    lfw_neg_path = os.path.join(data_path, 'trainImageList.txt')
    data_list = get_landmark_from_lfw_neg(lfw_neg_path, data_path)
    # 获取celeba，关键点
    # celeba_data_list = get_landmark_from_celeba(data_path)
    # data_list.extend(celeba_data_list)
    crop_landmark_image(data_path, data_list, 24, argument=True)
    # 合并数据列表
    print('开始合成数据列表')
    combine_data_list(os.path.join(data_path, '24'))
    # 合并图像数据
    print('开始合成图像文件')
    convert_data(os.path.join(data_path, '24'), os.path.join(data_path, '24', 'all_data'))
    # 删除旧数据
    print('开始删除就得图像文件')
    delete_old_img(data_path, 24)
