import pickle
import shutil
import numpy as np
import random
import os
import cv2
from tqdm import tqdm


class BBox:
    # 人脸的box
    def __init__(self, box):
        self.left = box[0]
        self.top = box[1]
        self.right = box[2]
        self.bottom = box[3]

        self.x = box[0]
        self.y = box[1]
        self.w = box[2] - box[0]
        self.h = box[3] - box[1]

    def project(self, point):
        """将关键点的绝对值转换为相对于左上角坐标偏移并归一化
        参数：
          point：某一关键点坐标(x,y)
        返回值：
          处理后偏移
        """
        x = (point[0] - self.x) / self.w
        y = (point[1] - self.y) / self.h
        return np.asarray([x, y])

    def reproject(self, point):
        """将关键点的相对值转换为绝对值，与project相反
        参数：
          point:某一关键点的相对归一化坐标
        返回值：
          处理后的绝对坐标
        """
        x = self.x + self.w * point[0]
        y = self.y + self.h * point[1]
        return np.asarray([x, y])

    def reprojectLandmark(self, landmark):
        """对所有关键点进行reproject操作"""
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.reproject(landmark[i])
        return p

    def projectLandmark(self, landmark):
        """对所有关键点进行project操作"""
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.project(landmark[i])
        return p


# 预处理数据，转化图像尺度并对像素归一
def processed_image(img, scale):
    height, width, channels = img.shape
    new_height = int(height * scale)
    new_width = int(width * scale)
    new_dim = (new_width, new_height)
    img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)
    # 把图片转换成numpy值
    image = np.array(img_resized).astype(np.float32)
    # 转换成CHW
    image = image.transpose((2, 0, 1))
    # 归一化
    image = (image - 127.5) / 128
    return image


def IOU(box, boxes):
    """裁剪的box和图片所有人脸box的iou值
    参数：
      box：裁剪的box,当box维度为4时表示box左上右下坐标，维度为5时，最后一维为box的置信度
      boxes：图片所有人脸box,[n,4]
    返回值：
      iou值，[n,]
    """
    # box面积
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    # boxes面积,[n,]
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    # 重叠部分左上右下坐标
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # 重叠部分长宽
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    # 重叠部分面积
    inter = w * h
    return inter / (box_area + area - inter + 1e-10)


def get_landmark_from_lfw_neg(txt, data_path, with_landmark=True):
    """获取txt中的图像路径，人脸box，人脸关键点
    参数：
      txt：数据txt文件
      data_path:数据存储目录
      with_landmark:是否留有关键点
    返回值：
      result包含(图像路径，人脸box，关键点)
    """
    with open(txt, 'r') as f:
        lines = f.readlines()
    result = []
    for line in lines:
        line = line.strip()
        components = line.split(' ')
        # 获取图像路径
        img_path = os.path.join(data_path, components[0]).replace('\\', '/')
        # 人脸box
        box = (components[1], components[3], components[2], components[4])
        box = [float(_) for _ in box]
        box = list(map(int, box))

        if not with_landmark:
            result.append((img_path, BBox(box)))
            continue
        # 五个关键点(x,y)
        landmark = np.zeros((5, 2))
        for index in range(5):
            rv = (float(components[5 + 2 * index]), float(components[5 + 2 * index + 1]))
            landmark[index] = rv
        result.append((img_path, BBox(box), landmark))
    return result


def get_landmark_from_celeba(data_path, with_landmark=True):
    """获取celeba的脸box，人脸关键点
    参数：
      bbox_txt：数据bbox文件
      landmarks_txt：数据landmarks文件
      data_path:数据存储目录
      with_landmark:是否留有关键点
    返回值：
      result包含(图像路径，人脸box，关键点)
    """
    bbox_txt = os.path.join(data_path, 'list_bbox_celeba.txt')
    landmarks_txt = os.path.join(data_path, 'list_landmarks_celeba.txt')
    # 获取图像路径，box，关键点
    if not os.path.exists(bbox_txt):
        return []
    with open(bbox_txt, 'r') as f:
        bbox_lines = f.readlines()
    with open(landmarks_txt, 'r') as f:
        landmarks_lines = f.readlines()
    result = []
    for i in range(2, len(bbox_lines)):
        bbox_line = bbox_lines[i]
        landmarks_line = landmarks_lines[i]
        bbox_components = bbox_line.strip().split()
        landmarks_components = landmarks_line.strip().split()
        # 获取图像路径
        img_path = os.path.join(data_path, 'img_celeba', bbox_components[0]).replace('\\', '/')
        # 人脸box
        box = (bbox_components[1], bbox_components[2], bbox_components[3], bbox_components[4])
        box = [float(_) for _ in box]
        box = list(map(int, box))
        box = [box[0], box[1], box[2] + box[0], box[3] + box[1]]

        if not with_landmark:
            result.append((img_path, BBox(box)))
            continue
        # 五个关键点(x,y)
        landmark = np.zeros((5, 2))
        for index in range(5):
            rv = (float(landmarks_components[1 + 2 * index]), float(landmarks_components[1 + 2 * index + 1]))
            landmark[index] = rv
        result.append((img_path, BBox(box), landmark))
    return result


def combine_data_list(data_dir):
    """把每个数据列表放在同一个文件上
    参数：
      data_dir：已经裁剪后的文件夹
    """
    npr = np.random
    with open(os.path.join(data_dir, 'positive.txt'), 'r') as f:
        pos = f.readlines()
    with open(os.path.join(data_dir, 'negative.txt'), 'r') as f:
        neg = f.readlines()
    with open(os.path.join(data_dir, 'part.txt'), 'r') as f:
        part = f.readlines()
    with open(os.path.join(data_dir, 'landmark.txt'), 'r') as f:
        landmark = f.readlines()
    with open(os.path.join(data_dir, 'all_data_list.txt'), 'w') as f:
        base_num = len(pos) // 1000 * 1000
        s1 = '整理前的数据：neg数量：{} pos数量：{} part数量:{} landmark: {} 基数:{}'.format(len(neg), len(pos), len(part),
                                                                            len(landmark), base_num)
        print(s1)
        # 打乱写入的数据顺序，并这里这里设置比例，设置size参数的比例就能得到数据集比例, 论文比例为：3:1:1:2
        neg_keep = npr.choice(len(neg), size=base_num * 3, replace=base_num * 3 > len(neg))
        part_keep = npr.choice(len(part), size=base_num, replace=base_num > len(part))
        pos_keep = npr.choice(len(pos), size=base_num, replace=base_num > len(pos))
        landmark_keep = npr.choice(len(landmark), size=base_num * 2, replace=base_num * 2 > len(landmark))

        s2 = '整理后的数据：neg数量：{} pos数量：{} part数量:{} landmark数量：{}'.format(len(neg_keep), len(pos_keep),
                                                                       len(part_keep), len(landmark_keep))
        print(s2)
        with open(os.path.join(data_dir, 'temp.txt'), 'a', encoding='utf-8') as f_temp:
            f_temp.write('%s\n' % s1)
            f_temp.write('%s\n' % s2)
            f_temp.flush()
            
        # 开始写入列表数据
        for i in pos_keep:
            f.write(pos[i].replace('\\', '/'))
        for i in neg_keep:
            f.write(neg[i].replace('\\', '/'))
        for i in part_keep:
            f.write(part[i].replace('\\', '/'))
        for i in landmark_keep:
            f.write(landmark[i].replace('\\', '/'))


def crop_landmark_image(data_dir, data_list, size, argument=True):
    """裁剪并保存带有人脸关键点的图片
    参数：
      data_dir：数据目录
      size:裁剪图片的大小
      argument:是否进行数据增强
    """
    npr = np.random
    image_id = 0

    # 数据输出路径
    output = os.path.join(data_dir, str(size))
    if not os.path.exists(output):
        os.makedirs(output)

    # 图片处理后输出路径
    dstdir = os.path.join(output, 'landmark')
    if not os.path.exists(dstdir):
        os.mkdir(dstdir)

    # 记录label的txt
    f = open(os.path.join(output, 'landmark.txt'), 'w')
    idx = 0
    for (imgPath, box, landmarkGt) in tqdm(data_list):
        # 存储人脸图片和关键点
        F_imgs = []
        F_landmarks = []
        img = cv2.imread(imgPath)

        img_h, img_w, img_c = img.shape
        # 转换成numpy值
        gt_box = np.array([box.left, box.top, box.right, box.bottom])
        # 裁剪人脸图片
        f_face = img[box.top:box.bottom + 1, box.left:box.right + 1]
        try:
            # resize成网络输入大小
            f_face = cv2.resize(f_face, (size, size))
        except Exception as e:
            print(e)
            print('resize成网络输入大小，跳过')
            continue

        # 创建一个空的关键点变量
        landmark = np.zeros((5, 2))
        for index, one in enumerate(landmarkGt):
            # 关键点相对于左上坐标偏移量并归一化，这个就保证了关键点都处于box内
            rv = ((one[0] - gt_box[0]) / (gt_box[2] - gt_box[0]), (one[1] - gt_box[1]) / (gt_box[3] - gt_box[1]))
            landmark[index] = rv

        F_imgs.append(f_face)
        F_landmarks.append(landmark.reshape(10))

        # 做数据增强处理
        if argument:
            landmark = np.zeros((5, 2))
            # 对图像变换
            idx = idx + 1
            x1, y1, x2, y2 = gt_box
            gt_w = x2 - x1 + 1
            gt_h = y2 - y1 + 1
            # 除去过小图像
            if max(gt_w, gt_h) < 40 or x1 < 0 or y1 < 0:
                continue
            for i in range(10):
                # 随机裁剪图像大小
                box_size = npr.randint(int(min(gt_w, gt_h) * 0.8), np.ceil(1.25 * max(gt_w, gt_h)))
                # 随机左上坐标偏移量
                try:
                    delta_x = npr.randint(-gt_w * 0.2, gt_w * 0.2)
                    delta_y = npr.randint(-gt_h * 0.2, gt_h * 0.2)
                except Exception as e:
                    print(e)
                    print('随机裁剪图像大小，跳过')
                    continue
                # 计算左上坐标
                nx1 = int(max(x1 + gt_w / 2 - box_size / 2 + delta_x, 0))
                ny1 = int(max(y1 + gt_h / 2 - box_size / 2 + delta_y, 0))
                nx2 = nx1 + box_size
                ny2 = ny1 + box_size
                # 除去超过边界的
                if nx2 > img_w or ny2 > img_h:
                    continue
                # 裁剪边框，图片
                crop_box = np.array([nx1, ny1, nx2, ny2])
                cropped_im = img[ny1:ny2 + 1, nx1:nx2 + 1, :]
                resized_im = cv2.resize(cropped_im, (size, size))
                # 计算iou值
                iou = IOU(crop_box, np.expand_dims(gt_box, 0))

                # 只保留pos图像
                if iou > 0.65:
                    F_imgs.append(resized_im)
                    # 关键点相对偏移
                    for index, one in enumerate(landmarkGt):
                        rv = ((one[0] - nx1) / box_size, (one[1] - ny1) / box_size)
                        landmark[index] = rv
                    F_landmarks.append(landmark.reshape(10))
                    landmark = np.zeros((5, 2))
                    landmark_ = F_landmarks[-1].reshape(-1, 2)
                    box = BBox([nx1, ny1, nx2, ny2])
                    # 镜像
                    if random.choice([0, 1]) > 0:
                        face_flipped, landmark_flipped = flip(resized_im, landmark_)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(10))
                    # 逆时针翻转
                    if random.choice([0, 1]) > 0:
                        face_rotated_by_alpha, landmark_rorated = rotate(img, box, box.reprojectLandmark(landmark_), 5)
                        # 关键点偏移
                        landmark_rorated = box.projectLandmark(landmark_rorated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rorated.reshape(10))

                        # 左右翻转
                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rorated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(10))
                    # 顺时针翻转
                    if random.choice([0, 1]) > 0:
                        face_rotated_by_alpha, landmark_rorated = rotate(img, box, box.reprojectLandmark(landmark_), -5)
                        # 关键点偏移
                        landmark_rorated = box.projectLandmark(landmark_rorated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rorated.reshape(10))

                        # 左右翻转
                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rorated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(10))
        F_imgs, F_landmarks = np.asarray(F_imgs), np.asarray(F_landmarks)

        # 开始保存裁剪的图片和标注信息
        for i in range(len(F_imgs)):
            # 剔除数据偏移量在[0,1]之间
            if np.sum(np.where(F_landmarks[i] <= 0, 1, 0)) > 0:
                continue
            if np.sum(np.where(F_landmarks[i] >= 1, 1, 0)) > 0:
                continue
            # 保存裁剪带有关键点的图片
            cv2.imwrite(os.path.join(dstdir, '%d.jpg' % (image_id)), F_imgs[i])
            # 把图片路径和label，还有关键点保存到数据列表上
            landmarks = list(map(str, list(F_landmarks[i])))
            f.write(os.path.join(dstdir, '%d.jpg' % (image_id)) + ' -2 ' + ' '.join(landmarks) + '\n')
            image_id += 1
    f.close()


# 镜像处理
def flip(face, landmark):
    face_flipped_by_x = cv2.flip(face, 1)
    landmark_ = np.asarray([(1 - x, y) for (x, y) in landmark])
    landmark_[[0, 1]] = landmark_[[1, 0]]
    landmark_[[3, 4]] = landmark_[[4, 3]]
    return face_flipped_by_x, landmark_


# 旋转处理
def rotate(img, box, landmark, alpha):
    center = ((box.left + box.right) / 2, (box.top + box.bottom) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)
    img_rotated_by_alpha = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))
    landmark_ = np.asarray([(rot_mat[0][0] * x + rot_mat[0][1] * y + rot_mat[0][2],
                             rot_mat[1][0] * x + rot_mat[1][1] * y + rot_mat[1][2]) for (x, y) in landmark])
    face = img_rotated_by_alpha[box.top:box.bottom + 1, box.left:box.right + 1]
    return face, landmark_


def convert_to_square(box):
    """将box转换成更大的正方形
    参数：
      box：预测的box,[n,5]
    返回值：
      调整后的正方形box，[n,5]
    """
    square_box = box.copy()
    h = box[:, 3] - box[:, 1] + 1
    w = box[:, 2] - box[:, 0] + 1
    # 找寻正方形最大边长
    max_side = np.maximum(w, h)

    square_box[:, 0] = box[:, 0] + w * 0.5 - max_side * 0.5
    square_box[:, 1] = box[:, 1] + h * 0.5 - max_side * 0.5
    square_box[:, 2] = square_box[:, 0] + max_side - 1
    square_box[:, 3] = square_box[:, 1] + max_side - 1
    return square_box


def read_annotation(data_path, label_path):
    """
    从原标注数据中获取图片路径和标注box
    :param data_path: 数据的根目录
    :param label_path: 标注数据的文件
    :return:
    """
    data = dict()
    images = []
    bboxes = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        labels = line.strip().split(' ')
        # 图像地址
        imagepath = labels[0]
        # 如果有一行为空，就停止读取
        if not imagepath:
            break
        # 获取图片的路径
        imagepath = data_path + 'WIDER_train/images/' + imagepath + '.jpg'
        images.append(imagepath)
        # 根据人脸的数目开始读取所有box
        one_image_bboxes = []
        for i in range(0, len(labels) - 1, 4):
            xmin = float(labels[1 + i])
            ymin = float(labels[2 + i])
            xmax = float(labels[3 + i])
            ymax = float(labels[4 + i])

            one_image_bboxes.append([xmin, ymin, xmax, ymax])

        bboxes.append(one_image_bboxes)

    data['images'] = images
    data['bboxes'] = bboxes
    return data


def pad(bboxes, w, h):
    """将超出图像的box进行处理
    参数：
      bboxes:人脸框
      w,h:图像长宽
    返回值：
      dy, dx : 为调整后的box的左上角坐标相对于原box左上角的坐标
      edy, edx : n为调整后的box右下角相对原box左上角的相对坐标
      y, x : 调整后的box在原图上左上角的坐标
      ex, ex : 调整后的box在原图上右下角的坐标
      tmph, tmpw: 原始box的长宽
    """
    # box的长宽
    tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1, bboxes[:, 3] - bboxes[:, 1] + 1
    num_box = bboxes.shape[0]

    dx, dy = np.zeros((num_box,)), np.zeros((num_box,))
    edx, edy = tmpw.copy() - 1, tmph.copy() - 1
    # box左上右下的坐标
    x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    # 找到超出右下边界的box并将ex,ey归为图像的w,h
    # edx,edy为调整后的box右下角相对原box左上角的相对坐标
    tmp_index = np.where(ex > w - 1)
    edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
    ex[tmp_index] = w - 1

    tmp_index = np.where(ey > h - 1)
    edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
    ey[tmp_index] = h - 1
    # 找到超出左上角的box并将x,y归为0
    # dx,dy为调整后的box的左上角坐标相对于原box左上角的坐标
    tmp_index = np.where(x < 0)
    dx[tmp_index] = 0 - x[tmp_index]
    x[tmp_index] = 0

    tmp_index = np.where(y < 0)
    dy[tmp_index] = 0 - y[tmp_index]
    y[tmp_index] = 0

    return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
    return_list = [item.astype(np.int32) for item in return_list]

    return return_list


def calibrate_box(bbox, reg):
    """校准box
    参数：
      bbox:pnet生成的box

      reg:rnet生成的box偏移值
    返回值：
      调整后的box是针对原图的绝对坐标
    """

    bbox_c = bbox.copy()
    w = bbox[:, 2] - bbox[:, 0] + 1
    w = np.expand_dims(w, 1)
    h = bbox[:, 3] - bbox[:, 1] + 1
    h = np.expand_dims(h, 1)
    reg_m = np.hstack([w, h, w, h])
    aug = reg_m * reg
    bbox_c[:, 0:4] = bbox_c[:, 0:4] + aug
    return bbox_c


def py_nms(dets, thresh, mode="Union"):
    """
    贪婪策略选择人脸框
    keep boxes overlap <= thresh
    rule out overlap > thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap <= thresh
    :return: indexes to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 将概率值从大到小排列
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "Union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        # 保留小于阈值的下标，因为order[0]拿出来做比较了，所以inds+1是原来对应的下标
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def generate_bbox(cls_map, reg, scale, threshold):
    """
     得到对应原图的box坐标，分类分数，box偏移量
    """
    # pnet大致将图像size缩小2倍
    stride = 2

    cellsize = 12

    # 将置信度高的留下
    t_index = np.where(cls_map > threshold)

    # 没有人脸
    if t_index[0].size == 0:
        return np.array([])
    # 偏移量
    dx1, dy1, dx2, dy2 = [reg[i, t_index[0], t_index[1]] for i in range(4)]

    reg = np.array([dx1, dy1, dx2, dy2])
    score = cls_map[t_index[0], t_index[1]]
    # 对应原图的box坐标，分类分数，box偏移量
    boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),
                             np.round((stride * t_index[0]) / scale),
                             np.round((stride * t_index[1] + cellsize) / scale),
                             np.round((stride * t_index[0] + cellsize) / scale),
                             score,
                             reg])
    # shape[n,9]
    return boundingbox.T


# 合并图像后删除原来的文件
def delete_old_img(old_image_folder, image_size):
    shutil.rmtree(os.path.join(old_image_folder, str(image_size), 'positive'), ignore_errors=True)
    shutil.rmtree(os.path.join(old_image_folder, str(image_size), 'negative'), ignore_errors=True)
    shutil.rmtree(os.path.join(old_image_folder, str(image_size), 'part'), ignore_errors=True)
    shutil.rmtree(os.path.join(old_image_folder, str(image_size), 'landmark'), ignore_errors=True)
    
    # 删除原来的数据列表文件
    os.remove(os.path.join(old_image_folder, str(image_size), 'positive.txt'))
    os.remove(os.path.join(old_image_folder, str(image_size), 'negative.txt'))
    os.remove(os.path.join(old_image_folder, str(image_size), 'part.txt'))
    os.remove(os.path.join(old_image_folder, str(image_size), 'landmark.txt'))


def save_hard_example(data_path, save_size):
    """
    根据预测的结果裁剪下一个网络所需要训练的图片的标注数据
    :param data_path: 数据的根目录
    :param save_size: 裁剪图片的大小
    :return:
    """
    # 获取原数据集中的标注数据
    filename = os.path.join(data_path, 'wider_face_train.txt')
    data = read_annotation(data_path, filename)

    # 获取原数据集中的图像路径和标注信息
    im_idx_list = data['images']
    gt_boxes_list = data['bboxes']

    # 保存裁剪图片数据文件夹
    pos_save_dir = os.path.join(data_path, '%d/positive' % save_size)
    part_save_dir = os.path.join(data_path, '%d/part' % save_size)
    neg_save_dir = os.path.join(data_path, '%d/negative' % save_size)

    # 创建文件夹
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(pos_save_dir):
        os.mkdir(pos_save_dir)
    if not os.path.exists(part_save_dir):
        os.mkdir(part_save_dir)
    if not os.path.exists(neg_save_dir):
        os.mkdir(neg_save_dir)

    # 保存图片数据的列表文件
    neg_file = open(os.path.join(data_path, '%d/negative.txt' % save_size), 'w')
    pos_file = open(os.path.join(data_path, '%d/positive.txt' % save_size), 'w')
    part_file = open(os.path.join(data_path, '%d/part.txt' % save_size), 'w')

    # 读取预测结果
    det_boxes = pickle.load(open(os.path.join(data_path, '%d/detections.pkl' % save_size), 'rb'))

    # 保证预测结果和本地数据数量是一样的
    assert len(det_boxes) == len(im_idx_list), "预测结果和真实数据数量不一致"

    # 图片的命名
    n_idx = 0
    p_idx = 0
    d_idx = 0

    # 开始裁剪下一个网络的训练图片
    pbar = tqdm(total=len(im_idx_list))
    for im_idx, dets, gts in zip(im_idx_list, det_boxes, gt_boxes_list):
        pbar.update(1)
        # 把原标注数据集以4个数据作为一个box进行变形
        gts = np.array(gts, dtype=np.float32).reshape(-1, 4)

        # 如果没有预测到数据就调成本次循环
        if dets.shape[0] == 0:
            continue

        # 读取原图像
        img = cv2.imread(im_idx)

        # 把预测数据转换成正方形
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        neg_num = 0
        for box in dets:
            # 获取预测结果中单张图片中的单个人脸坐标，和人脸的宽高
            x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # 除去过小的
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue

            # 计算iou值
            Iou = IOU(box, gts)

            # 裁剪并统一大小图片
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (save_size, save_size), interpolation=cv2.INTER_LINEAR)

            # 划分种类
            if np.max(Iou) < 0.3 and neg_num < 60:
                # 保存negative图片，同时也避免产生太多的negative图片
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                # 指定label为0
                neg_file.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1
            else:
                # 或者最大iou值的真实box坐标数据
                idx = np.argmax(Iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt

                # 计算偏移量
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                # pos和part
                if np.max(Iou) >= 0.65:
                    # 保存positive图片，同时也避免产生太多的positive图片
                    save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                    # 指定label为1
                    pos_file.write(
                        save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1

                elif np.max(Iou) >= 0.4:
                    # 保存part图片，同时也避免产生太多的part图片
                    save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                    # 指定label为-1
                    part_file.write(
                        save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
    pbar.close()
    neg_file.close()
    part_file.close()
    pos_file.close()
