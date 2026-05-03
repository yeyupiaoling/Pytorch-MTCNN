import mmap

import cv2
import numpy as np
from torch.utils.data import Dataset


class ImageData(object):
    def __init__(self, data_path):
        self.offset_dict = {}
        for line in open(data_path + '.header', 'rb'):
            key, val_pos, val_len = line.split('\t'.encode('ascii'))
            self.offset_dict[key] = (int(val_pos), int(val_len))
        self.fp = open(data_path + '.data', 'rb')
        self.m = mmap.mmap(self.fp.fileno(), 0, access=mmap.ACCESS_READ)
        print('正在加载数据标签...')
        # 获取label
        self.label = {}
        self.box = {}
        self.landmark = {}
        label_path = data_path + '.label'
        for line in open(label_path, 'rb'):
            key, bbox, landmark, label = line.split(b'\t')
            self.label[key] = int(label)
            self.box[key] = [float(x) for x in bbox.split()]
            self.landmark[key] = [float(x) for x in landmark.split()]
        print('数据加载完成，总数据量为：%d' % len(self.label))

    # 获取图像数据
    def get_img(self, key):
        p = self.offset_dict.get(key, None)
        if p is None:
            return None
        val_pos, val_len = p
        return self.m[val_pos:val_pos + val_len]

    # 获取图像标签
    def get_label(self, key):
        return self.label.get(key)

    # 获取人脸box
    def get_bbox(self, key):
        return self.box.get(key)

    # 获取关键点
    def get_landmark(self, key):
        return self.landmark.get(key)

    # 获取所有keys
    def get_keys(self):
        return self.label.keys()


# 数据加载器
class CustomDataset(Dataset):
    def __init__(self, data_path, is_train=True, augment_prob=0.8):
        super(CustomDataset, self).__init__()
        self.is_train = is_train
        self.augment_prob = augment_prob
        self.imageData = ImageData(data_path)
        self.keys = self.imageData.get_keys()
        self.keys = list(self.keys)
        np.random.shuffle(self.keys)

    @staticmethod
    def _clip_uint8(img):
        return np.clip(img, 0, 255).astype(np.uint8)

    @staticmethod
    def _random_horizontal_flip(img, label, bbox, landmark):
        if np.random.rand() >= 0.5:
            return img, bbox, landmark

        img = cv2.flip(img, 1)

        # bbox 是相对裁剪框的偏移量，水平翻转后需要交换左右边并取反。
        if abs(label) == 1:
            bbox = np.array([-bbox[2], bbox[1], -bbox[0], bbox[3]], dtype=np.float32)

        # landmark 采用 [左眼, 右眼, 鼻子, 左嘴角, 右嘴角]，翻转后要同步交换左右语义。
        if label == -2:
            landmark = landmark.reshape(5, 2).copy()
            landmark[:, 0] = 1.0 - landmark[:, 0]
            landmark[[0, 1]] = landmark[[1, 0]]
            landmark[[3, 4]] = landmark[[4, 3]]
            landmark = landmark.reshape(10).astype(np.float32)

        return img, bbox, landmark

    @staticmethod
    def _random_brightness_contrast(img):
        if np.random.rand() < 0.5:
            alpha = np.random.uniform(0.8, 1.2)
            beta = np.random.uniform(-18, 18)
            img = CustomDataset._clip_uint8(img.astype(np.float32) * alpha + beta)
        return img

    @staticmethod
    def _random_grayscale(img):
        if np.random.rand() < 0.15:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return img

    @staticmethod
    def _random_blur(img):
        if np.random.rand() < 0.2:
            img = cv2.GaussianBlur(img, (3, 3), sigmaX=np.random.uniform(0.2, 1.2))
        return img

    @staticmethod
    def _random_noise(img):
        if np.random.rand() < 0.2:
            noise = np.random.normal(0, np.random.uniform(2.0, 8.0), img.shape)
            img = CustomDataset._clip_uint8(img.astype(np.float32) + noise)
        return img

    @staticmethod
    def _random_jpeg_compression(img):
        if np.random.rand() < 0.15:
            quality = int(np.random.uniform(55, 95))
            success, encoded = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if success:
                decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if decoded is not None:
                    img = decoded
        return img

    @staticmethod
    def _random_occlusion(img):
        if np.random.rand() < 0.15:
            h, w = img.shape[:2]
            occ_w = max(1, int(w * np.random.uniform(0.1, 0.25)))
            occ_h = max(1, int(h * np.random.uniform(0.1, 0.25)))
            x1 = np.random.randint(0, max(1, w - occ_w + 1))
            y1 = np.random.randint(0, max(1, h - occ_h + 1))
            fill_value = np.random.randint(0, 256, size=(1, 1, 3), dtype=np.uint8)
            img[y1:y1 + occ_h, x1:x1 + occ_w] = fill_value
        return img

    def _apply_train_augmentations(self, img, label, bbox, landmark):
        # 训练期同时做几何增强和光照/压缩扰动，提升模型对翻转、模糊、噪声和遮挡的鲁棒性。
        img, bbox, landmark = self._random_horizontal_flip(img, label, bbox, landmark)
        img = self._random_brightness_contrast(img)
        img = self._random_grayscale(img)
        img = self._random_blur(img)
        img = self._random_noise(img)
        img = self._random_jpeg_compression(img)
        img = self._random_occlusion(img)
        return img, bbox, landmark

    def __getitem__(self, idx):
        key = self.keys[idx]
        img = self.imageData.get_img(key)
        assert (img is not None)
        label = self.imageData.get_label(key)
        assert (label is not None)
        bbox = self.imageData.get_bbox(key)
        landmark = self.imageData.get_landmark(key)
        img = np.frombuffer(img, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        assert (img is not None), 'img is None'
        bbox = np.array(bbox, np.float32)
        landmark = np.array(landmark, np.float32)

        if self.is_train and np.random.rand() < self.augment_prob:
            img, bbox, landmark = self._apply_train_augmentations(img, int(label), bbox, landmark)

        # 把图片转换成numpy值
        img = np.array(img).astype(np.float32)
        # 转换成CHW
        img = img.transpose((2, 0, 1))
        # 归一化
        img = (img - 127.5) / 128
        label = np.array([label], np.int64)
        return img, label, bbox, landmark

    def __len__(self):
        return len(self.keys)
