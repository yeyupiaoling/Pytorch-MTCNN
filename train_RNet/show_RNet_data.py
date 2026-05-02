import cv2
import numpy as np
import os
import sys
sys.path.append("../")
from utils.data import ImageData

def show_rnet_data(data_path, num_samples=10):
    """
    显示RNet训练数据，随机选择每种类型的样本并拼接成一个大图
    参数：
      data_path: 数据路径
      num_samples: 每种类型选择的样本数量
    """
    image_data = ImageData(data_path)
    keys = list(image_data.get_keys())
    np.random.shuffle(keys)
    
    # 分类收集样本
    samples = {
        'positive': [],
        'negative': [],
        'part': [],
        'landmark': []
    }
    
    # 标签映射
    label_map = {
        1: 'positive',
        0: 'negative',
        -1: 'part',
        -2: 'landmark'
    }
    
    # 遍历所有数据并分类
    for key in keys:
        label = image_data.get_label(key)
        if label is None:
            continue
        
        label_name = label_map.get(label)
        if label_name is None:
            continue
            
        if len(samples[label_name]) < num_samples:
            img_bytes = image_data.get_img(key)
            if img_bytes is None:
                continue
            
            # 解码图像
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is not None and len(samples[label_name]) < num_samples:
                samples[label_name].append(img)
    
    # 创建显示图像
    display_images = []
    labels_text = []
    
    for label_name, imgs in samples.items():
        if len(imgs) > 0:
            display_images.extend(imgs)
            labels_text.extend([label_name] * len(imgs))
    
    if not display_images:
        print("没有找到可显示的数据")
        return
    
    # 图像尺寸
    img_height, img_width = display_images[0].shape[:2]
    
    # 计算网格布局 - 每行显示num_samples张图
    rows = 4  # 4种类型
    cols = num_samples
    
    # 根据图像尺寸设置单元格大小，确保不会太小
    min_cell_size = 60
    cell_width = max(img_width * 2, min_cell_size)  # RNet图像放大2倍
    cell_height = max(img_height * 2, min_cell_size)
    
    total_width = cols * cell_width
    total_height = rows * cell_height
    
    # 创建白色背景的大图
    collage = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
    
    # 放置图像和标签
    for row_idx, label_name in enumerate(['positive', 'negative', 'part', 'landmark']):
        imgs = samples[label_name]
        for col_idx in range(num_samples):
            if col_idx < len(imgs):
                img = imgs[col_idx]
                # 放大图像到单元格大小
                display_size = (cell_width - 10, cell_height - 20)
                img_resized = cv2.resize(img, display_size, interpolation=cv2.INTER_NEAREST)
                
                y_offset = row_idx * cell_height + 15
                x_offset = col_idx * cell_width + 5
                
                # 放置图像
                collage[y_offset:y_offset + img_resized.shape[0], 
                       x_offset:x_offset + img_resized.shape[1]] = img_resized
                
                # 添加类别标签
                label_text = f"{label_name[:3].upper()}"
                cv2.putText(collage, label_text, 
                           (x_offset, y_offset - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 0, 255), 1)
    
    # 显示统计信息
    stats_text = f"Pos:{len(samples['positive'])} Neg:{len(samples['negative'])} Part:{len(samples['part'])} Landmark:{len(samples['landmark'])}"
    cv2.putText(collage, stats_text,
               (total_width // 2 - 150, total_height - 10),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.5, (0, 0, 0), 1)
    
    # 显示图像
    cv2.imshow('RNet Training Data', collage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 保存图像
    output_path = os.path.join(os.path.dirname(data_path), 'RNet_data_preview.png')
    cv2.imwrite(output_path, collage)
    print(f"图像已保存到: {output_path}")


if __name__ == '__main__':
    # RNet数据路径
    data_path = '../dataset/24/all_data'
    show_rnet_data(data_path, num_samples=10)
