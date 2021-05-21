# 前言

MTCNN，Multi-task convolutional neural network（多任务卷积神经网络），将人脸区域检测与人脸关键点检测放在了一起，总体可分为P-Net、R-Net、和O-Net三层网络结构。它是2016年中国科学院深圳研究院提出的用于人脸检测任务的多任务神经网络模型，该模型主要采用了三个级联的网络，采用候选框加分类器的思想，进行快速高效的人脸检测。这三个级联的网络分别是快速生成候选窗口的P-Net、进行高精度候选窗口过滤选择的R-Net和生成最终边界框与人脸关键点的O-Net。和很多处理图像问题的卷积神经网络模型，该模型也用到了图像金字塔、边框回归、非最大值抑制等技术。


# 环境
 - Pytorch 1.8.1
 - Python 3.7

# 文件介绍
 - `models/Loss.py` MTCNN所使用的损失函数，包括分类损失函数、人脸框损失函数、关键点损失函数
 - `models/PNet.py` PNet网络结构
 - `models/RNet.py` RNet网络结构
 - `models/ONet.py` ONet网络结构
 - `utils/data_format_converter.py` 把大量的图片合并成一个文件
 - `utils/data.py` 训练数据读取器
 - `utils/utils.py` 各种工具函数
 - `train_PNet/generate_PNet_data.py` 生成PNet训练的数据
 - `train_PNet/train_PNet.py` 训练PNet网络模型
 - `train_RNet/generate_RNet_data.py` 生成RNet训练的数据
 - `train_RNet/train_RNet.py` 训练RNet网络模型
 - `train_ONet/generate_ONet_data.py` 生成ONet训练的数据
 - `train_ONet/train_ONet.py` 训练ONet网络模型
 - `infer_path.py` 使用路径预测图像，检测图片上人脸的位置和关键的位置，并显示
 - `infer_camera.py` 预测图像程序，检测图片上人脸的位置和关键的位置实时显示


# 数据集下载
 - [WIDER Face](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) 下载训练数据WIDER Face Training Images，解压的WIDER_train文件夹放置到dataset下。并下载 [Face annotations](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip) ，解压把里面的 wider_face_train_bbx_gt.txt 文件放在dataset目录下，
 - [Deep Convolutional Network Cascade for Facial Point Detection](http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm) 。下载 Training set 并解压，将里面的 lfw_5590 和 net_7876 文件夹放置到dataset下
 - 解压数据集之后，`dataset`目录下应该有文件夹`lfw_5590`，`net_7876`，`WIDER_train`，有标注文件`testImageList.txt`，`trainImageList.txt`，`wider_face_train.txt`


# 训练模型

训练模型一共分为三步，分别是训练PNet模型、训练RNet模型、训练ONet模型，每一步训练都依赖上一步的结果。

## 第一步 训练PNet模型
PNet全称为Proposal Network，其基本的构造是一个全卷积网络，P-Net是一个人脸区域的区域建议网络，该网络的将特征输入结果三个卷积层之后，通过一个人脸分类器判断该区域是否是人脸，同时使用边框回归。

![PNet](https://img-blog.csdnimg.cn/2021031622070120.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMzMjAwOTY3,size_16,color_FFFFFF,t_70)
 - `cd train_PNet` 切换到`train_PNet`文件夹
 - `python3 generate_PNet_data.py` 首先需要生成PNet模型训练所需要的图像数据
 - `python3 train_PNet.py` 开始训练PNet模型

## 第二步 训练RNet模型
全称为Refine Network，其基本的构造是一个卷积神经网络，相对于第一层的P-Net来说，增加了一个全连接层，因此对于输入数据的筛选会更加严格。在图片经过P-Net后，会留下许多预测窗口，我们将所有的预测窗口送入R-Net，这个网络会滤除大量效果比较差的候选框，最后对选定的候选框进行Bounding-Box Regression和NMS进一步优化预测结果。

![RNet模型](https://img-blog.csdnimg.cn/20210316221211297.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMzMjAwOTY3,size_16,color_FFFFFF,t_70)
 - `cd train_RNet` 切换到`train_RNet`文件夹
 - `python3 generate_RNet_data.py` 使用上一步训练好的PNet模型生成RNet训练所需的图像数据
 - `python3 train_RNet.py` 开始训练RNet模型


## 第三步 训练ONet模型
ONet全称为Output Network，基本结构是一个较为复杂的卷积神经网络，相对于R-Net来说多了一个卷积层。O-Net的效果与R-Net的区别在于这一层结构会通过更多的监督来识别面部的区域，而且会对人的面部特征点进行回归，最终输出五个人脸面部特征点。

![ONet模型](https://img-blog.csdnimg.cn/20210316221433363.png)
 - `cd train_ONet` 切换到`train_ONet`文件夹
 - `python3 generate_ONet_data.py` 使用上两部步训练好的PNet模型和RNet模型生成ONet训练所需的图像数据
 - `python3 train_ONet.py` 开始训练ONet模型

# 预测

 - `python3 infer_path.py` 使用图像路径，识别图片中人脸box和关键点，并显示识别结果
![识别结果](https://img-blog.csdnimg.cn/2021040721044636.jpg)


 - `python3 infer_camera.py` 使用相机捕获图像，识别图片中人脸box和关键点，并显示识别结果


## 参考资料

1. https://github.com/AITTSMD/MTCNN-Tensorflow
2. https://blog.csdn.net/qq_36782182/article/details/83624357
