## 项目介绍

使用了YuNet模型用于检测人脸,基于Siamese Network实现人脸识别，以达到one-shot learning的效果。

整体框架
![](https://img.qylh.xyz/blog/1674898695967.png)

## 使用方法

1. 安装依赖包：`pip install -r requirements.txt`
2. 将需要检测的人脸集放置到`faceGallery`目录下,命名方式为`name.xxx`
3. 运行`main.py`

注意：Siamese模型基于Siamese-tf2训练实现，数据集使用CASIA人脸集，如需重新训练，请参考https://github.com/bubbliiiing/Siamese-tf2进行。（重新训练时主要保持输入图片大小相同）

## 人脸检测

此部分使用了南方科技大学于仕琪老师的YuNet实现，之所以选择YuNet是由于其检测速度很快，适合于运行在低性能的设备上，且OpenCV支持了这个模型。

该模型使用起来还是很简单的，具体可参考项目仓库：[ShiqiYu/libfacedetection](https://github.com/ShiqiYu/libfacedetection) 以及opencv_zoo仓库[face_detection_yunet](https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet)

但是要注意ONNX文件，是无法直接clone下来的需要用到`git lfs`进行pull参见[opencv_zoo使用说明](https://github.com/opencv/opencv_zoo),在我pull的过程中发现速度极慢，因此从其他地方找到了一个文件可以从我这里直接下载：链接：https://pan.baidu.com/s/1LW8frXu0fEe7tDXEHEO7Mg?pwd=vbnb 提取码：vbnb

注意这部分需要OpenCV >= 4.5.4以上版本

## 人脸识别

人脸识别部分是使用tf2做的Siamese孪生网络，基本原理是：训练出一个模型用于提取人脸特征，然后将两张图片输入模型计算相似度，相似度越高越有可能是同一个人。这样我们就可以通过比对样本与数据库中的图片之间的相似度，来判断人员的身份。

由于之前没怎么接触过AI的项目，这里是参考[孪生神经网络在tf2（tensorflow2）当中的实现](https://github.com/bubbliiiing/Siamese-tf2)进行训练实现的，但是训练完成后的模型需要输入两张图片进行对比，如果数据库中有100个人脸图片，我们除了对样本进行一次运算外，还需要反复运算100张现有的图片。

这样的话效率会很低下，因此我改进了一下模型，将模型拆分成了：提取特征和计算相似度两个部分，这样我们就可以将数据库中的图片提前运算好放到内存中，需要检测时，只需要运算测试图片，然后计算与所有图片之间的相似度即可。

其实拆分后少了一层（lambda）用于计算两张图片的差值，由于我不熟悉神经网络，没研究明白怎么设置input为两张照片，所以删除了这一层，改为在代码中实现。（其实就是向量相减再取绝对值）

### 原模型
原模型结构如下：

![原模型](https://img.qylh.xyz/blog/model.png)


### 拆分模型

分割为两个模型：

#### 模型一:特征提取

模型一用于提取特征：

![](https://img.qylh.xyz/blog/model1.png)


#### 模型二:相似度计算

模型二用于计算相似度：

![](https://img.qylh.xyz/blog/model2.png)



