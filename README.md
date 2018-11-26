#### 一.什么是人脸识别

​	人脸识别，是基于人的脸部特征信息进行身份识别的一种生物识别技术。用摄像机或摄像头采集含有人脸的图像或视频流，并自动在图像中检测和跟踪人脸，进而对检测到的人脸进行脸部识别的一系列相关技术，通常也叫做人像识别、面部识别。

​	人脸识别系统主要包括四个组成部分，分别为：人脸图像采集及检测、人脸图像预处理、人脸图像特征提取以及匹配与识别。

#### 二.相关库和配置环境

​	1.安装CMake

​	如果你是Ubuntu用户就可以跳过你也可以选择升级CMake

```bash

```

​	2.Dlib下载

​	这是Dlib的github网站https://github.com/davisking/dlib

​	![在这里插入图片描述](C:\\Users11193\\Desktop\\通过人脸识别来返回学号信息(一卡通)\a1.png)

​	箭头1就是clone地址如果你有GitHub可以输入以下代码(Ubuntu系统)

​	

```bash
$ mkdir Dlib
$ cd Dlib
$ git init
$ git clone #复制clone地址
```



​	箭头2就是直接下载ZIP压缩包进行解压就好了

​	下好后进入Dlib文件

```bash
$ cd Dlib
$ python setup.py install
#等待安装
```

​	3.安装 scikit-image

```bash
$ pip install scikit-image
```

​	三.开始人脸识别

​	这里，shape_predictor_68_face_landmarks.dat是已经训练好的人脸关键点检测器。

​	dlib_face_recognition_resnet_model_v1.dat是训练好的ResNet人脸识别模型。

​	所有文件都可以在以下网址下载：<http://dlib.net/files/>。

![](C:\Users\11193\Desktop\通过人脸识别来返回学号信息(一卡通)\a2.png)

​	其中all_faces文件夹是放我们学校所有人一卡通上的证件照

​	首先我们程序运行的思路是通过all_face_128D.py这个程序来先将所有证件照的描述子提取（128D向量）然后与照片文件的名字组成字典，最后将这个字典存入本地。这时我们就可以用face_rec.py来对证件照识别通过计算传入的人脸描述子与保存的字典中的描述子进行计算欧式距离。

​	以下是all_face_128D.py代码：



```python
#coding=utf-8
import sys
import dlib
import numpy as np
import os
import glob
from skimage import io
import re
import json

current_path = os.getcwd() #获取当前路径
predictor_path = current_path + '/model/shape_predictor_68_face_landmarks.dat'  # 1.人脸关键点检测器
face_rec_model_path = current_path + '/model/dlib_face_recognition_resnet_model_v1.dat'   # 2.人脸识别模型
face_folder_path = current_path + '/all_faces'    # 3.所有人脸文件夹


detector = dlib.get_frontal_face_detector() #加载正脸识别器
shaper = dlib.shape_predictor(predictor_path) #加载人脸关键识别器
facerec = dlib.face_recognition_model_v1(face_rec_model_path) #加载人脸模型

all_face_d = {}
for img_path in glob.glob(os.path.join(face_folder_path,"*.jpg")):
    print("processing file: {}".format(img_path))
    a = os.path.basename(img_path)
    b = re.compile('\d+')
    filename = b.findall(a)[0]
    print(filename)
    img = io.imread(img_path)
    dets = detector(img,1)

    print("number of faces detected: {}".format(len(dets)))
    for index, face in enumerate(dets):
        shape = shaper(img, face)      # 2.关键点检测
        # 3.描述子提取，128D向量
        face_desciptor = facerec.compute_face_descriptor(img, shape)  
        print(face_desciptor)
        vector = np.array(face_desciptor)    #转换为numpy array
        all_face_d.setdefault(filename,vector)

np.save('all_face_vectors.npy',all_face_d)
```



然后是face_rec.py的代码：

```python
#coding=utf-8
import sys
import dlib
import numpy as np
import os
import glob
from skimage import io
import base64


current_path = os.getcwd() #获取当前路径
predictor_path = current_path + '/model/shape_predictor_68_face_landmarks.dat'  # 1.人脸关键点检测器
face_rec_model_path = current_path + '/model/dlib_face_recognition_resnet_model_v1.dat'   # 2.人脸识别模型
face_folder_path = current_path + '/all_face'    # 3.所有人脸文件夹
imgs_path = io.imread('img/img.jpg')    # 4.图片地址


detector = dlib.get_frontal_face_detector() #加载正脸识别器
shaper = dlib.shape_predictor(predictor_path) #加载人脸关键识别器
facerec = dlib.face_recognition_model_v1(face_rec_model_path) #加载人脸模型 

all_face_vectors = np.load('all_face_vectors.npy')
data = all_face_vectors.item()




def face_recognition(img_path):
    img = img_path
    dets = detector(img, 1)
    dist = {}
    for index, face in enumerate(dets):
        shape = shaper(img, face)      # 2.关键点检测
        # 3.描述子提取，128D向量
        face_desciptor = facerec.compute_face_descriptor(img, shape)   
        d_test = np.array(face_desciptor)    #转换为numpy array
        
        for key,value in data.items():
            dist_ = np.linalg.norm(value-d_test)
            dist.setdefault(key,dist_)
            
    
    cd_sorted = sorted(dist.items(), key=lambda d:d[1])
    #print('\n The person ID is: %s' % ( cd_sorted[0][0] )  ) 
    return cd_sorted[0][0]

if __name__ == '__main__':
        face_recognition(imgs_path)
```

  以上就是所有代码和基本思路
