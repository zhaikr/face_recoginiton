#coding=utf-8
import sys
import dlib
import numpy as np
import os
import glob
from skimage import io
import re

current_path = os.getcwd() #获取当前路径
predictor_path = current_path + '/model/shape_predictor_68_face_landmarks.dat'  # 1.人脸关键点检测器
face_rec_model_path = current_path + '/model/dlib_face_recognition_resnet_model_v1.dat'   # 2.人脸识别模型
face_folder_path = current_path + '/all_face'    # 3.所有人脸文件夹


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
        face_desciptor = facerec.compute_face_descriptor(img, shape)   # 3.描述子提取，128D向量
        print(face_desciptor)
        vector = np.array(face_desciptor)    #转换为numpy array
        all_face_d.setdefault(filename,vector)

np.save('all_face_vectors.npy',all_face_d)

