import sys
import dlib
import numpy as np
import os
import glob
from skimage import io

current_path = os.getcwd() #获取当前路径
predictor_path = current_path + '/model/shape_predictor_68_face_landmarks.dat'  # 1.人脸关键点检测器
face_rec_model_path = current_path + '/model/dlib_face_recognition_resnet_model_v1.dat'   # 2.人脸识别模型
face_folder_path = current_path + '/all_face'    # 3.所有人脸文件夹
imgs_path = io.imread('/home/jackray/face_recoginition/20181117_165256.jpg')    # 4.图片地址


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
        face_desciptor = facerec.compute_face_descriptor(img, shape)   # 3.描述子提取，128D向量
        d_test = np.array(face_desciptor)    #转换为numpy array
        
        for key,value in data.items():
            dist_ = np.linalg.norm(value-d_test)
            dist.setdefault(key,dist_)
            
    
    cd_sorted = sorted(dist.items(), key=lambda d:d[1])
    print('\n The person ID is: %s' % ( cd_sorted[0][0] )  ) 

if __name__ == '__main__':
        face_recognition(imgs_path)