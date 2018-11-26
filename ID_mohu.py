import dlib
from skimage import io
import base64
from PIL import ImageDraw,Image

lujing = "20181117_165256.jpg"
with open(lujing,"rb") as f:
# b64enode是编码，b64decode是解码  
        base64_data = base64.b64encode(f.read())  
# base6.b64decode(base64data)
fh = open("img/ID_img.jpg", "wb")
fh.write(base64.b64decode(base64_data))
fh.close()
detector = dlib.get_frontal_face_detector() #加载正脸识别器

face_coordinate = []

def tezhengtiqu(img_path):
    img = io.imread(img_path)
    face = detector(img, 1)
    for index, face in enumerate(face):
        face_coordinate.append(face.top())
        face_coordinate.append(face.bottom())
        face_coordinate.append(face.left())
        face_coordinate.append(face.right())

def mohu(imgs_path):
    ymax = face_coordinate[1]
    ymin = face_coordinate[0]
    xmin = face_coordinate[2]
    xmax = face_coordinate[3]
    #print(ymax,ymin,xmax,xmin)
    bili = round((xmax-xmin)/132,2)
    #print(bili)
    bj1 = round(ymax+140*bili,2)
    #print(bj1)
    bj2 = round(ymax+102*bili,2)
    #print(bj2)
    img = Image.open(imgs_path)
    draw = ImageDraw.Draw(img)
    draw.rectangle((xmin,bj2,xmax,bj1), fill = (255,0,0))
    #img.show()
    img.save('mohu_ID.jpg')


#tezhengtiqu('/home/jackray/face_recoginition/timg.jpg')
#mohu('/home/jackray/face_recoginition/timg.jpg')
