import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

xml_path1 = "C:\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
xml_path2 = "C:\Anaconda3\Lib\site-packages\cv2\data\haarcascade_profileface.xml"
xml_path3 = "C:\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml"
xml_path4 = "C:\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_alt_tree.xml"
xml_path5 = "C:\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml"

face_classifier  = cv2.CascadeClassifier(xml_path1)
face_classifier2 = cv2.CascadeClassifier(xml_path2)
face_classifier3 = cv2.CascadeClassifier(xml_path3)
face_classifier4 = cv2.CascadeClassifier(xml_path4)
face_classifier5 = cv2.CascadeClassifier(xml_path5)

def face_extractor(img):
    try:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)   
        if faces is():
            faces = face_classifier2.detectMultiScale(gray,1.3,5)   
            if faces is():
                faces = face_classifier3.detectMultiScale(gray,1.3,5)          
                if faces is():
                    faces = face_classifier4.detectMultiScale(gray,1.3,5)     
                    if faces is():
                        faces = face_classifier5.detectMultiScale(gray,1.3,5)                   
                        if faces is():
                            return None
    except:
        return None
    try:    
        if faces[1] is not None:
            print("ifif")
            return None
    except:
        pass
    #얼굴들이 있으면 
    cntt = 0
    for (x,y,w,h) in faces:    
        cropped_face = img[y:y+h, x:x+w]
    return cropped_face


data_path = "./MJK/data/image/3/"

#faces폴더에 있는 파일 리스트 얻기 
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

#파일 개수 만큼 루프 
count = 1
for i, files in enumerate(onlyfiles):
    img = cv2.imread(data_path+str(i+1)+".jpg")
    if face_extractor(img) is not None:
            count+=1  
            face = cv2.resize(face_extractor(img),(200,200))        
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)          
            file_name_path = './Common_data/image_all/3/'+str(count)+'.jpg'          
            cv2.imwrite(file_name_path,face)       
            count = count + 1    
    else:
        print("Face not Found")

print('Colleting Samples Complete!!!')