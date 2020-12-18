  
# 종합_2 load_model
import cv2
import numpy as np
import time
import math

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

def dd(video,time2):
    from tensorflow.keras.models import load_model
    model =  load_model("./MJK/data/model/cp-rmsprop-75-0.428727.hdf5")
    
    def face_detector(img, size = 0.5):
        # color = cv2.cvtColor(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
                            return img,[] 
                        
        rr = [] 
        xx = []   
        yy = []   

        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
            roi = img[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200,200))
            rr.append(roi)
            xx.append(x)
            yy.append(y)                    
        return img,rr,xx,yy 
        

    # ==================================================================================

    #파일 경로
    FilePath = './teamProject/video/'+video
    saveFilePath = './teamProject/static/cascade.mp4'
    
    #Open the File
    movie = cv2.VideoCapture(FilePath) #동영상 핸들 얻기

    width = movie.get(cv2.CAP_PROP_FRAME_WIDTH) # 또는 movie.get(3)
    height = movie.get(cv2.CAP_PROP_FRAME_HEIGHT) # 또는 movie.get(4)
    fps = movie.get(cv2.CAP_PROP_FPS) # 또는 movie.get(4)
    print('프레임 너비: %d, 프레임 높이: %d, 초당 프레임 수: %d' %(width, height, fps))

    fourcc = cv2.VideoWriter_fourcc('H','2','6','4') # 코덱 정의
    out = cv2.VideoWriter(saveFilePath, fourcc, 24.0, (int(width), int(height))) # VideoWriter 객체 정의

    #Check that the file is opened
    if movie.isOpened() == False: #동영상 핸들 확인
        print('Can\'t open the File' + (FilePath))
        exit()

    movie2 = cv2.VideoCapture(saveFilePath)
    time2 = int(time2)*int(math.ceil(fps))
    print(time2)
    count = 0
    while True:
        #카메라로 부터 사진 한장 읽기 
        
        ret, frame = movie.read()
        
        try:
            image, face, x, y = face_detector(frame)
        except:
            image, face = face_detector(frame)
        try:
            Training_Data = []    
            for i in range(len(face)): 
                face[i] = cv2.cvtColor(face[i], cv2.COLOR_BGR2GRAY)
            face = np.array(face)
            face = face.reshape(face.shape[0],200,200,1)

            #학습한 모델로 예측시도
            result = model.predict(face)
            result = np.argmax(result,axis=1)
            bss1 = []
            for i in range(len(result)):
                if result[i] == 0:
                    bss1.append([cv2.putText(image, "Bae", (x[i], y[i]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)])

                elif result[i] == 1:
                    bss1.append([cv2.putText(image, "Nam", (x[i], y[i]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)])

                elif result[i] == 2:
                    bss1.append([cv2.putText(image, "Ha", (x[i], y[i]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)])

                elif result[i] == 3:
                    bss1.append([cv2.putText(image, "Kang", (x[i], y[i]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)])

                else:
                    bss1.append([cv2.putText(image, "U", (x[i], y[i]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)])

            bss1[:]
            cv2.imshow('Face Cropper', image)        
        except:
            #얼굴 검출 안됨 
            cv2.putText(image, "Face Not Found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Face Cropper', image)
            pass
        out.write(image)    
        if cv2.waitKey(1)==13:
            break
        count=count+1
        print(count)
        if count == time2:
            break
    
    movie.release()
    movie2.release()
    out.release()
    cv2.destroyAllWindows()
