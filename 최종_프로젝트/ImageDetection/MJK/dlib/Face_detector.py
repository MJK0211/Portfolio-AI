import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

detector = dlib.get_frontal_face_detector()
predictor  = dlib.shape_predictor('./MJK/data/photo/shape_predictor_68_face_landmarks.dat')

# 이미지
image = cv2.imread('./MJK/data/photo/15.jpg')
faces = detector(image)
face = faces[0]

image = cv2.rectangle(image, pt1=(face.left(), face.top()), pt2=(face.right(), face.bottom()), color=(0,0,255), thickness=2, lineType=cv2.LINE_AA)

#오픈cv rectangle 사용 얼굴의 좌상단좌표 우하단좌표

cv2.imshow('image', image)
cv2.waitKey()
