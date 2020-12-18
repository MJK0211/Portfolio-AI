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
dlib_shape = predictor(image, face)

shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

for s in shape_2d:
    cv2.circle(image, center=tuple(s), radius=2, color=(0,0,255), thickness=2, lineType=True)

cv2.imshow('image', image)
cv2.waitKey()
