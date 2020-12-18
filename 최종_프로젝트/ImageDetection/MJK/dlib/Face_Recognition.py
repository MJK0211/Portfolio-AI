import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./MJK/data/photo/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('./MJK/data/photo/dlib_face_recognition_resnet_model_v1.dat')

def find_faces(img):
    dets = detector(img, 1) #찾은 얼굴들
    if len(dets) == 0: #못찾으면
        return np.empty(0), np.empty(0), np.empty(0)

    rects, shapes = [], []
    shapes_np = np.zeros((len(dets), 68, 2), dtype=np.int) 
    for k, d in enumerate(dets): #얼굴을 찾았다면 갯수만큼 루프
        rect = ((d.left(), d.top()), (d.right(), d.bottom())) #얼굴 왼쪽 위 오른쪽 아래
        rects.append(rect)
        shape = predictor(img, d)
         
        for i in range(0, 68): #랜드마크 생성
            shapes_np[k][i] = (shape.part(i).x, shape.part(i).y)

        shapes.append(shape)    
    return rects, shapes, shapes_np

def encode_faces(img, shapes): #얼굴 인코드 -> 랜드마크 정보를 토대로 인코더에 넣어주게 되면 128개의 벡터로 변환됨(벡터의 특징들로 얼굴을 검출)
    face_descriptors = []
    for shape in shapes:
        face_descriptor = facerec.compute_face_descriptor(img, shape) #전체이미지와 랜드마크 결과가 저장
        face_descriptors.append(np.array(face_descriptor))
    return np.array(face_descriptors)
 
# 이미지 등록
img_paths = {
    'suzi': './MJK/data/photo/0/suzi.jpg',
    'juhyuk': './MJK/data/photo/1/juhyuk.jpg',
    'sunho': './MJK/data/photo/2/sunho.jpg',
    'hanna': './MJK/data/photo/3/hanna.jpg'
}
# 계산할 결과를 저장할 공간을 설정
descs = {
    'suzi': None,
    'juhyuk': None,
    'sunho': None,
    'hanna': None 
}

for name, img_path in img_paths.items():
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    _, img_shapes, _ = find_faces(img_rgb)
    descs[name] = encode_faces(img_rgb, img_shapes)[0]
    # print(descs[name])

np.save('./MJK/data/photo/startup.npy', descs)
print(len(descs['suzi']))  

'''
# 검증 사진 인풋 부분    
img_bgr = cv2.imread('./MJK/data/photo/11.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

rects, shapes, _ = find_faces(img_rgb)
descriptors = encode_faces(img_rgb, shapes)

# 결과 출력
fig, ax = plt.subplots(1, figsize=(20, 20))
ax.imshow(img_rgb)

for i, desc in enumerate(descriptors):
    
    found = False
    for name, saved_desc in descs.items():
        dist = np.linalg.norm([desc] - saved_desc, axis=1)

        if dist < 0.6:
            found = True

            text = ax.text(rects[i][0][0], rects[i][0][1], name,
                    color='b', fontsize=40, fontweight='bold')
            text.set_path_effects([path_effects.Stroke(linewidth=10, foreground='white'), path_effects.Normal()])
            rect = patches.Rectangle(rects[i][0],
                                 rects[i][1][1] - rects[i][0][1],
                                 rects[i][1][0] - rects[i][0][0],
                                 linewidth=2, edgecolor='w', facecolor='none')
            ax.add_patch(rect)

            break
    
    if not found:
        ax.text(rects[i][0][0], rects[i][0][1], 'unknown',
                color='r', fontsize=20, fontweight='bold')
        rect = patches.Rectangle(rects[i][0],
                             rects[i][1][1] - rects[i][0][1],
                             rects[i][1][0] - rects[i][0][0],
                             linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

plt.axis('off')
plt.show()
'''