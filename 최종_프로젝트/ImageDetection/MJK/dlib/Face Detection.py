import dlib, cv2
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./MJK/data/photo/shape_predictor_68_face_landmarks.dat')
recognition = dlib.face_recognition_model_v1('./MJK/data/photo/dlib_face_recognition_resnet_model_v1.dat')

descs = np.load('./MJK/data/photo/startup.npy', allow_pickle=True)[()]

def encode_face(img):
  dets = detector(img, 1) 

  if len(dets) == 0: #얼굴 못찾으면
    return np.empty(0)

  for k, d in enumerate(dets):
    shape = predictor(img, d)
    face_descriptor = recognition.compute_face_descriptor(img, shape)

    return np.array(face_descriptor)

video_path = './MJK/data/photo/testavi.mp4'
save_path = './MJK/wooseokfuck.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
  exit()

_, img_bgr = cap.read()
padding_size = 0
resized_width = 800
video_size = (resized_width, int(img_bgr.shape[0] * resized_width // img_bgr.shape[1]))
output_size = (resized_width, int(img_bgr.shape[0] * resized_width // img_bgr.shape[1] + padding_size * 2))

fourcc = cv2.VideoWriter_fourcc('H','2','6','4') 
width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH) # 또는 movie.get(3)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # 또는 movie.get(4)
fps    = cap.get(cv2.CAP_PROP_FPS)

writer = cv2.VideoWriter(save_path, fourcc, 24, (int(width), int(height)))
c = 0
while True:
  ret, img_bgr = cap.read()
  if not ret:
    break

  img_bgr = cv2.resize(img_bgr, video_size)
  img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  
  dets = detector(img_bgr, 1)

  for k, d in enumerate(dets):
    shape = predictor(img_rgb, d)
    face_descriptor = recognition.compute_face_descriptor(img_rgb, shape)

    last_found = {'name': 'unknown', 'dist': 0.6, 'color': (0,0,255)}

    for name, saved_desc in descs.items():
      dist = np.linalg.norm([face_descriptor] - saved_desc, axis=1)

      if dist < last_found['dist']:
        last_found = {'name': name, 'dist': dist, 'color': (0,255,0)}

    cv2.rectangle(img_bgr, pt1=(d.left(), d.top()), pt2=(d.right(), d.bottom()), color=last_found['color'], thickness=2)
    cv2.putText(img_bgr, last_found['name'], org=(d.left(), d.top()), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=last_found['color'], thickness=2)

  writer.write(img_bgr)
  cv2.imshow('img', img_bgr)
  c = c+1
  if c == 720:
        break
  if cv2.waitKey(1) == ord('q'):
        break
writer.release()
cap.release()
