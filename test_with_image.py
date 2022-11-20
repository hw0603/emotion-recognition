# pyright: reportMissingImports=false
from keras.models import load_model
import os
import numpy as np
import cv2


loaded_model = load_model("models/emotion_83.h5")
print("Loaded model from disk")

# 이미지 리사이징 파라미터 설정
WIDTH, HEIGHT = 256, 256
x, y = None, None
labels = ["Angry", "Embarrassed", "Happy", "Neutral", "Sad"]

# 테스트 이미지 로드
directory = 'integ_data/angry'
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        print(f)

        full_size_image = cv2.imread(f)
        # print("이미지 로드 완료")
        gray = cv2.cvtColor(full_size_image, cv2.COLOR_RGB2GRAY)
        face = cv2.CascadeClassifier("./models/haarcascade_frontalface_default.xml")
        faces = face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(
            100, 100), flags=cv2.CASCADE_SCALE_IMAGE)

        faces = [(0, 0, 256, 256)]

        # 얼굴 인식
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(
                cv2.resize(roi_gray, (256, 256)), -1), 0)
            cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1,
                        norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
            cv2.rectangle(full_size_image, (x, y), (x + w, y + h), (0, 255, 0), 1)

            # 표정 인식
            yhat = loaded_model.predict(cropped_img.astype(float) / 255)
            cv2.putText(full_size_image, labels[int(np.argmax(
                yhat))], (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

            print("인식 결과: " + labels[int(np.argmax(yhat))])
            for idx, emotion_type in enumerate(labels):
                print(f"{emotion_type}: {yhat[0][idx]:.2f}%")

        cv2.imshow("Emotion", full_size_image)
        cv2.waitKey()
