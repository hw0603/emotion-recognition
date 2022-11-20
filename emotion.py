import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
import time
import os
import PIL
# pyright: reportMissingImports=false
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import SGD
from keras.layers import Dense, Dropout, Flatten, Conv2D
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Activation, GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from numpy import argmax
import random
import imutils


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 명령행 인자 설정
ap = argparse.ArgumentParser()
ap.add_argument("--mode", help="train/display")
mode = ap.parse_args().mode


# 모델 학습 모드
if (mode == "train"):
    path = "/Users/hw0603/DATA/Project/emotion-recognition/integ_data"
    folder = ['angry', 'embarrassed', 'happy', 'neutral', 'sad']
    number = [0, 1, 2, 3, 4]
    imgs = []
    label = []
    size = 256  # 이미지 사이즈
    num = 400  # 전체 이미지 개수 중 train 이미지 개수

    for (name, labeling) in zip(folder, number):
        data = "%s/%s" % (path, name)
        count = os.listdir("%s/" % (data))

        for i in count:
            if i.endswith(".png"):
                img = PIL.Image.open("%s/%s" % (data, i))
                Processed_img = img.resize((size, size))
                Processed_img_data = np.array(Processed_img)
                imgs.append(Processed_img_data)
                label.append(labeling)

    imgs = np.array(imgs)
    label = np.array(label)
    idx = np.arange(imgs.shape[0])
    np.random.shuffle(idx)

    imgs = imgs[idx]
    label = label[idx]
    print(f"이미지 개수: {len(imgs)}, 라벨 개수: {len(label)}")
    print(f"학습 데이터 개수: {num}, 검증 데이터 개수: {len(imgs) - num}")

    x_train = imgs[:num]
    y_train = label[:num]
    x_test = imgs[num:]
    y_test = label[num:]
    x_train, x_test = x_train / 255.0, x_test / 255.0

    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    # 모델 정의
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=(size, size, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))
    # 모델 컴파일
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    # 모델 학습
    hist = model.fit(
        x_train, y_train, epochs=100, batch_size=64,
        validation_data=(x_train, y_train), validation_split=0.1
    )
    loss, acc = model.evaluate(x_test, y_test, verbose=1)

    # 학습 결과 출력
    print(f"loss: {loss}, acc: {acc}")

    # 학습 결과 그래프 출력
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()

    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
    acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
    acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuracy')

    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')
    plt.show()

    # 모델 서머리 출력
    model.summary()

    # 모델 저장
    model_file_name = f"model_a{acc:.3f}_l{loss:.3f}.h5"
    choice = input("모델 저장? (y/n) : ")
    if choice.upper() == "Y":
        model.save(model_file_name)
        print(f"h5 모델 저장 완료. 파일명={model_file_name}")
    else:
        print("모델 저장 취소")

    # 랜덤 테스트
    pred_count = 10
    pred_list = []
    true_list = []
    correct = 0

    for i in range(pred_count):
        th = len(y_test)
        th = random.randint(0, th-1)

        xhat = x_test[th]
        xhat = np.array([xhat])
        y_prob = model.predict(xhat, verbose=0)
        pred = y_prob.argmax(axis=-1)
        true_list.append(str(y_test[th]))
        pred_list.append(str(pred[0]))
        print('True : ' + str(y_test[th]) + ', Predict : ' + str(pred[0]))
        if str(y_test[th]) == str(pred[0]):
            correct = correct + 1

    print(true_list)
    print(pred_list)
    print(f"{pred_count}개 중 {correct}개 맞춤")



# Display 모드
elif (mode == "display"):
    # 모델 파일 이름 설정
    model_file_name = "models/emotion_83.h5"

    # 이미지 리사이징 파라미터 설정
    WIDTH, HEIGHT = 256, 256
    x, y = None, None

    # 표정 매핑 딕셔너리
    labels = {
        0: "Angry", 1: "Embarrassed", 2: "Happy", 3: "Neutral", 4: "Sad"
    }

    # 히스토리 딕셔너리
    hist = {
        "Angry": 0,
        "Embarrassed": 0,
        "Happy": 0,
        "Neutral": 0,
        "Sad": 0
    }
    prev_res = "Neutral"

    # 저장된 HDF5 모델 불러옴
    loaded_model = load_model(model_file_name)
    print(f"모델 로드 완료: {model_file_name}")

    # OpenCL 사용 설정
    cv2.ocl.setUseOpenCL(True)

    # DataFrame 생성
    df = pd.DataFrame(index=labels.values())

    # 카메라 피드 시작
    camera = cv2.VideoCapture(0)  # Camera_0 열기
    while (True):
        frame = camera.read()[1]
        frame = imutils.resize(frame, width=500)

        # 테스트 이미지 로드
        full_size_image = frame
        gray = cv2.cvtColor(full_size_image, cv2.COLOR_RGB2GRAY)
        
        # 얼굴 위치 인식
        # face = cv2.CascadeClassifier("./models/haarcascade_frontalface_default.xml")
        # faces = face.detectMultiScale(
        #     gray, scaleFactor=1.1, minNeighbors=5,
        #     minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
        # )
        faces = [(200, 50, 100, 150)] # 일단 얼굴 위치 하드코딩(X, Y, W, H)

        # frameClone = frame.copy()
        canvas = np.zeros((250, 300, 3), dtype="uint8")

        # 인식된 얼굴들의 위치를 기준으로 가장 크기가 큰 얼굴 하나를 골라서 표정 인식
        if (len(faces) > 0):
            default_face = sorted(faces, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            x, y, w, h = default_face

            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (256, 256)), -1), 0)
            cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
            cv2.rectangle(full_size_image, (x, y), (x + w, y + h), (0, 255, 0), 1)

            # 표정 인식
            yhat = loaded_model.predict(cropped_img.astype(float) / 255)
            pred_res = labels[int(np.argmax(yhat))]  # 인식된 label String

            hist[pred_res] += 1  # 인식된 표정을 히스토리에 추가
            if (pred_res != prev_res):  # 바로 직전 frame에 인식된 표정과 현재 인식된 표정이 다르면 hist 딕셔너리 초기화
                hist = {"Angry": 0, "Embarrassed": 0, "Happy": 0, "Neutral": 0, "Sad": 0}
            prev_res = pred_res

            if (hist[labels[int(np.argmax(yhat))]] > 2):  # 같은 표정이 2번 이상 인식되면 화면에 감정 레이블 표시
                cv2.putText(
                    full_size_image, pred_res,
                    (x, y+20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 1, cv2.LINE_AA
                )

            print("인식 결과: " + labels[int(np.argmax(yhat))])
            for idx, emotion_type in enumerate(labels):
                print(f"{labels[idx]}: {yhat[0][idx]*100:.2f}%")

            for (i, (emotion, prob)) in enumerate(zip(labels.values(), yhat[0])):
                # 레이블 String 만듦
                text = "{}: {:.2f}%".format(emotion, prob * 100)

                # label + 확률 바를 캔버스에 그림
                # emoji_face = feelings_faces[np.argmax(preds)]
                w = int(prob * 300)
                cv2.rectangle(
                    canvas, (7, (i * 35) + 5),
                    (w, (i * 35) + 35), (0, 0, 255), -1
                )
                cv2.putText(
                    canvas, text,
                    (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 2, cv2.LINE_AA
                )
        else: # 얼굴이 인식되지 않은 경우
            cv2.putText(
                canvas, "Face Not Detected",
                (10, (1 * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 2
            )

        cv2.imshow("Emotion", full_size_image)
        cv2.imshow("Probabilities", canvas)
        cv2.waitKey(1)

"""

        # 인식한 얼굴 주변으로 사각형을 그리기 위해 Haar Cascade 찾음
        ret, frame = cap.read()  # 성공여부, 현재 프레임(np.ndarray: x, y, w, h)
        if not (ret):
            break
        facecasc = cv2.CascadeClassifier("./models/haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3)

        # 얼굴을 인식한 경우
        if (len(faces) > 0):
            # 인식된 얼굴 데이터 중 크기가 가장 큰 데이터 하나를 기본 face로 지정
            default_face = sorted(faces, key=lambda x: (
                x[2] - x[0]) * (x[3] - x[1]))[0]

            x, y, w, h = default_face
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 5)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (256, 256)), -1), 0)  # 48x48 그레이스케일 이미지로 리사이징

            # 표정 추론
            prediction = model.predict(cropped_img.astype(float) / 255)  # 모델 추론 후 softmax 배열 반환받음
            maxindex = int(np.argmax(prediction))  # 추론 결과 max index 반환
            cv2.putText(
                frame, emotion_dict[maxindex],
                (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2, cv2.LINE_AA
            )  # 추론 결과를 프리뷰에 출력

            df = pd.concat([df, pd.Series(prediction.round(
                2)[0], index=emotion_dict.values(), name=time.time())])
            print(pd.Series(prediction.round(2)[0], index=emotion_dict.values()))
        else:
            # print(1)
            ...

        # 비디오 프리뷰
        cv2.imshow('Video', cv2.resize(
            frame, (854, 480), interpolation=cv2.INTER_CUBIC))

        if (cv2.waitKey(1) & 0xFF == ord('1')):  # "1" 누르면 종료
            break

    cap.release()
    cv2.destroyAllWindows()

    print(df.tail(10))
"""