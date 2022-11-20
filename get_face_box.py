#-*- encoding: utf8 -*-
from email.mime import image
from unicodedata import normalize
import json
import os, cv2, time
import numpy as np
pwd = "korean/Validation/"

emo_type_list = ["기쁨", "당황", "분노", "슬픔", "중립"]

annot_ABC = ['A', 'B', 'C']  # 어노테이터 3인
numbering = 1

for emo_type in emo_type_list:
    json_path = os.path.join(os.getcwd(), pwd + f'[라벨]EMOIMG_{emo_type}_VALID/img_emotion_validation_data({emo_type}).json')
    with open(json_path, encoding="UTF-8") as f:
        data = json.load(f)
        for i in range(0, len(data)-1):
            jpg_name = str(data[i]['filename']).replace("&", "_")
            img_path = os.path.join(os.getcwd(), pwd + f'[원천]EMOIMG_{emo_type}_VALID/'+jpg_name)
            img_array = np.fromfile(img_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            # 원본 이미지 위에 어노테이터 별로 설정한 ROI에 직사각형 표시
            # for j in annot_ABC:
            #     min_x, min_y = data[i]['annot_%s'%j]['boxes']['minX'], data[i]['annot_%s'%j]['boxes']['minY']
            #     max_x, max_y = data[i]['annot_%s'%j]['boxes']['maxX'], data[i]['annot_%s'%j]['boxes']['maxY']
            #     min_x, min_y = int(min_x), int(min_y)
            #     max_x, max_y = int(max_x), int(max_y)
            #     if j == 'A':
            #         img = cv2.rectangle(img, (min_x,min_y), (max_x, max_y), (0,0,255) ,5)
            #     elif j == 'B':
            #         img = cv2.rectangle(img, (min_x,min_y), (max_x, max_y), (0,255,0) ,5)
            #     elif j == 'C':
            #         img = cv2.rectangle(img, (min_x,min_y), (max_x, max_y), (255,0,0) ,5)

            # 일단 어노테이터 A의 데이터 사용
            min_x, min_y = int(data[i]['annot_A']['boxes']['minX']), int(data[i]['annot_A']['boxes']['minY'])
            max_x, max_y = int(data[i]['annot_A']['boxes']['maxX']), int(data[i]['annot_A']['boxes']['maxY'])
            img = cv2.rectangle(img, (min_x,min_y), (max_x, max_y), (0,0,255) , 5) # 어노테이터가 정한 부분에 빨간색 직사각형

            """
            # 어노테이터의 ROI 사이즈 구함
            roi_height = max_y - min_y
            roi_width = max_x - min_x
            roi_center = (min_x + roi_width // 2, min_y + roi_height // 2)
            roi_size = max(roi_width, roi_height)

            # ROI의 세로가 더 큰 경우 -> 정사각형으로 만들려면 가로를 늘려야 함
            if (roi_height > roi_width):
                min_x = roi_center[0] - (roi_size // 2)
                max_x = roi_center[0] + (roi_size // 2)
            # ROI의 가로가 더 긴 경우 -> 정사각형으로 만들려면 세로를 늘려야 함
            else:
                min_y = roi_center[1] - (roi_size // 2)
                max_y = roi_center[1] + (roi_size // 2)
            
            img = cv2.rectangle(img, (min_x,min_y), (max_x, max_y), (0,255,0), 5) # 정사각형으로 만든 부분에 초록색 직사각형

            """

            re_img = img[min_y:max_y, min_x:max_x]
            try:
                re_img = cv2.resize(re_img, (256, 256))
                # re_img = img

                # 이미지 타입은 numpy.ndarray
                re_img = cv2.cvtColor(re_img, cv2.COLOR_BGR2GRAY)
                cv2.imshow("picture", re_img)
                cv2.imwrite(f"data/val/{emo_type}/{numbering}.png", re_img)
                numbering += 1

                # cv2.imwrite('D:/hhh/img/%d_crop.jpg'%numbering, re_img)
                # numbering = numbering+1
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except:
                print(jpg_name)

            # 지금 ROI를 넓히려니까 특정 이미지들은 frame border에 걸리는 경우에 음수가 돼서 오류 발생....








# import json
# import os, cv2, time

# pwd = 'D:\\test\\BODYSHAKE\\20201118_dog-bodyshake-001668.mp4'
# # pwd = 'D:\\test\\BODYSHAKE\\20201202_dog-bodyshake-010101.mp4'
# json_path = os.path.join(os.getcwd(), pwd + '.json')



# with open(json_path, encoding="UTF-8") as f:
#     data = json.load(f)
#     # for i in range(len(data['annotations']) + 1):
#     for i in range(3):
#         print(i)
#         frame_number = data['annotations'][i]['frame_number']
#         timestamp = data['annotations'][i]['timestamp']
#         bounding_box = data['annotations'][i]['bounding_box']
#         keypoints = data['annotations'][i]['keypoints']
#         x, y, width, height = bounding_box['x'], bounding_box['y'], bounding_box['width'], bounding_box['height']
#    
#         image_path = os.path.join(os.getcwd(),
#         pwd + '\\frame_%s_timestamp_%s.jpg'
#         %(frame_number,timestamp))

#         image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
#         image = cv2.rectangle(image, (x,y), (x+width, y+height),(0,255,0) ,5)

#         cv2.imshow("picture", image)
#         # time.sleep(100)
#         cv2.waitKey()
#         cv2.destroyAllWindows()


train_dir = "korean/Training"
val_dir = "korean/Validation"