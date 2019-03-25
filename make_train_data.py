import cv2, utils
import os
import random

value = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
idx = 0

for file in os.listdir('./raw_train_data'):
    img = cv2.imread('./raw_train_data/' + file, cv2.IMREAD_COLOR)
    img_raw_width, img_raw_height = img.shape[:2]
    img = cv2.copyMakeBorder(img, (int)(img_raw_height * 0.5), (int)(img_raw_height * 0.5), (int)(img_raw_width * 0.5), (int)(img_raw_width * 0.5),
                             cv2.BORDER_CONSTANT, None, value)
    img_padded_width, img_padded_height = img.shape[:2]
    for i in range(7):
        idx += 1
        angle = random.randint(-20, 20)
        rotated_img = utils.rotate_img(img, angle)
        bottom_border = (int)(img_padded_height / 2 - img_raw_height / 2)
        left_border = (int)(img_padded_width / 2 - img_raw_width / 2)
        cropped_img = rotated_img[left_border:(left_border + img_raw_width), bottom_border:(bottom_border + img_raw_height)]
        cv2.imwrite('./train_data/' + str(angle) + '/' + str(idx) + '.jpg', cropped_img)