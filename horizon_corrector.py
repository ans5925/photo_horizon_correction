import model, utils
import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='사진의 수평을 맞춰줍니다')
parser.add_argument('-f', '--file', required=True)

args = parser.parse_args()
file_dir = args.file

_model = model.Model_for_training()
model = _model.build_model()
model = _model.load_weights('./save_weight/model.h5')

img = cv2.imread(file_dir)
img = np.asarray(img)

deg = model.predict(img)

print("Degree is : " + deg)