import cv2
from keras.preprocessing.image import ImageDataGenerator

def rotate_img(img, angle):
    height, width, channel = img.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    dst = cv2.warpAffine(img, matrix, (width, height))

    print("Image rotation complete")
    return dst

def make_train_datset(train_dat_dir, val_dat_dir):
    data_generator = ImageDataGenerator(rescale=1. / 255)

    train_generator = data_generator.flow_from_directory(train_dat_dir, target_size=(224, 224), batch_size=16, class_mode='sparse')
    validation_generator = data_generator.flow_from_directory(val_dat_dir, target_size=(224, 224), batch_size=16, class_mode='sparse')

    return train_generator, validation_generator