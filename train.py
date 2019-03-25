import model, utils
from keras.preprocessing.image import img_to_array, load_img

resnet = model.Model_for_training()
resnet = resnet.build_model()

test_dataset, validation_dataset = utils.make_train_datset('./train_data', './validation_data')

resnet.fit_generator(test_dataset, steps_per_epoch=358, epochs=16, validation_data=validation_dataset, validation_steps=10)

resnet.save_weights('./save_weight/model.h5')