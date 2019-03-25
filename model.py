from keras.applications import resnet50
from keras.optimizers import Adam
from keras.layers import GlobalAveragePooling2D, Dense
from keras import Model
import keras.backend as K

class Model_for_training:
    def __init__(self):
        self.model = resnet50.ResNet50(weights='imagenet', include_top=True, input_shape=(224, 224, 3))

    def build_model(self):
        session = K.get_session()
        idx = 0
        # Reset weights of 9 Residual Blocks for retraining
        # Total layer : 175 layers
        # Layers to be initialized : 94 layers
        for layer in self.model.layers:
            if idx < 94:
                continue
            else:
                layer.kernel.initializer.run(session=session)


        #for i in range(94):
            # 94개의 Layer의 가중치를 초기화해야 함. 근데 어떻게 해야 하지??

        self.model.layers.pop()

        x = Dense(256, activation='linear', name='fc')(self.model.layers[-1].output)
        x = Dense(1, activation='linear', name='predictions')(x)

        self.model = Model(input=self.model.input, output=x)

        optimizer = Adam(lr=0.001)
        self.model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=["accuracy"])
        self.model.summary()

        return self.model

    def load_weights(self, weight_dir):
        self.model.load_weights(weight_dir)

        return self.model

if __name__ == "__main__":
    model = Model_for_training()
    model = model.build_model()