import os
from keras.models import Model
from keras.layers import (
    Conv2D,
    Activation,
    Input,
    Add,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
)

from networks.models.RecognitionModel import RecognitionModel

def generate_model():
    myInput = Input(shape=(55, 47, 3))

    x = Conv2D(20, (4, 4), name="Conv1", activation="relu", input_shape=(55, 47, 3))(myInput)
    x = MaxPooling2D(pool_size=2, strides=2, name="Pool1")(x)
    x = Dropout(rate=0.99, name="D1")(x)

    x = Conv2D(40, (3, 3), name="Conv2", activation="relu")(x)
    x = MaxPooling2D(pool_size=2, strides=2, name="Pool2")(x)
    x = Dropout(rate=0.99, name="D2")(x)

    x = Conv2D(60, (3, 3), name="Conv3", activation="relu")(x)
    x = MaxPooling2D(pool_size=2, strides=2, name="Pool3")(x)
    x = Dropout(rate=0.99, name="D3")(x)

    x1 = Flatten()(x)
    fc11 = Dense(160, name="fc11")(x1)

    x2 = Conv2D(80, (2, 2), name="Conv4", activation="relu")(x)
    x2 = Flatten()(x2)
    fc12 = Dense(160, name="fc12")(x2)

    y = Add()([fc11, fc12])
    y = Activation("relu", name="deepid")(y)

    model = Model(inputs=[myInput], outputs=y)
    return model

class DeepID(RecognitionModel):
    def __init__(self):
        super(DeepID, self).__init__()
        self.model_name = 'DeepID'

        model = generate_model()

        path = os.path.dirname(os.path.abspath(__file__)) + "/weights/deepid_keras_weights.h5"
        model.load_weights(path)

        self.model = model