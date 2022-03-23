from typing import Tuple
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import tensorflow.keras.optimizers as ko
import numpy as np


class HourglassNetwork:
    def __init__(self,
                 num_classes: int,
                 num_stacks: int,
                 num_filters: int,
                 dim_in: Tuple[int, int],
                 dim_out: Tuple[int, int]) -> None:
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.model = create_hourglass_network(num_classes,
                                              num_stacks,
                                              num_filters,
                                              dim_in,
                                              dim_out)

    def load(self, path: str) -> None:
        self.model = km.load_model(path)

    def summary(self) -> None:
        self.model.summary()

    def fit(self, data_generator: int, dataset_size: int, batch_size: int, epochs: int):
        return self.model.fit(data_generator,
                              steps_per_epoch=dataset_size//batch_size,
                              epochs=epochs,
                              verbose=1)

    def predict(self, input: np.ndarray) -> np.ndarray:
        return self.model.predict(input)[0]


def create_hourglass_network(num_classes: int,
                             num_stacks: int,
                             num_filters: int,
                             dim_in: Tuple[int, int],
                             dim_out: Tuple[int, int]):
    input = kl.Input(shape=(*dim_in, 1))

    outputs = Hourglass(num_classes, num_filters, dim_in, dim_out)(input)

    model = km.Model(inputs=input, outputs=outputs)
    model.compile(optimizer="rmsprop",  # ko.RMSprop(learning_rate=5e-4),
                  loss="mean_squared_error",
                  metrics=["accuracy"])

    return model


def Hourglass(num_classes: int, num_filters: int, dim_in: Tuple[int, int], dim_out: Tuple[int, int]):

    def inner(x):
        # Front
        front = Front(num_filters//4)(x)
        dim_in_down = (dim_in[0]//4, dim_in[1]//4)
        dim_out_down = (dim_out[0]//4, dim_out[1]//4)

        # Create left features
        left_features = LeftHalf(num_filters, dim_in_down)(front)

        # Create right features, connect with left features
        right_features = RightHalf(num_filters, num_classes)(left_features)

        # add 1x1 conv with two heads, head_next_stage is sent to next stage
        # head_parts is used for intermediate supervision
        out = Output(num_classes, dim_out_down)(right_features)

        return out

    return inner


def Front(num_filters: int):
    # Front layer - reduces the resolution to 1/4:
    # 1 7x7 conv
    # 3 residual

    def inner(x):
        _x = kl.Conv2D(num_filters,
                       (7, 7),
                       (2, 2),
                       padding="same",
                       activation="relu",
                       data_format="channels_last")(x)
        _x = kl.BatchNormalization()(_x)
        return kl.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(_x)

    return inner


def Encoder(num_filters: int):
    # Encoder layer:
    # 2 3x3 conv
    # 1 2x2 max pooling

    def inner(x):
        _x = kl.Conv2D(num_filters,
                       (3, 3),
                       padding="same",
                       activation="relu",
                       data_format="channels_last")(x)
        _x = kl.Conv2D(num_filters,
                       (3, 3),
                       padding="same",
                       activation="relu",
                       data_format="channels_last")(_x)
        return kl.MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_last")(_x)

    return inner


def Bottleneck(num_filters: int, dim: Tuple[int, int]):
    # Residual Bottleneck Layer:
    # 1 conv height//4 x width//4
    # 1 conv 1x1

    def inner(x):
        _x = kl.Conv2D(num_filters,
                       (dim[0]//4, dim[1]//4),
                       padding="same",
                       activation="relu",
                       data_format="channels_last")(x)
        return kl.Conv2D(num_filters,
                         (1, 1),
                         padding="same",
                         activation="relu",
                         data_format="channels_last")(_x)

    return inner


def LeftHalf(num_filters: int, dim: Tuple[int, int]):
    # Left half layers for hourglass module

    def inner(x):
        f1 = Encoder(num_filters//2)(x)

        f2 = Encoder(num_filters)(f1)

        f3 = Bottleneck(num_filters, dim)(f2)

        return (f1, f2, f3)

    return inner


def RightHalf(num_filters: int, num_classes: int):
    # Right half layers for hourglass module

    def inner(x):
        lf1, _, _x = x
        rf1 = kl.Conv2DTranspose(num_filters//2,
                                 (2, 2),
                                 strides=(2, 2),
                                 use_bias=False,
                                 data_format="channels_last")(_x)
        _x = kl.Add()([rf1, lf1])
        return kl.Conv2DTranspose(num_classes,
                                  (2, 2),
                                  strides=(2, 2),
                                  use_bias=False,
                                  data_format="channels_last")(_x)

    return inner


def Output(num_classes: int, dim_out: Tuple[int, int]):

    def inner(x):
        return kl.Reshape((dim_out[1] * dim_out[0] * num_classes, 1))(x)

    return inner
