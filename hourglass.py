import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import tensorflow.keras.optimizers as ko
import numpy as np


class HourglassNetwork:
    def __init__(self,
                 num_classes: int,
                 num_stacks: int,
                 num_filters: int,
                 in_shape: tuple[int, int],
                 outers: tuple[int, int]) -> None:
        self.inres = in_shape
        self.outres = outers
        self.model = create_hourglass_network(num_classes,
                                              num_stacks,
                                              num_filters,
                                              in_shape)

    def summary(self) -> None:
        self.model.summary()

    def fit(self, data_generator: int, dataset_size: int, batch_size: int, epochs: int) -> None:
        self.model.fit(data_generator,
                       steps_per_epoch=dataset_size//batch_size,
                       epochs=epochs,
                       verbose=1)

    def predict(self, input: np.ndarray) -> np.ndarray:
        return self.model.predict(input)[0]


def create_hourglass_network(num_classes, num_stacks, num_filters, in_shape):
    input = kl.Input(shape=in_shape)

    out_next_stage = Front(num_filters)(input)

    outputs = []
    for _ in range(num_stacks):
        out_next_stage, out_to_loss = Hourglass(
            num_classes, num_filters)(out_next_stage)
        outputs.append(out_to_loss)

    model = km.Model(inputs=input, outputs=outputs)
    model.compile(optimizer=ko.RMSprop(learning_rate=5e-4),
                  loss="mean_squared_error",
                  metrics=["accuracy"])

    return model


def Hourglass(num_classes, num_filters):

    def inner(x):
        # Create left features , f1, f2, f4, and f8
        left_features = LeftHalf(num_filters)(x)

        # Create right features, connect with left features
        rf1 = RightHalf(num_filters)(left_features)

        # add 1x1 conv with two heads, head_next_stage is sent to next stage
        # head_parts is used for intermediate supervision
        out_next_stage, out_parts = Output(num_classes, num_filters)(x, rf1)

        return out_next_stage, out_parts

    return inner


def Front(num_filters: int):
    # Front layer - reduces the resolution to 1/4:
    # 1 7x7 conv
    # 3 residual

    def inner(x):
        _x = kl.Conv2D(num_filters//4,
                       (7, 7),
                       (2, 2),
                       padding="same",
                       activation="relu",
                       data_format="channels_last")(x)
        _x = kl.BatchNormalization()(_x)

        _x = ResidualBottleneck(num_filters//2)(_x)
        _x = kl.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(_x)

        _x = ResidualBottleneck(num_filters//2)(_x)
        return ResidualBottleneck(num_filters)(_x)

    return inner


def ResidualBottleneck(num_filters: int):
    # Residual Bottleneck Layer:
    # 3 conv
    # 1 add skip

    def inner(x):
        # Skip layer
        skip = kl.Conv2D(num_filters,
                         kernel_size=(1, 1),
                         padding="same",
                         activation="relu")(x)
        # 1st ConvLayer  num_filters -> num_filters/2
        _x = kl.Conv2D(num_filters//2,
                       kernel_size=(1, 1),
                       padding="same",
                       activation="relu")(x)
        _x = kl.BatchNormalization()(_x)
        # 2nd ConvLayer num_filters/2 -> num_filters/4
        _x = kl.Conv2D(num_filters//2,
                       kernel_size=(3, 3),
                       padding="same",
                       activation="relu")(_x)
        _x = kl.BatchNormalization()(_x)
        # 3rd ConvLayer num_filters/4 -> num_filters/4
        _x = kl.Conv2D(num_filters,
                       kernel_size=(1, 1),
                       padding="same",
                       activation="relu")(_x)
        _x = kl.BatchNormalization()(_x)
        return kl.Add()([skip, _x])

    return inner


def LeftHalf(num_filters: int):
    # Left half layers for hourglass module
    # f1, f2, f4 , f8 : 1, 1/2, 1/4 1/8 resolution

    def inner(x):
        f1 = ResidualBottleneck(num_filters)(x)
        _x = kl.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f1)

        f2 = ResidualBottleneck(num_filters)(_x)
        _x = kl.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f2)

        f4 = ResidualBottleneck(num_filters)(_x)
        _x = kl.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f4)

        f8 = ResidualBottleneck(num_filters)(_x)

        return (f1, f2, f4, f8)

    return inner


def RightHalf(num_filters):
    # Right half layers for hourglass module

    def inner(x):
        lf1, lf2, lf4, lf8 = x
        rf8 = ResidualBottleneck(num_filters)(lf8)
        rf4 = Connect(num_filters)(lf4, rf8)
        rf2 = Connect(num_filters)(lf2, rf4)
        return Connect(num_filters)(lf1, rf2)

    return inner


def Connect(num_filters):
    # Connection layer of left feature to right feature
    # left  -> 1 ResidualBottleneck
    # right -> Upsampling
    # Add   -> left + right

    def inner(left, right):
        _xleft = ResidualBottleneck(num_filters)(left)
        _xright = kl.UpSampling2D()(right)
        add = kl.Add()([_xleft, _xright])
        return ResidualBottleneck(num_filters)(add)

    return inner


def Output(num_classes, num_filters):

    def inner(pf, rf1):
        # two head, one head to next stage, one head to intermediate features
        out = kl.Conv2D(num_filters,
                        kernel_size=(1, 1),
                        padding="same",
                        activation="relu")(rf1)
        out = kl.BatchNormalization()(out)

        # for out as intermediate supervision, use 'linear' as activation.
        out_parts = kl.Conv2D(num_classes,
                              kernel_size=(1, 1),
                              padding="same",
                              activation="linear")(out)

        # use linear activation
        out = kl.Conv2D(num_filters,
                        kernel_size=(1, 1),
                        padding="same",
                        activation="linear")(out)
        out_m = kl.Conv2D(num_filters,
                          kernel_size=(1, 1),
                          padding="same",
                          activation="linear")(out_parts)

        out_next_stage = kl.Add()([out, out_m, pf])
        return out_next_stage, out_parts

    return inner
