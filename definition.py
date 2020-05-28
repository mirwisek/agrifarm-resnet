from keras.layers import Dense, Activation, Conv2D, MaxPool2D, GlobalAvgPool2D, BatchNormalization, add, Input
from keras.models import Model

# Complete Model
def ResNet50(input_shape, num_classes=10):
    input_object = Input(shape=input_shape)
    layers = [3, 4, 6, 3]
    channel_depths = [256, 512, 1024, 2048]

    output = Conv2D(64, kernel_size=7, strides=2, padding="same", kernel_initializer="he_normal")(input_object)
    output = BatchNormalization()(output)
    output = Activation("relu")(output)

    output = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(output)
    output = resnet_first_block_first_module(output, channel_depths[0])

    for i in range(4):
        channel_depth = channel_depths[i]
        num_layers = layers[i]

        strided_pool_first = True
        if (i == 0):
            strided_pool_first = False
            num_layers = num_layers - 1
        output = resnet_block(output, channel_depth=channel_depth, num_layers=num_layers,
                              strided_pool_first=strided_pool_first)

    output = GlobalAvgPool2D()(output)
    output = Dense(num_classes)(output)
    output = Activation("softmax")(output)

    model = Model(inputs=input_object, outputs=output)

    return model
def resnet_first_block_first_module(input, channel_depth):
    residual_input = input
    stride = 1

    residual_input = Conv2D(channel_depth, kernel_size=1, strides=1, padding="same", kernel_initializer="he_normal")(
        residual_input)
    residual_input = BatchNormalization()(residual_input)

    input = Conv2D(int(channel_depth / 4), kernel_size=1, strides=stride, padding="same",
                   kernel_initializer="he_normal")(input)
    input = BatchNormalization()(input)
    input = Activation("relu")(input)

    input = Conv2D(int(channel_depth / 4), kernel_size=3, strides=stride, padding="same",
                   kernel_initializer="he_normal")(input)
    input = BatchNormalization()(input)
    input = Activation("relu")(input)

    input = Conv2D(channel_depth, kernel_size=1, strides=stride, padding="same", kernel_initializer="he_normal")(input)
    input = BatchNormalization()(input)

    input = add([input, residual_input])
    input = Activation("relu")(input)

    return input
def resnet_block(input, channel_depth, num_layers, strided_pool_first=False):
    for i in range(num_layers):
        pool = False
        if (i == 0 and strided_pool_first):
            pool = True
        input = resnet_module(input, channel_depth, strided_pool=pool)

    return input
def resnet_module(input, channel_depth, strided_pool=False):
    residual_input = input
    stride = 1

    if (strided_pool):
        stride = 2
        residual_input = Conv2D(channel_depth, kernel_size=1, strides=stride, padding="same",
                                kernel_initializer="he_normal")(residual_input)
        residual_input = BatchNormalization()(residual_input)

    input = Conv2D(int(channel_depth / 4), kernel_size=1, strides=stride, padding="same",
                   kernel_initializer="he_normal")(input)
    input = BatchNormalization()(input)
    input = Activation("relu")(input)

    input = Conv2D(int(channel_depth / 4), kernel_size=3, strides=1, padding="same", kernel_initializer="he_normal")(
        input)
    input = BatchNormalization()(input)
    input = Activation("relu")(input)

    input = Conv2D(channel_depth, kernel_size=1, strides=1, padding="same", kernel_initializer="he_normal")(input)
    input = BatchNormalization()(input)

    input = add([input, residual_input])
    input = Activation("relu")(input)

    return input
