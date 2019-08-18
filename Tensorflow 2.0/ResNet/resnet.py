import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential


class BasicBlock(keras.Model):
    def __init__(self, kernel_num, strides=1):
        super(BasicBlock, self).__init__()

        # conv-block-1
        self.cl1 = layers.Conv2D(kernel_num, (3, 3), strides=strides, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        self.dl1 = layers.Dropout(0.5)

        # conv-block-2
        self.cl2 = layers.Conv2D(kernel_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.dl2 = layers.Dropout(0.5)

        # only for stride != 1 in conv-layer1
        if strides != 1:
            # self.downsample = Sequential()
            # self.downsample.add(layers.Conv2D(kernel_num, (1, 1), strides=strides))
            self.downsample = layers.Conv2D(kernel_num, (1, 1), strides=strides)
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None):
        output = self.cl1(inputs)
        output = self.bn1(output, training)
        output = self.relu(output)
        output = self.dl1(output, training)

        output = self.cl2(output)
        output = self.bn2(output, training)
        output = self.dl2(output, training)

        # identity layer accepts inputs as input
        identity = self.downsample(inputs)

        # print('basic block output shape', output.shape, 'basic block identity shape', identity.shape)
        output = layers.add([output, identity])
        output = self.relu(output)
        # maybe shouldn't put dropout layer here, lose too much information
        # output = self.dl2(output, training)


        return output


class ResNet(keras.Model):
    # conv_block_size: [2, 2, 2, 2]
    def __init__(self, conv_block_size, class_num=100):
        super(ResNet, self).__init__()

        self.initial = Sequential([
            layers.Conv2D(64, (3, 3), strides=1),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=1, padding='same'),
            # layers.Dropout(0.5)
        ])

        self.layer1 = self.res_block(64, conv_block_size[0])
        # set strides to 2 to shrink size
        self.layer2 = self.res_block(128, conv_block_size[1], strides=2)
        self.layer3 = self.res_block(256, conv_block_size[2], strides=2)
        self.layer4 = self.res_block(512, conv_block_size[3], strides=2)

        # [b, 512]
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(class_num)

    def call(self, inputs, training=None):
        o = self.initial(inputs)

        o = self.layer1(o, training)
        o = self.layer2(o, training)
        o = self.layer3(o, training)
        o = self.layer4(o, training)

        o = self.avgpool(o)
        output = self.fc(o)

        return output

    def res_block(self, kernel_num, block_num, strides=1):
        res_blocks = Sequential()
        res_blocks.add(BasicBlock(kernel_num, strides))

        # since use default stride, don't need downsample anymore
        for _ in range(1, block_num):
            res_blocks.add(BasicBlock(kernel_num, strides=1))

        return res_blocks


def resnet_18():
    return ResNet([2, 2, 2, 2])


def resnet_32():
    return ResNet([3, 4, 6, 3])

