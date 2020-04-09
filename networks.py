import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU, UpSampling2D
from tensorflow.keras.applications import vgg16, VGG16

from tensorflow_addons.layers import InstanceNormalization


class ReflectionPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding=1, **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        self.padding = padding

    def compute_output_shape(self, s):
        return s[0], s[1] + 2 * self.padding, s[2] + 2 * self.padding, s[3]

    def call(self, x):
        return tf.pad(
            x,
            [
                [0, 0],
                [self.padding, self.padding],
                [self.padding, self.padding],
                [0, 0],
            ],
            "REFLECT",
        )


class ConvLayer(tf.keras.layers.Layer):
    def __init__(self, channels, kernel_size=3, strides=1):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = ReflectionPadding2D(reflection_padding)
        self.conv2d = Conv2D(channels, kernel_size, strides=strides)

    def call(self, x):
        x = self.reflection_pad(x)
        x = self.conv2d(x)
        return x


class UpsampleConvLayer(tf.keras.layers.Layer):
    def __init__(self, channels, kernel_size=3, strides=1, upsample=2):
        super(UpsampleConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = ReflectionPadding2D(reflection_padding)
        self.conv2d = Conv2D(channels, kernel_size, strides=strides)
        self.up2d = UpSampling2D(size=upsample)

    def call(self, x):
        x = self.up2d(x)
        x = self.reflection_pad(x)
        x = self.conv2d(x)
        return x


class ResidualBlock(tf.keras.Model):
    def __init__(self, channels, strides=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, kernel_size=3, strides=strides)
        self.in1 = InstanceNormalization()
        self.conv2 = ConvLayer(channels, kernel_size=3, strides=strides)
        self.in2 = InstanceNormalization()

    def call(self, inputs):
        residual = inputs

        x = self.in1(self.conv1(inputs))
        x = tf.nn.relu(x)

        x = self.in2(self.conv2(x))
        x = x + residual
        return x


class TransformerNet(tf.keras.Model):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.conv1 = ConvLayer(32, kernel_size=9, strides=1)
        self.in1 = InstanceNormalization()
        self.conv2 = ConvLayer(64, kernel_size=3, strides=2)
        self.in2 = InstanceNormalization()
        self.conv3 = ConvLayer(128, kernel_size=3, strides=2)
        self.in3 = InstanceNormalization()

        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)

        self.deconv1 = UpsampleConvLayer(
            64, kernel_size=3, strides=1, upsample=2
        )
        self.in4 = InstanceNormalization()
        self.deconv2 = UpsampleConvLayer(
            32, kernel_size=3, strides=1, upsample=2
        )
        self.in5 = InstanceNormalization()
        self.deconv3 = ConvLayer(3, kernel_size=9, strides=1)

        self.relu = ReLU()

    def call(self, x):
        x = self.relu(self.in1(self.conv1(x)))
        x = self.relu(self.in2(self.conv2(x)))
        x = self.relu(self.in3(self.conv3(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.relu(self.in4(self.deconv1(x)))
        x = self.relu(self.in5(self.deconv2(x)))
        x = self.deconv3(x)
        return x


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        vgg = VGG16(include_top=False, weights="imagenet")
        vgg.trainable = False

        style_outputs = [vgg.get_layer(name).output for name in style_layers]
        content_outputs = [
            vgg.get_layer(name).output for name in content_layers
        ]

        self.vgg = tf.keras.Model(
            [vgg.input], [style_outputs, content_outputs]
        )
        self.vgg.trainable = False

    def call(self, inputs):
        preprocessed_input = vgg16.preprocess_input(inputs)
        style_outputs, content_outputs = self.vgg(preprocessed_input)
        return style_outputs, content_outputs
