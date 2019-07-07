import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, models


class ResidualBlock(keras.Model):
    def __init__(self, channels, strides=1):
        super(ResidualBlock, self).__init__()

        self.channels = channels
        self.strides = strides

        self.conv1 = layers.Conv2D(
            channels, (3, 3), strides=strides, padding="same"
        )
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(
            channels, (3, 3), strides=strides, padding="same"
        )
        self.bn2 = layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        residual = inputs

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x = x + residual
        return x


class TransformerNet(keras.Model):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        self.conv1 = layers.Conv2D(
            32, kernel=(9, 9), strides=1, padding="same"
        )
        self.in1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(
            64, kernel=(3, 3), strides=2, padding="same"
        )
        self.in2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(
            128, kernel=(3, 3), strides=2, padding="same"
        )
        self.in3 = layers.BatchNormalization()
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = layers.Conv2DTranspose(
            64, kernel_size=(3, 3), strides=2, padding="same"
        )
        self.in4 = layers.BatchNormalization()
        self.deconv2 = layers.Conv2DTranspose(
            32, kernel_size=(3, 3), strides=2, padding="same"
        )
        self.in5 = layers.BatchNormalization()
        self.deconv3 = layers.Conv2D(
            3, kernel=(9, 9), strides=1, padding="same", activation="tanh"
        )
        # Non-linearities
        self.relu = layers.ReLU()

    def call(self, inputs, training=None, **kwargs):
        x = inputs / 255.0
        x = self.relu(self.in1(self.conv1(x), training=training))
        x = self.relu(self.in2(self.conv2(x), training=training))
        x = self.relu(self.in3(self.conv3(x), training=training))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.relu(self.in4(self.deconv1(x), training=training))
        x = self.relu(self.in5(self.deconv2(x), training=training))
        x = self.deconv3(x)
        return x * 127.5 + 127.5


def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


def gram_matrix(input_tensor):
    result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / (num_locations)


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs, **kwargs):
        inputs = inputs
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(
            inputs
        )
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (
            outputs[: self.num_style_layers],
            outputs[self.num_style_layers :],
        )

        style_outputs = [
            gram_matrix(style_output) for style_output in style_outputs
        ]

        content_dict = {
            content_name: value
            for content_name, value in zip(
                self.content_layers, content_outputs
            )
        }

        style_dict = {
            style_name: value
            for style_name, value in zip(self.style_layers, style_outputs)
        }

        return {"content": content_dict, "style": style_dict}
