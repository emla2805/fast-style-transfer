import os
import logging

import tensorflow as tf
import tensorflow_datasets as tfds

from networks import StyleContentModel, TransformerNet

logging.basicConfig(level=logging.INFO)
AUTOTUNE = tf.data.experimental.AUTOTUNE

total_variation_weight = 1e8
style_weight = 1e-2
content_weight = 1e4


def load_img(path_to_img):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize_with_crop_or_pad(img, 256, 256)
    img = tf.cast(img, tf.float32)
    img = img[tf.newaxis, :]
    return img


def style_content_loss(outputs, transformed_outputs):
    style_outputs = outputs["style"]
    content_outputs = outputs["content"]
    transformed_content_outputs = transformed_outputs["content"]

    style_loss = tf.add_n(
        [
            tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
            for name in style_outputs.keys()
        ]
    )
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n(
        [
            tf.reduce_mean(
                (transformed_content_outputs[name] - content_outputs[name])
                ** 2
            )
            for name in content_outputs.keys()
        ]
    )
    content_loss *= content_weight / num_content_layers
    return style_loss, content_loss


def high_pass_x_y(image):
    x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_var = image[:, 1:, :, :] - image[:, :-1, :, :]

    return x_var, y_var


def total_variation_loss(image):
    x_deltas, y_deltas = high_pass_x_y(image)
    return tf.reduce_mean(x_deltas ** 2) + tf.reduce_mean(y_deltas ** 2)


if __name__ == "__main__":
    IMAGE_SIZE = 256
    log_dir = "logs"
    learning_rate = 0.02

    style_path = tf.keras.utils.get_file(
        "kandinsky.jpg",
        "https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg",
    )
    style_image = load_img(style_path)
    test_content_path = tf.keras.utils.get_file(
        "turtle.jpg",
        "https://storage.googleapis.com/download.tensorflow.org/example_images/Green_Sea_Turtle_grazing_seagrass.jpg",
    )
    test_content_image = load_img(test_content_path)

    # Content layer where will pull our feature maps
    content_layers = ["block5_conv2"]

    # Style layer of interest
    style_layers = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1",
        "block5_conv1",
    ]

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)

    extractor = StyleContentModel(style_layers, content_layers)
    transformer = TransformerNet()

    # Precompute style_targets
    style_image = style_image
    style_targets = extractor(style_image)["style"]

    optimizer = tf.optimizers.Adam(
        learning_rate=learning_rate, beta_1=0.99, epsilon=1e-1
    )

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_style_loss = tf.keras.metrics.Mean(name="train_style_loss")
    train_content_loss = tf.keras.metrics.Mean(name="train_content_loss")
    train_va_loss = tf.keras.metrics.Mean(name="train_va_loss")

    train_summary_writer = tf.summary.create_file_writer(
        os.path.join(log_dir, "train")
    )

    @tf.function()
    def train_step(image):
        with tf.GradientTape() as tape:

            transformed_image = transformer(image)

            outputs = extractor(image)
            transformed_outputs = extractor(transformed_image)

            style_loss, content_loss = style_content_loss(
                outputs, transformed_outputs
            )
            va_loss = total_variation_weight * total_variation_loss(image)
            loss = style_loss + content_loss + va_loss

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, transformer.trainable_variables)
        )

        # Log the losses
        train_loss(loss)
        train_style_loss(style_loss)
        train_content_loss(content_loss)
        train_va_loss(va_loss)

    def _crop(features):
        image = tf.image.resize_with_crop_or_pad(features["image"], 256, 256)
        image = tf.cast(image, tf.float32)
        return image

    ds = tfds.load("coco2014", split=tfds.Split.TRAIN)
    ds = ds.map(_crop).shuffle(1000).batch(4).prefetch(AUTOTUNE)

    epochs = 2
    step = 0

    for epoch in range(epochs):
        for batch, image in enumerate(ds):
            train_step(image)

            step += 1

            if (step + 1) % 500 == 0:
                with train_summary_writer.as_default():
                    tf.summary.scalar("loss", train_loss.result(), step=step)
                    tf.summary.scalar(
                        "style_loss", train_style_loss.result(), step=step
                    )
                    tf.summary.scalar(
                        "content_loss", train_content_loss.result(), step=step
                    )
                    tf.summary.scalar(
                        "va_loss", train_va_loss.result(), step=step
                    )
                    tf.summary.image(
                        "Reference Image",
                        test_content_image / 255.0,
                        step=step,
                    )
                    test_styled_image = transformer(test_content_image)
                    tf.summary.image(
                        "Styled Image", test_styled_image / 255.0, step=step
                    )

                template = "Epoch {}, Batch {}, Loss: {}, Style Loss: {}, Content Loss: {}, VA Loss: {}"
                print(
                    template.format(
                        epoch + 1,
                        batch + 1,
                        train_loss.result(),
                        train_style_loss.result(),
                        train_content_loss.result(),
                        train_va_loss.result(),
                    )
                )

            train_loss.reset_states()
            train_style_loss.reset_states()
            train_content_loss.reset_states()
            train_va_loss.reset_states()
