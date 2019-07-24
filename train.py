import os
import logging
from argparse import ArgumentParser

import tensorflow as tf
import tensorflow_datasets as tfds

from networks import StyleContentModel, TransformerNet
from utils import load_img, gram_matrix, style_loss, content_loss

logging.basicConfig(level=logging.INFO)
AUTOTUNE = tf.data.experimental.AUTOTUNE


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log-dir", default="logs/style")
    parser.add_argument("--learning-rate", default=1e-3, type=float)
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=2, type=int)
    parser.add_argument("--content-weight", default=1e4, type=float)
    parser.add_argument("--style-weight", default=1e-2, type=float)
    parser.add_argument("--tv-weight", default=1, type=float)
    parser.add_argument("--style-image")
    parser.add_argument("--test-image")
    args = parser.parse_args()

    style_image = load_img(args.style_image)
    test_content_image = load_img(args.test_image)

    content_layers = ["block4_conv2"]
    style_layers = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1",
        "block5_conv1",
    ]

    extractor = StyleContentModel(style_layers, content_layers)
    transformer = TransformerNet()

    # Precompute gram for style image
    style_features, _ = extractor(style_image)
    gram_style = [gram_matrix(x) for x in style_features]

    optimizer = tf.optimizers.Adam(learning_rate=args.learning_rate)

    ckpt = tf.train.Checkpoint(
        step=tf.Variable(1), optimizer=optimizer, transformer=transformer
    )
    manager = tf.train.CheckpointManager(ckpt, args.log_dir, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_style_loss = tf.keras.metrics.Mean(name="train_style_loss")
    train_content_loss = tf.keras.metrics.Mean(name="train_content_loss")
    train_tv_loss = tf.keras.metrics.Mean(name="train_tv_loss")

    train_summary_writer = tf.summary.create_file_writer(
        os.path.join(args.log_dir, "train")
    )

    with train_summary_writer.as_default():
        tf.summary.image("Content Image", test_content_image / 255.0, step=0)
        tf.summary.image("Style Image", style_image / 255.0, step=0)

    @tf.function()
    def train_step(image):
        with tf.GradientTape() as tape:

            transformed_image = transformer(image)

            _, content_features = extractor(image)
            style_features_transformed, content_features_transformed = extractor(
                transformed_image
            )

            # va_loss = args.tv_weight * total_variation_loss(transformed_image)

            tot_style_loss = args.style_weight * style_loss(
                gram_style, style_features_transformed
            )
            tot_content_loss = args.content_weight * content_loss(
                content_features, content_features_transformed
            )

            # loss = tot_style_loss + tot_content_loss + va_loss
            loss = tot_style_loss + tot_content_loss

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, transformer.trainable_variables)
        )

        train_loss(loss)
        train_style_loss(tot_style_loss)
        train_content_loss(tot_content_loss)
        # train_tv_loss(va_loss)

    def _crop(features):
        image = tf.image.resize_with_crop_or_pad(
            features["image"], args.image_size, args.image_size
        )
        image = tf.cast(image, tf.float32)
        return image

    # Warning: Downloads the full coco2014 dataset
    ds = tfds.load(
        "coco2014", split=tfds.Split.TRAIN, data_dir="~/tensorflow_datasets"
    )
    ds = ds.map(_crop).shuffle(1000).batch(args.batch_size).prefetch(AUTOTUNE)

    for _ in range(args.epochs):
        for image in ds:
            train_step(image)

            ckpt.step.assign_add(1)
            step = int(ckpt.step)

            if step % 500 == 0:
                with train_summary_writer.as_default():
                    tf.summary.scalar("loss", train_loss.result(), step=step)
                    tf.summary.scalar(
                        "style_loss", train_style_loss.result(), step=step
                    )
                    tf.summary.scalar(
                        "content_loss", train_content_loss.result(), step=step
                    )
                    tf.summary.scalar(
                        "tv_loss", train_tv_loss.result(), step=step
                    )
                    test_styled_image = transformer(test_content_image)
                    test_styled_image = tf.clip_by_value(
                        test_styled_image, 0, 255
                    )
                    tf.summary.image(
                        "Styled Image", test_styled_image / 255.0, step=step
                    )

                template = "Step {}, Loss: {}, Style Loss: {}, Content Loss: {}, TV Loss: {}"
                print(
                    template.format(
                        step,
                        train_loss.result(),
                        train_style_loss.result(),
                        train_content_loss.result(),
                        train_tv_loss.result(),
                    )
                )
                save_path = manager.save()
                print(
                    "Saved checkpoint for step {}: {}".format(
                        int(ckpt.step), save_path
                    )
                )

        train_loss.reset_states()
        train_style_loss.reset_states()
        train_content_loss.reset_states()
        train_tv_loss.reset_states()
