import os
from argparse import ArgumentParser

import tensorflow as tf
import tensorflow_datasets as tfds

from networks import StyleContentModel, TransformerNet
from utils import load_img, gram_matrix, style_loss, content_loss

AUTOTUNE = tf.data.experimental.AUTOTUNE


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log-dir", default="models/style")
    parser.add_argument("--learning-rate", default=1e-3, type=float)
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--epochs", default=2, type=int)
    parser.add_argument("--content-weight", default=1e1, type=float)
    parser.add_argument("--style-weight", default=1e1, type=float)
    parser.add_argument("--style-image")
    parser.add_argument("--test-image")
    args = parser.parse_args()

    style_image = load_img(args.style_image)
    test_content_image = load_img(args.test_image)

    content_layers = ["block2_conv2"]
    style_layers = [
        "block1_conv2",
        "block2_conv2",
        "block3_conv3",
        "block4_conv3",
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

    log_dir = os.path.join(
        args.log_dir,
        "lr={lr}_bs={bs}_sw={sw}_cw={cw}".format(
            lr=args.learning_rate,
            bs=args.batch_size,
            sw=args.style_weight,
            cw=args.content_weight,
        ),
    )

    manager = tf.train.CheckpointManager(ckpt, log_dir, max_to_keep=1)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_style_loss = tf.keras.metrics.Mean(name="train_style_loss")
    train_content_loss = tf.keras.metrics.Mean(name="train_content_loss")

    summary_writer = tf.summary.create_file_writer(log_dir)

    with summary_writer.as_default():
        tf.summary.image("Content Image", test_content_image / 255.0, step=0)
        tf.summary.image("Style Image", style_image / 255.0, step=0)

    @tf.function
    def train_step(images):
        with tf.GradientTape() as tape:

            transformed_images = transformer(images)

            _, content_features = extractor(images)
            style_features_transformed, content_features_transformed = extractor(
                transformed_images
            )

            tot_style_loss = args.style_weight * style_loss(
                gram_style, style_features_transformed
            )
            tot_content_loss = args.content_weight * content_loss(
                content_features, content_features_transformed
            )

            loss = tot_style_loss + tot_content_loss

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, transformer.trainable_variables)
        )

        train_loss(loss)
        train_style_loss(tot_style_loss)
        train_content_loss(tot_content_loss)

    def _crop(features):
        image = tf.image.resize_with_crop_or_pad(
            features["image"], args.image_size, args.image_size
        )
        image = tf.cast(image, tf.float32)
        return image

    # Warning: Downloads the full coco/2014 dataset
    ds = tfds.load("coco/2014", split="train")
    ds = ds.map(_crop).batch(args.batch_size).prefetch(AUTOTUNE)

    for epoch in range(args.epochs):
        for images in ds:
            train_step(images)

            ckpt.step.assign_add(1)
            step = int(ckpt.step)

            if step % 500 == 0:
                with summary_writer.as_default():
                    tf.summary.scalar("loss", train_loss.result(), step=step)
                    tf.summary.scalar(
                        "style_loss", train_style_loss.result(), step=step
                    )
                    tf.summary.scalar(
                        "content_loss", train_content_loss.result(), step=step
                    )
                    test_styled_image = transformer(test_content_image)
                    test_styled_image = tf.clip_by_value(
                        test_styled_image, 0, 255
                    )
                    tf.summary.image(
                        "Styled Image", test_styled_image / 255.0, step=step
                    )

                template = "Epoch {}, Step {}, Loss: {}, Style Loss: {}, Content Loss: {}"
                print(
                    template.format(
                        epoch + 1,
                        step,
                        train_loss.result(),
                        train_style_loss.result(),
                        train_content_loss.result(),
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
