import os
from argparse import ArgumentParser

import tensorflow as tf
import tensorflow_datasets as tfds

import utils
from networks import StyleContentModel, TransformerNet, FastStyleTransfer

AUTOTUNE = tf.data.AUTOTUNE


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log-dir", default="models/style")
    parser.add_argument("--learning-rate", default=1e-3, type=float)
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=2, type=int)
    parser.add_argument("--content-weight", default=1e1, type=float)
    parser.add_argument("--style-weight", default=1e1, type=float)
    parser.add_argument("--style-image", required=True)
    parser.add_argument("--test-image", required=True)
    parser.add_argument("--log-freq", default=500, type=int)
    args = parser.parse_args()

    style_image = utils.load_img(args.style_image)
    test_content_image = utils.load_img(args.test_image)

    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        extractor = StyleContentModel()
        transformer = TransformerNet()
        model = FastStyleTransfer(transformer, extractor, style_image, args.style_weight, args.content_weight)
        optimizer = tf.optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(optimizer=optimizer)

    log_dir = os.path.join(
        args.log_dir,
        "lr={lr}_bs={bs}_sw={sw}_cw={cw}".format(
            lr=args.learning_rate,
            bs=args.batch_size,
            sw=args.style_weight,
            cw=args.content_weight,
        ),
    )

    def pre_process(features):
        image = features["image"]
        image = tf.image.resize(image, size=(args.image_size, args.image_size))
        image = tf.cast(image, tf.float32)
        return image

    # Warning: Downloads the full coco/2014 dataset
    ds = (
        tfds.load("coco/2014", split="train")
        .map(pre_process, num_parallel_calls=AUTOTUNE)
        .batch(args.batch_size)
        .prefetch(AUTOTUNE)
    )

    model.fit(
        ds,
        epochs=args.epochs,
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch=0),
            utils.TransferMonitor(
                log_dir=log_dir,
                update_freq=args.log_freq,
                content_images=test_content_image,
                style_images=style_image,
            ),
            tf.keras.callbacks.TerminateOnNaN(),
        ],
    )
