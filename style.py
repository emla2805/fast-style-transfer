import tensorflow as tf

from PIL import Image
from argparse import ArgumentParser

from networks import TransformerNet
from utils import load_img

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log-dir")
    parser.add_argument("--image-path")
    parser.add_argument("--output-path")
    args = parser.parse_args()

    image = load_img(args.image_path)

    transformer = TransformerNet()
    ckpt = tf.train.Checkpoint(transformer=transformer)
    ckpt.restore(tf.train.latest_checkpoint(args.log_dir)).expect_partial()

    transformed_image = transformer(image)
    transformed_image = tf.cast(
        tf.squeeze(transformed_image), tf.uint8
    ).numpy()

    img = Image.fromarray(transformed_image, mode="RGB")
    img.save(args.output_path)
