import tensorflow as tf


def load_img(path_to_img):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = img[tf.newaxis, :]
    return img


def gram_matrix(input_tensor):
    b, h, w, c = input_tensor.shape
    features = tf.reshape(input_tensor, [-1, c, w * h])
    gram = tf.matmul(features, features, transpose_b=True) / (c * h * w)
    return gram


def style_loss(gram_style, style_features_transformed):
    style_loss = tf.add_n(
        [
            tf.reduce_mean(tf.keras.losses.mean_squared_error(gram_matrix(sf_transformed), gm))
            for sf_transformed, gm in zip(
                style_features_transformed, gram_style
            )
        ]
    )
    return style_loss


def content_loss(content_features, content_features_transformed):
    content_loss = tf.add_n(
        [
            tf.reduce_mean(tf.keras.losses.mean_squared_error(cf_transformed, cf))
            for cf_transformed, cf in zip(
                content_features_transformed, content_features
            )
        ]
    )
    return content_loss


class TransferMonitor(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, update_freq, content_images, style_images):
        super().__init__()
        self.file_writer = tf.summary.create_file_writer(log_dir)
        self.update_freq = update_freq
        self.content_images = content_images
        self.style_images = style_images

    def on_train_begin(self, logs=None):
        with self.file_writer.as_default():
            tf.summary.image("Content Image", self.content_images / 255.0, step=0)
            tf.summary.image("Style Image", self.style_images / 255.0, step=0)

    def on_train_batch_end(self, batch, logs=None):
        if batch % self.update_freq == 0:
            stylized_images = self.model.model(self.content_images)
            with self.file_writer.as_default():
                tf.summary.image("Stylized Image", stylized_images / 255.0, step=batch)
