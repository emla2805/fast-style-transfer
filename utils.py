import tensorflow as tf


def load_img(path_to_img):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = img[tf.newaxis, :]
    return img


def gram_matrix(input_tensor):
    result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(
        input_shape[1] * input_shape[2] * input_shape[3], tf.float32
    )
    return result / num_locations


def style_loss(gram_style, style_features_transformed):
    style_loss = tf.add_n(
        [
            tf.reduce_mean((gram_matrix(sf_transformed) - gm) ** 2)
            for sf_transformed, gm in zip(
                style_features_transformed, gram_style
            )
        ]
    )
    return style_loss


def content_loss(content_features, content_features_transformed):
    content_loss = tf.add_n(
        [
            tf.reduce_mean((cf_transformed - cf) ** 2)
            for cf_transformed, cf in zip(
                content_features_transformed, content_features
            )
        ]
    )
    return content_loss
