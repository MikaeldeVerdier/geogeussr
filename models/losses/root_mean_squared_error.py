import tensorflow as tf

class RootMeanSquareError(tf.keras.losses.Loss):
    def __init__(self, name="root_mean_square_error", **kwargs):
        super(RootMeanSquareError, self).__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        return tf.math.sqrt(tf.reduce_mean(tf.math.square(y_true - y_pred), axis=-1))
