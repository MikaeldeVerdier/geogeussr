import tensorflow as tf

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, name="focal_loss", alpha=0.25, gamma=2, epsilon=1e-7, **kwargs):
        super(FocalLoss, self).__init__(name=name, **kwargs)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, self.epsilon, 1 - self.epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = self.alpha * tf.pow(1 - y_pred, self.gamma)

        total_loss = tf.reduce_sum(cross_entropy * weight, axis=-1)

        return total_loss
