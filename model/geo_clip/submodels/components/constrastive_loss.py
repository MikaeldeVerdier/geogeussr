import tensorflow as tf
from tf_keras.losses import Loss, sparse_categorical_crossentropy

class ContrastiveLoss(Loss):
    def __init__(self, **kwargs):
        super(ContrastiveLoss, self).__init__(**kwargs)

    def call(self, y_true, y_pred):  # could consider a custom training loop since y_true isn't used
        # logits_per_image, logits_per_text = y_pred

        logits_per_image = y_pred  # why does only logits_per_image get passed?
        logits_per_text = tf.transpose(y_pred)

        batch_size = tf.shape(logits_per_image)[0]
        labels = tf.range(batch_size)

        loss_image_to_text = tf.reduce_mean(sparse_categorical_crossentropy(labels, logits_per_image, from_logits=True))
        loss_text_to_image = tf.reduce_mean(sparse_categorical_crossentropy(labels, logits_per_text, from_logits=True))

        total_loss = (loss_image_to_text + loss_text_to_image) / 2

        return total_loss
