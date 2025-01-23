import tensorflow as tf
from keras.losses import Loss, sparse_categorical_crossentropy

class ContrastiveLoss(Loss):
    def __init__(self, temperature=0.07, **kwargs):
        super(ContrastiveLoss, self).__init__(**kwargs)

        self.temperature = temperature

    def call(self, y_true, y_pred):  # could consider a custom training loop since y_true isn't used
        similarity_matrix = y_pred
        similarity_matrix /= self.temperature

        batch_size = tf.shape(similarity_matrix)[0]
        labels = tf.range(batch_size)

        loss_image_to_text = tf.reduce_mean(sparse_categorical_crossentropy(labels, similarity_matrix, from_logits=True))
        loss_text_to_image = tf.reduce_mean(sparse_categorical_crossentropy(labels, tf.transpose(similarity_matrix), from_logits=True))

        total_loss = (loss_image_to_text + loss_text_to_image) / 2

        return total_loss
