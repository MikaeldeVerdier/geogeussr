import math
import tensorflow as tf

class HaversineLoss(tf.keras.losses.Loss):
    def __init__(self, name="haversine_loss", **kwargs):
        super(HaversineLoss, self).__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        lat1, lon1 = y_true[:, 0], y_true[:, 1]
        lat2, lon2 = y_pred[:, 0], y_pred[:, 1]
        
        pi_div_180 = tf.constant(math.pi / 180)

        lat1_rad = lat1 * pi_div_180
        lon1_rad = lon1 * pi_div_180
        lat2_rad = lat2 * pi_div_180
        lon2_rad = lon2 * pi_div_180
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = tf.square(tf.sin(dlat / 2.0)) + tf.cos(lat1_rad) * tf.cos(lat2_rad) * tf.square(tf.sin(dlon / 2.0))
        c = 2 * tf.atan2(tf.sqrt(a), tf.sqrt(1 - a))
        
        earth_radius_km = 6371.0
        distance = earth_radius_km * c

        loss = tf.reduce_mean(distance)
        
        return loss
