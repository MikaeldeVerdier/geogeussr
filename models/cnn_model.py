from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.layers import Conv2D, GlobalAveragePooling2D, Dense
from keras.regularizers import L2

class ConvolutionalNeuralNetwork(Model):
    def __init__(self, input_shape, num_unfrozen_base_layers, num_layers, dense_layers, num_classes, kernel_initializer, l2_reg):
        super(ConvolutionalNeuralNetwork, self).__init__()

        self.preprocess_func = preprocess_input
        self.base_network = VGG16(include_top=False, weights="imagenet", input_shape=input_shape)
        for layer in self.base_network.layers[-num_unfrozen_base_layers:]:
            layer.trainable = True

        self.convolutional_layers = [Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer=kernel_initializer, kernel_regularizer=L2(l2_reg)) for _ in range(num_layers)]
        self.dense_head = [Dense(num_neuron, activation="relu") for num_neuron in dense_layers] + [Dense(num_classes)]

    def call(self, x):
        x = self.base_network(x)

        for convolutional_layer in self.convolutional_layers:
            x = convolutional_layer(x)
        x = GlobalAveragePooling2D()(x)

        for dense_layer in self.dense_head:
            x = dense_layer(x)

        return x
