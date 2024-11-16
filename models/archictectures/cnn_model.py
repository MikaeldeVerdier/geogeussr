from keras.applications import VGG16
from keras.models import Model
from keras.layers import Conv2D, GlobalAveragePooling2D, Dense
from keras.regularizers import L2

class ConvolutionalNeuralNetwork(Model):
    def __init__(self, input_shape, num_unfrozen_base_layers, num_layers, dense_layers, num_classes, final_activation, kernel_initializer, l2_reg):
        super(ConvolutionalNeuralNetwork, self).__init__()

        self.num_unfrozen_base_layers = num_unfrozen_base_layers

        self.convolutional_layers = [Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer=kernel_initializer, kernel_regularizer=L2(l2_reg)) for _ in range(num_layers)]
        self.pool = GlobalAveragePooling2D()
        self.dense_head = [Dense(num_neuron, activation="relu") for num_neuron in dense_layers] + [Dense(num_classes, activation=final_activation)]

        self.build(input_shape)

    def build(self, input_shape):  # could really define base network in initializer
        super(ConvolutionalNeuralNetwork, self).build(input_shape)

        self.base_network = VGG16(include_top=False, weights="imagenet", input_shape=input_shape)
        self.base_network.trainable = False
        for layer in self.base_network.layers[-self.num_unfrozen_base_layers:]:
            layer.trainable = True

        self.built = True

    def call(self, inputs):
        x = self.base_network(inputs)

        for convolutional_layer in self.convolutional_layers:
            x = convolutional_layer(x)
        x = self.pool(x)

        for dense_layer in self.dense_head:
            x = dense_layer(x)

        return x
