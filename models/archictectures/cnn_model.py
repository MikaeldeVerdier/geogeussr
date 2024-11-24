from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.layers import Conv2D, GlobalAveragePooling2D, Dense
from keras.regularizers import L2

from models.subclassed_model import SubclassedModel

class ConvolutionalNeuralNetwork(SubclassedModel):  # don't think this needs to be SubclassedModel...
    def __init__(self, input_shape, unfrozen_base_layers, num_layers, dense_layers, num_classes, final_activation, kernel_initializer, l2_reg):
        super().__init__(input_shape=input_shape, unfrozen_base_layers=unfrozen_base_layers, num_layers=num_layers, dense_layers=dense_layers, num_classes=num_classes,final_activation=final_activation, kernel_initializer=kernel_initializer, l2_reg=l2_reg)

        base_network = VGG16(include_top=False, weights="imagenet", input_shape=input_shape)

        self.base_layers = base_network.layers[-unfrozen_base_layers:]
        for layer in self.base_layers:
            layer.trainable = True

        self.convolutional_layers = [Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer=kernel_initializer, kernel_regularizer=L2(l2_reg)) for _ in range(num_layers)]
        self.pool = GlobalAveragePooling2D()
        self.dense_head = [Dense(num_neuron, activation="relu") for num_neuron in dense_layers] + [Dense(num_classes, activation=final_activation)]

    def call(self, inputs):
        x = inputs
        for layer in self.base_layers:
            x = layer(x)

        for convolutional_layer in self.convolutional_layers:
            x = convolutional_layer(x)
        x = self.pool(x)

        for dense_layer in self.dense_head:
            x = dense_layer(x)

        return x
