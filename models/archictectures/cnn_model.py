from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Conv2D, ReLU, GlobalAveragePooling2D, Dense
from keras.regularizers import L2

from models.subclassed_model import SubclassedModel

class ConvolutionalNeuralNetwork(SubclassedModel):  # Does it still need to be subclassed?
    def __init__(self, input_shape, unfrozen_base_layers, layers, dense_layers, num_classes, final_activation, kernel_initializer, l2_reg):
        super().__init__(input_shape=input_shape, unfrozen_base_layers=unfrozen_base_layers, layers=layers, dense_layers=dense_layers, num_classes=num_classes,final_activation=final_activation, kernel_initializer=kernel_initializer, l2_reg=l2_reg)

        self.image_size = input_shape
        self.num_classes = num_classes

        self.preprocess_func = preprocess_input
        self.base_network = VGG16(include_top=False, weights="imagenet", input_shape=input_shape)

        for i, layer in enumerate(self.base_network.layers):
            if i < len(self.base_network.layers) - unfrozen_base_layers:
                layer.trainable = False
            else:
                layer.trainable = True

        self.convolutional_layers = [Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation, kernel_initializer=kernel_initializer, kernel_regularizer=L2(l2_reg)) for filters, kernel_size, strides, padding, activation in layers]
        self.pool = GlobalAveragePooling2D()
        self.dense_head = [Dense(num_neuron, activation="relu") for num_neuron in dense_layers] + [Dense(num_classes, activation=final_activation)]

    def call(self, inputs):
        x = self.base_network(inputs)

        for convolutional_layer in self.convolutional_layers:
            x = convolutional_layer(x)
        x = self.pool(x)

        for dense_layer in self.dense_head:
            x = dense_layer(x)

        return x
