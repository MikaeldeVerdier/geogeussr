from tf_keras import layers, Model
from tf_keras.applications import ResNet50

class ModifiedResnet(Model):
    def __init__(self, embed_dim, **kwargs):
        super(ModifiedResnet, self).__init__(**kwargs)

        self.resnet = ResNet50(include_top=False, weights="imagenet")
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(embed_dim)

    def call(self, inputs):
        cnn_output = self.resnet(inputs)
        flattened_output = self.flatten(cnn_output)
        embed_output = self.dense(flattened_output)

        return embed_output
