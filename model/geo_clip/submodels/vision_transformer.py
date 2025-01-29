import tensorflow as tf
from tf_keras import layers, Model

from model.geo_clip.submodels.components.transformer_block import TransformerBlock

class VisionTransformer(Model):
    def __init__(self, patch_size, num_patches, embed_dim, num_heads, ff_dim, num_layers, **kwargs):  # , num_classes):
        super(VisionTransformer, self).__init__(**kwargs)

        self.patch_size = patch_size
        self.num_patches = num_patches
        self.projection = layers.Dense(embed_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches + 1, output_dim=embed_dim)
        self.class_token = tf.Variable(initial_value=tf.random.normal([1, 1, embed_dim]), trainable=True)
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ]
        # self.mlp_head = tf.keras.Sequential([
        #     layers.LayerNormalization(epsilon=1e-6),
        #     layers.Dense(num_classes, activation="softmax"),
        # ])

    def extract_patches(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        patches = tf.reshape(patches, [batch_size, self.num_patches, patches.shape[-1]])

        return patches

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        patches = self.extract_patches(inputs)
        projected_patches = self.projection(patches)

        class_token = tf.broadcast_to(self.class_token, [batch_size, 1, tf.shape(projected_patches)[-1]])
        tokens = tf.concat([class_token, projected_patches], axis=1)

        positions = tf.range(start=0, limit=self.num_patches + 1, delta=1)
        tokens += self.position_embedding(positions)

        for block in self.transformer_blocks:
            tokens = block(tokens)

        return tokens[:, 0]

        # class_representation = tokens[:, 0]

        # return self.mlp_head(class_representation)