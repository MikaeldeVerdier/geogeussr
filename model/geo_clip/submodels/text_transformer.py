import tensorflow as tf
from tf_keras import layers, Model

from model.geo_clip.submodels.components.transformer_block import TransformerBlock

class TextTransformer(Model):
    def __init__(self, vocab_size, max_len, embed_dim, num_heads, ff_dim, num_layers, **kwargs):
        super(TextTransformer, self).__init__(**kwargs)

        self.token_embedding = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.position_embedding = layers.Embedding(input_dim=max_len, output_dim=embed_dim)
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ]

    def call(self, inputs):
        positions = tf.range(start=0, limit=tf.shape(inputs)[-1], delta=1)
        x = self.token_embedding(inputs) + self.position_embedding(positions)

        for block in self.transformer_blocks:
            x = block(x)

        return x[:, 0]
