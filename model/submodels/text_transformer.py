import tensorflow as tf
from keras import layers, Model

from model.submodels.components.transformer_block import TransformerBlock

class TextTransformer(Model):
    def __init__(self, vocab_size, max_len, embed_dim, num_heads, ff_dim, num_layers):
        super(TextTransformer, self).__init__()

        self.token_embedding = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.position_embedding = layers.Embedding(input_dim=max_len, output_dim=embed_dim)
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ]
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        positions = tf.range(start=0, limit=tf.shape(inputs)[-1], delta=1)
        x = self.token_embedding(inputs) + self.position_embedding(positions)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.layernorm(x)

        return x[:, 0]
