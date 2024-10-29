import tensorflow as tf
from keras.models import Model
from keras.layers import Layer, Dense, Embedding, LayerNormalization, MultiHeadAttention

class PatchEmbedding(Layer):
    def __init__(self, num_patches, patch_size, d_model):
        super(PatchEmbedding, self).__init__()

        self.num_patches = num_patches
        self.patch_size = patch_size
        self.d_model = d_model

        self.projection = Dense(d_model)
        self.position_embedding = Embedding(input_dim=num_patches, output_dim=d_model)

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        num_patches = patches.shape[1] * patches.shape[2]
        patches = tf.reshape(patches, [batch_size, num_patches, patches.shape[3]])  # flattens patch-grid (patches[1] and patches[2])
        patches = self.projection(patches)

        positions = tf.range(start=0, limit=num_patches, delta=1)
        embedded_positions = self.position_embedding(positions)
        all_patches = patches + embedded_positions

        return all_patches
    

class TransformerBlock(Layer):
    def __init__(self, d_model, num_heads, mlp_dim):
        super(TransformerBlock, self).__init__()

        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)

        self.mlp = tf.keras.Sequential([
            Dense(mlp_dim, activation="relu"),
            Dense(d_model),
        ])

    def call(self, x):
        attn_output = self.mha(x, x)
        x = self.layernorm1(x + attn_output)

        mlp_output = self.mlp(x)
        norm_output = self.layernorm2(x + mlp_output)

        return norm_output
    

class VisionTransformer(Model):
    def __init__(self, num_patches, patch_size, d_model, num_layers, num_heads, mlp_dim, num_classes):
        super(VisionTransformer, self).__init__()

        self.d_model = d_model

        self.preprocess_function = lambda x: x / 255
        self.patch_embedding = PatchEmbedding(num_patches, patch_size, d_model)
        self.transformer_layers = [TransformerBlock(d_model, num_heads, mlp_dim) for _ in range(num_layers)]
        self.class_token = tf.Variable(tf.zeros([1, 1, d_model]), trainable=True)
        self.mlp_head = Dense(num_classes)

    def call(self, x):
        batch_size = tf.shape(x)[0]

        x = self.patch_embedding(x)
        class_token = tf.broadcast_to(self.class_token, [batch_size, 1, self.d_model])
        x = tf.concat([class_token, x], axis=1)

        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)

        class_token_output = x[:, 0]
        mlp_head_output = self.mlp_head(class_token_output)

        return mlp_head_output
