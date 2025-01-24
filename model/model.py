import tensorflow as tf
from keras.models import Model

import model.submodels.configs.shared_config as shr_cfg
import model.submodels.configs.vit_config as vit_cfg
import model.submodels.configs.ttt_config as ttt_cfg
from model.submodels.vision_transformer import VisionTransformer
from model.submodels.text_transformer import TextTransformer

class GeoCLIP(Model):
    def __init__(self, **kwargs):
        super(GeoCLIP, self).__init__(**kwargs)

        self.image_encoder = VisionTransformer(vit_cfg.patch_size, vit_cfg.num_patches, shr_cfg.embed_dim, shr_cfg.num_heads, shr_cfg.ff_dim, shr_cfg.num_layers)
        self.text_encoder = TextTransformer(ttt_cfg.vocab_size, ttt_cfg.max_len, shr_cfg.embed_dim, shr_cfg.num_heads, shr_cfg.ff_dim, shr_cfg.num_layers)

    def call(self, inputs):
        image_input, text_input = inputs
        image_embeddings = self.image_encoder(image_input)
        text_embeddings = self.text_encoder(text_input)

        sim_matrix = tf.matmul(image_embeddings, text_embeddings, transpose_b=True)

        return sim_matrix
