import tensorflow as tf
from tf_keras.models import Model, load_model
from model.constrastive_loss import ContrastiveLoss
# from tf_keras import Variable

import model.geo_clip.submodels.configs.shared_config as shr_cfg
import model.geo_clip.submodels.configs.vit_config as vit_cfg
import model.geo_clip.submodels.configs.ttt_config as ttt_cfg
from model.geo_clip.submodels.vision_transformer import VisionTransformer
from model.geo_clip.submodels.text_transformer import TextTransformer
from model.geo_clip.submodels.modified_resnet import ModifiedResnet

class GeoCLIP(Model):
    def __init__(self, image_encoder="vit", **kwargs):
        super(GeoCLIP, self).__init__(**kwargs)

        if image_encoder == "resnet":
            self.image_encoder = ModifiedResnet(shr_cfg.embed_dim)
        else:
            self.image_encoder = VisionTransformer(vit_cfg.patch_size, vit_cfg.num_patches, shr_cfg.embed_dim, shr_cfg.num_heads, shr_cfg.ff_dim, shr_cfg.num_layers)
        self.text_encoder = TextTransformer(ttt_cfg.vocab_size, ttt_cfg.max_len, shr_cfg.embed_dim, shr_cfg.num_heads, shr_cfg.ff_dim, shr_cfg.num_layers)

        self.temperature = tf.Variable(initial_value=1.0, trainable=True, name="temperature", dtype=tf.float32)
        # self.temperature = self.add_weight(name="temperature", shape=(), initializer="ones")

    @classmethod
    def from_loaded(cls, load_path, **kwargs):
        return load_model(load_path, custom_objects={"ContrastiveLoss": ContrastiveLoss})

    def infer(self, image_input, text_input, ret_np=False):
        image_embeddings = self.image_encoder(image_input)
        text_embeddings = self.text_encoder(text_input)

        norm_img_embeddings = tf.nn.l2_normalize(image_embeddings, axis=-1)
        norm_txt_embeddings = tf.nn.l2_normalize(text_embeddings, axis=-1)

        logits_per_image = tf.matmul(norm_img_embeddings, norm_txt_embeddings, transpose_b=True) * self.temperature

        if ret_np:
            return logits_per_image.numpy()

        return logits_per_image

    def call(self, inputs, ret_np=False):
        image_input, text_input = inputs

        return self.infer(image_input, text_input, ret_np=ret_np)
