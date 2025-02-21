import tensorflow as tf
print(f"All devices: {tf.config.list_logical_devices('TPU')}")

tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
tf.tpu.experimental.initialize_tpu_system(tpu)
tpu_strategy = tf.distribute.TPUStrategy(tpu)

print("TPU strategy loaded.")

from keras.optimizers import Adam, SGD
from keras.optimizers.schedules import ExponentialDecay

from keras.models import Model, load_model
from transformers import TFCLIPModel
from model.constrastive_loss import ContrastiveLoss

class ClipCLIP(Model):  # ClipCLIP references is the exact same as StreetCLIP references, but with a different model and processor
    def __init__(self, load_path=None, **kwargs):
        super(ClipCLIP, self).__init__(**kwargs)

        with tpu_strategy.scope():
            if load_path is not None:
                self.clip_model = load_model(load_path, custom_objects={"ContrastiveLoss": ContrastiveLoss})
            else:
                self.clip_model = TFCLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
                opt = self.build_optimizer(1e-4, 1000, 0.95, 0.9, 0.95, 0.1)  # :)
                self.compile(optimizer=opt, loss=ContrastiveLoss(), steps_per_execution=32)

    def build_optimizer(self, initial_lr, decay_steps, decay_factor, beta_1, beta_2, weight_decay, **kwargs):
        schedule = ExponentialDecay(initial_lr, decay_steps, decay_factor, staircase=True)
        # optimizer = Adam(learning_rate=schedule, beta_1=beta_1, beta_2=beta_2, weight_decay=weight_decay)
        optimizer = SGD(learning_rate=schedule, momentum=beta_1, weight_decay=weight_decay)

        return optimizer

    @classmethod
    def from_save(cls, load_path, **kwargs):
        return cls(load_path=load_path, **kwargs)

    def infer(self, inputs, ret_np=False):
        outputs = self.clip_model(inputs)  # **inputs (changed for compatibility)

        logits_per_image = outputs["logits_per_image"]  # .logits_per_image (changed for compatibility)

        if ret_np:
            return logits_per_image.numpy()

        return logits_per_image

    def call(self, inputs, ret_np=False):
        return self.infer(inputs, ret_np=ret_np)

    def fit(self, *args, **kwargs):
        self.clip_model.fit(*args, **kwargs)

    def compile(self, *args, **kwargs):
        self.clip_model.compile(*args, **kwargs)
