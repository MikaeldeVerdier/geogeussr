from tf_keras.models import Model, load_model
from transformers import TFCLIPModel
from model.constrastive_loss import ContrastiveLoss

class StreetCLIP(Model):  # this class is kinda useless, just uses clip_model basically
    def __init__(self, load_path=None, **kwargs):
        super(StreetCLIP, self).__init__(**kwargs)

        if load_path is not None:
            self.clip_model = load_model(load_path, custom_objects={"ContrastiveLoss": ContrastiveLoss})
        else:
            self.clip_model = TFCLIPModel.from_pretrained("geolocal/StreetCLIP", from_pt=True)

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
