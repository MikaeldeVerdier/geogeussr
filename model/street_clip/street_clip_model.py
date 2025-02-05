from tf_keras.models import Model
from transformers import TFCLIPModel

class StreetCLIP(Model):  # this class is kinda useless, just uses clip_model basically
    def __init__(self, **kwargs):
        super(StreetCLIP, self).__init__(**kwargs)

        self.clip_model = TFCLIPModel.from_pretrained("geolocal/StreetCLIP", from_pt=True)

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
