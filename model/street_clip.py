from keras.models import Model
from transformers import CLIPProcessor, TFCLIPModel

class StreetClip(Model):
    def __init__(self, **kwargs):
        super(StreetClip, self).__init__(**kwargs)

        # CLIPModel._backends = ["tf"]
        self.clip_model = TFCLIPModel.from_pretrained("geolocal/StreetCLIP", from_pt=True)
        self.clip_processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")

    def infer(self, image_input, text_input, ret_np=False):
        inputs = self.clip_processor(text=text_input, images=image_input, return_tensors="tf", padding=True, do_rescale=False)
        outputs = self.clip_model(**inputs)

        logits_per_image = outputs.logits_per_image

        if ret_np:
            return logits_per_image.numpy()

        return logits_per_image

    def call(self, inputs):
        image_input, text_input = inputs

        return self.infer(image_input, text_input)
