from model.geo_clip.geo_clip_model import GeoCLIP
from model.street_clip.street_clip_model import StreetCLIP
from model.clip_clip.clip_clip_model import ClipCLIP
from inference_components.inferencer import Inferencer

if __name__ == "__main__":
    clip_clip = ClipCLIP()
    inferencer = Inferencer(processor_method="Clip")

    inferencer(clip_clip, "test5.png")
