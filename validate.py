from model.geo_clip.geo_clip_model import GeoCLIP
from model.street_clip.street_clip_model import StreetCLIP
from model.clip_clip.clip_clip_model import ClipCLIP
from validate_components.validator import Validator

if __name__ == "__main__":
    # Validate a GeoCLIP model
    geo_clip = GeoCLIP(image_encoder="resnet")
    validator = Validator()

    validator.validate(geo_clip, use_com=True)


    # # Validate a StreetCLIP model with new prompt format
    # street_clip = StreetCLIP()
    # validator = Validator(processor_method="Street")

    # validator.validate(street_clip)


    # # Validate a StreetCLIP model with original prompt format
    # street_clip = StreetCLIP()
    # validator = Validator(processor_method="StreetOG")

    # validator.validate(street_clip)


    # # Validate a ClipCLIP model with new prompt format
    # clip_clip = ClipCLIP()
    # validator = Validator(processor_method="Clip")

    # validator.validate(clip_clip)


    # # Validate a ClipCLIP model with original prompt format
    # clip_clip = ClipCLIP()
    # validator = Validator(processor_method="ClipOG")

    # validator.validate(clip_clip)
