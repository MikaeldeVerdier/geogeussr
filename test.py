from model.geo_clip.geo_clip_model import GeoCLIP
from model.street_clip.street_clip_model import StreetCLIP
from model.clip_clip.clip_clip_model import ClipCLIP
from test_components.evalutator import Evaluator

if __name__ == "__main__":
    # Test a GeoCLIP model
    geo_clip = GeoCLIP(image_encoder="resnet")
    evaluator = Evaluator()

    evaluator.evaluate(geo_clip)

    # # Test a StreetCLIP model with new prompt format
    # street_clip = StreetCLIP()
    # evaluator = Evaluator(processor_method="Street")

    # evaluator.evaluate(street_clip)

    # # Test a StreetCLIP model with original prompt format
    # street_clip = StreetCLIP()
    # evaluator = Evaluator(processor_method="StreetOG")

    # evaluator.evaluate(street_clip)

    # # Test a ClipCLIP model with new prompt format
    # clip_clip = ClipCLIP()
    # evaluator = Evaluator(processor_method="Clip")

    # evaluator.evaluate(clip_clip)

    # # Test a ClipCLIP model with original prompt format
    # clip_clip = ClipCLIP()
    # evaluator = Evaluator(processor_method="ClipOG")

    # evaluator.evaluate(clip_clip)
