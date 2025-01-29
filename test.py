from model.geo_clip.geo_clip_model import GeoCLIP
from model.street_clip.street_clip_model import StreetCLIP
from test_components.evalutator import Evaluator

if __name__ == "__main__":
    # # Test a GeoCLIP model
    # geo_clip = GeoCLIP()
    # evaluator = Evaluator()

    # evaluator.evaluate(geo_clip)

    # # Test a StreetCLIP model with new prompt format
    # steet_clip = StreetCLIP()
    # evaluator = Evaluator(processor_method="Street")

    # evaluator.evaluate(steet_clip)

    # Test a StreetCLIP model with original prompt format
    steet_clip = StreetCLIP()
    evaluator = Evaluator(processor_method="StreetOG")

    evaluator.evaluate(steet_clip)

