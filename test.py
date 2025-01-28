from model.geo_clip import GeoCLIP
from model.street_clip import StreetClip
from test_components.evalutator import Evaluator

if __name__ == "__main__":
    # geo_clip = GeoCLIP()
    # evaluator = Evaluator()

    steet_clip = StreetClip()
    evaluator = Evaluator(tokenizer_method="Street")

    evaluator.evaluate(steet_clip)
