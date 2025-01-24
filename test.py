from model.model import GeoCLIP
from test_components.evalutator import Evaluator

if __name__ == "__main__":
    geo_clip = GeoCLIP()
    evaluator = Evaluator()

    evaluator.evaluate(geo_clip)
