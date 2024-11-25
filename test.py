import configs.testing_config as test
from evalutator import Evaluator
from models.full_model import FullModel

if __name__ == "__main__":
    model = FullModel.load_complete(test.SAVE_PATH)

    evaluator = Evaluator(test.DATASET_PATH)
    evaluator.evaluate(model, test.AMOUNT_ITERATIONS)
