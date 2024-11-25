import configs.testing_config as test
from evalutator import Evaluator
from models.full_model import FullModel

if __name__ == "__main__":
    model = FullModel.load_full(test.SAVE_PATH, test.MODEL_NAME)

    evaluator = Evaluator(test.DATASET_PATH)
    evaluator.evaluate(model, test.AMOUNT_ITERATIONS)
