from dataset_handler import DatasetHandler

class Evaluator:
    def __init__(self, dataset_path):
        self.dataset_handler = DatasetHandler(dataset_path, 1, 1)

    def evaluate(self, model, iteration_amount):  # n_iterations?
        generator = self.dataset_handler.create_generator(model.used_input_shape, model.preprocess_func)
        for _ in range(iteration_amount):
            batch_input, coord_gt = next(generator)

            coords = model.predict(batch_input)

            print(f"Model guessed: {coords[0]}")
            print(f"Correct answer: {coord_gt[0]}")
            print()
