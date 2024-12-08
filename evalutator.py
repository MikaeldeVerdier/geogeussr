from dataset_handler import DatasetHandler

class Evaluator:
    def __init__(self, dataset_path):
        self.dataset_handler = DatasetHandler(dataset_path, 1, 1)

    def evaluate(self, model, iteration_amount):  # n_iterations?
        generator = self.dataset_handler.create_generator(model.used_input_shape, model.preprocess_func, None, None)
        for _ in range(iteration_amount):
            batch_input, (coord_gt, country_gt) = next(generator)

            results = model.predict(batch_input)
            coord_result, country_result = self.dataset_handler.decode_predictions(results[0], results[1], ret_country=True)

            print(f"Model guessed: {coord_result[0]} ({country_result[0][0]}, confidence: {country_result[0][1]})")
            print(f"Correct answer: {coord_gt[0]} ({country_gt[0]})")
            print()
