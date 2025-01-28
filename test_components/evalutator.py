from math import radians, sin, cos, sqrt, atan2

import test_components.test_config as test_cfg
from shared_components.dataset_handler import DatasetHandler

class Evaluator:
    def __init__(self):
        self.dataset_handler = DatasetHandler(test_cfg.dataset_path, 1, 1, test_cfg.regions)

    def get_prompts(self):
        locations = [{"country": region, "lat": region_origin[1], "lng": region_origin[0]} for region, region_origin in zip(test_cfg.regions, test_cfg.region_origins)]
        prompts = []
        for location in locations:
            prompts.append(self.dataset_handler.generate_description(location))

        return prompts

    def great_circle_distance(self, lat1, lng1, lat2, lng2, r):
        dlat = lat2 - lat1
        dlon = lng2 - lng1

        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance = r * c

        return distance

    def evaluate_result(self, pred, gt):
        comp_pred = self.dataset_handler.tokenizer.get_components([pred])
        comp_gt = self.dataset_handler.tokenizer.get_components([gt])
        
        correct_region = comp_pred[0] == comp_gt[0]

        R = 6371.0  # Earth's radius in km
        lat1, lng1, lat2, lng2 = map(radians, comp_pred[1][0] + comp_gt[1][0])
        distance = self.great_circle_distance(lat1, lng1, lat2, lng2, r=R)  # (km)

        return correct_region, distance

    def evaluate(self, model):
        prompts = self.get_prompts()
        toknized_prompts = self.dataset_handler.tokenizer.encode_texts(prompts)
        # embedded_texts = model.text_encoder.predict(toknized_prompts)

        region_results = []
        distance_results = []
        generator = self.dataset_handler.create_generator(test_cfg.image_size, test_cfg.used_regions)
        for _ in range(test_cfg.iteration_amount):
            (image_input, text_input), _ = next(generator)
            logits_per_image = model.infer(image_input, toknized_prompts, ret_np=True)
            text_gt = self.dataset_handler.tokenizer.decode_texts(text_input)

            # similarities = model.compute_similarities(embedded_images, embedded_texts).numpy()
            best_prompt, conf = self.dataset_handler.decode_predictions_standard(logits_per_image, prompts)
            best_prompt_com = self.dataset_handler.decode_predictions_com(logits_per_image, toknized_prompts)

            print(f"Model guessed (standard): {best_prompt}, confidence: {conf})")
            # print(f"Model guessed (CoM): {best_prompt_com}")  # center-of-mass
            print(f"Correct answer: {text_gt}")

            correct_region, distance = self.evaluate_result(best_prompt[0], text_gt[0])
            region_results.append(correct_region)
            distance_results.append(distance)

            print(f"Country is {'correct' if correct_region else 'incorrect'}.")
            print(f"Distance is {distance:.2f}km.")

        print("Total results:")

        region_accuracy = sum(map(int, region_results)) / len(region_results)
        mean_distance = sum(distance_results) / len(distance_results)
        print(f"Country was correct {region_accuracy * 100:}% of the time.")
        print(f"Average distance was {mean_distance:2f}km.")
