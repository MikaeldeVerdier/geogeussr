from math import radians, sin, cos, sqrt, atan2

import test_components.test_config as test_cfg
from shared_components.dataset_handler import DatasetHandler
from shared_components.files import save_json

class Evaluator:
    def __init__(self, processor_method="Geo"):
        processor_kwargs = {
            "iamge_size": test_cfg.image_size,
            "max_len": test_cfg.max_len,
            "refinement_base": test_cfg.refinement_base
        }

        self.dataset_handler = DatasetHandler(test_cfg.dataset_path, 1, 1, test_cfg.regions, processor_method=processor_method, processor_kwargs=processor_kwargs, shapefile_path=test_cfg.shapefile_path)

    def great_circle_distance(self, lat1, lng1, lat2, lng2, r):
        dlat = lat2 - lat1
        dlon = lng2 - lng1

        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance = r * c

        return distance

    def evaluate_result(self, pred, gt):
        comp_pred = self.dataset_handler.preprocessor.get_components([pred])[0]
        comp_gt = self.dataset_handler.preprocessor.get_components([gt])[0]
        
        correct_region = comp_pred[0] == comp_gt[0]

        R = 6371.0  # Earth's radius in km
        lat1, lng1, lat2, lng2 = map(radians, comp_pred[1:] + comp_gt[1:])
        distance = self.great_circle_distance(lat1, lng1, lat2, lng2, r=R)  # (km)

        return correct_region, distance

    def evaluate(self, model, use_com=False):
        # prompts = self.get_prompts()
        prompts = self.dataset_handler.preprocessor.get_prompts()
        process_kwargs = {
            "passed_prompts": prompts
        }

        region_results = []
        distance_results = []
        generator = self.dataset_handler.create_generator(test_cfg.image_size, test_cfg.used_regions, processor_kwargs=process_kwargs)
        for _ in range(test_cfg.iteration_amount):
            inputs, gt = next(generator)
            used_prompts = prompts

            for refinement_level in range(test_cfg.refinement_steps + 1):
                logits_per_image = model(inputs, ret_np=True)

                if not use_com:
                    best_prompt, conf = self.dataset_handler.decode_predictions_standard(logits_per_image, used_prompts)
                    print(f"Model guessed (standard): {best_prompt}, confidence: {conf})")
                else:
                    best_prompt = self.dataset_handler.decode_predictions_com(logits_per_image, used_prompts)
                    print(f"Model guessed (CoM): {best_prompt}")  # center-of-mass

                if refinement_level == test_cfg.refinement_steps:
                    continue  # don't need to do the last ones

                img = self.dataset_handler.preprocessor.find_image(inputs)
                used_prompts = self.dataset_handler.preprocessor.get_refinement_prompts(best_prompt[0], refinement_amount=refinement_level)
                if not len(used_prompts):
                    break  # could try to continue if next refinement level is possible but would require a restructure

                inputs, _ = self.dataset_handler.preprocessor([], test_cfg.image_size, passed_processed_images=img, passed_prompts=used_prompts)

            print(f"Correct answer: {gt}")

            correct_region, distance = self.evaluate_result(best_prompt[0], gt[0])
            region_results.append(correct_region)
            distance_results.append(distance)

            print(f"Country is {'correct' if correct_region else 'incorrect'}.")
            print(f"Distance is {distance:.2f}km.")

        print("Total results:")

        region_accuracy = sum(map(int, region_results)) / len(region_results)
        mean_distance = sum(distance_results) / len(distance_results)
        print(f"Country was correct {region_accuracy * 100:}% of the time.")
        print(f"Average distance was {mean_distance:.2f}km.")

        test_results = {
            "correct_regions": region_results,
            "distances": distance_results
        }
        save_json(test_results, test_cfg.test_results_path)
