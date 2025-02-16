import os
from math import radians, sin, cos, sqrt, atan2

import validate_components.validator_config as val_cfg
from shared_components.inferencer import Inferencer
from shared_components.files import load_json, save_json

class Validator:
    def __init__(self, processor_method="Geo"):
        self.inferencer = Inferencer(val_cfg.image_size, val_cfg.refinement_steps, val_cfg.dataset_path, val_cfg.shapefile_path, processor_method=processor_method)

    def great_circle_distance(self, lat1, lng1, lat2, lng2, r):
        dlat = lat2 - lat1
        dlon = lng2 - lng1

        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance = r * c

        return distance

    def evaluate_result(self, pred, gt_location, regions):
        comp_pred = self.inferencer.dataset_handler.preprocessor.get_components([pred], regions)[0]

        correct_region = all([pred == gt for pred, gt in zip(comp_pred[0], gt_location[0])])  # zips different lengths

        R = 6371.0  # Earth's radius in km
        lat1, lng1, lat2, lng2 = map(radians, comp_pred[1] + gt_location[1])
        distance = self.great_circle_distance(lat1, lng1, lat2, lng2, r=R)  # (km)

        return correct_region, distance

    def validate(self, model, use_com=False):
        regions = load_json(val_cfg.regions_path)

        # prompts = self.get_prompts()
        save_inference_data = val_cfg.inference_results_path is not None
        if save_inference_data and not os.path.exists(val_cfg.inference_results_path):
            os.mkdir(val_cfg.inference_results_path)

        prompts = self.inferencer.dataset_handler.preprocessor.get_prompts(regions)
        process_kwargs = {"passed_prompts": prompts}

        rets = ["gt_location"]
        if save_inference_data:
            rets.append("raw_images")
        generator = self.inferencer.dataset_handler.create_generator(val_cfg.used_regions, shuffle=val_cfg.shuffle, rets=rets, processor_kwargs=process_kwargs)
        
        region_results = []
        distance_results = []
        for _ in range(val_cfg.iteration_amount):  # could process all the 1st refinement level promp images at once
            if not len(rets):
                inputs, gt = next(generator)
            else:
                inputs, gt, ret_values = next(generator)
            used_prompts = prompts

            best_prompt, sim_matrix = self.inferencer.infer(model, used_prompts, inputs=inputs, use_com=use_com)

            if save_inference_data:
                image = ret_values[0][0]  # saves unnormalized sometimes and sometimes normalized (depends on preprocessor). works though because inference_visualizer handles it
                inference_name = os.path.join(val_cfg.inference_results_path, f"inference_{gt[0]}.json")
                self.inferencer.save_inference(regions, used_prompts, image, best_prompt[0], sim_matrix[0], inference_name, correct_location=ret_values[1][0])

            print(f"Correct answer: {gt}")

            correct_region, distance = self.evaluate_result(best_prompt[0], ret_values[1][0], regions)
            region_results.append(correct_region)
            distance_results.append(distance)

            print(f"Country is {'correct' if correct_region else 'incorrect'}.")
            print(f"Distance is {distance:.2f}km.")

        print("Total results:")

        region_accuracy = sum(map(int, region_results)) / len(region_results)
        mean_distance = sum(distance_results) / len(distance_results)
        print(f"Country was correct {region_accuracy * 100:}% of the time.")
        print(f"Average distance was {mean_distance:.2f}km.")

        validation_results = {
            "correct_regions": region_results,
            "distances": distance_results
        }
        save_json(validation_results, val_cfg.vaidation_results_path)
