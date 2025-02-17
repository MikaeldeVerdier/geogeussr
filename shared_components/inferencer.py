# import test_components.test_config as test_cfg
from shared_components.dataset_handler import DatasetHandler
from shared_components.files import save_json

class Inferencer:
    def __init__(self, image_size, refinement_steps, dataset_path, gadm_path=None, city_path=None, processor_method="Geo"):
        self.refinement_steps = refinement_steps
        self.dataset_path = dataset_path

        self.dataset_handler = DatasetHandler(dataset_path, image_size, 1, 1, gadm_path=gadm_path, city_path=city_path, processor_method=processor_method)

    def infer(self, model, used_prompts, used_prompt_components, inputs=None, image=None, use_com=False):
        process_first = inputs is None and image is not None

        for refinement_level in range(self.refinement_steps + 1):
            if process_first:
                inputs, _ = self.dataset_handler.preprocessor([], passed_images=image, passed_prompts=used_prompts)

            # import numpy as np
            # inputs["input_ids"] = np.array(inputs["input_ids"], dtype=np.int32)
            # inputs["attention_mask"] = np.array(inputs["attention_mask"], dtype=np.int32)

            logits_per_image = model(inputs, ret_np=True)

            if refinement_level == 0:
                first_sim_matrix = self.dataset_handler.softmax(logits_per_image)

            if not use_com:
                best_prompt, best_prompt_components, conf = self.dataset_handler.decode_predictions_standard(logits_per_image, used_prompts, used_prompt_components)
                print(f"Model guessed (standard): {best_prompt}, confidence: {conf})")
            else:
                best_prompt, best_prompt_components = self.dataset_handler.decode_predictions_com(logits_per_image, used_prompts, used_prompt_components)
                print(f"Model guessed (CoM): {best_prompt}")  # center-of-mass

            if refinement_level == self.refinement_steps:
                continue  # don't need to do the last ones

            img = self.dataset_handler.preprocessor.find_image(inputs)
            used_prompts = self.dataset_handler.preprocessor.get_refinement_prompts(best_prompt[0], refinement_amount=refinement_level)
            if not len(used_prompts):
                break  # could try to continue if next refinement level is possible but would require a restructure

            if not process_first:
                inputs, _ = self.dataset_handler.preprocessor([], passed_processed_images=img, passed_prompts=used_prompts)

        return best_prompt, best_prompt_components, first_sim_matrix

    def save_inference(self, prompt_components, image, best_prompt_components, sim_matrix, path, correct_components=None):
        inference_results = {
            "prompts": prompt_components,
            "image": image.tolist(),
            "confs": sim_matrix.tolist(),
            "best_prompt": best_prompt_components
        }
        if correct_components is not None:
            inference_results["correct_prompt"] = correct_components

        save_json(inference_results, path)
