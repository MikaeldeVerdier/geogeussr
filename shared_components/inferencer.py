# import test_components.test_config as test_cfg
from shared_components.dataset_handler import DatasetHandler
from shared_components.files import save_json

class Inferencer:
    def __init__(self, image_size, refinement_steps, dataset_path, gadm_path=None, city_path=None, processor_method="Geo"):
        self.refinement_steps = refinement_steps
        self.dataset_path = dataset_path

        self.dataset_handler = DatasetHandler(dataset_path, image_size, 1, 1, gadm_path=gadm_path, city_path=city_path, processor_method=processor_method)

    def infer(self, model, used_prompts, used_prompt_components, inputs=None, image=None, regions=None, use_com=False):
        process_first = inputs is None and image is not None

        refinement_results = []
        for refinement_level in range(self.refinement_steps + 1):
            if process_first:
                inputs, _ = self.dataset_handler.preprocessor([], passed_images=image, passed_prompts=used_prompts)

            # import numpy as np
            # inputs["input_ids"] = np.array(inputs["input_ids"], dtype=np.int32)
            # inputs["attention_mask"] = np.array(inputs["attention_mask"], dtype=np.int32)

            logits_per_image = model(inputs, ret_np=True)

            if not use_com:
                best_prompt, best_prompt_components, conf = self.dataset_handler.decode_predictions_standard(logits_per_image, used_prompts, used_prompt_components)
                print(f"Model guessed (standard): {best_prompt} (confidence: {conf})")
            else:
                best_prompt, best_prompt_components = self.dataset_handler.decode_predictions_com(logits_per_image, used_prompts, used_prompt_components)
                print(f"Model guessed (hedging): {best_prompt}")  # center-of-mass

            confs = self.dataset_handler.softmax(logits_per_image)  # done again in decode_predictions_standard but whatever
            refinement_results.append({
                "prompts": used_prompts,
                "prompt_components": used_prompt_components,
                "used_prompt": best_prompt[0],
                "used_prompt_components": best_prompt_components[0],
                "confidences": confs[0].tolist()
            })

            if refinement_level == self.refinement_steps:
                continue  # don't need to do the last ones

            # this loses batch generality, but it needs to, there is not other way to do it. just weird to have this hybrid generality
            used_prompts, used_prompt_components = self.dataset_handler.preprocessor.get_prompts(regions, best_prompt_components[0][0].copy(), refinement_amount=refinement_level + 2)
            if not len(used_prompts):
                break  # could try to continue if next refinement level is possible but would require a restructure

            img = self.dataset_handler.preprocessor.find_image(inputs)
            inputs, _ = self.dataset_handler.preprocessor([], passed_processed_images=img, passed_prompts=used_prompts)

        return refinement_results

    def save_inference(self, path, image, refinement_results, correct_components=None):
        inference_results = {
            "image": image.tolist(),
            "refinement_results": refinement_results
        }
        if correct_components is not None:
            inference_results["correct_prompt"] = correct_components

        save_json(inference_results, path)
