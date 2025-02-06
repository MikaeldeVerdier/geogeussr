import inference_components.inference_config as inference_config
from shared_components.dataset_handler import DatasetHandler

class Inferencer:
    def __init__(self, processor_method="Geo"):
        processor_kwargs = {
            "iamge_size": inference_config.image_size,
            "max_len": inference_config.max_len,
            "refinement_base": inference_config.refinement_base
        }

        self.dataset_handler = DatasetHandler("", 1, 1, inference_config.regions, processor_method=processor_method, processor_kwargs=processor_kwargs)

    def __call__(self, model, image_path, use_com=False):
        used_prompts = self.dataset_handler.preprocessor.get_prompts()
        image = self.dataset_handler.preprocessor.get_image(image_path, inference_config.image_size)[None]

        for refinement_level in range(inference_config.refinement_steps + 1):
            inputs, _ = self.dataset_handler.preprocessor([], inference_config.image_size, passed_images=image, passed_prompts=used_prompts)

            logits_per_image = model(inputs, ret_np=True)

            if not use_com:
                best_prompt, conf = self.dataset_handler.decode_predictions_standard(logits_per_image, used_prompts)
                print(f"Model guessed (standard): {best_prompt}, confidence: {conf})")
            else:
                best_prompt = self.dataset_handler.decode_predictions_com(logits_per_image, used_prompts)
                print(f"Model guessed (CoM): {best_prompt}")  # center-of-mass

            if refinement_level == inference_config.refinement_steps:
                continue  # don't need to do the last ones

            used_prompts = self.dataset_handler.preprocessor.get_refinement_prompts(best_prompt[0], refinement_amount=refinement_level)
            if not len(used_prompts):
                break  # could try to continue if next refinement level is possible but would require a restructure
