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

    def evaluate(self, model):
        prompts = self.get_prompts()
        toknized_prompts = self.dataset_handler.tokenizer.encode_texts(prompts)
        # embedded_texts = model.text_encoder.predict(toknized_prompts)

        generator = self.dataset_handler.create_generator(test_cfg.image_size, test_cfg.used_regions)
        for _ in range(test_cfg.iteration_amount):
            (image_input, text_input), _ = next(generator)
            logits_per_image = model.infer(image_input, toknized_prompts, ret_np=True)
            text_gt = self.dataset_handler.tokenizer.decode_texts(text_input)

            # similarities = model.compute_similarities(embedded_images, embedded_texts).numpy()
            best_prompt, conf = self.dataset_handler.decode_predictions_standard(logits_per_image, prompts)
            best_prompt_com = self.dataset_handler.decode_predictions_com(logits_per_image, toknized_prompts)

            print(f"Model guessed (standard): {best_prompt}, confidence: {conf})")
            print(f"Model guessed (CoM): {best_prompt_com}")  # center-of-mass
            print(f"Correct answer: {text_gt}")
            print()
