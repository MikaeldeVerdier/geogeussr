import os

import test_components.test_config as test_config
from shared_components.inferencer import Inferencer
from shared_components.files import save_json

class Tester:
    def __init__(self, processor_method="Geo"):
        self.inferencer = Inferencer(test_config.image_size, test_config.tokens_len, test_config.refinement_base, test_config.refinement_steps, "", test_config.regions, test_config.shapefile_path, processor_method=processor_method)

        if not os.path.exists(test_config.test_save_path):
            os.mkdir(test_config.test_save_path)

    def perform_test(self, model, image_path, use_com=False):
        prompts = self.inferencer.dataset_handler.preprocessor.get_prompts()
        image = self.inferencer.dataset_handler.preprocessor.get_image(image_path, test_config.image_size)[None]

        best_prompt, sim_matrix = self.inferencer.infer(model, prompts, image=image, use_com=use_com)

        prompt_components = self.inferencer.dataset_handler.preprocessor.get_components(prompts)
        best_prompt_components = self.inferencer.dataset_handler.preprocessor.get_components(best_prompt)[0]
        test_results = {
            "prompts": prompt_components,
            "image": image[0].tolist(),
            "confs": sim_matrix.tolist(),
            "best_prompt": best_prompt_components
        }
        image_name = image_path.split(os.sep)[-1].split(".")[0]
        test_path = os.path.join(test_config.test_save_path, f"test_{image_name}.json")
        save_json(test_results, test_path)

    def test(self, model, file=None, dir=None, use_com=False):
        if file is not None:
            self.perform_test(model, file, use_com=use_com)
        elif dir is not None:
            for file in os.listdir(dir):
                if file.endswith(".png"):
                    self.perform_test(model, os.path.join(dir, file), use_com=use_com)
