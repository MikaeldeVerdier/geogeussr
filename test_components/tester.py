import os

import test_components.test_config as test_cfg
from shared_components.inferencer import Inferencer
from shared_components.files import load_json

class Tester:
    def __init__(self, processor_method="Geo"):
        self.inferencer = Inferencer(test_cfg.image_size, test_cfg.refinement_steps, "", test_cfg.shapefile_path, processor_method=processor_method)

        if not os.path.exists(test_cfg.inference_save_path):
            os.mkdir(test_cfg.inference_save_path)

    def perform_test(self, model, image_path, use_com=False):
        regions = load_json(test_cfg.regions_path)
        prompts, prompt_components = self.inferencer.dataset_handler.preprocessor.get_prompts(regions)
        image = self.inferencer.dataset_handler.preprocessor.get_image(image_path)[None]

        best_prompt, best_prompt_components, sim_matrix = self.inferencer.infer(model, prompts, prompt_components, image=image, use_com=use_com)

        file_name = os.path.basename(image_path).replace(".png", "")
        inference_name = os.path.join(test_cfg.inference_save_path, f"inference_{file_name}.json")
        self.inferencer.save_inference(prompt_components, image[0], best_prompt_components[0], sim_matrix[0], inference_name)

    def test(self, model, use_com=False):
        if test_cfg.data_file is not None and os.path.exists(test_cfg.data_file):
            self.perform_test(model, test_cfg.data_file, use_com=use_com)
        elif test_cfg.data_dir is not None:
            for file in os.listdir(test_cfg.data_dir):
                if file.endswith(".png"):
                    print(f"Testing: {file}")

                    self.perform_test(model, os.path.join(test_cfg.data_dir, file), use_com=use_com)  # nested!
