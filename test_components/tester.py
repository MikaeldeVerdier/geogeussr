import os

import test_components.test_config as test_config
from shared_components.inferencer import Inferencer

class Tester:
    def __init__(self, processor_method="Geo"):
        self.inferencer = Inferencer(test_config.image_size, test_config.refinement_steps, "", test_config.shapefile_path, processor_method=processor_method)

        if not os.path.exists(test_config.inference_save_path):
            os.mkdir(test_config.inference_save_path)

    def perform_test(self, model, image_path, use_com=False):
        prompts = self.inferencer.dataset_handler.preprocessor.get_prompts()
        image = self.inferencer.dataset_handler.preprocessor.get_image(image_path, test_config.image_size)[None]

        best_prompt, sim_matrix = self.inferencer.infer(model, prompts, image=image, use_com=use_com)
        inference_name = os.path.join(test_config.inference_save_path, f"inference_{os.path.basename(image_path).replace('.png', '')}.json")
        self.inferencer.save_inference(prompts, image[0], best_prompt[0], sim_matrix[0], inference_name)

    def test(self, model, use_com=False):
        if test_config.data_file is not None and os.path.exists(test_config.data_file):
            self.perform_test(model, test_config.data_file, use_com=use_com)
        elif test_config.data_dir is not None:
            for file in os.listdir(test_config.data_dir):
                if file.endswith(".png"):
                    print(f"Testing: {file}")

                    self.perform_test(model, os.path.join(test_config.data_dir, file), use_com=use_com)  # nested!
