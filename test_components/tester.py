import test_components.test_config as test_config
from shared_components.inferencer import Inferencer

class Tester:
    def __init__(self, processor_method="Geo"):
        self.inferencer = Inferencer(test_config.image_size, test_config.tokens_len, test_config.refinement_base, test_config.refinement_steps, "", test_config.regions, test_config.shapefile_path, processor_method=processor_method)

    def test(self, model, image_path, use_com=False):
        used_prompts = self.inferencer.dataset_handler.preprocessor.get_prompts()
        image = self.inferencer.dataset_handler.preprocessor.get_image(image_path, test_config.image_size)[None]

        self.inferencer.infer(model, used_prompts, image=image, use_com=use_com)
