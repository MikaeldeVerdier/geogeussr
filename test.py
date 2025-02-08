from model.geo_clip.geo_clip_model import GeoCLIP
from model.street_clip.street_clip_model import StreetCLIP
from model.clip_clip.clip_clip_model import ClipCLIP
from test_components.tester import Tester

if __name__ == "__main__":
    # Test a GeoCLIP model
    geo_clip = GeoCLIP(image_encoder="resnet")
    tester = Tester()

    tester.test(geo_clip, dir="test_imgs", use_com=True)


    # # Test a StreetCLIP model with new prompt format
    # street_clip = StreetCLIP()
    # tester = Tester(processor_method="Street")

    # tester.test(street_clip, "test.png")


    # # Test a StreetCLIP model with original prompt format
    # street_clip = StreetCLIP()
    # tester = Tester(processor_method="StreetOG")

    # tester.test(street_clip, "test.png")


    # # Test a ClipCLIP model with new prompt format
    # clip_clip = ClipCLIP(load_path="GeoCLIP/model")
    # tester = Tester(processor_method="Clip")

    # tester.test(clip_clip, "test.png")


    # # Test a ClipCLIP model with original prompt format
    # clip_clip = ClipCLIP()
    # tester = Tester(processor_method="ClipOG")

    # tester.test(clip_clip, "test.png")

