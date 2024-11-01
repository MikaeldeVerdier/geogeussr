import os

from files import save_annotations
from image_fetcher import ImageFetcher

class DatasetGenerator:
    def __init__(self, shapefile_path, api_key, location_tolerance, secret=None, generator_method="scrape"):
        self.shapefile_path = shapefile_path
        self.api_key = api_key
        self.location_tolerance = location_tolerance
        self.secret = secret
        self.generator_method = generator_method

        self.streetview_base_url = "https://maps.googleapis.com/maps/api/streetview?"

    def create_folder(self, dir):
        if not os.path.exists(dir):
            os.mkdir(dir)

    def generate_dataset(self, dir, amount_images, image_size, save_ratio=0.1):
        self.create_folder(dir)

        fetcher = ImageFetcher(self.shapefile_path, self.api_key, self.secret, self.streetview_base_url, image_size, self.location_tolerance, self.generator_method)

        annotations = []
        for i in range(amount_images):
            image_path, location = fetcher.generate_image(dir)
            lat, lng = location.split(",")
            annotation = {
                "image_path": image_path,
                "location": {
                    "lat": float(lat),
                    "lng": float(lng)
                }
            }
            annotations.append(annotation)

            if not i % int(save_ratio * amount_images):  # will save the first iteration but that's okay!
                save_annotations(annotations, dir)

        save_annotations(annotations, dir)
