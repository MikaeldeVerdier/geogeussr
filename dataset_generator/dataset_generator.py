import os
import json

from image_fetcher import ImageFetcher

class DatasetGenerator:
    def __init__(self, shapefile_path, api_key, location_tolerance, secret=None):
        self.shapefile_path = shapefile_path
        self.api_key = api_key
        self.location_tolerance = location_tolerance
        self.secret = secret

        self.streetview_base_url = "https://maps.googleapis.com/maps/api/streetview?"

    def create_folder(self, dir):
        if not os.path.exists(dir):
            os.mkdir(dir)

    def save_annotations(self, annotations, dir):
        with open(os.path.join(dir, "_annotations.json"), "w") as json_file:
            json.dump(annotations, json_file)

    def generate_dataset(self, dir, amount_images, image_size):
        self.create_folder(dir)

        fetcher = ImageFetcher(self.shapefile_path, self.api_key, self.secret, self.streetview_base_url, image_size, self.location_tolerance)

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
    
        self.save_annotations(annotations, dir)
