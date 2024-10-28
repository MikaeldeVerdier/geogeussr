import requests
import random

class ImageFetcher:
    def __init__(self, api_key, base_url, return_size, location_tolerance):
        self.api_key = api_key
        self.base_url = base_url
        self.return_size = return_size
        self.location_tolerance = location_tolerance

    def query_metadata(self, location, radius):
        metadata_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
        params = {
            "key": self.api_key,
            "location": location,
            "radius": radius
        }

        response = requests.get(metadata_url, params=params)
        metadata = response.json()

        return metadata

    def generate_location(self, lat_range=(-90, 90), lng_range=(-180, 180)):  # POSSIBLE BIAS WITH THIS APPROACH: Initial location will often be in the middle of the ocean, meaning this will be more likely to generate beach locations.
        num_attempts = 0
        metadata = {"status": ""}
        while metadata["status"] != "OK":  # try a random point until a location within location_tolerance is accepted
            lat = random.uniform(lat_range[0], lat_range[1])
            lng = random.uniform(lng_range[0], lng_range[1])

            initial_location = f"{lat},{lng}"
            # print(initial_location)
            metadata = self.query_metadata(initial_location, self.location_tolerance)

            num_attempts += 1

        print()
        print(f"In {num_attempts} attempts, a valid location was found:")
        print(f"Initial location: {initial_location}")
        exact_location = f"{metadata['location']['lat']},{metadata['location']['lng']}"
        print(f"Exact location: {exact_location}")

        return exact_location, metadata

    def query_image(self, location):
        params = {
            "key": self.api_key,
            "location": location,
            "size": f"{self.return_size[0]}x{self.return_size[1]}",
            "radius": 10  # should theoretically be able to be 0? 
        }

        response = requests.get(self.base_url, params=params)
        image = response.content

        return image

    def generate_path(self, location, dir):
        path = f"{dir}/{location}_{self.return_size}.jpg"

        return path

    def save_image(self, image, path):
        with open(path, "wb") as file:
            file.write(image)

    def generate_image(self, dir):
        location, metadata = self.generate_location()
        image = self.query_image(location)
        
        image_path = self.generate_path(location, dir)
        self.save_image(image, image_path)

        return image_path, location
