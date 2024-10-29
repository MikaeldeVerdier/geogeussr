import requests
import random
import io
import hashlib
import hmac
import base64
import urllib.parse as urlparse
import geopandas as gpd
from PIL import Image

class ImageFetcher:
    def __init__(self, shapefile_path, api_key, secret, base_url, return_size, location_tolerance):
        self.api_key = api_key
        self.secret = secret
        self.base_url = base_url
        self.return_size = return_size
        self.location_tolerance = location_tolerance

        self.geodf = gpd.read_file(shapefile_path)
        self.geodf.dissolve(by="GID_0")

        areadf = self.geodf.to_crs("EPSG:6933")  # For accruate area, an equal-area projection is used
        self.areas = areadf.geometry.area

        if self.geodf.crs != "EPSG:4326":
            self.geodf = self.geodf.to_crs("EPSG:4326")  # Lat, lng

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

    def generate_location(self):  # , lat_range=(-90, 90), lng_range=(-180, 180)):
        num_attempts = 0
        metadata = {"status": ""}
        while metadata["status"] != "OK":  # try a random point until a location within location_tolerance is accepted
            # lat = random.uniform(lat_range[0], lat_range[1])
            # lng = random.uniform(lng_range[0], lng_range[1])
            chosen_id = random.choices(self.geodf["UID"], weights=self.areas)[0]  # Could generate all locations that will be used in one
            chosen_entry = self.geodf[self.geodf["UID"] == chosen_id]
            points = chosen_entry.sample_points(1)

            lat = points.geometry.y._values[0]
            lng = points.geometry.x._values[0]

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
    
    def encode_signature(self, input_url):
        url = urlparse.urlparse(input_url)
        url_to_sign = f"{url.path}?{url.query}"

        decoded_key = base64.urlsafe_b64decode(self.secret)
        signature = hmac.new(decoded_key, str.encode(url_to_sign), hashlib.sha1)
        encoded_signature = base64.urlsafe_b64encode(signature.digest())

        return encoded_signature

    def query_image(self, location):
        params = {
            "key": self.api_key,
            "location": location,
            "size": f"{self.return_size[0]}x{self.return_size[1]}",
            "radius": 10  # should theoretically be able to be 0? 
        }
        if self.secret is not None:
            url = self.base_url + urlparse.urlencode(params)
            signature = self.encode_signature(url)
            params |= {"signature": signature}

        response = requests.get(self.base_url, params=params)
        image = response.content

        return image

    def generate_path(self, location, dir):
        path = f"{dir}/{location}_{self.return_size}.jpg"

        return path

    def save_image(self, byte_image, path):
        image = Image.open(io.BytesIO(byte_image))
        image.save(path)

    def generate_image(self, dir):
        location, metadata = self.generate_location()
        image = self.query_image(location)
        
        image_path = self.generate_path(location, dir)
        self.save_image(image, image_path)

        return image_path, location
