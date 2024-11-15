import os
import requests
import random
import io
import hashlib
import hmac
import base64
import urllib.parse as urlparse
import geopandas as gpd
from PIL import Image
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

class ImageFetcher:
    def __init__(self, shapefile_path, api_key, secret, base_url, return_size, location_tolerance, generator_method):
        self.api_key = api_key
        self.secret = secret
        self.base_url = base_url
        self.return_size = return_size
        self.location_tolerance = location_tolerance
        self.generator_method = generator_method

        self.use_shapefile = shapefile_path is not None

        if self.use_shapefile:
            self.geodf = gpd.read_file(shapefile_path)
            self.geodf = self.geodf.dissolve(by="GID_0")

            areadf = self.geodf.to_crs("EPSG:6933")  # For accurate area, an equal-area projection is used
            self.areas = areadf.geometry.area

            if self.geodf.crs != "EPSG:4326":
                self.geodf = self.geodf.to_crs("EPSG:4326")  # Lat, lng

        if generator_method == "scrape":
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument(f"--window-size={return_size[0]},{return_size[1] + int(278 / 2)}")  # DEVICE DEPENDANT: 278 is search bar height. Image will be 2x this for some reason.

            self.driver = webdriver.Chrome(options=chrome_options)  # Never quits but whatever

    def query_metadata(self, location, radius):
        metadata_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
        params = {
            "key": self.api_key,
            "location": location,
            "radius": radius
        }

        try:
            response = requests.get(metadata_url, params=params)
            metadata = response.json()
        except requests.ConnectTimeout:
            metadata = {"status": "TIMEOUT"}
        except requests.ConnectionError:
            metadata = {"status": "TIMEOUT"}

        return metadata

    def generate_location(self, lat_range=(-90, 90), lng_range=(-180, 180)):
        num_attempts = 0
        timed_out_attempts = 0
        metadata = {"status": ""}
        while metadata["status"] != "OK":  # try a random point until a location within location_tolerance is accepted
            if self.use_shapefile:
                chosen_id = random.choices(self.geodf["UID"]._values, weights=self.areas)[0]  # Could generate all locations that will be used in one
                chosen_entry = self.geodf[self.geodf["UID"] == chosen_id]
                points = chosen_entry.sample_points(1)

                lat = points.geometry.y._values[0]
                lng = points.geometry.x._values[0]
            else:
                lat = random.uniform(lat_range[0], lat_range[1])
                lng = random.uniform(lng_range[0], lng_range[1])

            initial_location = f"{lat},{lng}"
            # print(initial_location)
            metadata = self.query_metadata(initial_location, self.location_tolerance)
            if metadata["status"] == "TIMEOUT":
                timed_out_attempts += 1
                sleep(1.5 ** timed_out_attempts)
                print(f"Timed out, sleeping for {1.5 ** timed_out_attempts}")
            else:
                timed_out_attempts = 0

            num_attempts += 1

        country_name = chosen_entry.index.values[0]

        print()
        print(f"In {num_attempts} attempts, a valid location was found in {country_name}:")
        print(f"Initial location: {initial_location}")
        exact_location = f"{metadata['location']['lat']},{metadata['location']['lng']}"
        print(f"Exact location: {exact_location}")

        heading = random.uniform(0, 360)
        print(f"Heading: {heading}")

        return country_name, exact_location, metadata, heading

    def generate_path(self, location, dir, heading):
        file_name = f"{location}ll_{heading}h_{self.return_size}s.png"
        file_path = os.path.join(dir, file_name)

        return file_name, file_path

    def scrape_image(self, location, pano_id, image_path, heading=0):
        url = f"https://www.google.com/maps/@{location},3a,90y,{heading}h,90t/data=!3m6!1e1!3m4!1s{pano_id}!2e0!7i16384!8i8192"  #a: idk, y: fov, h: heading, t: idk
        print(f"URL to scrape is: {url}")

        self.driver.get(url)

        wait = WebDriverWait(self.driver, 10)  # 10s timeout

        try:
            if self.driver.find_elements(By.CSS_SELECTOR, "button.XWZjwc"):
                click_injection = "document.querySelectorAll('button.XWZjwc')[1].click()"  # Why is this even needed?
                self.driver.execute_script(click_injection)

            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "canvas")))
            js_injection = """
                canvas = document.querySelector('canvas');
                context = canvas.getContext('webgl');
                if (context == null) {
                    context = canvas.getContext('webgl2');
                }
                context.drawArrays = function() { }
            """
            self.driver.execute_script(js_injection)

            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#minimap div div:nth-child(2)")))
            css_injection = "document.styleSheets[0].insertRule('.app-viewcard-strip, .scene-footer, .id-omnibox-container, .hdeJwf, #titlecard, #watermark, #image-header { display: none !important; }')"  # ; document.querySelector('.scene-footer').style.display = 'none'"  # Don't know why scene-footer needs to be seperate
            self.driver.execute_script(css_injection)

            self.driver.save_screenshot(image_path)
        except TimeoutException:
            print("Skipping location!")

    def encode_signature(self, input_url):
        url = urlparse.urlparse(input_url)
        url_to_sign = f"{url.path}?{url.query}"

        decoded_key = base64.urlsafe_b64decode(self.secret)
        signature = hmac.new(decoded_key, str.encode(url_to_sign), hashlib.sha1)
        encoded_signature = base64.urlsafe_b64encode(signature.digest())

        return encoded_signature

    def query_image(self, location, heading=0):
        params = {
            "key": self.api_key,
            "location": location,
            "size": f"{self.return_size[0]}x{self.return_size[1]}",
            "radius": 10,  # should theoretically be able to be 0?
            "heading": heading
        }
        if self.secret is not None:
            url = self.base_url + urlparse.urlencode(params)
            signature = self.encode_signature(url)
            params |= {"signature": signature}

        response = requests.get(self.base_url, params=params)
        image = response.content

        return image

    def save_image(self, byte_image, path):
        image = Image.open(io.BytesIO(byte_image))
        image.save(path)

    def generate_image(self, dir):
        country, location, metadata, heading = self.generate_location()
        image_name, image_path = self.generate_path(location, dir, heading)

        if self.generator_method == "scrape":
            self.scrape_image(location, metadata["pano_id"], image_path, heading=heading)
        elif self.generator_method == "query":
            image = self.query_image(location, heading=heading)
            self.save_image(image, image_path)

        return image_name, country, location
