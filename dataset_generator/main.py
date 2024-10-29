from dataset_generator import DatasetGenerator

API_KEY = "API_KEY"
SECRET = "SECRET"

SHAPEFILE_PATH = "dataset_generator/gadm_410.gpkg"
DATASET_DIR = "datasets/dataset"
AMOUNT_IMAGES = 10
IMAGE_SIZE = (640, 640)
LOCATION_TOLERANCE = 5000  # The maximum distance (in meters) from a random point allowed. Smaller -> more uniform distribution, longer generation time; Larger -> less uniform distribution, shorter generation time


if __name__ == "__main__":
    dataset_generator = DatasetGenerator(SHAPEFILE_PATH, API_KEY, LOCATION_TOLERANCE, secret=SECRET)
    dataset_generator.generate_dataset(DATASET_DIR, AMOUNT_IMAGES, IMAGE_SIZE)
