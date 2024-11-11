from dataset_generator import DatasetGenerator

API_KEY = "API_KEY"  # str
SECRET = None  # str | None

SHAPEFILE_PATH = "dataset_generator/gadm_410.gpkg"  # str | None
DATASET_DIR = "datasets/dataset"  # str
AMOUNT_IMAGES = 100  # int
IMAGE_SIZE = (640, 640)  # (int int)
LOCATION_TOLERANCE = 5000  # int  # The maximum distance (in meters) from a random point allowed. Smaller -> more uniform distribution, longer generation time; Larger -> less uniform distribution, shorter generation time

GENERATION_METHOD = "scrape"  # "scrape" | "query"


if __name__ == "__main__":
    dataset_generator = DatasetGenerator(SHAPEFILE_PATH, API_KEY, LOCATION_TOLERANCE, secret=SECRET, load_dataset_path=DATASET_DIR)
    dataset_generator.generate_dataset(DATASET_DIR, AMOUNT_IMAGES, IMAGE_SIZE)
