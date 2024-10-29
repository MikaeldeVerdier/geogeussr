from dataset_generator import DatasetGenerator

API_KEY = "API_KEY"

DATASET_DIR = "./datasets/dataset10"
AMOUNT_IMAGES = 10
IMAGE_SIZE = (640, 640)
LOCATION_TOLERANCE = 5000  # The maximum distance (in meters) from a random point allowed. Smaller -> more uniform distribution, longer generation time; Larger -> less uniform distribution, shorter generation time


if __name__ == "__main__":
    dataset_generator = DatasetGenerator(API_KEY, LOCATION_TOLERANCE)
    dataset_generator.generate_dataset(DATASET_DIR, AMOUNT_IMAGES, IMAGE_SIZE)
