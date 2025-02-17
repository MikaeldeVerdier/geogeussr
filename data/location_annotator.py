import annotator_config as anno_cfg
from data_handler import DataHandler
from files import load_annotations, save_annotations

class LocationAnnotator:
    def __init__(self, csv_path=anno_cfg.csv_path, gpkg_path=anno_cfg.gpkg_path):
        self.data_handler = DataHandler(csv_path, gpkg_path)

    def annotate_dataset(self, dataset_path=anno_cfg.dataset_path, save_path=anno_cfg.save_path):
        annotations = load_annotations(dataset_path)
        for anno in annotations:
            coding = self.data_handler.annotate_point(anno["location"]["lat"], anno["location"]["lng"], force_point=True)
            anno["location"]["coding"] = coding

        save_annotations(annotations, save_path)


if __name__ == "__main__":
    annotator = LocationAnnotator()
    annotator.annotate_dataset()
