import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import Point

import annotator_config as anno_cfg
from files import load_annotations, save_annotations

class LocationAnnotator:
    def __init__(self, csv_path=anno_cfg.csv_path, gpkg_path=anno_cfg.gpkg_path):
        self.csv_path = csv_path
        self.gpkg_path = gpkg_path

        self.mercator_crs = "EPSG:3857"
        self.geodesic_crs = "EPSG:4326"

        self.prepare_df(csv_path)
        self.prepare_geodf(gpkg_path)

    def estimate_radius(self, population):
        return 1000 + 3000 * np.log10(population + 1)

    def prepare_df(self, csv_path):
        city_df = pd.read_csv(csv_path)
        city_df = gpd.GeoDataFrame(
            city_df, geometry=gpd.points_from_xy(city_df["lng"], city_df["lat"]), crs=self.geodesic_crs
        )

        city_df = city_df.to_crs(self.mercator_crs)
        city_df["estimated_radius"] = city_df["population"].apply(self.estimate_radius)
        city_df["geometry"] = city_df.geometry.buffer(city_df["estimated_radius"])
        city_df = city_df.to_crs(self.geodesic_crs)

        self.city_df = city_df

    def prepare_geodf(self, gpkg_path):
        self.geodf = gpd.read_file(gpkg_path)[["GID_0", "NAME_0", "NAME_1", "CONTINENT", "geometry"]]
        if self.geodf.crs != self.geodesic_crs:
            self.geodf = self.geodf.to_crs(self.geodesic_crs)

    def annotate_dataset(self, dataset_path=anno_cfg.dataset_path, save_path=anno_cfg.save_path):
        annotations = load_annotations(dataset_path)
        for anno in annotations:
            anno["location"]["coding"] = self.annotate(anno["location"]["lat"], anno["location"]["lng"])

        save_annotations(annotations, save_path)

    def annotate(self, lat, lng):
        point = Point(lng, lat)
        location = self.geodf[self.geodf.contains(point)]
        if location.empty:
            location_annotation = {
                "continent": None,
                "country": None,
                "province": None
            }
        else:
            loc_0 = location.iloc[0]
            location_annotation = {
                "continent": loc_0["CONTINENT"],
                "country": loc_0["NAME_0"],
                "province": loc_0["NAME_1"]  # could consider NAME_2 as well (county maybe (only available for some), would be good for us states maybe)
            }

        city_match = self.city_df[self.city_df.contains(point)]
        if not city_match.empty:
            city_name = city_match.iloc[0]["city"]
            location_annotation["city"] = city_name

        return location_annotation


if __name__ == "__main__":
    annotator = LocationAnnotator()
    annotator.annotate_dataset()
