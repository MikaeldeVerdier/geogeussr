import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import Point

import annotator_config as anno_cfg
from files import load_annotations, save_annotations

class LocationAnnotator:
    def __init__(self, csv_path=anno_cfg.csv_path, gpkg_path=anno_cfg.gpkg_path):
        self.mercator_crs = "EPSG:3857"
        self.geodesic_crs = "EPSG:4326"

        self.prepare_city_df(csv_path)
        self.prepare_gadm_df(gpkg_path)  # could sjoin these dataframes (like in the generator)

    def modify_city_df(self, city_df):
        city_df.loc[city_df["country"] == "Korea, South", "country"] = "South Korea"  # Name: Korea, South -> South Korea
        city_df.loc[city_df["country"] == "Korea, North", "country"] = "North Korea"  # Name: Korea, North -> North Korea
        city_df.loc[city_df["country"] == "Burma", "country"] = "Myanmar"  # Name: Burma -> Myanmar
        city_df.loc[city_df["country"] == "Islas Canarias", "country"] = "Canary Islands"  # Name: Islas Canarias -> Canary Islands
        # cities_gdf.loc[cities_gdf["country"] == "Falkland Islands (Islas Malvinas).", "country"] = "Falkland Islands"  # Name: Falkland Islands (Islas Malvinas) -> Falkland Islands
        city_df.loc[city_df["country"] == "Micronesia, Federated States of", "country"] = "Micronesia"  # Name: Micronesia, Federated States of -> Micronesia  # should it be "Federated States of Micronesia"?
        city_df.loc[city_df["country"] == "Peloponnese, Western Greece and", "country"] = "Western Greece and Peloponnese"  # Name: Peloponnese, Western Greece and -> Western Greece and Peloponnese
        city_df.loc[city_df["country"] == "Virgin Islands, British", "country"] = "British Virgin Islands"  # Name: Virgin Islands, British -> British Virgin Islands
        city_df.loc[city_df["country"] == "Bahamas, The", "country"] = "The Bahamas"  # Name: Bahamas, The -> The Bahamas
        city_df.loc[city_df["country"] == "Gambia, The", "country"] = "The Gambia"  # Name: Gambia, The -> The Gambia
        city_df.loc[city_df["country"] == "South Georgia And South Sandwich Islands", "country"] = "South Georgia and South Sandwich Islands"  # Name: South Georgia And South Sandwich Islands -> South Georgia and South Sandwich Islands  # (captilization, for some reason wrong)

        return city_df

    def estimate_radius(self, population):
        return 1000 + 3000 * np.log10(population + 1)

    def prepare_city_df(self, csv_path):
        city_df = pd.read_csv(csv_path)
        city_df = gpd.GeoDataFrame(
            city_df, geometry=gpd.points_from_xy(city_df["lng"], city_df["lat"]), crs=self.geodesic_crs
        )

        city_df = city_df.to_crs(self.mercator_crs)
        city_df["estimated_radius"] = city_df["population"].apply(self.estimate_radius)
        city_df["geometry"] = city_df.geometry.buffer(city_df["estimated_radius"])
        city_df = city_df.to_crs(self.geodesic_crs)

        self.city_df = self.modify_city_df(city_df)

    def modify_gadm_df(self, gadm_df):
        gadm_df.loc[gadm_df["CONTINENT"] == "Australia", "CONTINENT"] = "Oceania" # Continent: Australia -> Oceania
        # Could consider modifying countries in multiple continents ([Saint Helena, Ascension and Tristan da Cunha, Bonaire, Sint Eustatius, and Saba, Russia, Turkey, Spain, Azerbaijan, Kazakhstan])

        return gadm_df

    def prepare_gadm_df(self, gpkg_path):
        gadm_df = gpd.read_file(gpkg_path)[["GID_0", "NAME_0", "NAME_1", "CONTINENT", "geometry"]]
        if gadm_df.crs != self.geodesic_crs:
            gadm_df = gadm_df.to_crs(self.geodesic_crs)

        self.gadm_df = self.modify_gadm_df(gadm_df)
        self.mercator_gadm_df = self.gadm_df.to_crs(self.mercator_crs)

    def annotate_dataset(self, dataset_path=anno_cfg.dataset_path, save_path=anno_cfg.save_path):
        annotations = load_annotations(dataset_path)
        for anno in annotations:
            anno["location"]["coding"] = self.annotate(anno["location"]["lat"], anno["location"]["lng"])

        save_annotations(annotations, save_path)

    def annotate(self, lat, lng):
        point = Point(lng, lat)
        location = self.gadm_df[self.gadm_df.contains(point)]
        if location.empty:  # TODO: test this
            mercator_point = gpd.GeoSeries([point], crs=self.geodesic_crs).to_crs(self.mercator_crs).geometry[0]
            location = self.gadm_df.loc[self.mercator_gadm_df.distance(mercator_point).idxmin()]  # should it be forced to be in a continent/country/province, city isn't forced...
        else:
            location = location.iloc[0]

        location_annotation = {
            "continent": location["CONTINENT"],
            "country": location["NAME_0"],
            "province": location["NAME_1"]  # could consider NAME_2 as well (county maybe (only available for some), would be good for us states maybe)
        }

        city_match = self.city_df[self.city_df.contains(point)]
        if not city_match.empty:
            city_name = city_match.iloc[0]["city"]
            location_annotation["city"] = city_name

        return location_annotation


if __name__ == "__main__":
    annotator = LocationAnnotator()
    annotator.annotate_dataset()
