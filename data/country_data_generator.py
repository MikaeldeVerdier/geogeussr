import pandas as pd
import geopandas as gpd

import os
import sys
parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_name)  # This sucks!

import generator_config as gen_cfg
from shared_components.files import save_json

class CountryDataGenerator:
    def __init__(self, save_path, csv_path, gpkg_path, top_k=30, is_predissolved=False):
        self.save_path = save_path

        self.prepare_df(csv_path, top_k)
        self.prepare_geodf(gpkg_path, is_predissolved)

    def prepare_df(self, csv_path, top_k):
        df = pd.read_csv(csv_path)
        df["population"] = pd.to_numeric(df["population"], errors="coerce")
        df = df.dropna(subset=["population"])

        df_sorted = df.sort_values(by=["iso3", "population"], ascending=[True, False])
        self.city_df = df_sorted.groupby("iso3").head(top_k)

    def prepare_geodf(self, gpkg_path, is_predissolved):
        self.geodf = gpd.read_file(gpkg_path)
        if not is_predissolved:
            self.geodf = self.geodf.dissolve(by="GID_0")
        if self.geodf.crs != "EPSG:3857":
            self.geodf = self.geodf.to_crs("EPSG:3857")

    def get_countries(self):
        return (self.geodf.index.tolist(), self.geodf["NAME_0"].values.tolist())

    def get_cities(self, country_df, country_name=None):
        if country_df.empty:
            print(f"Cities not found for {country_name}!")

            return []

        city_names = country_df["city"].values.tolist()
        city_lngs = country_df["lng"].values
        city_lats = country_df["lat"].values

        cities_data = []  # country_df.values[:, (0, 2, 3)]
        for name, lng, lat in zip(city_names, city_lngs, city_lats):
            city_data = {
                "name": name,
                "origin": [lng, lat]  # Longitude, Latitude
            }

            cities_data.append(city_data)

        return cities_data

    def get_bounding_box(self, country_geodf, country_name=None):
        if country_geodf.empty:
            print(f"Bounding box not found for {country_name}!")

            return [None, None, None, None]

        return country_geodf.geometry.to_crs("EPSG:4326").total_bounds.tolist()  # think this could just as well be 4326 from the beginning

    def get_origin(self, country_geodf, country_name=None):  # can be inferred from the bounding box (probably inaccurately though because of projection)
        if country_geodf.empty:
            print(f"Origin not found for {country_name}!")

            return [None, None]

        origin_point = country_geodf.geometry.centroid.to_crs("EPSG:4326")._values[0]

        return [origin_point.x, origin_point.y]  # Longitude, Latitude

    def get_bordering_countries(self, country_geodf, country_name=None):  # is only direct borders, not ocean borders
        if country_geodf.empty:
            print(f"Bordering countries not found for {country_name}!")

            return []
        
        bordering_countries = self.geodf[self.geodf.geometry.touches(country_geodf.geometry, align=True)]

        return bordering_countries.values

    def generate_data(self):
        country_codes, country_names = self.get_countries()

        country_objs = []
        for country_code, country_name in zip(country_codes, country_names):
            country_df = self.city_df[self.city_df["iso3"] == country_code]
            country_cities = self.get_cities(country_df, country_name=country_name)

            country_geodf = self.geodf[self.geodf.index == country_code]
            country_bounding_box = self.get_bounding_box(country_geodf, country_name=country_name)
            country_origin = self.get_origin(country_geodf, country_name=country_name)
            country_bordering_countries = self.get_bordering_countries(country_geodf, country_name=country_name)

            country_obj = {
                "code": country_code,
                "name": country_name,
                "cities": country_cities,
                "bounding_box": country_bounding_box,
                "origin": country_origin,
                "bordering_countries": country_bordering_countries
            }
            country_objs.append(country_obj)

        self.save_data(country_objs)

        return country_objs

    def save_data(self, data):
        save_json(data, self.save_path)


if __name__ == "__main__":
    country_data_generator = CountryDataGenerator(gen_cfg.save_path, gen_cfg.csv_path, gen_cfg.gpkg_path, top_k=gen_cfg.top_k, is_predissolved=gen_cfg.is_predissolved)
    country_data = country_data_generator.generate_data()
