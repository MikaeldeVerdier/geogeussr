import pandas as pd
import geopandas as gpd

import generator_config as gen_cfg
from files import save_json

class GeoDataGenerator:
    def __init__(self, csv_path=gen_cfg.csv_path, gpkg_path=gen_cfg.gpkg_path, top_k=gen_cfg.top_k):
        self.mercator_crs = "EPSG:3857"
        self.geodesic_crs = "EPSG:4326"

        self.prepare_df(csv_path)
        self.prepare_geodf(gpkg_path)

    def prepare_df(self, csv_path):
        city_df = pd.read_csv(csv_path)
        city_df = gpd.GeoDataFrame(
            city_df, geometry=gpd.points_from_xy(city_df["lng"], city_df["lat"]), crs=self.geodesic_crs
        )

        self.city_df = city_df

    def prepare_geodf(self, gpkg_path):
        self.geodf = gpd.read_file(gpkg_path)[["GID_0", "NAME_0", "NAME_1", "CONTINENT", "geometry"]]
        if self.geodf.crs != self.geodesic_crs:
            self.geodf = self.geodf.to_crs(self.geodesic_crs)

    def generate_geo_data(self, save_path=gen_cfg.save_path, top_k=gen_cfg.top_k):
        cities_gdf = gpd.sjoin(self.city_df, self.geodf, how="left", predicate="within")
        cities_gdf = cities_gdf[["city", "country", "population", "NAME_1", "CONTINENT"]]

        cities_gdf = cities_gdf.sort_values(by="population", ascending=False)
        cities_gdf = cities_gdf.groupby("country")
        cities_gdf = cities_gdf.head(top_k)
    
        result = {}
        for _, row in cities_gdf.iterrows():
            continent = row["CONTINENT"]
            country = row["country"]
            province = row["NAME_1"]
            city = row["city"]
            
            result.setdefault(continent, {}).setdefault(country, {}).setdefault(province, []).append(city)

        save_json(result, save_path)


if __name__ == "__main__":
    geo_data_generator = GeoDataGenerator()
    geo_data = geo_data_generator.generate_geo_data()
