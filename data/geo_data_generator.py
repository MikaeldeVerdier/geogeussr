import numpy as np
import geopandas as gpd

import generator_config as gen_cfg
from data_handler import DataHandler
from files import save_json

class GeoDataGenerator:
    def __init__(self, csv_path=gen_cfg.csv_path, gpkg_path=gen_cfg.gpkg_path):
        self.data_handler = DataHandler(csv_path, gpkg_path, buffer_cities=False)

    def head_by_area(self, gdf):
        region = gdf["GID_0"].iloc[0]
        # region_geom = self.gadm_df[self.gadm_df["GID_0"] == region].dissolve().geometry.iloc[0]
        area = self.data_handler.mercator_gadm_df[self.data_handler.mercator_gadm_df["GID_0"] == region].geometry.area.sum()
        scaled_area = area * 1e-9

        k = int(np.ceil(2 * scaled_area ** 0.4))

        return gdf.head(k)

    def generate_geo_data(self, save_path=gen_cfg.save_path):
        joined_gdf = gpd.sjoin_nearest(self.data_handler.mercator_city_df, self.data_handler.mercator_gadm_df, how="left", distance_col="dist")
        joined_gdf = joined_gdf[["GID_0", "city", "country", "population", "NAME_1", "CONTINENT", "geometry"]]

        joined_gdf = joined_gdf.to_crs(self.data_handler.geodesic_crs)

        joined_gdf = joined_gdf.sort_values(by="population", ascending=False)
        joined_gdf = joined_gdf.groupby("country")
        joined_gdf = joined_gdf.apply(self.head_by_area, include_groups=False).reset_index()

        continent_centroids = joined_gdf.dissolve(by="CONTINENT").geometry.apply(self.data_handler.spherically_averaged_centroid)
        country_centroids = joined_gdf.dissolve(by="country").geometry.apply(self.data_handler.spherically_averaged_centroid)
        province_centroids = joined_gdf.dissolve(by="NAME_1").geometry.apply(self.data_handler.spherically_averaged_centroid)  # could do something when these are empty

        result = {}
        for _, row in joined_gdf.iterrows():
            continent = row["CONTINENT"]
            country = row["country"]
            province = row["NAME_1"]  # TODO: For some reason, some have provinces that are empty strings. These should probably be ignored, look into it.
            city = row["city"]

            continent_origin = (continent_centroids[continent].x, continent_centroids[continent].y) if continent in continent_centroids.index else None
            country_origin = (country_centroids[country].x, country_centroids[country].y) if country in country_centroids.index else None
            province_origin = (province_centroids[province].x, province_centroids[province].y) if province in province_centroids.index else None
            city_origin = (row.geometry.centroid.x, row.geometry.centroid.y)  # hasattr(row, "geometry") ?

            """  Opted out of this because key order become inconsistent and unpreferred
            result.setdefault(continent, {
                "origin": continent_origin
            }).setdefault("countries", {}).setdefault(country, {
                "origin": country_origin
            }).setdefault("provinces", {}).setdefault(province, {
                "origin": province_origin
            }).setdefault("cities", []).append({
                "name": city,
                "origin": city_origin
            })
            """
            result.setdefault(continent, {}).setdefault("countries", {}).setdefault(country, {}).setdefault("provinces", {}).setdefault(province, {}).setdefault("cities", []).append({
                "name": city,
                "origin": city_origin
            })
            result[continent]["origin"] = continent_origin
            result[continent]["countries"][country]["origin"] = country_origin
            result[continent]["countries"][country]["provinces"][province]["origin"] = province_origin
            # result[continent]["countries"][country]["provinces"][province]["cities"][city]["origin"] = city_origin

        save_json(result, save_path)


if __name__ == "__main__":
    geo_data_generator = GeoDataGenerator()
    geo_data = geo_data_generator.generate_geo_data()
