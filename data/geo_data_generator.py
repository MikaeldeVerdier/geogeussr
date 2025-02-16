import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import Point

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

        self.city_df = city_df.to_crs(self.mercator_crs)

    def prepare_geodf(self, gpkg_path):
        self.geodf = gpd.read_file(gpkg_path)[["GID_0", "NAME_0", "NAME_1", "CONTINENT", "geometry"]]
        if self.geodf.crs != self.mercator_crs:
            self.geodf = self.geodf.to_crs(self.mercator_crs)

    def spherically_averaged_centroid(self, geom):
        if isinstance(geom, Point):
            return geom

        lon_lat = np.array([(p.x, p.y) for p in geom.geoms])

        lon = np.radians(lon_lat[:, 0])
        lat = np.radians(lon_lat[:, 1])

        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)

        x_mean = np.mean(x)
        y_mean = np.mean(y)
        z_mean = np.mean(z)

        lon_c = np.arctan2(y_mean, x_mean)
        hyp = np.sqrt(x_mean ** 2 + y_mean ** 2)
        lat_c = np.arctan2(z_mean, hyp)

        new_point = Point(np.degrees(lon_c), np.degrees(lat_c))

        return new_point

    def generate_geo_data(self, save_path=gen_cfg.save_path, top_k=gen_cfg.top_k):
        cities_gdf = gpd.sjoin_nearest(self.city_df, self.geodf, how="left", distance_col="dist")
        cities_gdf = cities_gdf[["city", "country", "population", "NAME_1", "CONTINENT", "geometry"]]

        cities_gdf = cities_gdf.to_crs(self.geodesic_crs)

        # Some modifications and improvements for exceptions and bad data
        cities_gdf.loc[cities_gdf["country"] == "Korea, South", "country"] = "South Korea"  # Name: Korea, South -> South Korea
        cities_gdf.loc[cities_gdf["country"] == "Korea, North", "country"] = "North Korea"  # Name: Korea, North -> North Korea
        cities_gdf.loc[cities_gdf["country"] == "Burma", "country"] = "Myanmar"  # Name: Burma -> Myanmar
        cities_gdf.loc[cities_gdf["country"] == "Islas Canarias", "country"] = "Canary Islands"  # Name: Islas Canarias -> Canary Islands
        # cities_gdf.loc[cities_gdf["country"] == "Falkland Islands (Islas Malvinas).", "country"] = "Falkland Islands"  # Name: Falkland Islands (Islas Malvinas) -> Falkland Islands
        cities_gdf.loc[cities_gdf["country"] == "Micronesia, Federated States of", "country"] = "Micronesia"  # Name: Micronesia, Federated States of -> Micronesia
        cities_gdf.loc[cities_gdf["country"] == "Virgin Islands, British", "country"] = "British Virgin Islands"  # Name: Virgin Islands, British -> British Virgin Islands
        cities_gdf.loc[cities_gdf["country"] == "Bahamas, The", "country"] = "The Bahamas"  # Name: Bahamas, The -> The Bahamas
        cities_gdf.loc[cities_gdf["country"] == "South Georgia And South Sandwich Islands", "country"] = "South Georgia and South Sandwich Islands"  # Name: South Georgia And South Sandwich Islands -> South Georgia and South Sandwich Islands  # (captilization, for some reason wrong)
        cities_gdf.loc[cities_gdf["country"] == "Australia", "CONTINENT"] = "Oceania"  # Continent: Australia -> Oceania
        # Could consider modifying countries in multiple continents ([Saint Helena, Ascension and Tristan da Cunha, Bonaire, Sint Eustatius, and Saba, Russia, Turkey, Spain, Azerbaijan, Kazakhstan])

        cities_gdf = cities_gdf.sort_values(by="population", ascending=False)
        cities_gdf = cities_gdf.groupby("country")
        cities_gdf = cities_gdf.head(top_k)

        continent_centroids = cities_gdf.dissolve(by="CONTINENT").geometry.apply(self.spherically_averaged_centroid)
        country_centroids = cities_gdf.dissolve(by="country").geometry.apply(self.spherically_averaged_centroid)
        province_centroids = cities_gdf.dissolve(by="NAME_1").geometry.apply(self.spherically_averaged_centroid)

        result = {}
        for _, row in cities_gdf.iterrows():
            continent = row["CONTINENT"]
            country = row["country"]
            province = row["NAME_1"]
            city = row["city"]

            continent_origin = (continent_centroids[continent].x, continent_centroids[continent].y) if continent in continent_centroids.index else None
            country_origin = (country_centroids[country].x, country_centroids[country].y) if country in country_centroids.index else None
            province_origin = (province_centroids[province].x, province_centroids[province].y) if province in province_centroids.index else None
            city_origin = (row.geometry.centroid.x, row.geometry.centroid.y)  # hasattr(row, "geometry") ?

            result.setdefault(continent, {}).setdefault("countries", {}).setdefault(country, {}).setdefault("provinces", {}).setdefault(province, {}).setdefault("cities", []).append({"name": city, "origin": city_origin})
            result[continent]["origin"] = continent_origin
            result[continent]["countries"][country]["origin"] = country_origin
            result[continent]["countries"][country]["provinces"][province]["origin"] = province_origin
            # result[continent]["countries"][country]["provinces"][province]["cities"][city]["origin"] = city_origin

        save_json(result, save_path)


if __name__ == "__main__":
    geo_data_generator = GeoDataGenerator()
    geo_data = geo_data_generator.generate_geo_data()
