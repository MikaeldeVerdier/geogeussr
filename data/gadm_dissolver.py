import geopandas as gpd

import dissolver_config as dis_cfg

class GADMDissolver:
    def __init__(self, save_path=dis_cfg.save_path, shapefile_path=dis_cfg.shapefile_path):
        self.save_path = save_path
        self.shapefile_path = shapefile_path

    def dissolve(self):
        gadm = gpd.read_file(self.shapefile_path)
        gadm_dissolved = gadm.dissolve(by="GID_0")
        gadm_dissolved.to_file(self.save_path)


if __name__ == "__main__":
    dissolver = GADMDissolver()
    dissolver.dissolve()
