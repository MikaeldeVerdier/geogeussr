from clusterer import Clusterer

import configs.tool_configs.cluster_config as clu_cfg
from visualizer.visualize_gmm import visualize_gmm

if __name__ == "__main__":
    clusterer = Clusterer(clu_cfg.NUM_CLUSTERS)

    clusterer.cluster_location(clu_cfg.DATASET_PATH)
    clusterer.save_gmm(clu_cfg.SAVE_PATH)

    # visualize_gmm(clusterer.gm, clusterer.coords, "test.png", "dissolved_gadm.gpkg")
