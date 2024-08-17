from __future__ import annotations

import os
import time
from typing import Optional

import hydra
import numpy as np
import rootutils
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("eval", eval)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import RankedLogger, rich_utils

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="calib_camera.yaml"
)
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)

    log.info(f"Instantiating cameras <{cfg.camera._target_}>")
    camera = hydra.utils.instantiate(cfg.camera)

    calib_dicts = camera.calibrate(visualize=cfg.visualize)
    calib_dicts = {calib_dict["serial"]: calib_dict for calib_dict in calib_dicts}
    if cfg.visualize:
        import matplotlib.pyplot as plt

        from src.utils.pointcloud_utils import construct_pcd, vis_pcds

        images, pcds = [], []
        for calib_dict in calib_dicts.values():
            images.append(calib_dict.pop("color_image"))
            pcds.append(
                construct_pcd(calib_dict.pop("points"), calib_dict.pop("colors"))
            )
        for image in images:
            plt.imshow(image)
            plt.show()
        vis_pcds(*pcds)

    timestamp = int(time.time() * 1000)
    os.makedirs(cfg.paths.data_dir, exist_ok=True)
    np.save(os.path.join(cfg.paths.data_dir, f"{timestamp}.npy"), calib_dicts)
    print(f"Calib data have been saved to {cfg.paths.data_dir}/{timestamp}.npy")
    camera.stop()


if __name__ == "__main__":
    main()
