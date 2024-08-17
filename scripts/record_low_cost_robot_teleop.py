from __future__ import annotations

from typing import Optional

import hydra
import rootutils
from omegaconf import DictConfig, OmegaConf
from pynput import keyboard

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

from src.utils import RankedLogger, clear_shared_memory, extras

log = RankedLogger(__name__, rank_zero_only=True)

has_stop = False


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="record_low_cost_robot_teleop.yaml",
)
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    extras(cfg)

    log.info(f"Instantiating cameras <{cfg.camera._target_}>")
    camera = hydra.utils.instantiate(cfg.camera)

    log.info(f"Instantiating tele operator <{cfg.teleop._target_}>")
    tele_operator = hydra.utils.instantiate(cfg.teleop)

    log.info(f"Instantiating collector <{cfg.collector._target_}>")
    collector = hydra.utils.instantiate(cfg.collector)

    def _on_press(key):
        global has_stop
        try:
            if key.char == "q":
                if not has_stop:
                    collector.stop()
                    camera.stop()
                    tele_operator.stop()
                    has_stop = True
        except AttributeError:
            pass

    def _on_release(key):
        pass

    try:
        listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)

        camera.start()
        tele_operator.start()
        collector.start()
        listener.start()

        # breakpoint()
        # tele_operator.start2()

        while not has_stop:
            pass
        listener.stop()
        clear_shared_memory("record_low_cost_robot_teleop")
    except:
        collector.stop()
        camera.stop()
        tele_operator.stop()
        listener.stop()
        clear_shared_memory("record_low_cost_robot_teleop")
        raise


if __name__ == "__main__":
    clear_shared_memory(None)
    main()
