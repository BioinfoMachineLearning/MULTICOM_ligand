import logging
import os
import subprocess  # nosec

import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import register_custom_omegaconf_resolvers

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


@hydra.main(
    version_base="1.3",
    config_path="../../../configs/model/baselines",
    config_name="diffdock_inference.yaml",
)
def main(cfg: DictConfig):
    """Run inference using a trained DiffDock model checkpoint.

    :param cfg: Configuration dictionary from the hydra YAML file.
    """
    try:
        cmd = [
            cfg.python_exec_path,
            os.path.join(cfg.diffdock_exec_dir, "inference.py"),
            "--config",
            cfg.inference_config_path,
            "--protein_ligand_csv",
            cfg.input_csv_path,
            "--out_dir",
            cfg.output_dir,
            "--inference_steps",
            str(cfg.inference_steps),
            "--samples_per_complex",
            str(cfg.samples_per_complex),
            "--batch_size",
            str(cfg.batch_size),
            "--actual_steps",
            str(cfg.actual_steps),
            "--cuda_device_index",
            str(cfg.cuda_device_index),
            "--model_dir",
            cfg.model_dir,
            "--confidence_model_dir",
            cfg.confidence_model_dir,
        ]
        if cfg.no_final_step_noise:
            cmd.append("--no_final_step_noise")
        if cfg.skip_existing:
            cmd.append("--skip_existing")
        subprocess.run(cmd, check=True)  # nosec
    except Exception as e:
        raise e
    log.info(f"DiffDock inference for CSV input file `{cfg.input_csv_path}` complete.")


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
