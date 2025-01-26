import glob
import logging
import os
import subprocess  # nosec

import hydra
import pandas as pd
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import register_custom_omegaconf_resolvers

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


@hydra.main(
    version_base="1.3",
    config_path="../../../configs/model/baselines",
    config_name="neuralplexer_inference.yaml",
)
def main(cfg: DictConfig):
    """Run inference using a trained NeuralPLexer model checkpoint.

    :param cfg: Configuration dictionary from the hydra YAML file.
    """
    os.makedirs(cfg.out_path, exist_ok=True)
    for _, row in pd.read_csv(cfg.input_csv_path).iterrows():
        out_dir = os.path.join(cfg.out_path, row.id)
        os.makedirs(out_dir, exist_ok=True)
        out_protein_filepaths = list(glob.glob(os.path.join(out_dir, "prot_all.pdb")))
        out_ligand_filepaths = list(glob.glob(os.path.join(out_dir, "lig_all.sdf")))
        if cfg.skip_existing and out_protein_filepaths and out_ligand_filepaths:
            log.info(f"Skipping inference for completed complex `{row.id}`.")
            continue
        try:
            subprocess_args = [
                str(cfg.python_exec_path),
                os.path.join(str(cfg.neuralplexer_exec_dir), "neuralplexer", "inference.py"),
                "--task",
                str(cfg.task),
                "--sample-id",
                str(cfg.sample_id),
                "--template-id",
                str(cfg.template_id),
                "--cuda-device-index",
                str(cfg.cuda_device_index),
                "--model-checkpoint",
                str(cfg.model_checkpoint),
                "--input-ligand",
                str(row.input_ligand),
                "--input-receptor",
                str(row.input_receptor),
                "--input-template",
                str(row.input_template),
                "--out-path",
                str(out_dir),
                "--n-samples",
                str(cfg.n_samples),
                "--chunk-size",
                str(cfg.chunk_size),
                "--num-steps",
                str(cfg.num_steps),
                "--sampler",
                str(cfg.sampler),
                "--start-time",
                str(cfg.start_time),
                "--max-chain-encoding-k",
                str(cfg.max_chain_encoding_k),
            ]
            if cfg.latent_model:
                subprocess_args.extend(["--latent-model", cfg.latent_model])
            if cfg.exact_prior:
                subprocess_args.extend(["--exact-prior"])
            if cfg.discard_ligand:
                subprocess_args.extend(["--discard-ligand"])
            if cfg.discard_sdf_coords:
                subprocess_args.extend(["--discard-sdf-coords"])
            if cfg.detect_covalent:
                subprocess_args.extend(["--detect-covalent"])
            if cfg.use_template:
                subprocess_args.extend(["--use-template"])
            if cfg.separate_pdb:
                subprocess_args.extend(["--separate-pdb"])
            if cfg.rank_outputs_by_confidence:
                subprocess_args.extend(["--rank-outputs-by-confidence"])
            if cfg.csv_path:
                subprocess_args.extend(["--csv-path", cfg.csv_path])
            subprocess.run(subprocess_args, check=True)  # nosec
        except Exception as e:
            log.error(
                f"NeuralPLexer inference for complex `{row.id}` failed with error: {e}. Skipping..."
            )
            continue
        log.info(f"NeuralPLexer inference for complex `{row.id}` complete.")


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
