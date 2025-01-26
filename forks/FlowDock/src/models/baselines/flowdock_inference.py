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
logger = logging.getLogger(__name__)


@hydra.main(
    version_base="1.3",
    config_path="../../../configs/model/baselines",
    config_name="flowdock_inference.yaml",
)
def main(cfg: DictConfig):
    """Run inference using a trained FlowDock model checkpoint.

    :param cfg: Configuration dictionary from the hydra YAML file.
    """
    os.makedirs(cfg.out_path, exist_ok=True)
    if cfg.input_ligand is not None and cfg.input_receptor is not None:
        out_dir = os.path.join(cfg.out_path, cfg.sample_id)
        os.makedirs(out_dir, exist_ok=True)
        try:
            subprocess_args = [
                str(cfg.python_exec_path),
                os.path.join(str(cfg.flowdock_exec_dir), "src", "sample.py"),
                f"sampling_task={cfg.sampling_task}",
                f"sample_id='{cfg.sample_id if cfg.sample_id is not None else 0}'",
                f"input_ligand='{cfg.input_ligand}'",
                f"input_receptor='{cfg.input_receptor}'",
                f"trainer={'gpu' if cfg.cuda_device_index is not None else 'default'}",
                f"{f'trainer.devices=[{int(cfg.cuda_device_index)}]' if cfg.cuda_device_index is not None else ''}",
                f"ckpt_path={cfg.model_checkpoint}",
                f"out_path={out_dir}",
                f"n_samples={int(cfg.n_samples)}",
                f"chunk_size={int(cfg.chunk_size)}",
                f"num_steps={int(cfg.num_steps)}",
                f"sampler={cfg.sampler}",
                f"sampler_eta={cfg.sampler_eta}",
                f"start_time={str(cfg.start_time)}",
                f"max_chain_encoding_k={int(cfg.max_chain_encoding_k)}",
                f"plddt_ranking_type={cfg.plddt_ranking_type}",
            ]
            if cfg.input_template:
                subprocess_args.append(f"input_template='{cfg.input_template}'")
            if cfg.latent_model:
                subprocess_args.append(f"latent_model={cfg.latent_model}")
            if cfg.exact_prior:
                subprocess_args.append(f"exact_prior={cfg.exact_prior}")
            if cfg.discard_ligand:
                subprocess_args.append(f"discard_ligand={cfg.discard_ligand}")
            if cfg.discard_sdf_coords:
                subprocess_args.append(f"discard_sdf_coords={cfg.discard_sdf_coords}")
            if cfg.detect_covalent:
                subprocess_args.append(f"detect_covalent={cfg.detect_covalent}")
            if cfg.use_template:
                subprocess_args.append(f"use_template={cfg.use_template}")
            if cfg.separate_pdb:
                subprocess_args.append(f"separate_pdb={cfg.separate_pdb}")
            if cfg.rank_outputs_by_confidence:
                subprocess_args.append(
                    f"rank_outputs_by_confidence={cfg.rank_outputs_by_confidence}"
                )
            if cfg.auxiliary_estimation_only:
                subprocess_args.append(
                    f"auxiliary_estimation_only={cfg.auxiliary_estimation_only}"
                )
            if cfg.csv_path:
                subprocess_args.append(f"csv_path={cfg.csv_path}")
            if cfg.esmfold_chunk_size:
                subprocess_args.append(f"esmfold_chunk_size={int(cfg.esmfold_chunk_size)}")
            subprocess.run(subprocess_args, check=True)  # nosec
        except Exception as e:
            logger.error(
                f"FlowDock inference for complex with protein `{cfg.input_receptor}` and ligand `{cfg.input_ligand}` failed with error: {e}."
            )
            raise e
        logger.info(
            f"FlowDock inference for complex with protein `{cfg.input_receptor}` and ligand `{cfg.input_ligand}` complete."
        )
    else:
        for _, row in pd.read_csv(cfg.input_csv_path).iterrows():
            out_dir = os.path.join(cfg.out_path, row.id)
            os.makedirs(out_dir, exist_ok=True)
            out_protein_filepath = os.path.join(out_dir, "prot_all.pdb")
            out_ligand_filepath = os.path.join(out_dir, "lig_all.sdf")
            if (
                cfg.skip_existing
                and os.path.exists(out_protein_filepath)
                and os.path.exists(out_ligand_filepath)
            ):
                logger.info(
                    f"Skipping inference for completed complex with protein `{out_protein_filepath}` and ligand `{out_ligand_filepath}`."
                )
                continue
            try:
                subprocess_args = [
                    str(cfg.python_exec_path),
                    os.path.join(str(cfg.flowdock_exec_dir), "src", "sample.py"),
                    f"sampling_task={cfg.sampling_task}",
                    f"sample_id='{cfg.sample_id if cfg.sample_id is not None else row.id}'",
                    f"input_ligand='{row.input_ligand}'",
                    f"input_receptor='{row.input_receptor}'",
                    f"input_template='{row.input_template}'",
                    f"trainer={'gpu' if cfg.cuda_device_index is not None else 'default'}",
                    f"{f'trainer.devices=[{int(cfg.cuda_device_index)}]' if cfg.cuda_device_index is not None else ''}",
                    f"ckpt_path={cfg.model_checkpoint}",
                    f"out_path={out_dir}",
                    f"n_samples={int(cfg.n_samples)}",
                    f"chunk_size={int(cfg.chunk_size)}",
                    f"num_steps={int(cfg.num_steps)}",
                    f"sampler={cfg.sampler}",
                    f"sampler_eta={cfg.sampler_eta}",
                    f"start_time={str(cfg.start_time)}",
                    f"max_chain_encoding_k={int(cfg.max_chain_encoding_k)}",
                    f"plddt_ranking_type={cfg.plddt_ranking_type}",
                ]
                if cfg.latent_model:
                    subprocess_args.append(f"latent_model={cfg.latent_model}")
                if cfg.exact_prior:
                    subprocess_args.append(f"exact_prior={cfg.exact_prior}")
                if cfg.discard_ligand:
                    subprocess_args.append(f"discard_ligand={cfg.discard_ligand}")
                if cfg.discard_sdf_coords:
                    subprocess_args.append(f"discard_sdf_coords={cfg.discard_sdf_coords}")
                if cfg.detect_covalent:
                    subprocess_args.append(f"detect_covalent={cfg.detect_covalent}")
                if cfg.use_template:
                    subprocess_args.append(f"use_template={cfg.use_template}")
                if cfg.separate_pdb:
                    subprocess_args.append(f"separate_pdb={cfg.separate_pdb}")
                if cfg.rank_outputs_by_confidence:
                    subprocess_args.append(
                        f"rank_outputs_by_confidence={cfg.rank_outputs_by_confidence}"
                    )
                if cfg.csv_path:
                    subprocess_args.append(f"csv_path={cfg.csv_path}")
                if cfg.esmfold_chunk_size:
                    subprocess_args.append(f"esmfold_chunk_size={int(cfg.esmfold_chunk_size)}")
                subprocess.run(subprocess_args, check=True)  # nosec
            except Exception as e:
                logger.error(
                    f"FlowDock inference for complex with protein `{out_protein_filepath}` and ligand `{out_ligand_filepath}` failed with error: {e}. Skipping..."
                )
                # Create stub files to indicate failure
                with open(out_protein_filepath, "w") as f:
                    f.write(f"Failed to generate protein file due to error: {e}.")
                with open(out_ligand_filepath, "w") as f:
                    f.write(f"Failed to generate ligand file due to error: {e}.")
                continue
        logger.info(
            f"FlowDock inference for complex with protein `{out_protein_filepath}` and ligand `{out_ligand_filepath}` complete."
        )


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
