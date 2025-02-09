# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for MULTICOM_ligand: (https://github.com/BioinfoMachineLearning/MULTICOM_ligand)
# -------------------------------------------------------------------------------------------------------------------------------------

import argparse
import ast
import logging
import os
import shutil
import subprocess  # nosec

import pandas as pd
import rootutils
from omegaconf import DictConfig, OmegaConf
from rdkit import Chem
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from multicom_ligand.models.ensemble_generation import (
    AFFINITY_UNIT_CONVERSION_DICT,
    create_flowdock_bash_script,
)

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main(
    input_dir: str,
    output_script_dir: str,
    auxiliary_estimation_input_dir: str,
    output_prediction_filepath: str,
    skip_existing: bool,
    prediction_cfg: DictConfig,
):
    """Predict binding affinities for CASP ligand L-target experimental
    protein-ligand structures."""
    assert os.path.isdir(input_dir), f"Invalid input directory: {input_dir}"

    auxiliary_estimation_csv_filepaths = []
    for input_id in tqdm(os.listdir(input_dir), desc="Predicting binding affinities"):
        # Organize input and output filepaths
        target_dir_path = os.path.join(input_dir, input_id)

        protein_pdb_filepath = os.path.join(target_dir_path, "protein_aligned.pdb")
        ligand_sdf_filepath = os.path.join(target_dir_path, "ligand.sdf")

        output_script_filepath = os.path.join(
            output_script_dir, f"flowdock_{input_id}_affinity_prediction.sh"
        )

        assert os.path.isfile(
            protein_pdb_filepath
        ), f"Invalid protein PDB file: {protein_pdb_filepath}"
        assert os.path.isfile(
            ligand_sdf_filepath
        ), f"Invalid ligand SDF file: {ligand_sdf_filepath}"

        target_auxiliary_estimation_input_dir = os.path.join(
            auxiliary_estimation_input_dir, input_id
        )
        os.makedirs(target_auxiliary_estimation_input_dir, exist_ok=True)

        auxiliary_estimation_csv_filepath = os.path.join(
            target_auxiliary_estimation_input_dir, "auxiliary_estimation.csv"
        )
        auxiliary_estimation_csv_filepaths.append(auxiliary_estimation_csv_filepath)

        if skip_existing and os.path.isfile(auxiliary_estimation_csv_filepath):
            logger.info(f"Skipping existing affinity prediction for {input_id}")
            continue

        if not os.path.isfile(
            os.path.join(target_auxiliary_estimation_input_dir, "exper_rank1_rmsd0.0.pdb")
        ):
            shutil.copy(
                protein_pdb_filepath,
                os.path.join(target_auxiliary_estimation_input_dir, "exper_rank1_rmsd0.0.pdb"),
            )
        if not os.path.isfile(
            os.path.join(target_auxiliary_estimation_input_dir, "exper_rank1_rmsd0.0.sdf")
        ):
            shutil.copy(
                ligand_sdf_filepath,
                os.path.join(target_auxiliary_estimation_input_dir, "exper_rank1_rmsd0.0.sdf"),
            )

        # Load (fragment) ligand SMILES string
        mol = Chem.MolFromMolFile(ligand_sdf_filepath, sanitize=False)
        ligand_smiles = Chem.MolToSmiles(mol)

        # Create prediction bash script
        create_flowdock_bash_script(
            protein_pdb_filepath,
            ligand_smiles.replace(
                ".", "|"
            ),  # NOTE: FlowDock supports multi-ligands using the separator "|"
            input_id,
            output_script_filepath,
            prediction_cfg,
            generate_hpc_scripts=False,
            auxiliary_estimation_input_dir=target_auxiliary_estimation_input_dir,
        )

        # Run the prediction bash script
        subprocess.run(["bash", output_script_filepath], check=True)  # nosec

    # Combine all auxiliary estimation CSV files into a CASP-compliant prediction format
    output_lines = []
    combined_auxiliary_estimation_df = pd.concat(
        [pd.read_csv(item) for item in auxiliary_estimation_csv_filepaths]
    )
    if not combined_auxiliary_estimation_df["affinity_ligs"].isnull().any():
        combined_auxiliary_estimation_df["affinity_ligs"] = combined_auxiliary_estimation_df[
            "affinity_ligs"
        ].apply(lambda x: ast.literal_eval(x))
        combined_auxiliary_estimation_df["average_affinity_ligs"] = (
            combined_auxiliary_estimation_df["affinity_ligs"].apply(lambda x: sum(x) / len(x))
        )
        for row in combined_auxiliary_estimation_df.itertuples():
            ligand_affinity_value = row.average_affinity_ligs

            # Convert predicted pK to nanomoles/liter (nM)
            ligand_affinity_value_in_moles = 10 ** (-ligand_affinity_value)
            ligand_affinity_value_in_nanomoles = (
                ligand_affinity_value_in_moles / AFFINITY_UNIT_CONVERSION_DICT["nM"]
            )
            ligand_affinity_value = ligand_affinity_value_in_nanomoles

            output_lines.append(
                f"{row.sample_id.split('_rank')[0]} {ligand_affinity_value:.3f} aa"
            )
        with open(output_prediction_filepath, "w") as f:
            f.write("\n".join(output_lines))
        logger.info(f"Saved CASP-compliant affinity predictions to {output_prediction_filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict binding affinities for CASP ligand L-target experimental protein-ligand structures."
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default=os.path.join(
            "data", "casp16_set", "superligand_targets", "L1000_exper", "L1000_prepared"
        ),
        help="Input directory containing CASP ligand L-target experimental structure files.",
    )
    parser.add_argument(
        "--output_script_dir",
        type=str,
        default="affinity_prediction_scripts",
        help="Output directory in which to store each CASP ligand L-target's binding affinity prediction bash script.",
    )
    parser.add_argument(
        "--auxiliary_estimation_input_dir",
        type=str,
        default=os.path.join(
            "data",
            "test_cases",
            "casp16",
            "superligand_affinity_predictions",
            "L1000_exper",
        ),
        help="Directory in which to store each CASP ligand L-target's predicted binding affinity CSV output file.",
    )
    parser.add_argument(
        "--output_prediction_filepath",
        type=str,
        default=os.path.join(
            "data",
            "test_cases",
            "casp16",
            "superligand_affinity_predictions",
            "L2000_exper",
            "LG207_L2000.affinities",
        ),
        help="Output text filepath to store all CASP ligand L-target binding affinity predictions.",
    )
    parser.add_argument(
        "--cuda_device_index",
        type=int,
        default=0,
        help="CUDA device index to use for FlowDock affinity predictions.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip existing CASP ligand L-target binding affinity predictions.",
    )

    args = parser.parse_args()

    if args.skip_existing and os.path.isfile(args.output_prediction_filepath):
        logger.info("Skipping existing CASP ligand L-target binding affinity predictions.")

    os.makedirs(args.output_script_dir, exist_ok=True)
    os.makedirs(args.auxiliary_estimation_input_dir, exist_ok=True)

    # Reference the string above for the configuration options
    prediction_cfg = OmegaConf.create(
        {
            "cuda_device_index": args.cuda_device_index,
            "max_method_predictions": 1,
            "flowdock_python_exec_path": "forks/FlowDock/FlowDock/bin/python3",
            "flowdock_exec_dir": "forks/FlowDock",
            "flowdock_input_data_dir": None,
            "flowdock_input_receptor_structure_dir": None,
            "flowdock_input_csv_path": "forks/FlowDock/inference/flowdock_ensemble_inputs.csv",
            "flowdock_skip_existing": True,
            "flowdock_sampling_task": "batched_structure_sampling",
            "flowdock_sample_id": 0,
            "flowdock_model_checkpoint": "forks/FlowDock/checkpoints/best_ep_d8ef2baz_epoch_189.ckpt",
            "flowdock_out_path": "forks/FlowDock/inference/flowdock_ensemble_outputs",
            "flowdock_n_samples": 1,
            "flowdock_chunk_size": 1,
            "flowdock_num_steps": 40,
            "flowdock_latent_model": None,
            "flowdock_sampler": "VDODE",
            "flowdock_sampler_eta": 1.0,
            "flowdock_start_time": "1.0",
            "flowdock_max_chain_encoding_k": -1,
            "flowdock_exact_prior": False,
            "flowdock_discard_ligand": False,
            "flowdock_discard_sdf_coords": False,
            "flowdock_detect_covalent": False,
            "flowdock_use_template": True,
            "flowdock_separate_pdb": True,
            "flowdock_rank_outputs_by_confidence": True,
            "flowdock_plddt_ranking_type": "ligand",
            "flowdock_visualize_sample_trajectories": False,
            "flowdock_auxiliary_estimation_only": True,
            "flowdock_auxiliary_estimation_input_dir": args.auxiliary_estimation_input_dir,
            "flowdock_esmfold_chunk_size": None,
        }
    )

    main(
        args.input_dir,
        args.output_script_dir,
        args.auxiliary_estimation_input_dir,
        args.output_prediction_filepath,
        args.skip_existing,
        prediction_cfg,
    )
