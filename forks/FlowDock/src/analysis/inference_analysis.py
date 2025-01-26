import glob
import logging
import os
from pathlib import Path

import hydra
import pandas as pd
import rootutils
from omegaconf import DictConfig, open_dict
from posebusters import PoseBusters

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import register_custom_omegaconf_resolvers, resolve_method_title

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

pd.options.mode.copy_on_write = True

BUST_TEST_COLUMNS = [
    # accuracy
    "rmsd_≤_2å",
    # chemical validity and consistency
    "mol_pred_loaded",
    "mol_true_loaded",
    "mol_cond_loaded",
    "sanitization",
    "molecular_formula",
    "molecular_bonds",
    "tetrahedral_chirality",
    "double_bond_stereochemistry",
    # intramolecular validity
    "bond_lengths",
    "bond_angles",
    "aromatic_ring_flatness",
    "double_bond_flatness",
    "internal_steric_clash",
    "internal_energy",
    # intermolecular validity
    "minimum_distance_to_protein",
    "minimum_distance_to_organic_cofactors",
    "minimum_distance_to_inorganic_cofactors",
    "volume_overlap_with_protein",
    "volume_overlap_with_organic_cofactors",
    "volume_overlap_with_inorganic_cofactors",
]

RANKED_METHODS = ["diffdock", "flowdock", "neuralplexer"]


def create_mol_table(
    input_csv_path: Path,
    input_data_dir: Path,
    inference_dir: Path,
    cfg: DictConfig,
    relaxed: bool = False,
) -> pd.DataFrame:
    """Create a table of molecules and their corresponding ligand files.

    :param input_csv_path: Path to the input CSV file.
    :param input_data_dir: Path to the input data directory.
    :param inference_dir: Path to the inference directory.
    :param mol_table_filepath: Molecule table DataFrame.
    :param cfg: Hydra configuration dictionary.
    :param relaxed: Whether to use the relaxed poses.
    :return: Molecule table DataFrame.
    """
    input_table = pd.read_csv(input_csv_path)
    if "id" in input_table.columns:
        input_table.rename(columns={"id": "pdb_id"}, inplace=True)
    if "name" in input_table.columns:
        input_table.rename(columns={"name": "pdb_id"}, inplace=True)
    if "pdb_id" not in input_table.columns:
        input_table["pdb_id"] = input_table["complex_name"].copy()

    # parse molecule (e.g., protein-)conditioning files
    mol_table = pd.DataFrame()
    if cfg.method in ["flowdock", "neuralplexer"]:
        mol_table["mol_cond"] = input_table["pdb_id"].apply(
            lambda x: list(
                (Path(str(inference_dir).replace("_relaxed", "")) / x).rglob(
                    "prot_rank1_*_aligned.pdb"
                )
            )[0]
            if len(
                list(
                    (Path(str(inference_dir).replace("_relaxed", "")) / x).rglob(
                        "prot_rank1_*_aligned.pdb"
                    )
                )
            )
            else None
        )
    else:
        protein_structure_input_dir = (
            os.path.join(
                Path(cfg.input_data_dir).parent, f"{cfg.dataset}_holo_aligned_esmfold_structures"
            )
            if os.path.exists(
                os.path.join(
                    Path(cfg.input_data_dir).parent,
                    f"{cfg.dataset}_holo_aligned_esmfold_structures",
                )
            )
            else os.path.join(Path(cfg.input_data_dir).parent, f"{cfg.dataset}_esmfold_structures")
        )
        protein_structure_file_postfix = (
            "_holo_aligned_esmfold_protein"
            if os.path.exists(
                os.path.join(
                    Path(cfg.input_data_dir).parent,
                    f"{cfg.dataset}_holo_aligned_esmfold_structures",
                )
            )
            else ""
        )
        mol_table["mol_cond"] = input_table["pdb_id"].apply(
            lambda x: os.path.join(
                protein_structure_input_dir, f"{x}{protein_structure_file_postfix}.pdb"
            )
            if os.path.exists(
                os.path.join(
                    protein_structure_input_dir, f"{x}{protein_structure_file_postfix}.pdb"
                )
            )
            else None
        )
    # parse true molecule files
    mol_table["mol_true"] = input_table["pdb_id"].apply(
        lambda x: os.path.join(
            input_data_dir, x, f"{x}_ligand{'.mol2' if cfg.dataset == 'pdbbind' else '.pdb'}"
        )
    )
    # parse predicted molecule files
    if cfg.method in RANKED_METHODS:
        mol_table["mol_pred"] = input_table["pdb_id"].apply(
            lambda x: glob.glob(
                os.path.join(
                    (
                        Path(str(inference_dir).replace("_relaxed", ""))
                        if cfg.method in ["flowdock", "neuralplexer"]
                        else inference_dir
                    ),
                    x,
                    (
                        "lig_rank1*_relaxed_aligned.sdf"
                        if cfg.method in ["flowdock", "neuralplexer"]
                        else f"{x}_relaxed.sdf"
                    ),
                )
                if relaxed
                else os.path.join(
                    (
                        Path(str(inference_dir).replace("_relaxed", ""))
                        if cfg.method in ["flowdock", "neuralplexer"]
                        else inference_dir
                    ),
                    x,
                    (
                        "lig_rank1*_aligned.sdf"
                        if cfg.method in ["flowdock", "neuralplexer"]
                        else "rank1.sdf"
                    ),
                )
            )[0]
            if len(
                glob.glob(
                    os.path.join(
                        (
                            Path(str(inference_dir).replace("_relaxed", ""))
                            if cfg.method in ["flowdock", "neuralplexer"]
                            else inference_dir
                        ),
                        x,
                        (
                            "lig_rank1*_relaxed_aligned.sdf"
                            if cfg.method in ["flowdock", "neuralplexer"]
                            else f"{x}_relaxed.sdf"
                        ),
                    )
                    if relaxed
                    else os.path.join(
                        (
                            Path(str(inference_dir).replace("_relaxed", ""))
                            if cfg.method in ["flowdock", "neuralplexer"]
                            else inference_dir
                        ),
                        x,
                        (
                            "lig_rank1*_aligned.sdf"
                            if cfg.method in ["flowdock", "neuralplexer"]
                            else "rank1.sdf"
                        ),
                    )
                )
            )
            else None
        )
    else:
        mol_table["mol_pred"] = input_table["pdb_id"].apply(
            lambda x: glob.glob(
                os.path.join(
                    inference_dir,
                    f"{x}_*{'_relaxed' if relaxed else ''}{'_aligned' if cfg.method in ['flowdock', 'neuralplexer'] else ''}.sdf",
                )
            )[0]
            if len(
                glob.glob(
                    os.path.join(
                        inference_dir,
                        f"{x}_*{'_relaxed' if relaxed else ''}{'_aligned' if cfg.method in ['flowdock', 'neuralplexer'] else ''}.sdf",
                    )
                )
            )
            else None
        )

    # drop rows with missing conditioning inputs or true ligand structures
    missing_true_indices = mol_table["mol_cond"].isna() | mol_table["mol_true"].isna()
    mol_table = mol_table.dropna(subset=["mol_cond", "mol_true"])
    input_table = input_table[~missing_true_indices]

    # check for missing (relaxed) predictions
    if mol_table["mol_pred"].isna().sum() > 0:
        if relaxed:
            missing_pred_indices = mol_table["mol_pred"].isna()
            unrelaxed_inference_dir = Path(str(inference_dir).replace("_relaxed", ""))
            if cfg.method == "diffdock":
                mol_table.loc[missing_pred_indices, "mol_pred"] = input_table.loc[
                    missing_pred_indices, "pdb_id"
                ].apply(
                    lambda x: glob.glob(os.path.join(unrelaxed_inference_dir, x, "rank1.sdf"))[0]
                    if len(glob.glob(os.path.join(unrelaxed_inference_dir, x, "rank1.sdf")))
                    else None
                )
            elif cfg.method in ["flowdock", "neuralplexer"]:
                mol_table.loc[missing_pred_indices, "mol_pred"] = input_table.loc[
                    missing_pred_indices, "pdb_id"
                ].apply(
                    lambda x: glob.glob(
                        os.path.join(
                            Path(str(inference_dir).replace("_relaxed", "")),
                            x,
                            "lig_rank1_aligned.sdf",
                        )
                    )[0]
                    if len(
                        glob.glob(
                            os.path.join(
                                Path(str(inference_dir).replace("_relaxed", "")),
                                x,
                                "lig_rank1_aligned.sdf",
                            )
                        )
                    )
                    else None
                )
            else:
                mol_table.loc[missing_pred_indices, "mol_pred"] = input_table.loc[
                    missing_pred_indices, "pdb_id"
                ].apply(
                    lambda x: glob.glob(os.path.join(unrelaxed_inference_dir, f"{x}_*.sdf"))[0]
                    if len(glob.glob(os.path.join(unrelaxed_inference_dir, f"{x}_*.sdf")))
                    else None
                )
            if mol_table["mol_pred"].isna().sum() > 0:
                if cfg.method == "diffdock":
                    logger.warning(
                        f"Skipping imputing missing (relaxed) predictions for {mol_table['mol_pred'].isna().sum()} molecules regarding the following conditioning inputs: {mol_table[mol_table['mol_pred'].isna()]['mol_cond'].tolist()}."
                    )
                    mol_table = mol_table.dropna(subset=["mol_pred"])
                else:
                    raise ValueError(
                        f"After imputing missing (relaxed) predictions, still missing predictions for {mol_table['mol_pred'].isna().sum()} molecules regarding the following conditioning inputs: {mol_table[mol_table['mol_pred'].isna()]['mol_cond'].tolist()}."
                    )
        else:
            if cfg.method == "diffdock":
                logger.warning(
                    f"Skipping missing predictions for {mol_table['mol_pred'].isna().sum()} molecules regarding the following conditioning inputs: {mol_table[mol_table['mol_pred'].isna()]['mol_cond'].tolist()}."
                )
                mol_table = mol_table.dropna(subset=["mol_pred"])
            else:
                raise ValueError(
                    f"Missing predictions for {mol_table['mol_pred'].isna().sum()} molecules regarding the following conditioning inputs: {mol_table[mol_table['mol_pred'].isna()]['mol_cond'].tolist()}."
                )

    return mol_table


@hydra.main(
    version_base="1.3",
    config_path="../../configs/analysis",
    config_name="inference_analysis.yaml",
)
def main(cfg: DictConfig):
    """Analyze the inference results of a trained model checkpoint.

    :param cfg: Configuration dictionary from the hydra YAML file.
    """
    with open_dict(cfg):
        if cfg.dataset == "pdbbind":
            cfg.input_data_dir = os.path.join(
                Path(cfg.input_data_dir).parent, "pdbbind", "PDBBind_processed"
            )
        elif cfg.dataset == "moad":
            cfg.input_data_dir = os.path.join(
                Path(cfg.input_data_dir).parent, "DockGen", "processed_files"
            )
        else:
            raise ValueError(f"Dataset `{cfg.dataset}` not supported.")

    for config in [""]:
        output_dir = cfg.output_dir + config
        if not os.path.exists(output_dir) or (
            os.path.exists(output_dir) and (Path(output_dir) / "bust_results.csv").is_file()
        ):
            output_dir = Path(str(output_dir).replace("_relaxed", ""))
        bust_results_filepath = Path(str(output_dir) + config) / "bust_results.csv"
        os.makedirs(bust_results_filepath.parent, exist_ok=True)

        # collect test results
        if os.path.exists(bust_results_filepath):
            logger.info(
                f"{resolve_method_title(cfg.method)}{config} bust results for inference directory `{output_dir}` already exist at `{bust_results_filepath}`. Directly analyzing..."
            )
            bust_results = pd.read_csv(bust_results_filepath)
        else:
            mol_table = create_mol_table(
                Path(cfg.input_csv_path),
                Path(cfg.input_data_dir),
                Path(output_dir),
                cfg,
                relaxed="relaxed" in config,
            )

            # NOTE: we use the `redock` mode here since with each method we implicitly perform cognate (e.g., apo or ab initio) docking,
            # and we have access to the ground-truth ligand structures
            buster = PoseBusters(config="redock", top_n=None)
            bust_results = buster.bust_table(mol_table, full_report=cfg.full_report)

            bust_results.to_csv(bust_results_filepath, index=False)
            logger.info(
                f"{resolve_method_title(cfg.method)}{config} bust results for inference directory `{output_dir}` successfully saved to `{bust_results_filepath}`."
            )

        # report test results
        logger.info(
            f"{resolve_method_title(cfg.method)}{config} rmsd_≤_2å: {bust_results['rmsd_≤_2å'].mean()}"
        )
        tests_table = bust_results[BUST_TEST_COLUMNS]
        tests_table.loc[:, "pb_valid"] = tests_table.iloc[:, 1:].all(axis=1)
        logger.info(
            f"{resolve_method_title(cfg.method)}{config} rmsd_≤_2å and pb_valid: {tests_table[tests_table['pb_valid']]['rmsd_≤_2å'].sum() / len(tests_table)}"
        )


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
