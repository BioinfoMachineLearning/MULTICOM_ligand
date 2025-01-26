import argparse
import glob
import logging
import os

import pandas as pd
import rootutils

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def create_casp16_superligand_ensemble_input_csv(args: argparse.Namespace):
    """Create a CSV file with the predicted protein filepaths and ligand SMILES strings for the
    superligand CASP16 targets.

    :param args: The command line arguments.
    """
    rows = []
    for item in os.listdir(args.predicted_structures_dir):
        if not item.endswith(".pdb"):
            continue

        item_id = os.path.splitext(item)[0]
        predicted_protein_filepath = os.path.join(args.predicted_structures_dir, f"{item_id}.pdb")
        ligand_smiles_tsv_filepaths = list(
            glob.glob(os.path.join(args.superligand_targets_dir, item_id, "*.tsv"))
        )
        if not os.path.exists(predicted_protein_filepath) or not ligand_smiles_tsv_filepaths:
            logger.warning(
                f"Skipping {item_id} because the predicted protein file or ligand SMILES TSV is missing."
            )
            continue

        for ligand_smiles_tsv_filepath in ligand_smiles_tsv_filepaths:
            assert os.path.exists(
                predicted_protein_filepath
            ), f"Predicted protein file not found: {predicted_protein_filepath}"
            assert os.path.exists(
                ligand_smiles_tsv_filepath
            ), f"Ligand SMILES TSV file not found: {ligand_smiles_tsv_filepath}"

            subitem_id = os.path.splitext(os.path.basename(ligand_smiles_tsv_filepath))[0]
            try:
                ligand_smiles_df = pd.read_csv(ligand_smiles_tsv_filepath, delimiter="\t")
                mol_numbers = ligand_smiles_df["ID"].tolist()
                mol_names = ligand_smiles_df["Name"].tolist()
                mol_smiles = ":".join(ligand_smiles_df["SMILES"].tolist())
                mol_tasks = ligand_smiles_df["Task"].tolist()
                assert (
                    len(set(mol_tasks)) == 1
                ), "All ligands in the TSV file must be assigned the same task."
                mol_tasks = mol_tasks[0]
            except Exception as e:
                logger.error(
                    f"Failed to read ligand SMILES TSV file {ligand_smiles_tsv_filepath} due to: {e}. Skipping..."
                )
                continue

            rows.append(
                (
                    predicted_protein_filepath,
                    mol_smiles,
                    subitem_id,
                    mol_numbers,
                    mol_names,
                    mol_tasks,
                )
            )

    df = pd.DataFrame(
        rows,
        columns=[
            "protein_input",
            "ligand_smiles",
            "name",
            "ligand_numbers",
            "ligand_names",
            "ligand_tasks",
        ],
    )
    assert (
        df["name"].nunique() == df["name"].count()
    ), "All values in the `name` column must be unique."
    df.to_csv(args.output_csv_filepath, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a CSV file with the predicted protein filepaths and ligand SMILES strings for the superligand CASP16 targets."
    )
    parser.add_argument(
        "-t",
        "--superligand-targets-dir",
        "--superligand_targets_dir",
        type=str,
        default="data/casp16_set/superligand_targets",
        help="The directory containing the CASP16 superligand targets.",
    )
    parser.add_argument(
        "-p" "--predicted-structures-dir",
        "--predicted_structures_dir",
        type=str,
        default="data/casp16_set/af3_predicted_structures",
        help="The directory containing the predicted protein structures for the CASP16 superligand targets.",
    )
    parser.add_argument(
        "-o",
        "--output-csv-filepath",
        "--output_csv_filepath",
        type=str,
        default="data/test_cases/casp16/ensemble_superligand_inputs.csv",
        help="The output CSV file.",
    )
    args = parser.parse_args()

    create_casp16_superligand_ensemble_input_csv(args)
