import logging
import os
from pathlib import Path

import hydra
import rootutils
from beartype import beartype
from beartype.typing import Any, List, Optional, Tuple
from omegaconf import DictConfig, open_dict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import register_custom_omegaconf_resolvers
from src.utils.data_utils import parse_inference_inputs_from_dir

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# NOTE: the following sequence is derived from `5S8I_2LY.pdb` of the PoseBusters Benchmark set
LIGAND_ONLY_RECEPTOR_PLACEHOLDER_SEQUENCE = "DSLFAGLVGEYYGTNSQLNNISDFRALVDSKEADATFEAANISYGRGSSDVAKGTHLQEFLGSDASTLSTDPGDNTDGGIYLQGYVYLEAGTYNFKVTADDGYEITINGNPVATVDNNQSVYTVTHASFTISESGYQAIDMIWWDQGGDYVFQPTLSADGGSTYFVLDSAILSSTGETPY"


@beartype
def write_input_csv(
    smiles_and_pdb_id_list: Optional[List[Tuple[Any, str]]],
    output_csv_path: str,
    input_receptor_structure_dir: Optional[str],
    input_receptor: Optional[str] = None,
    input_ligand: Optional[Any] = None,
    input_template: Optional[str] = None,
    input_id: Optional[str] = None,
):
    """Write a NeuralPLexer inference CSV file.

    :param smiles_and_pdb_id_list: A list of tuples each containing a SMILES string and a PDB ID.
    :param output_csv_path: Path to the output CSV file.
    :param input_receptor_structure_dir: Path to the directory containing the protein structure
        input files.
    :param input_receptor: Optional path to a single input protein sequence.
    :param input_ligand: Optional single input ligand SMILES string.
    :param input_template: Path to the optional template protein structure.
    :param input_id: Optional input ID.
    """
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, "w") as f:
        f.write("id,input_receptor,input_ligand,input_template\n")
        if input_ligand is not None:
            input_id = (
                (
                    "_".join(os.path.splitext(os.path.basename(input_template))[0].split("_")[:2])
                    if input_template
                    else "ensemble_input_ligand"
                )
                if input_id is None
                else input_id
            )
            # NOTE: a placeholder protein sequence is used when making ligand-only predictions
            if not input_receptor:
                input_receptor = LIGAND_ONLY_RECEPTOR_PLACEHOLDER_SEQUENCE
            if not input_template:
                input_template = LIGAND_ONLY_RECEPTOR_PLACEHOLDER_SEQUENCE
            f.write(f"{input_id},{input_receptor},{input_ligand},{input_template}\n")
        else:
            for smiles, pdb_id in smiles_and_pdb_id_list:
                if os.path.isdir(os.path.join(input_receptor_structure_dir, pdb_id)):
                    input_receptor = os.path.join(
                        input_receptor_structure_dir, pdb_id, f"{pdb_id}_ligand.pdb"
                    )
                else:
                    input_receptor = (
                        os.path.join(
                            input_receptor_structure_dir,
                            f"{pdb_id}_holo_aligned_esmfold_protein.pdb",
                        )
                        if os.path.exists(
                            os.path.join(
                                input_receptor_structure_dir,
                                f"{pdb_id}_holo_aligned_esmfold_protein.pdb",
                            )
                        )
                        else os.path.join(input_receptor_structure_dir, f"{pdb_id}.pdb")
                    )
                if not os.path.exists(input_receptor):
                    logger.warning(f"Skipping input protein which was not found: {input_receptor}")
                    continue
                f.write(f"{pdb_id},{input_receptor},{smiles},{input_receptor}\n")


@hydra.main(
    version_base="1.3",
    config_path="../../../configs/data/baselines",
    config_name="neuralplexer_input_preparation.yaml",
)
def main(cfg: DictConfig):
    """Parse a data directory containing subdirectories of protein-ligand complexes and prepare
    corresponding inference CSV file for the NeuralPLexer model.

    :param cfg: Configuration dictionary from the hydra YAML file.
    """
    pdb_ids = None
    with open_dict(cfg):
        if cfg.dataset == "pdbbind":
            cfg.input_data_dir = os.path.join(
                Path(cfg.input_data_dir).parent, "pdbbind", "PDBBind_processed"
            )
            with open(cfg.pdbbind_test_ids_filepath) as f:
                pdb_ids = set(f.read().splitlines())
        elif cfg.dataset == "moad":
            cfg.input_data_dir = os.path.join(
                Path(cfg.input_data_dir).parent, "DockGen", "processed_files"
            )
            cfg.input_receptor_structure_dir = os.path.join(
                Path(cfg.input_data_dir).parent, "dockgen_holo_aligned_esmfold_structures"
            )
            with open(cfg.moad_test_ids_filepath) as f:
                pdb_ids = {line.replace(" ", "-") for line in f.read().splitlines()}
        else:
            raise ValueError(f"Dataset `{cfg.dataset}` not supported.")
    if cfg.input_receptor is not None and cfg.input_ligand is not None:
        write_input_csv(
            [],
            output_csv_path=cfg.output_csv_path,
            input_receptor_structure_dir=cfg.input_receptor_structure_dir,
            input_receptor=cfg.input_receptor,
            input_ligand=cfg.input_ligand,
            input_template=cfg.input_template,
            input_id=cfg.input_id,
        )
    else:
        smiles_and_pdb_id_list = parse_inference_inputs_from_dir(
            cfg.input_data_dir,
            pdb_ids=pdb_ids,
        )
        write_input_csv(
            smiles_and_pdb_id_list=smiles_and_pdb_id_list,
            output_csv_path=cfg.output_csv_path,
            input_receptor_structure_dir=cfg.input_receptor_structure_dir,
            input_receptor=cfg.input_receptor,
            input_ligand=cfg.input_ligand,
            input_template=cfg.input_template,
            input_id=cfg.input_id,
        )

    logger.info(f"NeuralPLexer input CSV preparation for dataset `{cfg.dataset}` complete.")


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
