import logging
import os
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import rootutils
from biopandas.pdb import PandasPdb
from omegaconf import DictConfig, open_dict
from rdkit import Chem
from rdkit.Geometry import Point3D
from scipy.optimize import Bounds, minimize
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import register_custom_omegaconf_resolvers
from src.data.components.esmfold_apo_to_holo_alignment import (
    align_prediction,
    read_molecule,
)
from src.utils.data_utils import pdb_filepath_to_protein

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def save_aligned_complex(
    predicted_protein_pdb: str,
    predicted_ligand_sdf: Optional[str],
    reference_protein_pdb: str,
    reference_ligand_sdf: str,
    save_protein: bool = True,
    save_ligand: bool = True,
    aligned_filename_postfix: str = "_aligned",
    atom_df_name: str = "ATOM",
):
    """Align the predicted protein-ligand structures to the reference protein-ligand structures and
    save the aligned results.

    :param predicted_protein_pdb: Path to the predicted protein structure in PDB format
    :param predicted_ligand_sdf: Optional path to the predicted ligand structure in SDF format
    :param reference_protein_pdb: Path to the reference protein structure in PDB format
    :param reference_ligand_sdf: Path to the reference ligand structure in SDF format
    :param save_protein: Whether to save the aligned protein structure
    :param save_ligand: Whether to save the aligned ligand structure
    :param aligned_filename_postfix: Postfix to append to the aligned files
    :param atom_df_name: Name of the atom dataframe in the PDB file
    """
    # Load protein and ligand structures
    try:
        predicted_rec = pdb_filepath_to_protein(predicted_protein_pdb)
        predicted_calpha_coords = predicted_rec.atom_positions[:, 1, :]
    except Exception as e:
        logger.warning(
            f"Unable to parse predicted protein structure {predicted_protein_pdb} due to the error: {e}. Skipping..."
        )
        return
    try:
        reference_rec = pdb_filepath_to_protein(reference_protein_pdb)
        reference_calpha_coords = reference_rec.atom_positions[:, 1, :]
    except Exception as e:
        logger.warning(
            f"Unable to parse reference protein structure {reference_protein_pdb} due to the error: {e}. Skipping..."
        )
        return
    if predicted_ligand_sdf is not None:
        predicted_ligand = read_molecule(predicted_ligand_sdf, remove_hs=True, sanitize=True)
        if predicted_ligand is None:
            predicted_ligand = read_molecule(predicted_ligand_sdf, remove_hs=True, sanitize=False)
    reference_ligand = read_molecule(reference_ligand_sdf, remove_hs=True, sanitize=True)
    if reference_ligand is None:
        reference_ligand = read_molecule(reference_ligand_sdf, remove_hs=True, sanitize=False)
    if predicted_ligand_sdf is not None:
        try:
            predicted_ligand_conf = predicted_ligand.GetConformer()
        except Exception as e:
            logger.warning(
                f"Unable to extract predicted ligand conformer for {predicted_ligand_sdf} due to the error: {e}. Skipping..."
            )
            return
    try:
        reference_ligand_coords = reference_ligand.GetConformer().GetPositions()
    except Exception as e:
        logger.warning(
            f"Unable to extract reference ligand structure for {reference_ligand_sdf} due to the error: {e}. Skipping..."
        )
        return

    if reference_calpha_coords.shape != predicted_calpha_coords.shape:
        logger.warning(
            f"Receptor structures differ for prediction {predicted_protein_pdb}. Skipping due to shape mismatch:",
            reference_calpha_coords.shape,
            predicted_calpha_coords.shape,
        )
        return

    # Optimize the alignment
    res = minimize(
        align_prediction,
        [0.1],
        bounds=Bounds([0.0], [1.0]),
        args=(reference_calpha_coords, predicted_calpha_coords, reference_ligand_coords),
        tol=1e-8,
    )
    smoothing_factor = res.x
    rotation, reference_calpha_centroid, predicted_calpha_centroid = align_prediction(
        smoothing_factor,
        reference_calpha_coords,
        predicted_calpha_coords,
        reference_ligand_coords,
        return_rotation=True,
    )

    # Transform and record protein
    predicted_protein = PandasPdb().read_pdb(predicted_protein_pdb)
    predicted_protein_pre_rot = (
        predicted_protein.df[atom_df_name][["x_coord", "y_coord", "z_coord"]]
        .to_numpy()
        .squeeze()
        .astype(np.float32)
    )
    predicted_protein_aligned = (
        rotation.apply(predicted_protein_pre_rot - predicted_calpha_centroid)
        + reference_calpha_centroid
    )
    predicted_protein.df[atom_df_name][
        ["x_coord", "y_coord", "z_coord"]
    ] = predicted_protein_aligned
    if save_protein:
        predicted_protein.to_pdb(
            path=predicted_protein_pdb.replace(".pdb", f"{aligned_filename_postfix}.pdb"),
            records=[atom_df_name],
            gz=False,
        )

    # Transform and record ligand
    if predicted_ligand_sdf is not None:
        predicted_ligand_aligned = (
            rotation.apply(predicted_ligand_conf.GetPositions() - predicted_calpha_centroid)
            + reference_calpha_centroid
        )
        for i in range(predicted_ligand.GetNumAtoms()):
            x, y, z = predicted_ligand_aligned[i]
            predicted_ligand_conf.SetAtomPosition(i, Point3D(x, y, z))
        if save_ligand:
            with Chem.SDWriter(
                predicted_ligand_sdf.replace(".sdf", f"{aligned_filename_postfix}.sdf")
            ) as f:
                f.write(predicted_ligand)


@hydra.main(
    version_base="1.3",
    config_path="../../configs/analysis",
    config_name="complex_alignment.yaml",
)
def main(cfg: DictConfig):
    """Align the predicted protein-ligand structures to the reference protein-ligand structures.

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

    input_data_dir = Path(cfg.input_data_dir)
    for config in [""]:
        output_dir = Path(cfg.output_dir + config)
        if not output_dir.exists() or cfg.method in ["flowdock", "neuralplexer"]:
            output_dir = Path(str(output_dir).replace("_relaxed", ""))

        # parse ligand files
        if cfg.method == "diffdock":
            output_ligand_files = sorted(
                list(output_dir.rglob(f"*rank{cfg.rank_to_align}_confidence*{config}.sdf"))
            )
        elif cfg.method in ["flowdock", "neuralplexer"]:
            output_ligand_files = list(
                output_dir.rglob(f"lig_rank{cfg.rank_to_align}_*{config}.sdf")
            )
            output_ligand_files = sorted(
                [
                    file
                    for file in output_ligand_files
                    if config == "_relaxed"
                    or (config == "" and "_relaxed" not in file.stem)
                    and "_aligned" not in file.stem
                ]
            )
        else:
            raise ValueError(f"Invalid method: {cfg.method}")

        # parse protein files
        if cfg.method == "diffdock":
            output_protein_files = sorted(
                list(
                    (
                        input_data_dir.parent / f"{cfg.dataset}_holo_aligned_esmfold_structures"
                    ).rglob("*.pdb")
                )
            )
            output_protein_files = sorted(
                [
                    file
                    for file in output_protein_files
                    if file.stem in [item.parent.stem for item in output_ligand_files]
                ]
            )
        elif cfg.method in ["flowdock", "neuralplexer"]:
            output_protein_files = sorted(
                [
                    file
                    for file in list(output_dir.rglob(f"prot_rank{cfg.rank_to_align}_*.pdb"))
                    if "_aligned" not in file.stem
                ]
            )
        else:
            raise ValueError(f"Invalid method: {cfg.method}")

        assert len(output_protein_files) == len(
            output_ligand_files
        ), f"Numbers of protein ({len(output_protein_files)}) and ligand ({len(output_ligand_files)}) files do not match."

        # align protein-ligand complexes
        for protein_file, ligand_file in tqdm(zip(output_protein_files, output_ligand_files)):
            protein_id, ligand_id = protein_file.stem, ligand_file.stem
            if protein_id != ligand_id:
                protein_id, ligand_id = protein_file.stem, ligand_file.parent.stem
            if protein_id != ligand_id:
                protein_id, ligand_id = protein_file.parent.stem, ligand_file.parent.stem
            if protein_id != ligand_id:
                raise ValueError(f"Protein and ligand IDs do not match: {protein_id}, {ligand_id}")
            reference_protein_pdbs = [
                item
                for item in (input_data_dir / protein_id).rglob(
                    f"{protein_id}_protein_processed.pdb"
                )
                if "esmfold_structures" not in str(item)
            ]
            reference_ligand_sdfs = [
                item for item in (input_data_dir / ligand_id).rglob(f"{ligand_id}_ligand.mol2")
            ]
            if not reference_ligand_sdfs:
                reference_ligand_sdfs = [
                    item for item in (input_data_dir / ligand_id).rglob(f"{ligand_id}_ligand.sdf")
                ]
            if not reference_ligand_sdfs:
                reference_ligand_sdfs = [
                    item for item in (input_data_dir / ligand_id).rglob(f"{ligand_id}_ligand.pdb")
                ]
            assert (
                len(reference_protein_pdbs) == 1
            ), f"Expected 1 reference protein PDB file, but found {len(reference_protein_pdbs)}."
            assert (
                len(reference_ligand_sdfs) == 1
            ), f"Expected 1 reference ligand SDF file, but found {len(reference_ligand_sdfs)}."
            reference_protein_pdb, reference_ligand_sdf = (
                reference_protein_pdbs[0],
                reference_ligand_sdfs[0],
            )
            if (
                cfg.force_process
                or not os.path.exists(
                    str(protein_file).replace(".pdb", f"{cfg.aligned_filename_postfix}.pdb")
                )
                or not os.path.exists(
                    str(ligand_file).replace(".sdf", f"{cfg.aligned_filename_postfix}.sdf")
                )
            ):
                save_aligned_complex(
                    str(protein_file),
                    str(ligand_file),
                    str(reference_protein_pdb),
                    str(reference_ligand_sdf),
                    save_protein=cfg.method != "diffdock",
                    aligned_filename_postfix=cfg.aligned_filename_postfix,
                )


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
