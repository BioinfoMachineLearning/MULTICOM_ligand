import argparse
import glob
import logging
import os
import tempfile

import numpy as np
import pandas as pd
from beartype.typing import List, Optional, Tuple
from posebusters import PoseBusters
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdFingerprintGenerator
from tqdm import tqdm

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def align_to_binding_site(
    predicted_protein: str,
    predicted_ligand: Optional[str],
    reference_protein: str,
    reference_ligand: Optional[str],
    aligned_filename_suffix: str = "_aligned",
    cutoff: float = 10.0,
    save_protein: bool = True,
    save_ligand: bool = True,
    verbose: bool = True,
):
    """Align the predicted protein-ligand complex to the reference complex
    using the reference protein's heavy atom ligand binding site residues.

    :param predicted_protein: File path to the predicted protein (PDB).
    :param predicted_ligand: File path to the optional predicted ligand
        (SDF).
    :param reference_protein: File path to the reference protein (PDB).
    :param reference_ligand: File path to the optional reference ligand
        (SDF).
    :param output_dir: Directory to which to save the aligned protein
        and ligand structures.
    :param dataset: Dataset name (e.g., "dockgen", "casp15",
        "posebusters_benchmark", or "astex_diverse").
    :param aligned_filename_suffix: Suffix to append to the aligned
        files (default "_aligned").
    :param cutoff: Distance cutoff in Å to define the binding site
        (default 10.0).
    :param save_protein: Whether to save the aligned protein structure
        (default True).
    :param save_ligand: Whether to save the aligned ligand structure
        (default True).
    :param verbose: Whether to print the alignment RMSD and number of
        aligned atoms (default True).
    """
    from pymol import cmd

    reference_target = os.path.splitext(os.path.basename(reference_protein))[0].split("_protein")[
        0
    ]
    prediction_target = os.path.basename(os.path.dirname(predicted_protein))

    # Refresh PyMOL
    cmd.reinitialize()

    # Load structures
    cmd.load(reference_protein, "ref_protein")
    cmd.load(predicted_protein, "pred_protein")

    if reference_ligand is not None:
        cmd.load(reference_ligand, "ref_ligand")

    if predicted_ligand is not None:
        cmd.load(predicted_ligand, "pred_ligand")

    # Group predicted protein and ligand(s) together for alignment
    cmd.create(
        "pred_complex",
        ("pred_protein or pred_ligand" if predicted_ligand is not None else "pred_protein"),
    )

    # Select heavy atoms in the reference protein
    cmd.select("ref_protein_heavy", "ref_protein and not elem H")

    # Select heavy atoms in the reference ligand(s)
    cmd.select("ref_ligand_heavy", "ref_ligand and not elem H")

    # Define the reference binding site(s) based on the reference ligand(s)
    cmd.select("binding_site", f"ref_protein_heavy within {cutoff} of ref_ligand_heavy")

    # Align the predicted protein to the reference binding site(s)
    align_cmd = cmd.align
    alignment_result = align_cmd("pred_complex", "binding_site")

    # Report alignment RMSD and number of aligned atoms
    if verbose:
        logger.info(
            f"Alignment RMSD for {reference_target} with {alignment_result[1]} aligned atoms: {alignment_result[0]:.3f} Å"
        )

    # Apply the transformation to the individual objects
    cmd.matrix_copy("pred_complex", "pred_protein")
    cmd.matrix_copy("pred_complex", "pred_ligand")

    # # Maybe prepare to visualize the computed alignments
    # import shutil
    # assert (
    #     reference_target == prediction_target
    # ), f"Reference target {reference_target} does not match prediction target {prediction_target}"
    # complex_alignment_viz_dir = os.path.join("complex_alignment_viz", reference_target)
    # os.makedirs(complex_alignment_viz_dir, exist_ok=True)

    # Save the aligned protein
    if save_protein:
        cmd.save(
            predicted_protein.replace(".pdb", f"{aligned_filename_suffix}.pdb"),
            "pred_protein",
        )

        # # Maybe visualize the computed protein alignments
        # cmd.save(
        #     os.path.join(
        #         complex_alignment_viz_dir,
        #         os.path.basename(predicted_protein).replace(
        #             ".pdb", f"{aligned_filename_suffix}.pdb"
        #         ),
        #     ),
        #     "pred_protein",
        # )
        # shutil.copyfile(
        #     reference_protein,
        #     os.path.join(
        #         complex_alignment_viz_dir, os.path.basename(reference_protein)
        #     ),
        # )

    # Save the aligned ligand
    if save_ligand and predicted_ligand is not None:
        cmd.save(
            predicted_ligand.replace(".sdf", f"{aligned_filename_suffix}.sdf"),
            "pred_ligand",
        )

        # # Maybe visualize the computed ligand alignments
        # cmd.save(
        #     os.path.join(
        #         complex_alignment_viz_dir,
        #         os.path.basename(predicted_ligand).replace(
        #             ".sdf", f"{aligned_filename_suffix}.sdf"
        #         ),
        #     ),
        #     "pred_ligand",
        # )
        # shutil.copyfile(
        #     reference_ligand,
        #     os.path.join(complex_alignment_viz_dir, os.path.basename(reference_ligand)),
        # )


def find_most_similar_frag(
    mol_true_frag: Chem.Mol, mol_pred_frags: List[Chem.Mol]
) -> Tuple[Chem.Mol, float, float]:
    """Find the most similar fragment to the true fragment among the predicted
    fragments.

    :param mol_true_frag: True fragment molecule.
    :param mol_pred_frags: List of predicted fragment molecules.
    :return: Tuple of the most similar fragment molecule, the Tanimoto
        similarity, and the RMSD.
    """
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    # Generate the fingerprint for the true fragment
    mol_true_frag.UpdatePropertyCache()
    Chem.GetSymmSSSR(mol_true_frag)  # Perceive rings for fingerprinting
    fp_true = mfpgen.GetFingerprint(mol_true_frag)

    max_similarity = -1
    min_rmsd = float("inf")
    most_similar_frag = None

    for mol_pred_frag in mol_pred_frags:
        # Skip fragments with different number of atoms
        if mol_pred_frag.GetNumAtoms() != mol_true_frag.GetNumAtoms():
            continue

        # Generate the fingerprint for the predicted fragment
        mol_pred_frag.UpdatePropertyCache()
        Chem.GetSymmSSSR(mol_pred_frag)  # Perceive rings for fingerprinting
        fp_pred = mfpgen.GetFingerprint(mol_pred_frag)

        # Calculate the Tanimoto similarity
        similarity = DataStructs.TanimotoSimilarity(fp_true, fp_pred)

        # Calculate the RMSD
        rmsd = (
            AllChem.GetBestRMS(mol_true_frag, mol_pred_frag) if similarity > 0.5 else float("inf")
        )

        # Update the most similar fragment if the current one is more similar or has a lower RMSD
        if similarity > max_similarity or (similarity == max_similarity and rmsd < min_rmsd):
            max_similarity = similarity
            min_rmsd = rmsd
            most_similar_frag = mol_pred_frag

    return most_similar_frag, max_similarity, min_rmsd


def select_primary_ligands_in_df(
    mol_table: pd.DataFrame, select_most_similar_pred_frag: bool = True
) -> pd.DataFrame:
    """Select the primary ligands predictions from the molecule table
    DataFrame.

    NOTE: This function is used for single-primary-ligand datasets such as Astex Diverse, PoseBusters Benchmark, and DockGen
    to identify a method's (most likely) prediction for a specific primary ligand crystal structure when the method is tasked
    with predicting all cofactors as well (to enhance its molecular context for primary ligand predictions).

    :param mol_table: Molecule table DataFrame.
    :param select_most_similar_pred_frag: Whether to select the predicted ligand fragment most similar (chemically and structurally) to the true ligand fragment.
    :return: Molecule table DataFrame with primary ligand predictions.
    """
    new_rows = []
    for row in mol_table.itertuples():
        try:
            mol_true_file_fn = (
                Chem.MolFromPDBFile if str(row.mol_true).endswith(".pdb") else Chem.MolFromMolFile
            )

            mol_true = mol_true_file_fn(str(row.mol_true), removeHs=False)
            mol_pred = Chem.MolFromMolFile(str(row.mol_pred), removeHs=False)

            assert mol_true is not None, f"Failed to load the true molecule from {row.mol_true}."
            assert (
                mol_pred is not None
            ), f"Failed to load the predicted molecule from {row.mol_pred}."

            mol_true_frags = Chem.GetMolFrags(mol_true, asMols=True, sanitizeFrags=False)
            mol_pred_frags = Chem.GetMolFrags(mol_pred, asMols=True, sanitizeFrags=False)

            if select_most_similar_pred_frag:
                mol_pred_frags = [
                    find_most_similar_frag(mol_true_frag, mol_pred_frags)[0]
                    for mol_true_frag in mol_true_frags
                ]
                if not any(mol_pred_frags):
                    logger.warning(
                        f"None of the predicted fragments are similar enough to the true fragments for row {row.Index}. Skipping this row."
                    )
                    continue

            assert len(mol_true_frags) == len(
                mol_pred_frags
            ), "The number of fragments should be the same."

            for frag_index, (mol_true_frag, mol_pred_frag) in enumerate(
                zip(mol_true_frags, mol_pred_frags)
            ):
                new_row = row._asdict()
                new_row["mol_cond"] = row.mol_cond
                with tempfile.NamedTemporaryFile(
                    suffix=".sdf", delete=False
                ) as temp_true, tempfile.NamedTemporaryFile(
                    suffix=".sdf", delete=False
                ) as temp_pred:
                    assert (
                        mol_true_frag.GetNumAtoms() == mol_pred_frag.GetNumAtoms()
                    ), "The number of atoms in each fragment should be the same."

                    Chem.MolToMolFile(mol_true_frag, temp_true.name)
                    Chem.MolToMolFile(mol_pred_frag, temp_pred.name)
                    true_smiles = Chem.MolToSmiles(Chem.MolFromMolFile(temp_true.name))
                    pred_smiles = Chem.MolToSmiles(Chem.MolFromMolFile(temp_pred.name))

                    if true_smiles != pred_smiles:
                        logger.warning(
                            f"The SMILES strings of the index {frag_index} fragments ({true_smiles} vs. {pred_smiles}) differ for row {row.Index} after post-processing."
                        )

                    new_row["mol_true"] = temp_true.name
                    new_row["mol_pred"] = temp_pred.name
                new_rows.append(new_row)

        except Exception as e:
            logger.warning(
                f"An error occurred while splitting fragments for row {row.Index}: {e}. Skipping this row."
            )

    return pd.DataFrame(new_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score FlowDock's CASP16 protein-ligand structure predictions using PyMOL and PoseBusters."
    )
    parser.add_argument(
        "--predictions_path",
        type=str,
        help="Path to the directory containing FlowDock's CASP16 prediction files.",
        default=os.path.join("forks", "FlowDock", "inference", "flowdock_ensemble_outputs"),
    )
    parser.add_argument(
        "--references_path",
        type=str,
        help="Path to the directory containing the crystal CASP16 structure files.",
        default=os.path.join("data", "casp16_set", "superligand_targets"),
    )
    args = parser.parse_args()

    bust_results_list = []
    for superligand_dir in glob.glob(os.path.join(args.references_path, "*_exper_prepared")):
        for superligand_target_dir in tqdm(
            glob.glob(os.path.join(superligand_dir, "*")),
            desc=f"Processing superligand targets of {os.path.basename(superligand_dir)}...",
        ):
            superligand_target = os.path.basename(superligand_target_dir)
            logger.info(f"Processing {superligand_target}...")

            # Find the predicted protein and ligand files
            predicted_proteins = [
                f
                for f in glob.glob(
                    os.path.join(args.predictions_path, superligand_target, "prot_rank1*.pdb")
                )
                if not f.endswith("_aligned.pdb")
            ]
            predicted_ligands = [
                f
                for f in glob.glob(
                    os.path.join(
                        args.predictions_path,
                        superligand_target,
                        "lig_rank1_*_ensemble_relaxed.sdf",
                    )
                )
                if "_aligned" not in f
            ]

            assert (
                len(predicted_proteins) == 1
            ), f"Expected one predicted protein for {superligand_target}, found {len(predicted_proteins)}"
            assert (
                len(predicted_ligands) == 1
            ), f"Expected one predicted ligand for {superligand_target}, found {len(predicted_ligands)}"

            predicted_protein = predicted_proteins[0]
            predicted_ligand = predicted_ligands[0]

            # Find the reference protein and ligand files
            reference_protein = os.path.join(superligand_target_dir, "protein_aligned.pdb")
            reference_ligand = os.path.join(superligand_target_dir, "ligand.sdf")

            assert os.path.exists(
                reference_protein
            ), f"Reference protein not found: {reference_protein}"
            assert os.path.exists(
                reference_ligand
            ), f"Reference ligand not found: {reference_ligand}"

            # Check if aligned files already exist
            if os.path.exists(predicted_ligand.replace(".sdf", "_aligned.sdf")):
                logger.info(f"Aligned files already exist for {superligand_target}, skipping...")
            else:
                # Save the aligned structures
                align_to_binding_site(
                    predicted_protein=predicted_protein,
                    predicted_ligand=predicted_ligand,
                    reference_protein=reference_protein,
                    reference_ligand=reference_ligand,
                    aligned_filename_suffix="_aligned",
                    cutoff=10.0,
                    save_protein=True,
                    save_ligand=True,
                    verbose=True,
                )

            # Score aligned structures using PoseBusters
            mol_table = pd.DataFrame(
                {
                    "mol_true": [reference_ligand],
                    "mol_pred": [predicted_ligand.replace(".sdf", "_aligned.sdf")],
                    "mol_cond": [predicted_protein.replace(".pdb", "_aligned.pdb")],
                }
            )
            mol_table = select_primary_ligands_in_df(mol_table)

            # NOTE: we use the `redock` mode here since with each method we implicitly perform cognate (e.g., apo or ab initio) docking,
            # and we have access to the ground-truth ligand structures in SDF format
            try:
                buster = PoseBusters(config="redock", top_n=None)
                buster.config["loading"]["mol_true"]["load_all"] = False

                bust_results = buster.bust_table(mol_table, full_report=True)
                bust_results["mol_id"] = superligand_target
                bust_results["lig_idx"] = np.arange(len(bust_results), dtype=int)

                bust_results_list.append(bust_results)
                logger.info(f"PoseBusters results for {superligand_target} collected")
            except Exception as e:
                logger.warning(
                    f"An error occurred while scoring {superligand_target} with PoseBusters: {e}"
                )

    # Concatenate all results and save to CSV
    bust_results_df = pd.concat(bust_results_list, ignore_index=True)
    bust_results_filepath = os.path.join(
        args.predictions_path, "casp16_flowdock_bust_results_relaxed.csv"
    )
    bust_results_df.to_csv(bust_results_filepath, index=False)
    logger.info(f"PoseBusters results for all CASP16 targets saved to {bust_results_filepath}")
