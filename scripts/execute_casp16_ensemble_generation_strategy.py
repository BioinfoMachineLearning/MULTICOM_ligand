# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for MULTICOM_ligand: (https://github.com/BioinfoMachineLearning/MULTICOM_ligand)
# -------------------------------------------------------------------------------------------------------------------------------------

import argparse
import glob
import os
import shutil
from pathlib import Path

from posebusters import PoseBusters
from rdkit import Chem
from tqdm import tqdm

NATIVE_MULTI_LIGAND_METHODS = {"neuralplexer", "rfaa"}


def count_num_pb_valid_criteria(ligand_filepath: str, protein_filepath: str) -> int:
    """Count the number of PB-valid criteria met by the given protein-ligand
    complex.

    :param ligand_filepath: The filepath to the ligand SDF file.
    :param protein_filepath: The filepath to the protein PDB file.
    :return: The number of PB-valid criteria met by the given protein-
        ligand complex.
    """
    buster = PoseBusters(config="dock", top_n=None)
    bust_results = buster.bust(
        mol_pred=ligand_filepath, mol_cond=protein_filepath, full_report=True
    )
    num_pb_valid_criteria = (
        bust_results["mol_pred_loaded"].astype(int).item()
        + bust_results["mol_cond_loaded"].astype(int).item()
        + bust_results["sanitization"].astype(int).item()
        + bust_results["all_atoms_connected"].astype(int).item()
        + bust_results["bond_lengths"].astype(int).item()
        + bust_results["bond_angles"].astype(int).item()
        + bust_results["internal_steric_clash"].astype(int).item()
        + bust_results["aromatic_ring_flatness"].astype(int).item()
        + bust_results["double_bond_flatness"].astype(int).item()
        + bust_results["internal_energy"].astype(int).item()
        + bust_results["protein-ligand_maximum_distance"].astype(int).item()
        + bust_results["minimum_distance_to_protein"].astype(int).item()
        + bust_results["minimum_distance_to_organic_cofactors"].astype(int).item()
        + bust_results["minimum_distance_to_inorganic_cofactors"].astype(int).item()
        + bust_results["minimum_distance_to_waters"].astype(int).item()
        + bust_results["volume_overlap_with_protein"].astype(int).item()
        + bust_results["volume_overlap_with_organic_cofactors"].astype(int).item()
        + bust_results["volume_overlap_with_inorganic_cofactors"].astype(int).item()
        + bust_results["volume_overlap_with_waters"].astype(int).item()
    )
    return num_pb_valid_criteria


def is_close_to_original_ligand_structure(
    modified_ligand_filepath: str,
    original_ligand_filepath: str,
    closeness_threshold: float,
    verbose: bool = True,
) -> bool:
    """Check if the modified ligand prediction is close to the original
    ligand's structure.

    :param modified_ligand_filepath: The filepath to the modified ligand
        SDF file.
    :param original_ligand_filepath: The filepath to the original ligand
        SDF file.
    :param closeness_threshold: The (fragment-averaged) threshold
        distance (in Angstroms) under which to consider a modified
        (relaxed) ligand prediction close to its original (unrelaxed)
        ligand structure counterpart.
    :param verbose: Whether to print the average RMSD between the
        modified and original ligand structures if the modified ligand
        prediction is not close to the original ligand's structure.
    :return: True if the modified ligand prediction is close to the
        original ligand's structure, False otherwise.
    """
    modified_mol = Chem.MolFromMolFile(modified_ligand_filepath, sanitize=False)
    original_mol = Chem.MolFromMolFile(original_ligand_filepath, sanitize=False)

    if modified_mol is None:
        raise ValueError(
            f"Could not load modified ligand prediction from file `{modified_ligand_filepath}`."
        )
    if original_mol is None:
        raise ValueError(
            f"Could not load original ligand prediction from file `{original_ligand_filepath}`."
        )

    modified_mol_frags = Chem.GetMolFrags(modified_mol, asMols=True, sanitizeFrags=False)
    original_mol_frags = Chem.GetMolFrags(original_mol, asMols=True, sanitizeFrags=False)

    assert len(modified_mol_frags) == len(
        original_mol_frags
    ), f"Number of fragments in modified ({modified_ligand_filepath}) and original ({original_ligand_filepath}) ligand predictions do not match."

    frag_rmsds = []
    for modified_mol_frag, original_mol_frag in zip(modified_mol_frags, original_mol_frags):
        frag_rmsd = Chem.rdMolAlign.CalcRMS(modified_mol_frag, original_mol_frag)
        frag_rmsds.append(frag_rmsd)
    avg_rmsd = sum(frag_rmsds) / len(frag_rmsds)

    is_close = avg_rmsd < closeness_threshold
    if verbose and not is_close:
        print(
            f"Fragment-averaged RMSD between the modified ({modified_ligand_filepath}) and original ({original_ligand_filepath}) ligand structures is too large: {avg_rmsd:.2f} Angstroms."
        )

    return is_close


def execute_step_3(prediction_dir: str):
    """Copy all subdirectories in the given directory that do not end with
    "_relaxed" to a new directory with the suffix "_unrelaxed" added to the
    original name.

    :param prediction_dir: The directory containing the ensemble
        predictions for which to copy unrelaxed versions.
    """
    # List all (unrelaxed) subdirectories to copy in the given directory
    unrelaxed_subdirs_to_copy = [
        os.path.join(prediction_dir, d)
        for d in os.listdir(prediction_dir)
        if os.path.isdir(os.path.join(prediction_dir, d))
        and not d.endswith("_relaxed")
        and not os.path.isdir(os.path.join(prediction_dir, d + "_unrelaxed"))
        and not d.endswith("_unrelaxed")
    ]

    # Copy each unrelaxed subdirectory to a new directory with the suffix "_unrelaxed"
    for subdir in tqdm(unrelaxed_subdirs_to_copy, desc="Copying unrelaxed subdirectories"):
        new_subdir = subdir + "_unrelaxed"
        if not os.path.exists(new_subdir):
            shutil.copytree(subdir, new_subdir)

    print("Step 3: All unrelaxed subdirectories have been duplicated for data provenance.")


def execute_step_4(
    prediction_dir: str, closeness_threshold: float, dry_run: bool, verbose: bool = False
):
    """Apply a two-stage heuristic to further rank the top-5 (consensus)
    complexes.

    1. If a top-n complex is already PB-valid, keep its top-n ranking as is.
    2. If a top-n complex is not PB-valid, check if its relaxed version is PB-valid or if it passes more of the PB-valid criteria than the unrelaxed version.
        1. If its relaxed version is PB-valid (or passes more of the PB-valid criteria than the unrelaxed version) and the relaxed version is close to (i.e., less than 1 Angstrom away from) the unrelaxed ligand's structure, switch to the relaxed version in-place within the top-n ranking. Otherwise, prefer to keep the unrelaxed complex in its original top-n ranking.

    :param prediction_dir: The directory containing the ensemble predictions for which to further rank the top-5 (consensus) complexes.
    :param closeness_threshold: The (fragment-averaged) threshold distance (in Angstroms) under which to consider a modified (relaxed) ligand prediction close to its original (unrelaxed) ligand structure counterpart.
    :param dry_run: Whether to perform a dry run (i.e., to not actually modify the top-5 unrelaxed (consensus) complexes).
    :param verbose: Whether to print additional information during the execution of this step.
    """
    finalized_status_filepath = (
        Path(prediction_dir).parent / f"{Path(prediction_dir).stem}_finalized_steps_3_to_4.done"
    )
    if finalized_status_filepath.exists():
        print("Step 4: All top-5 (consensus) complexes have already been finalized.")
        return

    # List all (unrelaxed) subdirectories in the given directory
    unrelaxed_subdirs_to_rank = [
        os.path.join(prediction_dir, d)
        for d in os.listdir(prediction_dir)
        if os.path.isdir(os.path.join(prediction_dir, d))
        and not d.endswith("_relaxed")
        and os.path.isdir(os.path.join(prediction_dir, d + "_unrelaxed"))
        and not d.endswith("_unrelaxed")
    ]

    # Iterate over each unrelaxed subdirectory to rank the top-5 (consensus) complexes
    for subdir in tqdm(unrelaxed_subdirs_to_rank, desc="Finalizing top-5 (consensus) complexes"):
        # List all ligand prediction files in the current subdirectory
        unrelaxed_ligand_filepaths = [
            os.path.join(subdir, f)
            for f in os.listdir(subdir)
            if os.path.isfile(os.path.join(subdir, f)) and f.endswith(".sdf")
        ]

        # Iterate over each ligand prediction to potentially swap the unrelaxed ligand prediction with its relaxed version
        for unrelaxed_ligand_filepath in unrelaxed_ligand_filepaths:
            unrelaxed_ligand_filename = os.path.basename(unrelaxed_ligand_filepath)
            ligand_method = unrelaxed_ligand_filename.split("_")[0]
            ligand_rank_index = int(unrelaxed_ligand_filename.split("_")[1].split("rank")[1])

            # NOTE: Multi-ligand predictions from single-ligand methods such as DiffDock or DynamicBind must always be relaxed prior to finalizing the top-5 (consensus) complexes
            num_ligand_frags = len(
                Chem.GetMolFrags(Chem.MolFromMolFile(unrelaxed_ligand_filepath), asMols=True)
            )
            force_relax_ligand_method_prediction = (
                num_ligand_frags > 1
            ) and ligand_method not in NATIVE_MULTI_LIGAND_METHODS

            # 1. Check if the ligand prediction is already PB-valid
            if (
                unrelaxed_ligand_filename.endswith("_pbvalid=True.sdf")
                and not force_relax_ligand_method_prediction
            ):
                if verbose:
                    print(
                        f"For subdirectory `{os.path.basename(subdir)}`, ligand prediction `{unrelaxed_ligand_filename}` is already PB-valid. Skipping..."
                    )
                continue

            # 2. If the unrelaxed ligand prediction is not PB-valid, check if the relaxed ligand prediction is PB-valid
            relaxed_ligand_file_dir = os.path.dirname(unrelaxed_ligand_filepath) + "_relaxed"
            relaxed_ligand_filepaths = glob.glob(
                os.path.join(
                    relaxed_ligand_file_dir, f"{ligand_method}_rank{ligand_rank_index}_*.sdf"
                )
            )
            assert (
                len(relaxed_ligand_filepaths) == 1
            ), f"Expected 1 relaxed ligand prediction file in directory `{relaxed_ligand_file_dir}` but found {len(relaxed_ligand_filepaths)}."
            relaxed_ligand_filepath = relaxed_ligand_filepaths[0]
            relaxed_ligand_filename = os.path.basename(relaxed_ligand_filepath)

            unrelaxed_protein_filepath = unrelaxed_ligand_filepath.replace(
                "_pbvalid=False.sdf", ".pdb"
            )
            relaxed_protein_filepath = os.path.join(
                relaxed_ligand_file_dir, os.path.basename(unrelaxed_protein_filepath)
            )
            assert os.path.exists(
                unrelaxed_protein_filepath
            ), f"Expected unrelaxed protein file `{unrelaxed_protein_filepath}` to exist for unrelaxed ligand prediction `{unrelaxed_ligand_filename}`. "
            assert os.path.exists(
                relaxed_protein_filepath
            ), f"Expected relaxed protein file `{relaxed_protein_filepath}` to exist for relaxed ligand prediction `{relaxed_ligand_filename}`. "

            unrelaxed_bust_results_filepath = unrelaxed_protein_filepath.replace(
                ".pdb", "_bust_results.csv"
            )
            relaxed_bust_results_filepath = (
                relaxed_protein_filepath.replace(".pdb", "_relaxed_bust_results.csv")
                if os.path.exists(
                    relaxed_protein_filepath.replace(".pdb", "_relaxed_bust_results.csv")
                )
                else relaxed_protein_filepath.replace(".pdb", "_bust_results.csv")
            )
            assert os.path.exists(
                unrelaxed_bust_results_filepath
            ), f"Expected unrelaxed bust results file `{unrelaxed_bust_results_filepath}` to exist for unrelaxed ligand prediction `{unrelaxed_ligand_filename}`. "
            assert os.path.exists(
                relaxed_bust_results_filepath
            ), f"Expected relaxed bust results file `{relaxed_bust_results_filepath}` to exist for relaxed ligand prediction `{relaxed_ligand_filename}`. "

            if (
                force_relax_ligand_method_prediction
                or relaxed_ligand_filename.endswith("_pbvalid=True.sdf")
                or (
                    count_num_pb_valid_criteria(relaxed_ligand_filepath, relaxed_protein_filepath)
                    > count_num_pb_valid_criteria(
                        unrelaxed_ligand_filepath, unrelaxed_protein_filepath
                    )
                )
            ):
                # 2.1. If the relaxed ligand prediction is PB-valid or passes more of the PB-valid criteria than the unrelaxed version, check if the relaxed version is close to the unrelaxed ligand's structure
                if force_relax_ligand_method_prediction or is_close_to_original_ligand_structure(
                    relaxed_ligand_filepath,
                    unrelaxed_ligand_filepath,
                    closeness_threshold,
                ):
                    # 2.1.1. If the relaxed version is close to the unrelaxed ligand's structure, switch to the relaxed version in-place within the top-n ranking
                    if verbose:
                        print(
                            f"For subdirectory `{os.path.basename(subdir)}`, swapping unrelaxed ligand prediction `{unrelaxed_ligand_filename}` with relaxed ligand prediction `{relaxed_ligand_filename}`."
                        )
                    if not dry_run:
                        os.remove(unrelaxed_ligand_filepath)
                        os.remove(unrelaxed_protein_filepath)
                        os.remove(unrelaxed_bust_results_filepath)

                        shutil.copy(relaxed_ligand_filepath, unrelaxed_ligand_filepath)
                        shutil.copy(relaxed_protein_filepath, unrelaxed_protein_filepath)
                        shutil.copy(relaxed_bust_results_filepath, unrelaxed_bust_results_filepath)

        if verbose:
            print(
                f"Step 4: Finalized top-5 (consensus) complexes in subdirectory `{os.path.basename(subdir)}`."
            )

    finalized_status_filepath.touch()
    print("Step 4: All top-5 (consensus) complexes have been finalized.")


def execute_steps(prediction_dir: str, closeness_threshold: float, dry_run: bool, verbose: bool):
    """Execute steps 3-4 of the CASP16 ensemble generation strategy, as
    described in detail below.

    CASP16 ensemble (consensus) ranking strategy:
    1. Generate 100 ligand conformations with DiffDock-L, 40 with DynamicBind and NeuralPLexer each, and 1 with RFAA.
    2. Run ensemble generation script with `max_method_predictions=100`, `method_top_n_to_select=3`, `export_top_n=5`, `superligand_inputs=false`, and without (with) `relax_method_ligands_post_ranking=false` (true). Note here that, for multi-ligand targets, NeuralPLexer's predictions will always be placed as the top-3 highest-ranked structures amongst the top-5 ensemble predictions.
    3. Create a copy of each (unrelaxed) complex directory with the suffix `_unrelaxed` for data provenance.
    4. Apply a three-stage heuristic to further rank these unrelaxed top-5 (consensus) complexes:
        1. If a top-n complex of a multi-ligand target was predicted by a single-ligand method such as DiffDock or DynamicBind, force swap the unrelaxed ligand prediction with its relaxed version.
        2. Otherwise, if a top-n complex is already PB-valid, keep its top-n ranking as is.
         - If a top-n complex is not PB-valid, check if its relaxed version is PB-valid or if it passes more of the PB-valid criteria than the unrelaxed version.
            1. If its relaxed version is PB-valid (or passes more of the PB-valid criteria than the unrelaxed version) and the relaxed version is visually close to (i.e., less than 5 Angstrom away from) the original complex's structure, switch to the relaxed version in-place within the top-n ranking. Otherwise, prefer to keep the unrelaxed complex in its original top-n ranking.
    5. Once all top-N (consensus) complexes are finalized (and once all "_unrelaxed" subdirectories have been removed) within a single directory (and after they have had any "_relaxed" substrings removed from their filenames via `for file in *_relaxed*; do mv "$file" "${file/_relaxed/}"; done` within the working directory e.g., `data/test_cases/casp16/top_consensus_ensemble_predictions_1_top_3`), run ensemble generation now with FlowDock using `ensemble_methods=[flowdock]`, `skip_existing=false`, `resume=false`, `superligand_inputs=false`, `flowdock_n_samples=1`, `flowdock_chunk_size=1`, `flowdock_discard_sdf_coords=false`, `flowdock_auxiliary_estimation_only=true`, and e.g., `flowdock_auxiliary_estimation_input_dir=data/test_cases/casp16/top_consensus_ensemble_predictions_1_top_3/` to generate FlowDock runtime scripts to be executed locally.
    6. After running FlowDock inference, re-run ensemble generation to export CASP16-formatted files using `ensemble_methods=[diffdock, dynamicbind, neuralplexer, rfaa]`, `skip_existing=false`, `resume=true`, `relax_method_ligands_post_ranking=false`, `export_file_format=casp16`, `combine_casp_output_files=true`, and `superligand_inputs=true` (if applicable), which will now attach FlowDock's estimated per-ligand confidence scores and affinity predictions to each rank-ordered top-N complex.
    7. For superligand targets, run the script `scripts/combine_casp16_ensemble_superligand_predictions.py` to combine all top-5 (consensus) complexes for each superligand target into a single prediction directory for each superligand.
    8. Optionally, first submit an intentionally-bugged ligand file format (for each model) to run CASP's protein geometry checker. If there are no severe residue-residue steric clashes found, then you are good to submit the top-5 models. If there are any severe steric clashes, go back to step 2 after decreasing the default value of `clash_cutoff=0.2` for the function `rerank_clashing_predictions()`. If no CASP-side clash checking is desired, directly submit the top-5 predictions for each target.

    :param prediction_dir: The directory containing the ensemble predictions for which to execute each step of the CASP16 ensemble generation strategy.
    :param closeness_threshold: The (fragment-averaged) threshold distance (in Angstroms) under which to consider a modified (relaxed) ligand prediction close to its original (unrelaxed) ligand structure counterpart.
    :param dry_run: Whether to perform a dry run (i.e., to not actually modify the top-5 unrelaxed (consensus) complexes).
    :param verbose: Whether to print additional information during the execution of each step.
    """
    assert os.path.exists(
        prediction_dir
    ), f"Prediction directory `{prediction_dir}` does not already exist. Please run steps 1 and 2 of the CASP16 ensemble generation strategy first."

    # Step 3 - NOTE: The first two steps are assumed to have already been executed by the user
    execute_step_3(prediction_dir)

    # Step 4 - NOTE: the user is expected to manually run the final steps (5-7) after executing this step
    execute_step_4(prediction_dir, closeness_threshold, dry_run, verbose=verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CASP16 ensemble generation strategy script (steps 3-4)."
    )
    parser.add_argument(
        "-i",
        "--prediction_dir",
        type=str,
        default=os.path.join(
            "data", "test_cases", "casp16", "top_consensus_ensemble_predictions_1_top_3"
        ),
        help="The directory containing the ensemble predictions.",
    )
    parser.add_argument(
        "-c",
        "--closeness_threshold",
        type=float,
        default=5.0,
        help="The (fragment-averaged) threshold distance (in Angstroms) under which to consider a modified (relaxed) ligand prediction close to its original (unrelaxed) ligand structure counterpart.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Whether to perform a dry run (i.e., to not actually modify the top-5 unrelaxed (consensus) complexes).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print additional information during the execution of each step.",
    )
    args = parser.parse_args()

    execute_steps(args.prediction_dir, args.closeness_threshold, args.dry_run, args.verbose)
