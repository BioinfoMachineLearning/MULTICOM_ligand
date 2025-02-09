# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for MULTICOM_ligand: (https://github.com/BioinfoMachineLearning/MULTICOM_ligand)
# -------------------------------------------------------------------------------------------------------------------------------------

import argparse
import os
import shutil

from tqdm import tqdm


def combine_superligand_predictions(prediction_dir: str, output_dir: str):
    """Execute step 7 of the CASP16 ensemble generation strategy, as described
    in detail within `scripts/execute_casp16_ensemble_generation_strategy.py`.

    :param prediction_dir: The directory containing the ensemble
        predictions for which to execute step 7 of the CASP16 ensemble
        generation strategy.
    :param output_dir: The directory to which to write the combined
        superligand predictions.
    """
    assert os.path.exists(
        prediction_dir
    ), f"Prediction directory `{prediction_dir}` does not already exist. Please perform steps 1-6 of the CASP16 ensemble generation strategy first."
    os.makedirs(args.output_dir, exist_ok=True)

    for target in tqdm(os.listdir(prediction_dir), desc="Combining superligand predictions"):
        if "_relaxed" in target:
            continue

        target_dir = os.path.join(prediction_dir, target)
        superligand_dir = os.path.join(output_dir, f"{target[:2]}000_models")
        os.makedirs(superligand_dir, exist_ok=True)

        submission_files = [f for f in os.listdir(target_dir) if f"{target}LG" in f]
        for submission_file in submission_files:
            shutil.copyfile(
                os.path.join(target_dir, submission_file),
                os.path.join(superligand_dir, submission_file),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CASP16 superligand prediction combination script (step 7 of the CASP16 ensemble generation strategy)."
    )
    parser.add_argument(
        "-i",
        "--prediction_dir",
        type=str,
        default=os.path.join(
            "data", "test_cases", "casp16", "top_consensus_ensemble_predictions_1_top_3"
        ),
        help="The directory containing the individual ensemble predictions.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=os.path.join(
            "data",
            "test_cases",
            "casp16",
            "top_consensus_ensemble_predictions_1_top_3_superligand_submission_files",
        ),
        help="The directory in which to store the combined ensemble predictions.",
    )
    args = parser.parse_args()

    combine_superligand_predictions(args.prediction_dir, args.output_dir)
