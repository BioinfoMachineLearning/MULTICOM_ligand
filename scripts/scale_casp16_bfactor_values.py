import argparse
import os

from tqdm import tqdm


def scale_bfactor_values(filepath: str):
    """Scale B-factor values in a CASP16 prediction file to the range of [0.0, 100.0]."""
    with open(filepath) as file:
        lines = file.readlines()

    modified_lines = []
    b_factors = []

    for line in lines:
        if line.startswith("ATOM"):
            b_factor = float(line[60:66].strip())
            b_factors.append(b_factor)

    if not b_factors:
        return

    min_b_factor = min(b_factors)
    max_b_factor = max(b_factors)

    if 0.0 <= min_b_factor <= 1.0 and 0.0 <= max_b_factor <= 1.0:
        scale = True
    elif 0.0 <= min_b_factor <= 100.0 and 0.0 <= max_b_factor <= 100.0:
        scale = False
    else:
        return

    for line in lines:
        if line.startswith("ATOM"):
            b_factor = float(line[60:66].strip())
            if scale:
                b_factor *= 100.0
            modified_line = f"{line[:60]}{b_factor:6.2f}{line[66:]}"
            modified_lines.append(modified_line)
        else:
            modified_lines.append(line)

    with open(filepath, "w") as file:
        file.writelines(modified_lines)


def scale_casp16_bfactor_values_in_file_directory(directory: str):
    """Scale B-factor values in all CASP16 prediction files within a given directory."""
    for root, _, files in os.walk(directory):
        for file in tqdm(files, desc="Scaling B-factor values in CASP16 prediction files"):
            _, ext = os.path.splitext(file)
            if not ext:
                # NOTE: The CASP16 prediction files are plain text (pseudo-PDB) files without a file extension.
                scale_bfactor_values(os.path.join(root, file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scale B-factor values in CASP16 prediction (i.e., plain text, pseudo-PDB) files."
    )
    parser.add_argument(
        "directory_path",
        type=str,
        help="Path to the directory containing CASP16 prediction files.",
    )
    args = parser.parse_args()

    scale_casp16_bfactor_values_in_file_directory(args.directory_path)
