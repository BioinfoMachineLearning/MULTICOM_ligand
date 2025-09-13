import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from plotly import graph_objects as go
from rdkit import Chem
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from typing import List, Literal


class BindingAffinityAnalyzer:
    """Analyzes predicted vs. true binding affinity data with standard regression metrics."""

    def __init__(self, output_dir: Path, stage: Literal[1, 2], first_model_only: bool, find_best_model: bool, find_worst_model: bool):
        """
        Initialize the analyzer with output directory for plots.

        Args:
            output_dir: Directory to save analysis outputs
            stage: Experiment stage number (1 or 2).
            first_model_only: Whether to analyze only the first model prediction for each target.
            find_best_model: Whether to find the best model prediction across all targets and molecules.
            find_worst_model: Whether to find the worst model prediction across all targets and molecules.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.stage = stage

        self.first_model_only = first_model_only
        self.find_best_model = find_best_model
        self.find_worst_model = find_worst_model

    @staticmethod
    def _process_target_ids(df: pd.DataFrame) -> pd.DataFrame:
        """
        Process complex target IDs to extract Model ID and base Target ID.

        Example input: "0 L1000.AFFNTY:L1001LG207_1"
        Extracts: Model ID = 1, Target ID = L1001

        Args:
            df: DataFrame containing Target ID column

        Returns:
            DataFrame with processed Target ID and new Model ID column
        """
        # Split the string on spaces and take the relevant part containing the target info
        df["Target Info"] = df["Target ID"].str.split().str[-1]

        # Extract Model ID from the end (number after underscore)
        try:
            df["Model ID"] = df["Target Info"].str.extract(r"_(\d+)$").astype(int)
        except ValueError:
            # If Model ID is not present (e.g., for Stage 2), set it to `1`
            df["Model ID"] = 1

        # Extract base Target ID (e.g., L1001 or L3231)
        if df["Target Info"].str.contains("AFFNTY").any():
            df["Target ID"] = df["Target Info"].str.extract(r"AFFNTY:(L\d+)")[0]
        else:
            # If AFFNTY is not present (e.g., for Stage 2), use the original target ID
            df["Target ID"] = df["Target Info"]

        # Drop temporary column
        df = df.drop("Target Info", axis=1)

        return df

    @staticmethod
    def ki_to_binding_affinity(
        ki: pd.Series, R: float = 1.987204258e-3, temperature: float = 298.15
    ):
        """
        Convert Ki (in M) to binding affinity (ΔG in kcal/mol)

        Args:
            ki: Ki values in M
            R: Gas constant in kcal/(mol·K) (default=1.987204258e-3 kcal/(mol·K))
            temperature: Temperature in Kelvin (default=298.15K/25°C)

        Returns:
            binding_affinity (i.e., ΔG) in kcal/mol
        """
        binding_affinity = R * temperature * np.log(ki)
        return binding_affinity

    def load_and_process_data(
        self,
        true_data_paths: Path | List[Path],
        predicted_data_path: Path,
        true_affinity_col: str,
        ligand_smiles_col: str,
        delimiter: str = ",",
    ) -> pd.DataFrame:
        """
        Load and process binding affinity data.

        Args:
            true_data_paths: Paths to ground truth data
            predicted_data_path: Path to predicted data (Kd values in nM/liter)
            true_affinity_col: Name of the binding affinity column in true data (ΔG values in kcal/mol)
            ligand_smiles_col: Name of the ligand SMILES column in true data
            delimiter: Delimiter for predicted data file

        Returns:
            DataFrame with processed and merged data
        """
        # Load ground truth data (ΔG values in kcal/mol)
        true_data_dfs = []
        if isinstance(true_data_paths, list):
            true_data_dfs = [pd.read_csv(path) for path in true_data_paths]
        else:
            true_data = pd.read_csv(true_data_paths)

        for true_data_df in true_data_dfs:
            if (
                "Structure" in true_data_df.columns
                and ligand_smiles_col not in true_data_df.columns
            ):
                # Standard ligand SMILES column name (e.g., for Stage 2)
                true_data_df.rename(
                    columns={"Structure": ligand_smiles_col}, inplace=True
                )

        # Combine all true data DataFrames
        true_data = pd.concat(true_data_dfs, ignore_index=True)

        # Load predicted data (Kd values in nM)
        predicted_data = pd.read_csv(
            predicted_data_path,
            delimiter=delimiter,
            header=None,
            names=["Target ID", "Predicted Affinity", "Units"],
        )

        # Process target IDs
        predicted_data = self._process_target_ids(predicted_data)

        # Filter out models other than the first one if needed
        if self.first_model_only:
            predicted_data = predicted_data[
                predicted_data["Model ID"] == 1
            ].reset_index(drop=True)

        # Convert predicted Kd values in nM to ΔG values in kcal/mol
        predicted_data["Predicted Affinity"] = (
            # First convert nM to M, and then assume Kd ≈ Ki
            # for calculation of binding affinity in kcal/mol
            self.ki_to_binding_affinity(
                pd.to_numeric(predicted_data["Predicted Affinity"], errors="coerce")
                / 1e9
            )
        )

        # Merge data
        merged_data = pd.merge(
            true_data.rename(columns={true_affinity_col: "True Affinity"}),
            predicted_data,
            on="Target ID",
            how="inner",
        )

        # Identify rows that were not matched
        unmatched_predictions = predicted_data[
            ~predicted_data["Target ID"].isin(merged_data["Target ID"])
        ]["Target ID"].unique()
        unmatched_true = true_data[
            ~true_data["Target ID"].isin(merged_data["Target ID"])
        ]["Target ID"].unique()

        if unmatched_predictions.any():
            print(
                f"WARNING: Unmatched predicted affinities ({len(unmatched_predictions)}): {unmatched_predictions}"
            )
        if unmatched_true.any():
            print(
                f"WARNING: Unmatched true affinities ({len(unmatched_true)}): {unmatched_true}"
            )

        # Ensure true affinity values are numeric
        merged_data["True Affinity"] = pd.to_numeric(
            merged_data["True Affinity"], errors="coerce"
        )

        # Remove any rows with NaN values
        merged_data = merged_data.dropna(subset=["True Affinity", "Predicted Affinity"])

        # Add ligand heavy atom and bond metadata to the DataFrame
        merged_data["Ligand Heavy Atoms"] = merged_data[ligand_smiles_col].apply(
            lambda x: (
                Chem.RemoveAllHs(Chem.MolFromSmiles(x)).GetNumAtoms()
                if x is not None and isinstance(x, str)
                else np.nan
            )
        )
        merged_data["Ligand Bonds"] = merged_data[ligand_smiles_col].apply(
            lambda x: (
                Chem.RemoveAllHs(Chem.MolFromSmiles(x)).GetNumBonds()
                if x is not None and isinstance(x, str)
                else np.nan
            )
        )

        # Find the best and worst model predictions
        if self.find_best_model:
            best_model = merged_data.loc[
                abs(merged_data["Predicted Affinity"] - merged_data["True Affinity"]).idxmin()
            ]
            print(f"\nBest model prediction for target {best_model['Target ID']}:\n{best_model}")

        if self.find_worst_model:
            worst_model = merged_data.loc[
                abs(merged_data["Predicted Affinity"] - merged_data["True Affinity"]).idxmax()
            ]
            print(f"\nWorst model prediction for target {worst_model['Target ID']}:\n{worst_model}")

        return merged_data

    def calculate_metrics(self, data: pd.DataFrame) -> dict:
        """
        Calculate standard regression metrics.

        Args:
            data: DataFrame with True Affinity and Predicted Affinity columns

        Returns:
            Dictionary of metrics
        """
        true_vals = data["True Affinity"]
        pred_vals = data["Predicted Affinity"]

        metrics = {
            "rmse": np.sqrt(mean_squared_error(true_vals, pred_vals)),
            "pearson_r": pearsonr(true_vals, pred_vals)[0],
            "r2": r2_score(true_vals, pred_vals),
            "mae": np.mean(np.abs(true_vals - pred_vals)),
            "median_ae": np.median(np.abs(true_vals - pred_vals)),
            "samples": len(data),
        }

        # Measure correlation between heavy atom and bond count and predicted affinities
        metrics["heavy_atom_corr"] = data["Predicted Affinity"].corr(
            data["Ligand Heavy Atoms"]
        )
        metrics["bond_corr"] = data["Predicted Affinity"].corr(data["Ligand Bonds"])

        return metrics

    def get_protein_target(self, target_id: str) -> str:
        """
        Extract protein target (e.g., L1000 or L3000) from ligand ID.

        Args:
            target_id: Ligand ID (e.g., L1000.AFFNTY:L1001LG207_1)

        Returns:
            Protein target ID (e.g., L1000)
        """
        return f"{target_id[:2]}000"

    def analyze_by_protein_target(self, data: pd.DataFrame) -> dict:
        """
        Analyze data separately for each protein target.

        Args:
            data: DataFrame with True Affinity, Predicted Affinity, and Target ID columns

        Returns:
            Dictionary of results for each protein target
        """
        results = {}

        # First calculate metrics and create correlation plot for the entire dataset
        all_metrics = self.calculate_metrics(data)
        self.plot_correlation(
            data,
            all_metrics,
            f"Binding Affinity Correlation (Stage {self.stage})",
        )

        results["All"] = {"data": data, "metrics": all_metrics}

        # Add protein target column
        data["Protein Target"] = data["Target ID"].apply(self.get_protein_target)

        for target in sorted(data["Protein Target"].unique()):
            target_data = data[data["Protein Target"] == target]

            # Calculate metrics
            metrics = self.calculate_metrics(target_data)

            # Create correlation plot
            self.plot_correlation(
                target_data,
                metrics,
                f"{target} Binding Affinity Correlation (Stage {self.stage})",
            )

            results[target] = {"data": target_data, "metrics": metrics}

        return results

    def plot_correlation(
        self,
        data: pd.DataFrame,
        metrics: dict,
        title: str,
        log_scale: bool = False,
        model_symbols: List[str] = [
            "circle",
            "square",
            "diamond",
            "cross",
            "x",
            "triangle-up",
            "star",
        ],
    ) -> None:
        """
        Create correlation plot of predicted vs. true binding affinity values.

        Args:
            data: DataFrame with True Affinity, Predicted Affinity, and Model ID columns
            title: Title of the plot
            log_scale: Whether to use log scale for axes
            model_symbols: List of symbols to use for each model
        """
        fig = go.Figure()

        # Get unique model IDs for symbol mapping
        model_ids = (
            sorted(data["Model ID"].unique()) if "Model ID" in data.columns else [1]
        )
        assert len(model_symbols) >= len(model_ids), "Not enough symbols for all models"

        for model_id in model_ids:
            model_data = (
                data[data["Model ID"] == model_id]
                if "Model ID" in data.columns
                else data
            )

            # Add points with consistent colors but different symbols per model
            # fig.add_trace(
            #     go.Scatter(
            #         x=model_data["True Affinity"],
            #         y=model_data["True Affinity"],
            #         mode="markers",
            #         name=f"True (Model {model_id})",
            #         marker=dict(
            #             color="royalblue",
            #             symbol=model_symbols[model_id % len(model_symbols)],
            #             size=10,
            #             opacity=0.7,
            #         ),
            #     )
            # )

            fig.add_trace(
                go.Scatter(
                    x=model_data["True Affinity"],
                    y=model_data["Predicted Affinity"],
                    mode="markers",
                    name=f"Predicted (Model {model_id})",
                    marker=dict(
                        color="crimson",
                        symbol=model_symbols[model_id % len(model_symbols)],
                        size=10,
                        opacity=0.7,
                    ),
                )
            )

        # Calculate and add identity line
        min_val = min(data["True Affinity"].min(), data["Predicted Affinity"].min())
        max_val = max(data["True Affinity"].max(), data["Predicted Affinity"].max())
        x_range = np.linspace(min_val, max_val, 100)

        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=x_range,
                mode="lines",
                name="Identity (y=x)",
                line=dict(color="gray", dash="dot", width=1),
            )
        )

        # Calculate and add correlation line
        z = np.polyfit(data["True Affinity"], data["Predicted Affinity"], 1)
        p = np.poly1d(z)

        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=p(x_range),
                mode="lines",
                name=f"Fit (y={z[0]:.2f}x + {z[1]:.2f})",
                line=dict(color="red", dash="dash"),
            )
        )

        # Update layout with axis titles
        pearson_r = metrics["pearson_r"]
        fig.update_layout(
            # title=title,
            xaxis_title=f"True ΔG (kcal/mol), Pearson's R={pearson_r:.2f}",
            yaxis_title="Predicted ΔG (kcal/mol)",
            showlegend=True,
        )

        if log_scale:
            fig.update_layout(xaxis_type="log", yaxis_type="log")

        fig.write_image(self.output_dir / f"{title.replace(' ', '_')}.png")


def main(output_dir: str, stage: int, first_model_only: bool, find_best_model: bool, find_worst_model: bool):
    """Using the BindingAffinityAnalyzer class, analyze CASP16 affinity prediction results."""
    analyzer = BindingAffinityAnalyzer(
        Path(output_dir), stage=stage, first_model_only=first_model_only, find_best_model=find_best_model, find_worst_model=find_worst_model,
    )

    # Load and process data
    stage_data = analyzer.load_and_process_data(
        true_data_paths=[
            Path("ref_L1000_affinity.csv"),
            Path("ref_L3000_affinity.csv"),
        ],
        predicted_data_path=Path(f"LG207.stage{stage}.affinities"),
        true_affinity_col="binding_affinity",
        ligand_smiles_col="ligand_smiles",
        delimiter=r"\s+",
    )

    # Analyze by protein target
    results = analyzer.analyze_by_protein_target(stage_data)

    # Create DataFrames to store results
    target_metrics = []
    model_metrics = []

    for target, result in results.items():
        # Store target-level metrics
        metrics = result["metrics"]
        metrics["target"] = target
        metrics["stage"] = stage
        target_metrics.append(metrics)

        # Store model-level metrics
        target_data = result["data"]
        for model_id in sorted(target_data["Model ID"].unique()):
            model_data = target_data[target_data["Model ID"] == model_id]
            model_result = analyzer.calculate_metrics(model_data)
            model_result["target"] = target
            model_result["stage"] = stage
            model_result["model_id"] = model_id
            model_metrics.append(model_result)

    # Convert to DataFrames and save
    target_df = pd.DataFrame(target_metrics)
    model_df = pd.DataFrame(model_metrics)

    # Reorder columns to put metadata first
    target_df = target_df[
        ["stage", "target"]
        + [col for col in target_df.columns if col not in ["stage", "target"]]
    ]
    model_df = model_df[
        ["stage", "target", "model_id"]
        + [
            col
            for col in model_df.columns
            if col not in ["stage", "target", "model_id"]
        ]
    ]

    # Save results
    output_path = Path(output_dir)
    target_df.to_csv(output_path / f"target_metrics_stage{stage}.csv", index=False)
    model_df.to_csv(output_path / f"model_metrics_stage{stage}.csv", index=False)

    return target_df, model_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze CASP16 affinity prediction results"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory in which to save analysis outputs",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=1,
        choices=[1, 2],
        help="Experiment stage number (1 or 2)",
    )
    parser.add_argument(
        "--first_model_only",
        action="store_true",
        help="Whether to analyze only the first model prediction for each target",
    )
    parser.add_argument(
        "--find_best_model",
        action="store_true",
        help="Whether to find the best model prediction across all targets and molecules",
    )
    parser.add_argument(
        "--find_worst_model",
        action="store_true",
        help="Whether to find the worst model prediction across all targets and molecules",
    )
    args = parser.parse_args()

    main(args.output_dir, args.stage, args.first_model_only, args.find_best_model, args.find_worst_model)
