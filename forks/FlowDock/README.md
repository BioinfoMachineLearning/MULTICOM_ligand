<div align="center">

# FlowDock

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

<!-- [![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539) -->

<!-- [![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020) -->

</div>

## Description

Official implementation of "FlowDock: Geometric Flow Matching for Switch-State Protein-Ligand Docking".

## Contents

- [Installation](#installation)
- [How to prepare data for FlowDock](#how-to-prepare-data-for-flowdock)
- [How to train FlowDock](#how-to-train-flowdock)
- [How to evaluate FlowDock](#how-to-evaluate-flowdock)
- [How to evaluate baseline methods](#how-to-evaluate-baseline-methods)
- [How to create comparative plots of evaluation results](#how-to-create-comparative-plots-of-evaluation-results)
- [How to predict new protein-ligand structures using FlowDock](#how-to-predict-new-protein-ligand-complex-structures-using-flowdock)
- [For developers](#for-developers)
- [Acknowledgements](#acknowledgements)
- [Citing this work](#citing-this-work)

## Installation

<details>

Install Mamba

```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh  # accept all terms and install to the default location
rm Mambaforge-$(uname)-$(uname -m).sh  # (optionally) remove installer after using it
source ~/.bashrc  # alternatively, one can restart their shell session to achieve the same result
```

Install dependencies

```bash
# clone project
git clone https://github.com/BioinfoMachineLearning/FlowDock
cd FlowDock

# create conda environments
# - FlowDock environment
mamba env create -f environments/flowdock_environment.yaml
conda activate FlowDock  # NOTE: one still needs to use `conda` to (de)activate environments
pip3 install -e . # install local project as package
# - (baseline) DiffDock environment
mamba env create -f environments/diffdock_environment.yaml --prefix forks/DiffDock/DiffDock/
conda activate forks/DiffDock/DiffDock/  # NOTE: one still needs to use `conda` to (de)activate environments
# - (baseline) NeuralPLexer environment
mamba env create -f environments/neuralplexer_environment.yaml --prefix forks/NeuralPLexer/NeuralPLexer/
conda activate forks/NeuralPLexer/NeuralPLexer/  # NOTE: one still needs to use `conda` to (de)activate environments
cd forks/NeuralPLexer/ && pip3 install -e . && cd ../../
```

Download checkpoints

```bash
# pretrained NeuralPLexer weights
cd checkpoints/
wget https://zenodo.org/records/10373581/files/neuralplexermodels_downstream_datasets_predictions.zip
unzip neuralplexermodels_downstream_datasets_predictions.zip
rm neuralplexermodels_downstream_datasets_predictions.zip
cd ../
```

</details>

## How to prepare data for `FlowDock`

<details>

Download data

```bash
# fetch preprocessed PDBBind, Binding MOAD & DockGen, as well as van der Mers (vdM) datasets
cd data/

wget https://zenodo.org/record/6408497/files/PDBBind.zip
wget https://zenodo.org/records/10656052/files/BindingMOAD_2020_processed.tar
wget https://zenodo.org/records/10656052/files/DockGen.tar
wget https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz

unzip PDBBind.zip
tar -xf BindingMOAD_2020_processed.tar
tar -xf DockGen.tar
tar -xzf pdb_2021aug02.tar.gz

rm PDBBind.zip BindingMOAD_2020_processed.tar DockGen.tar pdb_2021aug02.tar.gz

mkdir pdbbind/ moad/ pdbsidechain/
mv PDBBind_processed/ pdbbind/
mv BindingMOAD_2020_processed/ moad/
mv pdb_2021aug02/ pdbsidechain/

cd ../
```

### Generating ESM2 embeddings for each protein

To generate the ESM2 embeddings for the protein inputs,
first create all the corresponding FASTA files for each protein sequence

```bash
python multicom_ligand/data/components/esm_embedding_preparation.py --dataset pdbbind --data_dir data/pdbbind/PDBBind_processed/ --out_file data/pdbbind/pdbbind_sequences.fasta
python multicom_ligand/data/components/esm_embedding_preparation.py --dataset moad --data_dir data/moad/BindingMOAD_2020_processed/pdb_protein/ --out_file data/moad/moad_sequences.fasta
python multicom_ligand/data/components/esm_embedding_preparation.py --dataset dockgen --data_dir data/DockGen/processed_files/ --out_file data/DockGen/dockgen_sequences.fasta
python multicom_ligand/data/components/esm_embedding_preparation.py --dataset pdbsidechain --data_dir data/pdbsidechain/pdb_2021aug02/pdb/ --out_file data/pdbsidechain/pdbsidechain_sequences.fasta
```

Then, generate all ESM2 embeddings in batch using the ESM repository's helper script

```bash
python multicom_ligand/data/components/esm_embedding_extraction.py esm2_t33_650M_UR50D data/pdbbind/pdbbind_sequences.fasta data/pdbbind/embeddings_output --repr_layers 33 --include per_tok --truncation_seq_length 4096 --cuda_device_index 0
python multicom_ligand/data/components/esm_embedding_extraction.py esm2_t33_650M_UR50D data/moad/moad_sequences.fasta data/moad/embeddings_output --repr_layers 33 --include per_tok --truncation_seq_length 4096 --cuda_device_index 0
python multicom_ligand/data/components/esm_embedding_extraction.py esm2_t33_650M_UR50D data/DockGen/dockgen_sequences.fasta data/DockGen/embeddings_output --repr_layers 33 --include per_tok --truncation_seq_length 4096 --cuda_device_index 0
python multicom_ligand/data/components/esm_embedding_extraction.py esm2_t33_650M_UR50D data/pdbsidechain/pdbsidechain_sequences.fasta data/pdbsidechain/embeddings_output --repr_layers 33 --include per_tok --truncation_seq_length 4096 --cuda_device_index 0
```

### Predicting apo protein structures using ESMFold

To generate the apo version of each protein structure,
first create ESMFold-ready versions of the combined FASTA files
prepared above by the script `esm_embedding_preparation.py`
for the PDBBind, Binding MOAD, DockGen, and PDBSidechain datasets, respectively

```bash
python multicom_ligand/data/components/esmfold_sequence_preparation.py dataset=pdbbind
python multicom_ligand/data/components/esmfold_sequence_preparation.py dataset=moad
python multicom_ligand/data/components/esmfold_sequence_preparation.py dataset=dockgen
python multicom_ligand/data/components/esmfold_sequence_preparation.py dataset=pdbsidechain
```

Then, predict each apo protein structure using ESMFold's batch
inference script

```bash
# Note: Having a CUDA-enabled device available when running this script is highly recommended
python multicom_ligand/data/components/esmfold_batch_structure_prediction.py -i data/pdbbind/pdbbind_esmfold_sequences.fasta -o data/pdbbind/pdbbind_esmfold_structures --cuda-device-index 0 --skip-existing
python multicom_ligand/data/components/esmfold_batch_structure_prediction.py -i data/moad/moad_esmfold_sequences.fasta -o data/moad/moad_esmfold_structures --cuda-device-index 0 --skip-existing
python multicom_ligand/data/components/esmfold_batch_structure_prediction.py -i data/DockGen/dockgen_esmfold_sequences.fasta -o data/DockGen/dockgen_esmfold_structures --cuda-device-index 0 --skip-existing
python multicom_ligand/data/components/esmfold_batch_structure_prediction.py -i data/pdbsidechain/pdbsidechain_esmfold_sequences.fasta -o data/pdbsidechain/pdbsidechain_esmfold_structures --cuda-device-index 0 --skip-existing
```

Align each apo protein structure to its corresponding
holo protein structure counterpart in PDBBind, Binding MOAD, and PDBSidechain,
taking ligand conformations into account during each alignment

```bash
python multicom_ligand/data/components/esmfold_apo_to_holo_alignment.py dataset=pdbbind num_workers=1
python multicom_ligand/data/components/esmfold_apo_to_holo_alignment.py dataset=moad num_workers=1
python multicom_ligand/data/components/esmfold_apo_to_holo_alignment.py dataset=dockgen num_workers=1
python multicom_ligand/data/components/esmfold_apo_to_holo_alignment.py dataset=pdbsidechain num_workers=1
```

Lastly, assess the apo-to-holo alignments in terms of statistics and structural metrics
to enable runtime-dynamic dataset filtering using such information

```bash
python multicom_ligand/data/components/esmfold_apo_to_holo_assessment.py dataset=pdbbind usalign_exec_path=$MY_USALIGN_EXEC_PATH
python multicom_ligand/data/components/esmfold_apo_to_holo_assessment.py dataset=moad usalign_exec_path=$MY_USALIGN_EXEC_PATH
python multicom_ligand/data/components/esmfold_apo_to_holo_assessment.py dataset=dockgen usalign_exec_path=$MY_USALIGN_EXEC_PATH
python multicom_ligand/data/components/esmfold_apo_to_holo_assessment.py dataset=pdbsidechain usalign_exec_path=$MY_USALIGN_EXEC_PATH
```

</details>

## How to train `FlowDock`

<details>

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

For example, reproduce `FlowDock`'s model training run

```bash
python src/train.py experiment=flowdock_fm
```

**Note:** You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.data_cfg.batch_size=8
```

</details>

## How to evaluate `FlowDock`

<details>

For example, reproduce `FlowDock`'s evaluation results for structure prediction using the PDBBind and DockGen datasets, respectively

```bash
python3 multicom_ligand/data/baselines/flowdock_input_preparation.py dataset=pdbbind
python3 multicom_ligand/data/baselines/flowdock_input_preparation.py dataset=moad

python3 multicom_ligand/models/baselines/flowdock_inference.py dataset=pdbbind chunk_size=4 repeat_index=1
...
python3 multicom_ligand/models/baselines/flowdock_inference.py dataset=moad chunk_size=4 repeat_index=1
...

python3 multicom_ligand/analysis/complex_alignment.py method=flowdock dataset=pdbbind repeat_index=1
...
python3 multicom_ligand/analysis/complex_alignment.py method=flowdock dataset=moad repeat_index=1
...

python3 multicom_ligand/analysis/inference_analysis.py method=flowdock dataset=pdbbind repeat_index=1
...
python3 multicom_ligand/analysis/inference_analysis.py method=flowdock dataset=moad repeat_index=1
...
```

Or reproduce `FlowDock`'s evaluation results for binding affinity prediction using the PDBBind dataset

```bash
python src/eval.py data.test_datasets=[pdbbind] ckpt_path=checkpoints/esmfold_prior_paper_weights.ckpt trainer=gpu
... # re-run two more times to gather triplicate results
```

</details>

## How to evaluate baseline methods

<details>

For example, reproduce `DiffDock-L`'s evaluation results for structure prediction using the PDBBind and DockGen datasets, respectively

```bash
python3 multicom_ligand/data/baselines/diffdock_input_preparation.py dataset=pdbbind
python3 multicom_ligand/data/baselines/diffdock_input_preparation.py dataset=moad

python3 multicom_ligand/models/baselines/diffdock_inference.py dataset=pdbbind repeat_index=1
...
python3 multicom_ligand/models/baselines/diffdock_inference.py dataset=moad repeat_index=1
...

python3 multicom_ligand/analysis/inference_analysis.py method=diffdock dataset=pdbbind repeat_index=1
...
python3 multicom_ligand/analysis/inference_analysis.py method=diffdock dataset=moad repeat_index=1
...
```

Or reproduce `NeuralPLexer`'s evaluation results for structure prediction using the PDBBind and DockGen datasets, respectively

```bash
python3 multicom_ligand/data/baselines/neuralplexer_input_preparation.py dataset=pdbbind
python3 multicom_ligand/data/baselines/neuralplexer_input_preparation.py dataset=moad

python3 multicom_ligand/models/baselines/neuralplexer_inference.py dataset=pdbbind chunk_size=4 repeat_index=1
...
python3 multicom_ligand/models/baselines/neuralplexer_inference.py dataset=moad chunk_size=4 repeat_index=1
...

python3 multicom_ligand/analysis/complex_alignment.py method=neuralplexer dataset=pdbbind repeat_index=1
...
python3 multicom_ligand/analysis/complex_alignment.py method=neuralplexer dataset=moad repeat_index=1
...

python3 multicom_ligand/analysis/inference_analysis.py method=neuralplexer dataset=pdbbind repeat_index=1
...
python3 multicom_ligand/analysis/inference_analysis.py method=neuralplexer dataset=moad repeat_index=1
...
```

</details>

## How to create comparative plots of evaluation results

<details>

```bash
jupyter notebook notebooks/pdbbind_moad_inference_results_plotting.ipynb
```

</details>

## How to predict new protein-ligand structures using `FlowDock`

<details>

For example, generate new protein-ligand complexes for pairs of protein sequences and (multi-)ligand SMILES strings such as those of the CASP15 target `T1152`

```bash
python src/sample.py checkpoints/esmfold_prior_paper_weights.ckpt model.cfg.prior_type=esmfold sampling_task=batched_structure_sampling input_receptor='MYTVKPGDTMWKIAVKYQIGISEIIAANPQIKNPNLIYPGQKINIP|MYTVKPGDTMWKIAVKYQIGISEIIAANPQIKNPNLIYPGQKINIP|MYTVKPGDTMWKIAVKYQIGISEIIAANPQIKNPNLIYPGQKINIPN' input_ligand='CC(=O)NC1C(O)OC(CO)C(OC2OC(CO)C(OC3OC(CO)C(O)C(O)C3NC(C)=O)C(O)C2NC(C)=O)C1O' input_template=data/test_cases/predicted_structures/T1152.pdb sample_id='T1152' out_path='./T1152_sampled_structures/' n_samples=40 chunk_size=10 num_steps=40 sampler=VDODE sampler_eta=1.0 start_time='1.0' use_template=true separate_pdb=true visualize_sample_trajectories=true auxiliary_estimation_only=false esmfold_chunk_size=null trainer=gpu
```

If you do not already have a template protein structure available for your target of interest, set `input_template=null` to instead have the sampling script predict the ESMFold structure of your provided `input_protein` sequence before running the sampling pipeline. For more information regarding the input arguments available for sampling, please refer to the config at `configs/sample.yaml`.

</details>

## For developers

<details>

Set up `pre-commit` (one time only) for automatic code linting and formatting upon each `git commit`

```bash
pre-commit install
```

Manually reformat all files in the project, as desired

```bash
pre-commit run -a
```

Update dependencies in a `*_environment.yml` file

```bash
mamba env export > env.yaml # e.g., run this after installing new dependencies locally
diff environments/flowdock_environment.yaml env.yaml # note the differences and copy accepted changes back into e.g., `environments/flowdock_environment.yaml`
rm env.yaml # clean up temporary environment file
```

</details>

## Acknowledgements

`FlowDock` builds upon the source code and data from the following projects:

- [DiffDock](https://github.com/gcorso/DiffDock)
- [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)
- [NeuralPLexer](https://github.com/zrqiao/NeuralPLexer)

We thank all their contributors and maintainers!

## Citing this work

If you use the code associated with this repository or otherwise find this work useful, please cite:

```bibtex
@article{morehead2024flowdock,
  title={FlowDock: Geometric Flow Matching for Switch-State Protein-Ligand Docking},
  author={Morehead, Alex and Cheng, Jianlin},
  year={2024}
}
```
