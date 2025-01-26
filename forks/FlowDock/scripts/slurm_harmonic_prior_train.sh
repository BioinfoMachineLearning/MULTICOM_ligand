#!/bin/bash -l
######################### Batch Headers #########################
#SBATCH --partition chengji-lab-gpu  # use reserved partition `chengji-lab-gpu`
#SBATCH --account chengji-lab  # NOTE: this must be specified to use the reserved partition above
#SBATCH --nodes=1              # NOTE: this needs to match Lightning's `Trainer(num_nodes=...)`
#SBATCH --gres gpu:H100:4      # request H100 GPU resource(s)
#SBATCH --ntasks-per-node=4    # NOTE: this needs to be `1` on SLURM clusters when using Lightning's `ddp_spawn` strategy`; otherwise, set to match Lightning's quantity of `Trainer(devices=...)`
#SBATCH --mem=0                # NOTE: use `--mem=0` to request all memory "available" on the assigned node
#SBATCH -t 7-00:00:00          # time limit for the job (up to 7 days: `7-00:00:00`)
#SBATCH -J fd_fm_harmonic_prior_train  # job name
#SBATCH --output=R-%x.%j.out   # output log file
#SBATCH --error=R-%x.%j.err    # error log file

random_seconds=$(( (RANDOM % 100) + 1 ))
echo "Sleeping for $random_seconds seconds before starting run"
sleep "$random_seconds"

module purge
module load cuda/11.8.0_gcc_9.5.0

# determine location of the project directory
use_private_project_dir=false # NOTE: customize as needed
if [ "$use_private_project_dir" = true ]; then
    project_dir="/home/acmwhb/data/Repositories/Lab_Repositories/FlowDock"
else
    project_dir="/cluster/pixstor/chengji-lab/acmwhb/Repositories/Lab_Repositories/FlowDock"
fi

# shellcheck source=/dev/null
source /home/acmwhb/mambaforge/etc/profile.d/conda.sh
conda activate "$project_dir"/FlowDock/

echo "Calling src/train.py!"
cd "$project_dir" || exit
srun python3 src/train.py \
    callbacks.model_checkpoint.monitor='val_sampling/ligand_hit_score_2A_epoch' \
    ckpt_path="$(realpath 'logs/train/runs/2024-05-12_13-15-18/checkpoints/last.ckpt')" \
    data.edge_crop_size=400000 \
    data.batch_size=16 \
    data.num_workers=4 \
    data.pin_memory=true \
    data.train_datasets='[pdbbind, moad, pdbsidechain]' \
    data.double_val=false \
    data.remove_pdbbind=false \
    data.enforce_timesplit=true \
    data.limit_complexes=0 \
    data.moad_map_binding_affinities_to_superligands=false \
    data.pdbsidechain_postprocess_min_protein_length=100 \
    data.pdbsidechain_postprocess_max_protein_length=400 \
    data.vandermers_second_ligand_max_closeness=10.0 \
    data.vandermers_extract_second_ligand=true \
    data.vandermers_use_prob_as_surrogate_binding_affinity=false \
    experiment='flowdock_fm' \
    environment=slurm \
    logger=wandb \
    logger.wandb.entity='bml-lab' \
    logger.wandb.group='FlowDock-FM' \
    +logger.wandb.name='2024-05-12_12:00:00-Harmonic-Prior-Finetuning-ContactPredictorUnfrozen-PDBBind_MOAD_PDBSidechain-vdMSecondLigands' \
    +logger.wandb.id='urqhgps8' \
    model.cfg.prior_type=harmonic \
    model.cfg.mol_encoder.checkpoint_file='checkpoints/neuralplexermodels_downstream_datasets_predictions/models/complex_structure_prediction.ckpt' \
    model.cfg.mol_encoder.from_pretrained=true \
    model.cfg.protein_encoder.from_pretrained=true \
    model.cfg.relational_reasoning.from_pretrained=true \
    model.cfg.contact_predictor.from_pretrained=true \
    model.cfg.score_head.from_pretrained=true \
    model.cfg.confidence.from_pretrained=true \
    model.cfg.affinity.from_pretrained=false \
    model.cfg.affinity.dropout=0.01 \
    model.cfg.confidence.enabled=true \
    model.cfg.affinity.enabled=true \
    model.cfg.task.dropout=0.01 \
    model.cfg.task.freeze_mol_encoder=true \
    model.cfg.task.freeze_protein_encoder=false \
    model.cfg.task.freeze_relational_reasoning=false \
    model.cfg.task.freeze_contact_predictor=false \
    model.cfg.task.freeze_score_head=false \
    model.cfg.task.freeze_confidence=true \
    model.cfg.task.freeze_affinity=false \
    model.cfg.task.use_template=true \
    model.cfg.task.float32_matmul_precision=highest \
    model.cfg.task.affinity_loss_weight=0.1 \
    model.cfg.task.aux_batch_freq=10 \
    model.cfg.task.loss_mode='auxiliary_estimation' \
    model.optimizer.lr=2e-4 \
    paths.output_dir="$(realpath 'logs/train/runs/2024-05-12_13-15-18')" \
    seed=496 \
    strategy=ddp \
    trainer=ddp \
    trainer.devices=4 \
    trainer.num_nodes=1 \
    trainer.sync_batchnorm=true \
    trainer.gradient_clip_algorithm=norm \
    trainer.gradient_clip_val=1.0 \
    trainer.max_epochs=200
echo "Finished calling src/train.py!"

# NOTE: the following commands must be used to resume training from a checkpoint
# ckpt_path="$(realpath 'logs/train/runs/2024-05-12_13-15-18/checkpoints/last.ckpt')" \
# paths.output_dir="$(realpath 'logs/train/runs/2024-05-12_13-15-18')" \

# NOTE: the following commands may be used to speed up training
# model.compile=false \
# +trainer.precision=bf16-mixed
