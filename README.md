# Master Thesis: Inference-Time Guidance in Pocket-Conditioned Molecular Diffusion Models 
## Limits in Preventing Steric Clashes

TODO: provide a framework to place a mp4 file here to see: diffusion-animation.mp4

## Description

TODO: shorten the abstract below to be only three sentences, focusing on the technical contributions of this repo (no SBDD intro stuff just methods: MISATO, CDCD for cateogrical, and sidechain repuslive guidance (geometric). CDCD seemed to improve geometric guidance no improvement)
<abstract to use>
Abstract
Recent advances in AI-based protein structure prediction have accelerated structure-
based drug design (SBDD) by enabling computational models to exploit detailed
three-dimensional protein–ligand interactions. In parallel, geometric deep learning
has enabled generative models that directly design pocket-conditioned ligands with
physically plausible 3D structures. Among these approaches, SE(3)-equivariant dif-
fusion models have emerged as a powerful framework for jointly modeling atomic co-
ordinates and chemical features during molecular generation. However, these models
frequently produce unrealistic ligand–pocket interactions, particularly steric clashes
that violate fundamental geometric constraints. In this thesis, we investigate the
origin of steric clashes in diffusion-based ligand generation and analyze their struc-
tural characteristics using continuous distance-based and pocket-component–specific
metrics. Building on the score-based diffusion formulation, we introduce a side-chain
repulsive guidance mechanism that steers ligand atoms away from protein side chains
during sampling. Although the proposed model achieves competitive performance
compared to recent state-of-the-art methods, our experiments show that inference-
time repulsive guidance does not reduce steric clash frequency, suggesting that mit-
igating such failures likely requires stronger inductive biases incorporated directly
into the training objective

## Reproducibility

set up environment with conda install environment.yml

### Data Availablity

get the our dataset [here](https://drive.google.com/drive/folders/1OQCFzLhhrYmos3PmDKR5BXYZeDUDLbHN?usp=drive_link) or generate one yourself from pdb-sdf files in the following naming scheme

example_folder
|- 0_id.pdb # this is the pocket
|- 0_id.sdf # this is the ligand

python protlig_encoder.py \
    --granularity residue-level-fully-connected \ # Ca resolution / choose atom-level for full atom 
    --protein_source example_folder \
    --ligand_source example_dexample_folderataset \
    --output_dir data_extracted_graphs_atom_level

output files can be visualized as graphs:

python visualize_graph_all_atom.py --graph_path ../bioinformatics/data_extracted_graphs_atom_level/1a1e.pt

or concatenated to a dataset to train by: python Dataset.py --data_dir data_extracted_graphs_atom_level --save_path example_dataset_atom_level.pt --sidechains True

### Training

Set the training parameters in the CONFIG dictionary in main_optimize or set them as args for repeated experiments (don't forge to update the wandb subdicts entity if you want to stream to weights and biases), e.g., python main_optimize.py \
    --freeze_pocket_coords \ # conditional modelling
    --custom_wandb_prefix my-example-training-run- \
    --mode single \
    --n_max_virtual_nodes 0 \
    --edge_cutoff_pocket 8 \
    --num_epochs 500 \
    --learning_rate 0.0001 \
    --weight_decay 0.00000
    1 \
    --eval_interval 10 \
    --train_dataset datasets/dataset_pdbbind_train.pt \
    --eval_dataset datasets/dataset_pdbbind_validation.pt 

### Sampling

Sample new ligands from a trained checkpoint:
. note that this can be parallelized by e.g. when working on slurm:



#SBATCH --array=0-99 

EXP_NAME="guidance_reworked_50neg_logsumexp"
# (todo replace by example paths)
INPUT_FOLDER="../benchmarks/processed_crossdocked/test"
OUTPUT_FOLDER="../benchmarks/ours_${EXP_NAME}/ours_${EXP_NAME}_samples"
CKPT_PATH=/cluster/work/math/pbaertschi/molecular-diffusion/training_runs/RETRAIN_edge_cutoff_pocket81201_095837_datasets/datasetmd_pdbbind_train/checkpoint_epoch_230.pt

python generate_ligands.py \
    --protein_source $INPUT_FOLDER \
    --ligand_source $INPUT_FOLDER \
    --graph_dir $INPUT_FOLDER \
    --output_dir $OUTPUT_FOLDER \
    --ckpt_path $TMPDIR/model.pt \
    --n_samples 100 \
    --guidance_scale 50 \
    --job_id ${SLURM_ARRAY_TASK_ID} \
    --n_jobs ${SLURM_ARRAY_TASK_COUNT} 

### Analysis

The postprocessing and metrics calculation is inspired by the DrugFlow framework (https://github.com/LPDI-EPFL/DrugFlow) and can be parallized too: 

SAMPLES_DIR="/cluster/work/math/pbaertschi/benchmarks/${EXP_NAME}/${EXP_NAME}_samples"
EVALUATED_DATA_ALL="/cluster/work/math/pbaertschi/benchmarks/${EXP_NAME}/${EXP_NAME}_metrics/eval_tmp"
EVALUATED_DATA="/cluster/work/math/pbaertschi/benchmarks/${EXP_NAME}/${EXP_NAME}_metrics"
REFERENCE_SMILES="/cluster/work/math/pbaertschi/PDBbind_train/train_smiles.npy"
TRAINING_PDB_DIR="/cluster/work/math/pbaertschi/PDBbind_train"

python -m sbdd_metrics.evaluate_baselines \
       --in_dir "$SAMPLES_DIR" \
       --out_dir "$EVALUATED_DATA_ALL" \
       --job_id ${SLURM_ARRAY_TASK_ID} \
       --exclude interactions \
       --n_jobs ${SLURM_ARRAY_TASK_COUNT} \
       --overwrite-existing

# --- postprocessin ---
if [[ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]]; then
  echo "[INFO] Waiting for all shards to complete..."
  scontrol show job ${SLURM_ARRAY_JOB_ID} | grep -q "JobState=COMPLETED"

  echo "[INFO] Running postprocessing..."
  python -m sbdd_metrics.postprocess_metrics \
         --in_dir "$EVALUATED_DATA_ALL" \
         --out_dir "$EVALUATED_DATA"   \
         --trainingPDB_dir "$TRAINING_PDB_DIR" \
         --reference_smiles "$REFERENCE_SMILES" 

fi

voluntarily, export the results of severs analysied experiments below each other into a csv to produce tables by:

example_folder
- exp1
-- exp1_samples
--exp1_metrics
-exp2
--exp2_samples
--exp2_metrics

with 

python -m sbdd_metrics.compare_models example_folder

### Figures

To proceed to this section, make sure to have complete the workflow above.

#### Waterfall Survival Plots

To produce waterfall plots, run below. You can compare up to 6 experiments.

ython -m sbdd_visualize.posebusters_waterfall_plots --exp_name example1 example2 --labels 1 2 --out posebusters_waterfall_plots --global_only

#### Clash pattern analyiss

To analyze the clash pattern run the following script on any experiment and the result will also be saved into the metric folder

python -m sbdd_visualize.clash_eval --save_path example_folder/exp1_metrics/

## Citation

If you use this work, please cite it as software

@software{Bärtschi, Pascal is author and url is https://github.com/pascalbartschi/geom-diffusion-guidance/ and year is 2026}









