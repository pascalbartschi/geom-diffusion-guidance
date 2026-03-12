
# Master Thesis: Inference-Time Guidance in Pocket-Conditioned Molecular Diffusion Models
## Limits in Preventing Steric Clashes

<p align="center">
  <video src="diffusion-animation/diffusion-animation.mp4" controls width="700"></video>
</p>
Animation: Visualization of reverse-time variance exploding diffusion of a ligand conditioned on Glutathione S-Transferase P1-1 Apo Form 1 pocket (CrossDocked2020 cut). Reproduction guide in [diffusion-animation](diffusion-animation).

---

## Description

This repository contains the code accompanying the master's thesis **“Inference-Time Guidance in Pocket-Conditioned Molecular Diffusion Models: Limits in Preventing Steric Clashes.”**

The project builds on pocket‑conditioned **SE(3)-equivariant diffusion models** for 3D molecular generation and studies the structural failure mode of **steric clashes between generated ligands and protein pockets**. The implementation uses **MISATO-derived protein–ligand complexes** as training data and integrates **Continuous Diffusion for Categorical Data (CDCD)** for modeling atom types jointly with atomic coordinates.

Additionally, the repository implements **side‑chain repulsive guidance at inference time**, which introduces a geometric repulsive potential between ligand atoms and protein side chains during the reverse diffusion process. While CDCD stabilizes categorical sampling, experiments show that **repulsive geometric guidance does not significantly reduce steric clash frequency**, indicating that preventing such structural violations likely requires stronger inductive biases incorporated during model training rather than only during sampling.

The repository therefore provides:

- A full training pipeline for **pocket‑conditioned equivariant molecular diffusion**
- Tools for **dataset generation from protein–ligand complexes**
- Scripts for **parallel ligand generation**
- A reproducible **evaluation and benchmarking framework**
- Visualization tools for **clash analysis and PoseBusters survival plots**

---

# Reproducibility

Create the environment from the provided conda file:

```bash
conda env create -f environment.yml
conda activate sbdd
```

All experiments in the thesis were executed within this environment.

---

# Data Availability

A preprocessed dataset used in the thesis experiments can be downloaded [here](https://drive.google.com/drive/folders/1OQCFzLhhrYmos3PmDKR5BXYZeDUDLbHN).

Alternatively, datasets can be generated from protein–ligand pocket pairs using the following naming scheme:

```
example_folder/
 ├─ 0_id.pdb   # protein pocket
 └─ 0_id.sdf   # ligand
```

### Encoding protein–ligand graphs

```bash
python protlig_encoder.py     --granularity residue-level-fully-connected     --protein_source example_folder     --ligand_source example_folder     --output_dir data_extracted_graphs_atom_level
```

Notes:

- `residue-level-fully-connected` uses **Cα resolution**
- Use `atom-level` for **full atomic pocket representations**

### Visualizing encoded graphs

```bash
python visualize_graph_all_atom.py     --graph_path data_extracted_graphs_atom_level/1a1e.pt
```

### Creating a training dataset

```bash
python Dataset.py     --data_dir data_extracted_graphs_atom_level     --save_path example_dataset_atom_level.pt     --sidechains True
```

---

# Training

Training parameters can either be set in the `CONFIG` dictionary inside `main_optimize.py` or passed via command line arguments.

Example training run:

```bash
python main_optimize.py     --freeze_pocket_coords     --custom_wandb_prefix my-example-training-run     --mode single     --n_max_virtual_nodes 0     --edge_cutoff_pocket 8     --num_epochs 500     --learning_rate 1e-4     --weight_decay 0     --eval_interval 10     --train_dataset datasets/dataset_pdbbind_train.pt     --eval_dataset datasets/dataset_pdbbind_validation.pt
```

If logging to **Weights & Biases**, update the `wandb.entity` field accordingly.

---

# Sampling

New ligands can be generated from a trained checkpoint using the sampling script.

Sampling can be **parallelized**, for example using **SLURM job arrays** on HPC clusters.

Example SLURM setup:

```bash
#SBATCH --array=0-99

EXP_NAME="guidance_reworked"
INPUT_FOLDER="../benchmarks/processed_crossdocked/test"
OUTPUT_FOLDER="../benchmarks/ours_${EXP_NAME}/samples"
CKPT_PATH="checkpoint_epoch_230.pt"

python generate_ligands.py     --protein_source $INPUT_FOLDER     --ligand_source $INPUT_FOLDER     --graph_dir $INPUT_FOLDER     --output_dir $OUTPUT_FOLDER     --ckpt_path $CKPT_PATH     --n_samples 100     --guidance_scale 50     --job_id ${SLURM_ARRAY_TASK_ID}     --n_jobs ${SLURM_ARRAY_TASK_COUNT}
```

Each job processes a shard of the dataset and writes generated ligand structures to the output directory.

---

# Analysis

The evaluation pipeline is inspired by the **DrugFlow benchmarking framework**:

https://github.com/LPDI-EPFL/DrugFlow

Metric computation can also be **parallelized**, for example with SLURM arrays.

### Metric evaluation

```bash
python -m sbdd_metrics.evaluate_baselines     --in_dir SAMPLES_DIR     --out_dir EVALUATED_DATA_ALL     --job_id ${SLURM_ARRAY_TASK_ID}     --exclude interactions     --n_jobs ${SLURM_ARRAY_TASK_COUNT}     --overwrite-existing
```

### Postprocessing metrics

```bash
python -m sbdd_metrics.postprocess_metrics     --in_dir EVALUATED_DATA_ALL     --out_dir EVALUATED_DATA     --trainingPDB_dir TRAINING_PDB_DIR     --reference_smiles REFERENCE_SMILES
```

### Comparing multiple experiments

Results from multiple experiments can be aggregated into a single CSV table.

Expected directory structure:

```
example_folder/
 ├─ exp1/
 │   ├─ exp1_samples
 │   └─ exp1_metrics
 └─ exp2/
     ├─ exp2_samples
     └─ exp2_metrics
```

Run:

```bash
python -m sbdd_metrics.compare_models example_folder
```

---

# Graphics

The repository also contains scripts used to reproduce the figures in the thesis.

## Waterfall survival plots

```bash
python -m sbdd_visualize.posebusters_waterfall_plots     --exp_name example1 example2     --labels 1 2     --out posebusters_waterfall_plots     --global_only
```

Up to six experiments can be compared simultaneously.

## Clash pattern analysis

```bash
python -m sbdd_visualize.clash_eval     --save_path example_folder/exp1_metrics/
```

The results are written to the corresponding metrics folder.

---


# Citation

If you use this repository, please cite it as software:

```
@software{Baertschi2026GeomDiffusionGuidance,
  author = {Bärtschi, Pascal},
  title = {Inference-Time Guidance in Pocket-Conditioned Molecular Diffusion Models},
  year = {2026},
  url = {https://github.com/pascalbartschi/geom-diffusion-guidance}
}
```
