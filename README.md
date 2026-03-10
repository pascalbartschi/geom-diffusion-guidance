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

or concatenated to a dataset by: python Dataset.py --data_dir data_extracted_graphs_atom_level --save_path example_dataset_atom_level.pt --sidechains True

TODO: get the single_use_scripts back (before cleaning commit)



