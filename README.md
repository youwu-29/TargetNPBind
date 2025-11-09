# TargetNPBind: A Pharmacophore-Guided Dual-Stream Fusion Framework for Natural Product–Target Binding Affinity Prediction

This repository contains the official implementation of **TargetNPBind**, a computational framework for predicting natural product-target binding affinities.

## Model Architecture
![TargetNPBind Framework](model.png)

**TargetNPBind** features a dual-stream fusion framework that integrates 2D topological and 3D spatial information for accurate binding affinity prediction.

## Quick Start

### Using Pre-trained Model

1. **Install dependencies**:
```bash
conda create -n targetnpbind python=3.9
conda activate targetnpbind
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge rdkit
pip install dgl-cu118 scipy scikit-learn pandas numpy tqdm
```

2. **Run prediction with provided pre-trained model**:
```bash
python test.py
```
*The test script will load the pre-trained model from `results/train/model_weights.pth` and evaluate on the protein cold-start test set.*

## Training with Pre-processed Data

If you want to retrain the model using our pre-processed data files:

1. **Download the pre-processed data**:
   - Visit the Zenodo dataset: [DOI: 10.5281/zenodo.17562086](https://doi.org/10.5281/zenodo.17562086)
   - Download the `train.pt` and `test.pt` files

2. **Place files in the correct directory**:
   - Create the directory: `data/processed/`
   - Place the downloaded files in this directory:
     - `train.pt`
     - `test.pt`

3. **Train the model**:
```bash
python train.py
```
*This uses the pre-generated .pt files in `data/processed/` containing pharmacophore features, molecular graphs, and interaction graphs.*

## Complete Workflow (From Raw Data)

### Step 1: Download Data
Download TargetNP-4811 dataset from Zenodo ( [DOI: 10.5281/zenodo.17562086](https://doi.org/10.5281/zenodo.17562086)) and extract to `dataset/TargetNP-4811/`

### Step 2: Generate 3D Interaction Graphs
```bash
python generate_interaction_graphs.py
```
*Creates 3D atom-residue interaction graphs from molecular structures*
- Input: PDB and MOL2 files from TargetNP-4811
- Output: Interaction graphs in `data/interaction_graph/`

### Step 3: Preprocess Data
```bash
python create_data.py
```
*Processes SMILES, protein sequences, and integrates interaction graphs*
- Generates pharmacophore features from SMILES
- Combines 2D molecular graphs with 3D interaction graphs
- Creates PyTorch-ready datasets in `data/processed/`

### Step 4: Training
```bash
python train.py
```
*Trains the TargetNPBind model*

### Step 5: Testing
```bash
python test.py
```
*Evaluates model performance on test sets*

## Contact

For questions or issues, please contact: [wu.you@webmail.hzau.edu.cn]
