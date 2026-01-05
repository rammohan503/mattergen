# MatterGen: Complete Architecture & Workflow Guide

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture Overview](#architecture-overview)
3. [Core Technologies & Libraries](#core-technologies--libraries)
4. [Data Flow & Processing](#data-flow--processing)
5. [Key Classes & Components](#key-classes--components)
6. [Configuration System (Hydra)](#configuration-system-hydra)
7. [Execution Workflows](#execution-workflows)
8. [Important Files & Their Roles](#important-files--their-roles)

---

## Project Overview

**MatterGen** is a generative diffusion model for inorganic materials design that:
- Generates crystal structures across the periodic table
- Can be fine-tuned to condition on various properties (band gap, chemical system, space group, etc.)
- Uses graph neural networks to process crystal data
- Employs denoising diffusion probabilistic models (DDPM) for generation

### Key Capabilities
```
TRAIN      → Learn from crystal database
GENERATE   → Sample new crystal structures
EVALUATE   → Assess quality of generated materials
FINETUNE   → Adapt base model to specific properties
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    MatterGen Full Stack                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐   │
│  │          ENTRY POINTS (scripts/)                         │   │
│  │  generate.py, train/run.py, finetune.py, evaluate.py     │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│  ┌────────────────────────▼─────────────────────────────────┐   │
│  │      HYDRA CONFIGURATION MANAGER (conf/)                 │   │
│  │  - Merges YAML configs with CLI arguments                │   │
│  │  - Instantiates DictConfig objects                       │   │
│  │  - Handles composable configuration                      │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│  ┌────────────────────────▼─────────────────────────────────┐   │
│  │    GENERATOR/TRAINING ORCHESTRATOR                       │   │
│  │  CrystalGenerator | DiffusionLightningModule             │   │
│  └─────┬──────────────────┬──────────────────┬──────────────┘   │
│        │                  │                  │                  │
│  ┌─────▼──────────┐ ┌────▼──────────┐ ┌─────▼─────────────┐     │
│  │  DATA LAYER    │ │ DIFFUSION     │ │ GEM-NET ENCODER   │     │
│  │ (Dataset)      │ │ MODULE        │ │ (GeoMAN)          │     │
│  │                │ │ (Physics)     │ │ (Graph NN)        │     │
│  ├────────────────┤ ├───────────────┤ ├───────────────────┤     │
│  │ • Dataset      │ │ • Corruption  │ │ • GemNetT         │     │
│  │ • ChemGraph    │ │ • Scoring     │ │ • Edge Conv       │     │
│  │ • Collate      │ │ • Sampling    │ │ • Spherical Basis │     │
│  │ • Transforms   │ │               │ │ • Radial Basis    │     │
│  │ • Properties   │ │               │ │                   │     │
│  └────────────────┘ └───────────────┘ └───────────────────┘     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  PYTORCH LIGHTNING (Training Loop Management)            │   │
│  │  DiffusionLightningModule + Trainer                      │   │
│  │  - Training/Validation/Testing steps                     │   │
│  │  - Checkpoint saving/loading                             │   │
│  │  - EMA (Exponential Moving Average)                      │   │
│  │  - Distributed training (DDP)                            │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  OUTPUT LAYER                                            │   │
│  │  - CIF files, extxyz, trajectories                       │   │
│  │  - Metrics (stability, diversity, coverage)              │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Technologies & Libraries

### 1. **PyTorch Geometric (PyG)**
- **Purpose**: Graph Neural Network framework
- **Role**: Basis for batch processing and message passing
- **Usage**:
  - `ChemGraph` extends `torch_geometric.data.Data`
  - `Batch` for collating multiple graphs
  - Edge creation with `radius_graph_pbc` (periodic boundary conditions)

```python
# ChemGraph is a PyG Data object with:
ChemGraph(
    atomic_numbers,      # Node features (atom types)
    pos,                 # Node positions [num_atoms, 3]
    cell,                # Periodic boundary [1, 3, 3]
    edge_index,          # Graph connectivity [2, num_edges]
    edge_attr,           # Edge features
    **properties         # Additional properties (band gap, etc.)
)
```

### 2. **PyTorch Lightning**
- **Purpose**: High-level training framework
- **Role**: Manages training loops, validation, checkpointing
- **Key Components**:
  - `DiffusionLightningModule`: Wraps the diffusion model
  - `Trainer`: Orchestrates training (epochs, devices, logging)
  - `LightningDataModule`: Handles data loading

```python
# Training loop abstraction:
class DiffusionLightningModule(pl.LightningModule):
    def training_step(batch, batch_idx):
        loss = model.calc_loss(batch)
        return loss
    
    def validation_step(batch, batch_idx):
        # Evaluation metrics
        pass
```

### 3. **Hydra (Configuration Management)**
- **Purpose**: Flexible, composable configuration system
- **Why It's Important**:
  - Avoids hardcoding hyperparameters
  - CLI overrides: `python run.py learning_rate=0.001`
  - Config composition: Combine multiple YAML files
  - Automatic output directory management
  - Type safety with dataclasses

```yaml
# conf/default.yaml
defaults:
  - data_module: mp_20          # Which dataset
  - trainer: default             # Training config
  - lightning_module: default    # Model config
  - lightning_module/diffusion_module: default  # Diffusion config

hydra:
  run:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

### 4. **GemNet (Graph Equivariant Message Passing Network)**
- **Purpose**: Graph neural network for materials
- **Key Features**:
  - **Equivariance**: Respects rotational/translational symmetries
  - **Triplet Message Passing**: Node-edge-node interactions
  - **Radial Basis Functions (RBF)**: Smooth distance encoding
  - **Spherical Basis**: Angular information encoding
  - **Periodic Boundary Conditions**: Native PBC support

```python
class GemNetT(nn.Module):
    """
    GemNet-T: Triplets-Only Variant
    
    Processes:
    1. Atomic numbers → embeddings
    2. Distances → RBF encoding
    3. Angular info → Spherical basis
    4. Message passing: Node ↔ Edge ↔ Node
    5. Outputs: Energy, forces, stress
    """
```

### 5. **Diffusion Models (DDPM Framework)**
- **Purpose**: Generative process via iterative denoising
- **Core Idea**:
  ```
  Clean data → Add noise → Learn to denoise → Generate
  
  Forward:   x₀ ─noise→ x₁ ─noise→ ... ─noise→ xₜ
  Reverse:   xₜ ─denoise→ ... ─denoise→ x₁ ─denoise→ x₀
  ```
- **Components**:
  - **Corruption**: Add noise to data (discrete + continuous)
  - **Score Model**: Learn ∇log p(x_t) (gradient of log probability)
  - **Sampling**: Use predictor-corrector to reverse noise

### 6. **PyMatGen**
- **Purpose**: Materials science data structures and utilities
- **Usage**:
  - `Structure`: Crystal structure representation
  - `CifParser`: Read/write CIF files
  - `SpaceGroup`: Symmetry operations
  - `Lattice`: Crystallographic computations

---

## Data Flow & Processing

### **Complete Data Pipeline**

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. DATA LOADING & PREPARATION                                   │
├─────────────────────────────────────────────────────────────────┤

   CSV/CIF Files (POSCAR, CIF format)
        ↓
   CifParser (pymatgen)
        ↓
   Structure objects (pymatgen.core.Structure)
        │
        ├─ Primitive structure extraction
        ├─ Niggli reduction (standardize lattice)
        ├─ Extract fractional coordinates
        └─ Extract lattice matrix (3×3)
        ↓
   structures_to_numpy()
        │
        ├─ Flatten all atoms into single array
        ├─ Create index_offset mapping (structure→atoms)
        ├─ Store properties as separate arrays
        └─ Validate property dimensions
        ↓
   Cache to Disk
        ├─ pos.npy          [total_atoms, 3]
        ├─ cell.npy         [num_structures, 3, 3]
        ├─ atomic_numbers.npy [total_atoms]
        ├─ num_atoms.npy    [num_structures]
        ├─ structure_id.npy [num_structures]
        └─ property.json    [num_structures]

└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 2. DATASET BUILDING (CrystalDatasetBuilder)                     │
├─────────────────────────────────────────────────────────────────┤

   CrystalDatasetBuilder.from_cache_path()
        ↓
   Load numpy arrays (lazy via @cached_property)
        ↓
   Load properties (band gap, space group, etc.)
        ↓
   Create CrystalDataset instance
        │
        ├─ pos: [total_atoms, 3]
        ├─ cell: [num_structures, 3, 3]
        ├─ atomic_numbers: [total_atoms]
        ├─ num_atoms: [num_structures]
        ├─ structure_id: [num_structures]
        └─ properties: dict[prop_name → values]

└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 3. PER-SAMPLE RETRIEVAL & TRANSFORMATION                        │
├─────────────────────────────────────────────────────────────────┤

   DataLoader[CrystalDataset].__getitem__(idx)
        ↓
   Use index_offset[idx] to get atom range
        ↓
   Extract atoms for this structure:
        ├─ positions = pos[offset:offset+num_atoms]
        ├─ atomic_nums = atomic_numbers[offset:offset+num_atoms]
        ├─ cell = cell[idx]
        └─ properties = {prop: values[idx] for prop}
        ↓
   Create ChemGraph (PyG Data object)
        ├─ atomic_numbers (node features)
        ├─ pos (node coordinates, modulo 1)
        ├─ cell (graph-level property)
        ├─ num_atoms
        └─ properties (space_group, band_gap, etc.)
        ↓
   Apply per-sample transforms
        ├─ Normalize/symmetrize lattice
        ├─ Data augmentation
        └─ Coordinate transformations
        ↓
   Return: ChemGraph (single structure)

└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 4. BATCH COLLATION (collate.py)                                 │
├─────────────────────────────────────────────────────────────────┤

   Multiple ChemGraphs: [graph₁, graph₂, ..., graphₙ]
        ↓
   collate() function (PyG Batch)
        │
        ├─ Concatenate all pos: [total_nodes, 3]
        ├─ Concatenate all atomic_numbers: [total_nodes]
        ├─ Create batch indices for nodes
        ├─ Stack cells: [batch_size, 3, 3]
        ├─ Create batch indices for edges
        └─ Concatenate properties
        ↓
   ChemGraphBatch (Dynamic PyG Batch subclass)
        ├─ pos: [total_atoms_in_batch, 3]
        ├─ atomic_numbers: [total_atoms_in_batch]
        ├─ batch: [total_atoms_in_batch] → structure ID
        ├─ cell: [batch_size, 3, 3]
        ├─ num_atoms: [batch_size]
        ├─ num_graphs: batch_size
        └─ properties_batch indices
        ↓
   Optional: Build edge graphs
        ├─ radius_graph_pbc(): KNN with PBC
        ├─ edge_index: [2, num_edges]
        └─ edge_attr: distances, vectors, etc.

└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 5. DIFFUSION MODEL FORWARD PASS                                 │
├─────────────────────────────────────────────────────────────────┤

   ChemGraphBatch → DiffusionLightningModule
        ↓
   ┌─────────────────────────────────────────┐
   │ TRAINING (training_step)                │
   ├─────────────────────────────────────────┤
   │                                         │
   │ Step 1: Corruption (Add Noise)          │
   │ ─────────────────────────────────       │
   │  Noisy pos = pos + α·ε                  │
   │  Noisy atomic_nums = discrete corrupt   │
   │  Noisy cell = cell + β·ε                │
   │  Sample timestep t ∈ [0,T]              │
   │                                         │
   │ Step 2: Score Model (GemNet)            │
   │ ─────────────────────────────────       │
   │  pred_atom_types = ScoreModel(noisy, t) │
   │  pred_pos_noise = ScoreModel(noisy, t)  │
   │  pred_cell_noise = ScoreModel(noisy, t) │
   │                                         │
   │ Step 3: Loss Computation                │
   │ ─────────────────────────────────       │
   │  L = MSE(pred_noise, true_noise)        │
   │  + cross_entropy(atom_logits, true_atoms)
   │  + property_matching_loss               │
   │                                         │
   │ Step 4: Backward Pass (PyTorch)         │
   │ ─────────────────────────────────       │
   │  ∇θ L → Update weights                  │
   │                                         │
   └─────────────────────────────────────────┘
        ↓
   Return: loss, metrics

└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 6. GENERATION (INFERENCE)                                       │
├─────────────────────────────────────────────────────────────────┤

   CrystalGenerator.generate()
        ↓
   Load checkpoint (pretrained weights)
        ↓
   ConditionLoader (creates conditioning batches)
        ├─ Num atoms distribution
        ├─ Properties (e.g., "band_gap > 2.0")
        └─ Chemical system constraints
        ↓
   PredictorCorrector Sampler
        │
        ├─ Start: xₜ ∈ 𝒩(0, I)  [random noise]
        │
        ├─ For t = T to 0:
        │    │
        │    ├─ Predictor step:
        │    │    s(xₜ, t) = ScoreModel(xₜ, t)
        │    │    xₜ₋₁ = xₜ + drift(s) + diffusion
        │    │
        │    └─ Corrector step:
        │         Langevin dynamics: refine xₜ₋₁
        │
        └─ Final: x₀ ≈ clean structure
        ↓
   Convert to Cartesian coordinates
        ↓
   Post-process:
   ├─ Convert to Å units
   ├─ Create Structure object (pymatgen)
   └─ Validate (check PBC, etc.)
        ↓
   Save:
   ├─ CIF files (human-readable)
   ├─ extxyz (ASE format)
   ├─ Trajectories (full denoising path)
   └─ Metadata JSON

└─────────────────────────────────────────────────────────────────┘
```

---

## Key Classes & Components

### **1. ChemGraph (Atomic Structure Representation)**
```python
class ChemGraph(torch_geometric.data.Data):
    """
    PyG Data object for crystal structures.
    
    Attributes:
    - atomic_numbers: [num_atoms] - atom type (1-indexed)
    - pos: [num_atoms, 3] - fractional coordinates
    - cell: [1, 3, 3] - lattice matrix
    - edge_index: [2, num_edges] - graph edges
    - num_atoms: scalar - atoms in structure
    - num_nodes: scalar - same as num_atoms (PyG convention)
    - **properties: band_gap, space_group, etc.
    
    Methods:
    - replace(**kwargs): Create copy with updated fields
    - get_batch_idx(field_name): Get batch indices
    """
```

**Why Frozen?**
```python
def __setattr__(self, attr, value):
    if self.__dict__.get("_frozen", False):
        raise AttributeError("Use replace() instead")
```
Prevents accidental mutations; use `replace()` for immutability.

### **2. CrystalDataset (Data Loading)**
```python
class CrystalDataset(BaseDataset):
    """
    Efficient numpy-based dataset with flattened arrays.
    
    Storage:
    - pos: [total_atoms, 3] (single flat array)
    - atomic_numbers: [total_atoms]
    - cell: [num_structures, 3, 3]
    - num_atoms: [num_structures] (per-structure counts)
    - index_offset: [num_structures] → atom indices
    
    Access Pattern:
    __getitem__(idx):
        offset = index_offset[idx]
        count = num_atoms[idx]
        structure_atoms = pos[offset:offset+count]
    
    Methods:
    - subset(indices): Create subset dataset
    - repeat(n): Duplicate dataset n times
    - get_properties_dict(idx): Return properties as tensors
    """
```

### **3. CrystalDatasetBuilder (Factory Pattern)**
```python
class CrystalDatasetBuilder:
    """
    Manages dataset loading, caching, and property management.
    
    Workflow:
    1. from_csv() - Parse CIF/POSCAR files
    2. structures_to_numpy() - Convert to flat arrays
    3. Cache to disk - Save .npy and .json
    4. from_cache_path() - Lazy reload
    5. build() - Instantiate CrystalDataset
    
    Properties:
    - Sparse support: Missing properties → NaN
    - Dynamic addition: add_property_to_cache()
    - Validation: Check dimensions match
    """
```

### **4. DiffusionLightningModule (Training Manager)**
```python
class DiffusionLightningModule(pl.LightningModule):
    """
    PyTorch Lightning wrapper for diffusion model.
    
    Attributes:
    - diffusion_module: Core diffusion logic
    - optimizer_partial: Optimizer factory
    - scheduler_partials: LR scheduler factories
    
    Methods:
    - training_step(batch, idx): Loss calculation
    - validation_step(batch, idx): Evaluation
    - configure_optimizers(): Setup Adam, schedulers
    - load_from_checkpoint(): Restore from saved state
    
    Features:
    - EMA (Exponential Moving Average) for stability
    - Automatic checkpoint saving (best, last)
    - Multi-GPU support (DDP)
    - Distributed validation
    """
```

### **5. DiffusionModule (Core Physics)**
```python
class DiffusionModule(nn.Module):
    """
    Denoising diffusion probabilistic model.
    
    Components:
    - model: ScoreModel (GemNet) - learns ∇log p(x)
    - corruption: MultiCorruption - defines noise schedule
    - loss_fn: Loss function
    - timestep_sampler: Sample t ∈ [0, T]
    
    Process:
    1. Forward: x₀ → noisy_x_t (via corruption)
    2. Score: ŝ = model(noisy_x_t, t)
    3. Loss: L = ||ŝ - ∇log p(x_t||x₀)||²
    4. Optimize: θ ← θ - ∇θ L
    
    Methods:
    - calc_loss(batch): Compute training loss
    - _corrupt_batch(): Add noise
    """
```

### **6. GemNetT (Score Model/Denoiser)**
```python
class GemNetT(nn.Module):
    """
    Graph Equivariant Message Passing Network.
    
    Architecture:
    INPUT: noisy structure (pos, cell, atomic_nums) + timestep
    
    ├─ Timestep Encoding
    │  └─ sin/cos positional encoding of t
    │
    ├─ Atom Embedding
    │  └─ atomic_number → vector
    │
    ├─ Edge Creation
    │  └─ radius_graph_pbc (k-nearest neighbors + PBC)
    │
    ├─ Interaction Blocks (stacked)
    │  ├─ Radial Basis Functions (RBF)
    │  │  └─ Encode distances smoothly
    │  ├─ Spherical Basis
    │  │  └─ Encode angles
    │  └─ Triplet Message Passing
    │     ├─ Node → Edge messages
    │     └─ Edge → Node aggregation
    │
    └─ Output Block
       ├─ Predict noise in positions
       ├─ Predict noise in cell
       ├─ Predict logits for atom types
       └─ Predict forces/stress (optional)
    
    Key Properties:
    - Equivariance: E(R·x) = R·E(x) for rotations R
    - Covariance: E(x + τ) = E(x) + τ (translation)
    - Periodic: Handles PBC automatically
    """
```

### **7. Collate Function (Batching)**
```python
def collate(pytree: PyTree[ChemGraph]) -> ChemGraphBatch:
    """
    Merge multiple ChemGraphs into batch using PyG.
    
    Input: [graph₁, graph₂, ..., graphₙ]
    
    Process:
    1. Concatenate node features
       pos: [n₁+n₂+...+nₙ, 3]
       atomic_numbers: [n₁+n₂+...+nₙ]
    
    2. Create batch indices
       batch: [n₁+n₂+...+nₙ] → which graph each atom belongs to
    
    3. Stack graph-level features
       cell: [n, 3, 3] → [batch_size, 3, 3]
       num_atoms: [n] → [batch_size]
    
    4. Optional: Build edge graphs
       radius_graph_pbc + periodic distance matrix
    
    Result: ChemGraphBatch object with batch indices for:
    - Node attributes (pos, atomic_numbers)
    - Edge attributes
    - Graph attributes (cell, properties)
    """
```

---

## Configuration System (Hydra)

### **Hydra: Why It Matters**

Hydra provides **declarative configuration management** instead of scattered hyperparameters:

```bash
# Without Hydra: hardcoded in code
python train.py  # hidden hyperparams in code

# With Hydra: explicit and overridable
python train.py learning_rate=0.001 batch_size=32 dataset=mp_20
```

### **Configuration Hierarchy**

```
mattergen/conf/
├── default.yaml              # Main entry point
│   └── Specifies defaults for all subsystems
├── finetune.yaml             # Fine-tuning overrides
├── csp.yaml                  # Crystal structure prediction mode
├── data_module/
│   ├── mp_20.yaml            # MP-20 dataset config
│   ├── alex_mp_20.yaml       # Alex-MP-20 dataset config
│   └── custom.yaml           # Custom dataset config
├── lightning_module/
│   ├── default.yaml          # Base model config
│   └── diffusion_module/
│       ├── default.yaml      # Diffusion config
│       ├── model/
│       │   ├── mattergen.yaml   # GemNet params
│       │   └── baseline.yaml
│       └── corruption/
│           └── default.yaml  # Noise schedule
└── trainer/
    └── default.yaml          # PyTorch Lightning Trainer config
```

### **Config Resolution Process**

```
1. Load default.yaml
2. Load all defaults: (data_module, trainer, lightning_module, ...)
3. Merge YAML files → base config
4. Parse CLI args → overrides
5. Apply overrides to base config
6. Validate against Config dataclass
7. Instantiate all objects via Hydra
```

### **Example: default.yaml**
```yaml
hydra:
  run:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}  # Dynamic output dir

auto_resume: True

defaults:
  - data_module: mp_20                    # Load mp_20 dataset config
  - trainer: default                      # Default trainer settings
  - lightning_module: default             # Default model config
  - lightning_module/diffusion_module: default  # Default diffusion
  - lightning_module/diffusion_module/model: mattergen  # GemNet params
  - lightning_module/diffusion_module/corruption: default  # Noise schedule
```

### **Hydra Instantiation**

Hydra converts YAML → Python objects:

```yaml
# conf/data_module/mp_20.yaml
_target_: mattergen.common.data.data_module.CrystalDataModule
dataset_name: mp_20
split: train
batch_size: 64
num_workers: 4
```

```python
# In code:
from hydra.utils import instantiate

data_module = instantiate(cfg.data_module)
# → Creates: CrystalDataModule(dataset_name="mp_20", ...)
```

### **Key Hydra Features Used in MatterGen**

| Feature | Usage |
|---------|-------|
| **Composition** | Combine configs from different domains |
| **CLI Overrides** | `python train.py learning_rate=0.001` |
| **Interpolation** | `${other_key}` references within YAML |
| **Defaults List** | Order matters - later overrides earlier |
| **Instantiation** | Convert configs → Python objects |
| **Output Dir Management** | Auto-create `outputs/YYYY-MM-DD/HH-MM-SS/` |
| **Config Validation** | Type-checked with dataclasses |

---

## Execution Workflows

### **1. TRAIN Workflow**

```
$ python mattergen/scripts/run.py [config_overrides]

1. INITIALIZATION
   ├─ Hydra loads default.yaml + overrides
   ├─ Config merged and validated
   └─ Output directory created
        ↓

2. DATA MODULE SETUP
   ├─ instantiate(cfg.data_module)
   ├─ CrystalDataModule created
   │  ├─ setup(stage='fit')
   │  ├─ Load training dataset
   │  ├─ Load validation dataset
   │  └─ Create DataLoaders
   └─ Batches ready for training
        ↓

3. MODEL INITIALIZATION
   ├─ instantiate(cfg.lightning_module)
   ├─ Creates DiffusionLightningModule
   │  ├─ GemNetT score model
   │  ├─ MultiCorruption (noise schedule)
   │  ├─ DiffusionModule (forward diffusion)
   │  └─ Loss functions
   └─ Model on device (GPU/CPU)
        ↓

4. TRAINER SETUP
   ├─ instantiate(cfg.trainer)
   ├─ PyTorch Lightning Trainer
   │  ├─ Num epochs, devices, precision
   │  ├─ Checkpoint callbacks
   │  ├─ Validation frequency
   │  └─ EMA callbacks
   └─ Logger setup (TensorBoard, Weights&Biases)
        ↓

5. TRAINING LOOP
   ├─ For epoch = 0 to num_epochs:
   │  │
   │  ├─ FOR EACH BATCH:
   │  │  │
   │  │  ├─ training_step(batch):
   │  │  │  ├─ Batch collation
   │  │  │  ├─ Forward pass (add noise + denoise)
   │  │  │  ├─ Loss computation
   │  │  │  ├─ Backward pass
   │  │  │  └─ Optimizer.step()
   │  │  │
   │  │  └─ Log training metrics
   │  │
   │  ├─ VALIDATION (every N steps):
   │  │  │
   │  │  ├─ FOR EACH VAL BATCH:
   │  │  │  ├─ validation_step(batch)
   │  │  │  ├─ Compute validation loss
   │  │  │  └─ Track metrics
   │  │  │
   │  │  ├─ Aggregate validation metrics
   │  │  │
   │  │  └─ Save best checkpoint
   │  │
   │  └─ Update learning rate scheduler
   │
   └─ Save final checkpoint
        ↓

6. POST-TRAINING
   ├─ Save config.yaml in checkpoint dir
   ├─ Save training metrics
   ├─ Create checkpoint archive
   └─ Training complete!

KEY FILES INVOLVED:
├─ mattergen/scripts/run.py          [Entry point]
├─ mattergen/diffusion/run.py        [Main training logic]
├─ mattergen/diffusion/lightning_module.py  [Training step]
├─ mattergen/diffusion/diffusion_module.py  [Loss computation]
├─ mattergen/common/data/data_module.py    [Data loading]
├─ mattergen/common/data/dataset.py        [Dataset class]
└─ mattergen/conf/                   [Config files]
```

### **2. GENERATE Workflow**

```
$ mattergen-generate results/ --pretrained-name=mattergen_base --batch_size=16

1. CHECKPOINT LOADING
   ├─ Load checkpoint from:
   │  ├─ Hugging Face Hub (if pretrained-name)
   │  └─ Or local path (if model_path)
   │
   ├─ MatterGenCheckpointInfo.from_hf_hub("mattergen_base")
   │  ├─ Download from HuggingFace
   │  └─ Extract config.yaml
   │
   └─ DiffusionLightningModule.load_from_checkpoint()
      ├─ Load state_dict
      ├─ Reconstruct model architecture
      └─ Model in eval mode
        ↓

2. SAMPLING CONFIGURATION
   ├─ Load sampling config (default.yaml)
   ├─ ConditionLoader setup:
   │  ├─ Num atoms distribution (if unconditional)
   │  └─ Conditioning info (if conditional)
   │
   └─ Create batches:
      ├─ Batch 1: 16 structures
      ├─ Batch 2: 16 structures
      └─ ... (num_batches times)
        ↓

3. SAMPLING SETUP
   ├─ PredictorCorrector sampler
   │  ├─ Timesteps: 50, 100, 250 (configurable)
   │  └─ Noise schedule: linear, quadratic, etc.
   │
   └─ Optional guidance:
      ├─ Classifier-free guidance
      ├─ Scaling factor β
      └─ Property constraints
        ↓

4. DENOISING LOOP (Per Batch)
   ├─ Initialize: xₜ ~ 𝒩(0, I)  [random noise]
   │  ├─ Positions: [batch_size*avg_atoms, 3]
   │  ├─ Cell: [batch_size, 3, 3]
   │  └─ Atomic numbers: [batch_size*avg_atoms]
   │
   ├─ FOR t = T down to 0 (descending):
   │  │
   │  ├─ PREDICTOR STEP (Reverse SDE)
   │  │  ├─ Score = model(xₜ, t)
   │  │  │  (GemNet forward pass)
   │  │  │
   │  │  ├─ Update positions:
   │  │  │  xₜ₋₁ = xₜ + drift*dt + σ*dw
   │  │  │
   │  │  ├─ Update cell:
   │  │  │  cellₜ₋₁ = cellₜ + drift*dt
   │  │  │
   │  │  ├─ Update atom types:
   │  │  │  logits → sample with temperature
   │  │  │
   │  │  └─ Optional guidance step
   │  │
   │  ├─ CORRECTOR STEP (Langevin)
   │  │  └─ Refine xₜ₋₁ via auxiliary SDE
   │  │
   │  └─ Log trajectory (if record_trajectories=True)
   │
   └─ Final: x₀ ≈ clean structure
        ↓

5. POST-PROCESSING
   ├─ Convert to Cartesian coordinates
   │  └─ pos_cart = pos_frac @ cell
   │
   ├─ Wrap to unit cell (0 ≤ pos < 1)
   │
   ├─ Create Structure objects (pymatgen)
   │  ├─ lattice = cell
   │  ├─ species = [elem(Z) for Z in atomic_numbers]
   │  └─ coords = pos_cart
   │
   └─ Validate structures
      ├─ Check PBC
      ├─ Check atom overlaps
      └─ Remove invalid structures
        ↓

6. OUTPUT SAVING
   ├─ Generated structures:
   │  ├─ CIF files (one per structure)
   │  ├─ extxyz format (all in one file)
   │  └─ Zipped archive
   │
   ├─ Trajectories (if record_trajectories=True):
   │  ├─ Full denoising path for each structure
   │  └─ Time-evolved positions
   │
   └─ Metadata:
      ├─ generation_config.json
      ├─ sampled_properties.json
      └─ statistics.json

KEY FILES INVOLVED:
├─ mattergen/scripts/generate.py            [Entry point]
├─ mattergen/generator.py                   [Main generation logic]
├─ mattergen/diffusion/sampling/pc_sampler.py  [Denoising loop]
├─ mattergen/denoiser.py                    [GemNet wrapper]
├─ mattergen/common/utils/eval_utils.py    [Post-processing]
└─ mattergen/conf/sampling_conf/            [Sampling config]
```

### **3. FINETUNE Workflow**

```
$ python mattergen/scripts/finetune.py \
    --pretrained-name=mattergen_base \
    --property-name=band_gap \
    --train-data-path=/path/to/data.csv

1. BASE MODEL LOADING
   ├─ Load pretrained checkpoint
   ├─ MatterGenCheckpointInfo.from_hf_hub()
   │  ├─ Download weights
   │  └─ Extract config
   │
   └─ Parse original config:
      ├─ GemNet architecture
      ├─ Corruption schedule
      └─ Loss functions
        ↓

2. ADAPTER INITIALIZATION
   ├─ Create GemNetTCtrl (controlled variant)
   │  ├─ Same as GemNetT + adapter layers
   │  └─ Learnable property embeddings
   │
   ├─ Configure property to condition on
   │  └─ e.g., band_gap with embedding size 16
   │
   ├─ Transfer weights from pretrained:
   │  ├─ Copy matching parameters
   │  ├─ New adapter layers initialized randomly
   │  └─ Freeze or fine-tune base weights
   │
   └─ Setup new property embeddings
      ├─ PropertyEmbedding module
      └─ Map property values → vectors
        ↓

3. DATASET PREPARATION
   ├─ Load CSV with:
   │  ├─ CIF/POSCAR structures
   │  ├─ Material IDs
   │  └─ Property values (band_gap, etc.)
   │
   ├─ structures_to_numpy()
   │  └─ Convert to flat arrays
   │
   ├─ Add property to cache
   │  └─ PropertyValues.to_json()
   │
   └─ Create DataLoader
      ├─ Batch structures
      └─ Include property labels
        ↓

4. TRAINING LOOP (Similar to train, but)
   ├─ Lower learning rate (transfer learning)
   │  └─ Usually 10x smaller
   │
   ├─ Optional: Freeze base layers
   │  ├─ Only train property embeddings
   │  └─ Or train all with lower LR
   │
   ├─ New loss includes:
   │  ├─ Reconstruction loss (as before)
   │  ├─ Property matching loss
   │  │  L_prop = ||model(x, t) - target||²
   │  └─ Combined loss: L_total = L_recon + λ*L_prop
   │
   └─ Validation on held-out property data
        ↓

5. CHECKPOINT SAVING
   ├─ Save adapter weights
   ├─ Save property embeddings
   ├─ Save config with new property
   └─ New checkpoint ready for generation!

KEY FILES INVOLVED:
├─ mattergen/scripts/finetune.py            [Entry point]
├─ mattergen/adapter.py                     [Adapter logic]
├─ mattergen/property_embeddings.py         [Property conditioning]
├─ mattergen/common/data/data_module.py    [Data loading]
└─ mattergen/conf/finetune.yaml            [Config]
```

### **4. EVALUATE Workflow**

```
$ mattergen-evaluate results/generated.extxyz \
    --relax \
    --reference-dataset-path=/path/to/mp_20

1. STRUCTURE LOADING
   ├─ Load from:
   │  ├─ CIF files (directory)
   │  ├─ extxyz (single file)
   │  └─ ASE trajectory
   │
   └─ Parse with pymatgen/ASE
      └─ Create Structure objects
        ↓

2. OPTIONAL: STRUCTURE RELAXATION
   ├─ Use MACE or MatterSim potential
   │
   ├─ Relax atomic positions
   │  └─ Minimize forces
   │
   ├─ Relax cell
   │  └─ Minimize stress
   │
   └─ Extract relaxed energy
        ↓

3. EVALUATION METRICS
   ├─ Validity:
   │  ├─ Check for overlapping atoms
   │  ├─ Check composition feasibility
   │  └─ Check lattice parameters
   │
   ├─ Stability:
   │  ├─ Energy above hull (if reference provided)
   │  ├─ Phonon frequencies (if computed)
   │  └─ Formation energy
   │
   ├─ Diversity:
   │  ├─ Maximum pairwise distance (structure distance)
   │  ├─ Composition distribution
   │  └─ Crystal system distribution
   │
   ├─ Novelty:
   │  ├─ Comparison with reference dataset
   │  ├─ Structure matching (tolerance)
   │  └─ Novel compositions
   │
   └─ Coverage:
      ├─ Distribution match with training set
      └─ Property prediction accuracy
        ↓

4. COMPARISON WITH REFERENCE
   ├─ Load reference dataset (MP-20, ICSD, etc.)
   │
   ├─ For each generated structure:
   │  ├─ Find nearest neighbors in reference
   │  ├─ Structure match (default: disordered)
   │  └─ Calculate metrics
   │
   └─ Aggregate statistics
        ↓

5. OUTPUT
   ├─ JSON with all metrics:
   │  ├─ num_valid
   │  ├─ num_stable (Ehull < threshold)
   │  ├─ avg_distance_to_reference
   │  ├─ num_duplicates
   │  └─ property_MAE (if properties provided)
   │
   ├─ CSV with per-structure metrics
   │
   └─ Optionally save relaxed structures

KEY FILES INVOLVED:
├─ mattergen/scripts/evaluate.py            [Entry point]
├─ mattergen/evaluation/evaluate.py         [Main logic]
├─ mattergen/evaluation/utils/metrics.py   [Metrics computation]
├─ mattergen/evaluation/utils/structure_matcher.py  [Matching]
└─ mattergen/common/utils/eval_utils.py    [Utilities]
```

---

## Important Files & Their Roles

### **Core Architecture Files**

| File | Purpose | Key Classes |
|------|---------|-------------|
| `dataset.py` | Data loading & caching | `CrystalDataset`, `CrystalDatasetBuilder` |
| `chemgraph.py` | Structure representation | `ChemGraph` |
| `collate.py` | Batch creation | `collate()` function |
| `diffusion_module.py` | Core diffusion physics | `DiffusionModule` |
| `lightning_module.py` | Training orchestration | `DiffusionLightningModule` |
| `denoiser.py` | Score model wrapper | `get_chemgraph_from_denoiser_output()` |
| `generator.py` | Generation orchestration | `CrystalGenerator` |
| `property_embeddings.py` | Property conditioning | `ChemicalSystemMultiHotEmbedding` |

### **Geometry & Physics Files**

| File | Purpose | Key Functions |
|------|---------|----------------|
| `gemnet.py` | Graph NN architecture | `GemNetT` class |
| `gemnet_ctrl.py` | Adaptive GemNet | `GemNetTCtrl` class |
| `layers/` | Low-level GNN layers | Radial basis, spherical basis, convolutions |
| `diffusion_module.py` | Corruption & scoring | `DiffusionModule`, `MultiCorruption` |
| `sampling/pc_sampler.py` | Denoising loop | `PredictorCorrector` |

### **Utility Files**

| File | Purpose | Key Utilities |
|------|---------|---------------|
| `data_utils.py` | Coordinate conversions | `frac_to_cart_coords_with_lattice()` |
| `eval_utils.py` | Post-processing | `make_structure()`, `save_structures()` |
| `globals.py` | Constants | `PROPERTY_SOURCE_IDS`, `SELECTED_ATOMIC_NUMBERS` |
| `structure_matcher.py` | Structure comparison | `DefaultDisorderedStructureMatcher` |

### **Configuration Files**

| File | Purpose |
|------|---------|
| `conf/default.yaml` | Main entry point config |
| `conf/data_module/*.yaml` | Dataset selection |
| `conf/lightning_module/*.yaml` | Model architecture |
| `conf/trainer/*.yaml` | Training parameters |
| `sampling_conf/*.yaml` | Generation parameters |

---

## Advanced Topics

### **Batch Processing with PyTorch Geometric**

```python
# Single structure (ChemGraph):
graph = ChemGraph(
    atomic_numbers=[6, 8, 1],      # 3 atoms
    pos=[[0.1, 0.2, 0.3],          # fractional coords
         [0.5, 0.5, 0.5],
         [0.0, 0.0, 0.0]],
    cell=[[5, 0, 0],               # lattice (angstroms)
          [0, 5, 0],
          [0, 0, 5]]
)

# Multiple structures in batch:
batch = collate([graph1, graph2, graph3])

# batch.pos: [n1+n2+n3, 3] - concatenated atoms
# batch.batch: [n1+n2+n3] - which structure each atom belongs to
# batch.num_graphs: 3

# Access by structure:
structure_1_pos = batch.pos[batch.batch == 0]  # All atoms in structure 1
```

### **Equivariance in GemNet**

```python
"""
GemNet maintains equivariance to:

1. Rotations (3D rotations in space)
   E(R·x) = R·E(x)
   If you rotate structure, embeddings rotate accordingly

2. Translations (doesn't matter, PBC handled)
   E(x + τ) ≈ E(x)  (modulo periodicity)

3. Permutations (order of atoms)
   E([x1, x2, x3]) = E([x3, x1, x2])
   Permutation invariant via message passing

This is CRITICAL for materials:
- Learned features respect physical symmetries
- Model generalizes better to unseen structures
- Forces, stresses are naturally covariant
"""
```

### **Property Conditioning Mechanism**

```python
# How properties steer generation:

1. PROPERTY EMBEDDING (during training):
   band_gap_value = 2.5 eV
   ↓
   band_gap_embedding = PropertyEmbedding(2.5)  # → [16]
   ↓
   model inputs this embedding at each GemNet layer

2. UNCONDITIONAL EMBEDDING (for flexibility):
   If band_gap is NOT specified, use:
   special_vector = torch.ones(16)  # all ones or learnable
   
   This allows model to learn: "with this vector, any band gap is OK"

3. CLASSIFIER-FREE GUIDANCE (during generation):
   Two predictions:
   - With condition: ŝ_cond = model(x, t, embed_condition)
   - Without condition: ŝ_uncond = model(x, t, embed_null)
   
   Blended: ŝ = ŝ_uncond + β·(ŝ_cond - ŝ_uncond)
   
   Higher β → stronger conditioning effect
```

### **Noise Schedule (Corruption)**

```python
# During training:
t ∈ [0, 1]  (normalized timestep)

Corruption adds noise:
x_t = √(ᾱ_t) · x_0 + √(1-ᾱ_t) · ε

where ᾱ_t = ∏(1-β_i)  is cumulative variance

β_t can be:
- Linear: β_t = 0.0001 + t * (0.02 - 0.0001)
- Quadratic: β_t = (0.0001 + t * (0.02 - 0.0001))²
- Cosine: β_t = sin²(π·t/2)

This schedule:
- Early (small t): mostly signal + little noise
- Late (large t): mostly noise + little signal

Model learns: "Here's noisy structure. Remove this much noise."
```

### **Loss Functions in MatterGen**

```python
# Multi-part loss:

L_total = L_positions + L_cell + L_atoms + L_properties

1. L_positions:
   MSE between predicted and true noise in coordinates
   L = ||σ_pred - σ_true||²

2. L_cell:
   MSE for lattice noise prediction
   L = ||cell_pred - cell_true||²

3. L_atoms:
   Cross-entropy for discrete atom types
   L = -Σ_i log(p_i[true_atom_type])

4. L_properties (if conditioning on properties):
   Match property predictions to target values
   L_prop = ||property_pred - target||²

Final: L = Σ α_i * L_i  (weighted sum)
```

---

## Summary: The Complete Picture

```
MatterGen creates materials by:

1. TRAINING:
   ├─ Load real crystal structures (MP-20, etc.)
   ├─ Add noise iteratively (forward diffusion)
   ├─ Train neural network to reverse noise (denoising)
   ├─ Use GemNet (respects physics symmetries)
   └─ Condition on properties (band gap, space group, etc.)

2. GENERATION:
   ├─ Start with pure random noise
   ├─ Iteratively remove noise (reverse diffusion)
   ├─ Use trained network to guide denoising
   ├─ Apply property constraints (classifier-free guidance)
   └─ End up with plausible new structures

3. WHY THIS WORKS:
   ├─ Diffusion ≈ Maximum likelihood learning (proven mathematically)
   ├─ GemNet respects physical symmetries
   ├─ Property conditioning allows controlled generation
   ├─ Iterative denoising = high quality
   └─ Probabilistic → uncertainty quantification

4. KEY ADVANTAGES:
   ├─ Works for arbitrary materials (not limited to test set)
   ├─ Conditionable on multiple properties
   ├─ Physics-aware (equivariant) architecture
   ├─ Fast inference (50-250 steps to denoise)
   └─ Publicly available pre-trained checkpoints
```

