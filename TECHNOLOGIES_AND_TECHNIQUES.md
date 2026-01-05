# MatterGen: Complete Technology & Technique Stack

**Purpose:** Comprehensive reference of every technology, algorithm, library, and method used in MatterGen with their specific roles.

**Date:** 31 December 2025  
**Organization:** By category (frameworks, algorithms, data structures, utilities)

---

## Table of Contents
1. [Deep Learning Frameworks](#1-deep-learning-frameworks)
2. [Graph Neural Networks & Architectures](#2-graph-neural-networks--architectures)
3. [Diffusion & Generative Models](#3-diffusion--generative-models)
4. [Data Structures & Processing](#4-data-structures--processing)
5. [Crystal Structure Libraries](#5-crystal-structure-libraries)
6. [Force Fields & Relaxation](#6-force-fields--relaxation)
7. [Configuration & Experiment Management](#7-configuration--experiment-management)
8. [Utilities & Supporting Libraries](#8-utilities--supporting-libraries)
9. [Training & Optimization](#9-training--optimization)
10. [Sampling & Inference](#10-sampling--inference)
11. [Metrics & Evaluation](#11-metrics--evaluation)
12. [Distributed Training & Parallelization](#12-distributed-training--parallelization)

---

## 1. Deep Learning Frameworks

### PyTorch (v2.2.1 Linux / v2.4.1 macOS)
**What it is:** Core tensor computation and automatic differentiation library  
**Used for:**
- Tensor operations (pos, atomic_numbers, cell matrices)
- Forward/backward passes in training loops
- GPU acceleration with CUDA
- Gradient accumulation and optimization

**Files:** Nearly all files in `mattergen/`  
**Key modules:**
- `torch.nn`: Neural network layers (Linear, Conv, GRU, etc.)
- `torch.optim`: Optimizers (Adam, AdamW)
- `torch.nn.functional`: Activation functions, losses
- `torch.utils.data`: DataLoader, Dataset base classes

**Example usage:**
```python
import torch
pos = torch.randn(32, 3)  # atom coordinates
atomic_numbers = torch.randint(1, 100, (32,))  # atomic Z
cell = torch.eye(3).unsqueeze(0)  # lattice matrix
```

---

### PyTorch Lightning (v1.9 or later)
**What it is:** High-level training framework built on PyTorch  
**Used for:**
- Organized training loops (`LightningModule`)
- Multi-GPU distributed training
- Checkpoint/callback management
- Early stopping, learning rate scheduling

**Files:**
- `mattergen/diffusion/lightning_module.py` — main training loop
- `mattergen/scripts/run.py` — trainer instantiation

**Key classes:**
- `Trainer` — high-level training orchestrator
- `LightningModule` — train_step, val_step, test_step
- `Callback` — checkpointing, early stopping

---

### PyTorch Geometric (v2.5+)
**What it is:** Graph neural network library  
**Used for:**
- Graph data structures (`Data`, `Batch`)
- Graph convolutions (GCNConv, etc.)
- Message passing utilities
- Neighbor sampling, graph operations
- Edge index creation and manipulation

**Files:**
- `mattergen/common/gemnet/gemnet.py` — GemNetT graph operations
- `mattergen/common/data/chemgraph.py` — ChemGraph extends `Data`
- `mattergen/diffusion/` — batch collation uses `Batch`

**Key components:**
- `Data` class — single graph representation
- `Batch` class — collated multiple graphs
- `scatter_add`, `scatter_mean` — aggregation operations
- Edge index format: `(2, num_edges)` with src/dst nodes

**Example:**
```python
from torch_geometric.data import Data, Batch
data = Data(x=h, edge_index=edge_idx, pos=pos)
batch = Batch.from_data_list([data1, data2, data3])  # per-atom batch index
```

---

## 2. Graph Neural Networks & Architectures

### GemNetT (Geometric Message-Passing Neural Network)
**What it is:** Equivariant GNN designed for molecular/crystalline systems  
**Used for:**
- **Backbone denoiser** in the diffusion model
- Predicting denoised coordinates, atom types, lattice
- Incorporating geometric features (distances, angles, triplets)

**Files:**
- `mattergen/common/gemnet/gemnet.py` — GemNetT implementation
- `mattergen/denoiser.py` — uses GemNetT
- `mattergen/common/gemnet/layers/` — building blocks (interaction blocks, atom updates, edge updates)

**Key features:**
- **Radial basis functions (RBF):** encode pairwise distances into learnable features
- **Angle embeddings:** triplet (angle) information for 3-body interactions
- **Equivariance:** handles rotations and permutations correctly
- **Directional messages:** aware of atomic geometry

**Input:**
- Node features (atom embeddings) `(num_atoms, d_h)`
- Edge indices `(2, num_edges)`
- Edge distances, angle features
- Positional encodings

**Output:**
- Updated node representations
- Coordinate deltas (pos predictions) `(num_atoms, 3)`
- Atom type logits `(num_atoms, n_types)`
- Cell/lattice updates

---

### Message Passing (Edge → Node)
**What it is:** Core mechanism in all GNNs for aggregating neighborhood information  
**Used for:**
- Propagating chemical/structural information between atoms
- Computing messages along bonds and angles
- Iterative refinement of node embeddings

**Algorithm:**
1. Compute message per edge: `m_{u→v} = mlp([h_u; e_attr])`
2. Aggregate at destination: `m_v = sum_{u in neighbors(v)} m_{u→v}`
3. Update node: `h_v' = gru(h_v, m_v)` or `h_v' = h_v + mlp([h_v; m_v])`

**Files:**
- `mattergen/common/gemnet/layers/interaction_block.py` — message computation
- `mattergen/common/gemnet/layers/atom_update_block.py` — aggregation/node update

**Libraries used:**
- `torch_scatter.scatter_add` — efficient aggregation (sum over neighbors)
- `torch_scatter.scatter_mean` — mean aggregation

**Example dimensions:**
- Messages: `(num_edges, 64)` → aggregated to `(num_atoms, 64)`
- Per-node aggregated: `(num_atoms, 64)` (total from all neighbors)

---

### Graphormer & Transformer-Style Graph Models
**What it is:** Attention-based (transformer) message passing  
**Used for:**
- Alternative attention mechanism for property conditioning
- Long-range dependency modeling
- Can be used in model variants

**Files:**
- `mattergen/conf/lightning_module/diffusion_module/model/` — model choices
- Referenced in property embedding fusion (alternative implementation)

**Key ideas:**
- Attention weights replace explicit message aggregation
- Structural biases: edges/distances become attention bias terms
- Positional encodings for graphs (shortest path, Laplacian, degree)

**Formula:**
- score_{i,j} = (Q_i · K_j) / sqrt(d) + bias_{i,j}
- `bias_{i,j}` from edge types, distances

**Advantages over GNN:**
- Full pairwise mixing vs local neighborhood
- Fewer layers needed for long-range effects
- More expressive but O(n²) complexity

---

### Graph Convolution (GCN/GCNConv)
**What it is:** Simplified message-passing layer  
**Used for:**
- Baseline comparisons (optional)
- Secondary feature refinement

**Not primary:** MatterGen uses GemNetT, not basic GCN  

**Files:** If used, would be in `mattergen/common/gemnet/`

---

## 3. Diffusion & Generative Models

### Continuous Diffusion (Gaussian)
**What it is:** Noise-based generative process for continuous variables  
**Used for:**
- Coordinates (pos): gradually add noise, then denoise
- Cell/lattice vectors: similar process
- Atomic positions in fractional coordinates

**Algorithm:**
1. Forward: x_t = sqrt(α_t) * x_0 + sqrt(1 - α_t) * ε, where ε ~ N(0,I)
2. Reverse: predict ε (noise) or x_0 (clean) at each timestep
3. Sample: x_{t-1} from predicted distribution, iteratively from t=T to 0

**Files:**
- `mattergen/diffusion/` — diffusion modules
- `mattergen/diffusion/diffusion_module.py` — base diffusion class
- `mattergen/diffusion/losses.py` — loss computation per field

**Key hyperparameters:**
- `betas`: noise schedule (linear, cosine, etc.)
- `num_diffusion_timesteps`: T (e.g., 1000)
- `loss_weight_` per field: balance pos vs type vs cell losses

**Output shapes:**
- `x_t` same shape as `x_0` (continuous tensors)

---

### Discrete Diffusion (D3PM - Discrete Denoising Diffusion Probabilistic Models)
**What it is:** Diffusion adapted for categorical/discrete variables  
**Used for:**
- Atom types (atomic numbers): discrete choices {1, 2, ..., 118}
- Corruption: replace atom with random type or mask token
- Denoising: predict logits over atom types

**Algorithm:**
1. Forward: corrupt atom i to random type with probability depending on t
2. Reverse: predict distribution over atom types
3. Sample: argmax or categorical sampling from predicted logits

**Files:**
- `mattergen/diffusion/diffusion_module.py` — D3PM logic for atom types
- `mattergen/diffusion/losses.py` — cross-entropy loss for types

**Key features:**
- Mask token: special type indicating "unknown atom"
- Transition matrix: probabilities of type corruptions
- Output: logits `(num_atoms, num_atom_types)` → sample via argmax

---

### Predictor-Corrector (PC) Sampling
**What it is:** Two-stage iterative sampling for quality improvement  
**Used for:**
- Generating high-quality structures during inference
- Balancing speed vs quality

**Algorithm per timestep:**
1. **Predictor:** one-step prediction using learned model
   - x_{t-1}^pred = x_t + step_size * model_score(x_t, t)
2. **Corrector:** small MCMC/Langevin steps around prediction
   - refine x_{t-1}^pred with Langevin dynamics

**Files:**
- `mattergen/diffusion/sampling/pc_sampler.py` — main sampler
- `mattergen/generator.py` — orchestrates sampling

**Parameters:**
- `num_steps`: number of predictor-corrector iterations (e.g., 50)
- `step_size`: predictor step magnitude
- `snr`: signal-to-noise ratio for corrector

**Output:**
- Sampled structure (ChemGraph) at t=0

---

### Classifier-Free Guidance
**What it is:** Technique to steer generation toward or away from properties  
**Used for:**
- Conditional generation with property control
- No need for separate classifier network

**Algorithm:**
- Compute two predictions:
  - score_cond = denoiser(x_t, t, property_embedding)
  - score_uncond = denoiser(x_t, t, unconditional_embedding)
- Guided score = score_uncond + guidance_factor * (score_cond - score_uncond)
- guidance_factor ≥ 0: 0=uncond, 1=normal, >1=strong conditioning

**Files:**
- `mattergen/diffusion/sampling/pc_sampler.py` — guidance applied during sampling
- `mattergen/property_embeddings.py` — property embedding creation
- Training: `mattergen/diffusion/lightning_module.py` — random unconditional masking

**Training trick:**
- During training, randomly replace property embedding with zeros (unconditional) to learn both paths

**Inference:**
- `properties_to_condition_on={'dft_mag_density': 0.15}` → embedding created
- `diffusion_guidance_factor=2.0` → blended prediction

---

## 4. Data Structures & Processing

### ChemGraph (Custom Data Class)
**What it is:** Immutable representation of atomic/crystal structures as graphs  
**Used for:**
- Single structure: nodes=atoms, edges=interactions
- Batched multiple structures: all nodes/edges concatenated, with batch index
- Data contract for all pipeline stages

**Files:**
- `mattergen/common/data/chemgraph.py` — definition, methods

**Fields:**
- `atomic_numbers`: LongTensor `(num_atoms,)` — element type Z
- `pos`: Tensor `(num_atoms, 3)` — fractional coordinates
- `cell`: Tensor `(num_graphs, 3, 3)` or `(1, 3, 3)` — lattice vectors
- `num_atoms`: LongTensor `(num_graphs,)` — per-graph atom count (for unbatching)
- `batch`: LongTensor `(num_atoms,)` — per-atom graph index (added by collate)
- Optional properties: tensors like `dft_mag_density`, `dft_band_gap`, etc.

**Key pattern - Immutability:**
```python
# Correct: .replace() method
new_cg = cg.replace(pos=new_pos)

# Wrong: in-place assignment forbidden
cg.pos = new_pos  # raises AttributeError
```

**Batching:**
- Input: N separate ChemGraph objects
- Output: single batched ChemGraph with `.batch` indices

**Shape example:**
- Single structure (e.g., Fe2O3): `num_atoms=5`, `atomic_numbers.shape=(5,)`, `pos.shape=(5,3)`, `cell.shape=(1,3,3)`
- Batch of 3: total atoms M=15, `batch=(0,0,0,0,0,1,1,1,1,1,2,2,2,2,2)` — tells which atoms belong to which structure

---

### PyTorch Dataset & DataLoader
**What it is:** Standard PyTorch machinery for batching and parallel data loading  
**Used for:**
- Training data iteration
- Validation/test data

**Files:**
- `mattergen/common/data/` — dataset classes
- `mattergen/scripts/run.py` → DataModule → creates DataLoader

**Key classes:**
- `Dataset`: holds list of samples, implements `__getitem__`, `__len__`
- `DataLoader`: parallel I/O, batching, shuffling
- `Batch.from_data_list()`: collates list of ChemGraph into batched ChemGraph

**Parameters:**
- `batch_size`: number of structures per batch (e.g., 8)
- `num_workers`: parallel processes for I/O (e.g., 4)
- `shuffle`: randomize order during training

---

### Tensor & Array Operations
**What it is:** Core numerical computation primitives  
**Used for:**
- All numerical calculations (distances, angles, losses)
- In-place ops on CPU, GPU-accelerated on CUDA

**Libraries:**
- `torch.{linalg, nn.functional}` — matrix operations, activations
- `numpy` — occasional preprocessing (slower, used offline)
- `scipy` — eigenvector calculations (Laplacian)

---

## 5. Crystal Structure Libraries

### pymatgen (Materials Project)
**What it is:** Mature, comprehensive materials informatics library  
**Used for:**
- Convert ChemGraph ↔ Structure (molecular/crystal objects)
- Symmetry operations, space group detection
- Distance/angle calculations
- CIF/XYZ file I/O

**Files:**
- `mattergen/common/utils/eval_utils.py` — structure conversion
- `mattergen/evaluation/evaluate.py` — loading, matching
- `mattergen/diffusion/sampling/pc_sampler.py` — structure creation

**Key functions:**
- `Structure(lattice=cell, species=atoms, coords=fractional_pos)` — create structure
- `Structure.from_file('file.cif')` — load
- `structure.to(fmt='cif')` — save as CIF

**What it does:**
- Validates chemical composition
- Computes volume, density
- Normalizes coordinates to conventional cell

---

### Atomic Simulation Environment (ASE)
**What it is:** Atomic simulation toolkit  
**Used for:**
- Alternative structure I/O format (`.extxyz`)
- Visualization and analysis
- Integration with external calculators

**Files:**
- `mattergen/common/utils/eval_utils.py` — save_structures uses ASE for `.extxyz`
- `mattergen/generator.py` → sampling output saving

**Key functions:**
- `Atoms(symbols=..., positions=..., cell=..., pbc=True)` — create
- `.write('file.extxyz')` — trajectory format
- `.get_distance(i, j)` — pairwise distances

**Format:**
- `.extxyz`: extended XYZ format, supports multiple frames (trajectories), per-atom/frame metadata

---

## 6. Force Fields & Relaxation

### MatterSim MLFF (Machine Learning Force Field)
**What it is:** Pretrained universal neural network force field  
**Used for:**
- (Optional) Structure relaxation: refine generated structures to lower energy
- Energy prediction, force computation
- Stability assessment

**Files:**
- `mattergen/evaluation/evaluate.py` — calls relaxation if `relax=True`
- External dependency: `mattersim` package

**What it does:**
- Takes structure → predicts atomic forces and total energy
- Runs gradient descent on atomic positions to minimize energy
- Outputs: relaxed structure, final energy, relaxation trajectory

**Computational cost:**
- ~1-5 minutes per structure on GPU (M1/H100)
- Essential for structure validation but optional for generation

**Output:**
- Relaxed structure (lower/stable energy state)
- Energy per atom

---

## 7. Configuration & Experiment Management

### Hydra (v1.3+)
**What it is:** Hierarchical configuration management for ML  
**Used for:**
- Define model architecture, training hyperparameters, data settings
- CLI argument overrides with type safety
- Automatic output directory creation (`outputs/`)

**Files:**
- `mattergen/conf/` — YAML configuration tree
- `mattergen/scripts/run.py`, `generate.py`, `evaluate.py`, `finetune.py` — instantiate via Hydra

**Configuration hierarchy:**
```
default.yaml (top-level)
├── data_module: mp_20 | alex_mp_20
├── trainer: default (Lightning config)
├── lightning_module: default
│   └── diffusion_module: default
│       └── model: mattergen
│           └── property_embeddings: <individual .yaml>
└── adapter: default (fine-tuning only)
```

**CLI syntax:**
```bash
mattergen-train data_module=alex_mp_20 trainer.accumulate_grad_batches=4
# key=value overrides
# ~key removes entry
# +key=value adds new entry
```

**Composable config:**
- Separate YAML files for each component
- Instantiate with `hydra.utils.instantiate(config)` → Python objects

---

## 8. Utilities & Supporting Libraries

### scikit-learn (sklearn)
**What it is:** Machine learning utilities library  
**Used for:**
- Metrics (e.g., pairwise distances, clustering)
- Preprocessing (optional)

**Files:**
- `mattergen/evaluation/metrics.py` — metric computation (if used)

---

### NumPy
**What it is:** Array/numerical computing library  
**Used for:**
- Offline data preprocessing
- Occasional non-critical computations
- Data type conversions

**Files:**
- Throughout data loading/preprocessing

---

### Pandas
**What it is:** Data frame / tabular data library  
**Used for:**
- Reading CSV data (properties, structure metadata)
- Organizing datasets

**Files:**
- `mattergen/scripts/csv_to_dataset.py` — CSV → dataset conversion
- Data loading paths

**Example:**
```python
df = pd.read_csv('data/structures.csv')  # columns: formula, dft_mag_density, ...
```

---

### tqdm
**What it is:** Progress bar for loops  
**Used for:**
- Visual feedback during data loading, evaluation, generation

**Files:**
- Any loop processing multiple structures

---

### jsonschema / pyyaml
**What it is:** JSON/YAML validation and parsing  
**Used for:**
- Load configuration YAML files (Hydra)
- Validate config structure

---

## 9. Training & Optimization

### Optimizers (torch.optim)
**What it is:** Gradient-based parameter updates  
**Used for:**
- Minimize loss during training

**Common choices:**
- **Adam:** adaptive learning rates, momentum, momentum of second moments
- **AdamW:** Adam + weight decay (L2 regularization)
- **SGD:** stochastic gradient descent (simpler, less memory)

**Files:**
- Instantiated in Lightning config: `mattergen/conf/trainer/`
- Default: Adam with configurable learning rate

**Hyperparameters:**
- `lr`: learning rate (e.g., 1e-3)
- `weight_decay`: L2 penalty (e.g., 1e-5)
- `betas`: momentum coefficients for Adam

---

### Learning Rate Scheduling
**What it is:** Adjust learning rate during training  
**Used for:**
- Accelerate early training, refine later
- Prevent divergence

**Common schedules:**
- **Linear decay:** lr → 0 linearly
- **Cosine annealing:** smooth decay with restarts
- **Step decay:** drop every N epochs

**Files:**
- PyTorch Lightning `Trainer` config
- Callables in `mattergen/conf/trainer/`

---

### Loss Functions
**What it is:** Quantify prediction error  
**Used for:**
- Training objective

**Specific losses in MatterGen:**
- **L2 loss (MSE):** for continuous fields (coordinates, cell)
  - `||x_hat - x_0||^2` where x_hat is prediction, x_0 is target
- **Cross-entropy loss:** for discrete atom types
  - `-sum_i log(p_i[y_i])` where p_i is predicted distribution, y_i is true type
- **Weighted per-field loss:** balance multiple objectives
  - `total_loss = w_pos * loss_pos + w_type * loss_type + w_cell * loss_cell`

**Files:**
- `mattergen/diffusion/losses.py` — loss computation

---

### Gradient Clipping & Accumulation
**What it is:** Prevent training instability  
**Used for:**
- Gradient clipping: cap gradient norms to prevent exploding gradients
- Gradient accumulation: simulate larger batches with smaller GPU memory

**Files:**
- PyTorch Lightning Trainer: `gradient_clip_val`, `accumulate_grad_batches`

---

## 10. Sampling & Inference

### Scoring Networks (Score Functions)
**What it is:** Neural network predicting denoising direction  
**Used for:**
- At each timestep t, predict score ∇_x log p(x_t | x_0)
- Equivalent to predicting noise ε or clean x_0

**Denoiser variants:**
- **ε-prediction:** predict added noise
- **x_0 prediction:** predict clean target directly
- **v-prediction:** parameterization of velocity

**Files:**
- `mattergen/denoiser.py` — main scoring network (wraps GemNetT)

---

### Noise Schedules
**What it is:** Defines signal-to-noise ratio over time  
**Used for:**
- Control corruption level at each timestep
- Must be monotonic: most noise at t=T, least at t=0

**Common schedules:**
- **Linear:** α_t = 1 - t/T
- **Cosine:** α_t = cos²(π t / (2T))
- **Quadratic:** custom polynomial

**Files:**
- `mattergen/diffusion/diffusion_module.py` — noise schedule definition

**Use:**
- x_t = sqrt(α_t) * x_0 + sqrt(1 - α_t) * ε
- α_T ≈ 0 (all noise), α_0 ≈ 1 (all signal)

---

### Timestep Embeddings
**What it is:** Encode continuous time t into learnable features  
**Used for:**
- Denoiser must know "how noisy is current sample?"
- Helps model focus on appropriate denoising level

**Common methods:**
- **Sinusoidal positional encoding:** e.g., sin(t / 10000^(i/d)), cos(...)
- **Learned embedding:** t → embedding lookup
- **MLPs:** t → hidden layer → embedding

**Files:**
- `mattergen/common/gemnet/` — timestep embedding layers in GemNetT

**Dimensions:**
- Input: scalar t ∈ [0, T]
- Output: vector e_t ∈ R^d (e.g., d=256)

---

## 11. Metrics & Evaluation

### Structure Matching
**What it is:** Determine if two structures are identical or similar  
**Used for:**
- Novelty detection: is generated structure new vs. training set?
- Structure comparison

**Matchers:**
- **OrderedStructureMatcher:** assumes fixed atom ordering (periodic solids)
- **DisorderedStructureMatcher:** allows permutations, partial occupancy (more general)

**Files:**
- `mattergen/evaluation/evaluate.py` — instantiate matcher
- Uses pymatgen's `Matcher` classes

**Output:**
- Boolean match or similarity score

---

### Metrics Computed
**What it is:** Quantify structure quality  
**Used for:**
- Validate generated structures

**Common metrics:**
- **Validity:** is structure chemically/geometrically valid?
  - No negative volumes, reasonable atomic distances
- **Novelty:** how different from training set?
  - Percentage of structures with no match in reference
- **Stability:** how energetically favorable?
  - Energy per atom (lower = more stable)
  - Distance from convex hull (materials thermodynamics)
- **Property match (if conditioned):** does generated structure have desired property?
  - Correlation between predicted and actual property

**Files:**
- `mattergen/evaluation/evaluate.py` — MetricsEvaluator
- `mattergen/evaluation/metrics/` — individual metric implementations

---

### Energy Above Hull
**What it is:** Thermodynamic stability measure  
**Used for:**
- Compare generated structure energy to ground state energy at same composition
- Hull: convex hull of all known phases
- AH = E_predicted - E_convex_hull (lower = more stable)

**Files:**
- Computed if reference dataset (e.g., Materials Project) available

---

## 12. Distributed Training & Parallelization

### Distributed Data Parallel (DDP)
**What it is:** Training across multiple GPUs/TPUs  
**Used for:**
- Scale training to large datasets
- Synchronize gradients across devices

**Files:**
- PyTorch Lightning `Trainer` handles automatically
- Config: `trainer.devices`, `trainer.accelerator`

**How it works:**
- Each GPU gets a subset of batch
- Compute gradients in parallel
- All-reduce (sum gradients across devices)
- Update shared parameters

---

### Gradient Synchronization
**What it is:** Ensure all workers see same model after update  
**Used for:**
- Essential for multi-GPU consistency

**Handled by:** PyTorch Lightning (transparent to user)

---

### Checkpointing & Model Saving
**What it is:** Persist model weights and optimizer state  
**Used for:**
- Resume training
- Release pretrained models
- Evaluation/generation

**Files:**
- `mattergen/checkpoints/` — stored checkpoints
- PyTorch Lightning Trainer: auto-checkpointing

**Formats:**
- `.pt` or `.pth`: PyTorch state_dict
- `.ckpt`: Lightning checkpoint (includes optimizer, epoch, etc.)

**Metadata:**
- `config.yaml`: exact configuration used for training (reproducibility)

---

## Technology Stack Summary Table

| Category | Technology | Version | Purpose |
|----------|-----------|---------|---------|
| **Core DL** | PyTorch | 2.2.1 / 2.4.1 | Tensor ops, autodiff, GPU |
| **Training** | PyTorch Lightning | 1.9+ | Training loops, checkpoints |
| **Graphs** | PyTorch Geometric | 2.5+ | GNN, graph ops, scatter |
| **GNN Arch** | GemNetT | custom | Denoiser backbone |
| **Config** | Hydra | 1.3+ | Config management |
| **Structures** | pymatgen | 2024.6.4+ | Crystal I/O, symmetry |
| **Structures** | ASE | - | XYZ/EXTXYZ I/O |
| **Relaxation** | MatterSim MLFF | - | Energy/force prediction |
| **Data** | PyTorch Dataset/Loader | - | Batch iteration |
| **Optimization** | Adam/AdamW | torch.optim | Gradient updates |
| **Math** | NumPy | - | Numeric preprocessing |
| **Tabular** | Pandas | - | CSV/table ops |
| **Progress** | tqdm | - | Progress bars |
| **Utils** | scikit-learn | - | Metrics, distances |

---

## File-to-Technology Mapping

### Data & Loading
- `mattergen/common/data/chemgraph.py` — **ChemGraph** (PyTorch, PyTorch Geometric)
- `mattergen/scripts/csv_to_dataset.py` — **Pandas**, **Hydra**, **pymatgen**, **ASE**

### Model & Architecture
- `mattergen/denoiser.py` — **PyTorch**, **PyTorch Geometric**, **GemNetT**
- `mattergen/common/gemnet/gemnet.py` — **PyTorch**, **PyTorch Geometric**, **message passing**
- `mattergen/property_embeddings.py` — **PyTorch**, **Hydra**

### Training
- `mattergen/diffusion/lightning_module.py` — **PyTorch Lightning**, **diffusion (continuous + D3PM)**, **losses**, **gradient accumulation**
- `mattergen/diffusion/losses.py` — **L2 loss**, **cross-entropy loss**, **weighted loss**

### Sampling & Generation
- `mattergen/diffusion/sampling/pc_sampler.py` — **predictor-corrector**, **classifier-free guidance**, **noise schedules**
- `mattergen/generator.py` — **orchestration**, **PyTorch**, **pymatgen**, **ASE**

### Evaluation
- `mattergen/evaluation/evaluate.py` — **structure matching**, **metrics**, **MatterSim**, **pymatgen**
- `mattergen/evaluation/metrics/` — **validity**, **novelty**, **stability**

### Scripts
- `mattergen/scripts/run.py` — **Hydra**, **PyTorch Lightning**, **DDP**
- `mattergen/scripts/generate.py` — **Hydra**, **generator**, **pymatgen**, **ASE**
- `mattergen/scripts/evaluate.py` — **Hydra**, **evaluate**
- `mattergen/scripts/finetune.py` — **Hydra**, **adapter**, **PyTorch Lightning**

---

## Quick Reference: Technology → Problem Solved

| Problem | Technology |
|---------|-----------|
| Represent structures as graphs | ChemGraph + PyTorch Geometric |
| Learn from graph topology | Message passing + GemNetT |
| Generate structures | Diffusion + predictor-corrector + classifier-free guidance |
| Condition on properties | Property embeddings + Hydra config |
| Train efficiently | PyTorch Lightning + DDP + gradient accumulation |
| Validate structures | pymatgen + structure matching |
| Relax structures | MatterSim MLFF |
| Compute metrics | pymatgen + novelty/stability algorithms |
| Manage experiments | Hydra + checkpointing |
| Load data efficiently | PyTorch Dataset/Loader + parallel workers |

---

## Next Steps

1. **Understand each technology in isolation:** Pick one (e.g., "message passing") and read the relevant file.
2. **Trace the pipeline:** Follow a single structure from CSV → trained model → generated sample → metrics.
3. **Modify/extend:** Once comfortable, add custom property, new loss function, or alternative sampler.
4. **Debug:** Use the logging system added (DEBUGGING_*.md files) to trace execution step-by-step.

---

**Document Version:** 1.0  
**Created:** 31 December 2025  
**Scope:** MatterGen v0.1+
