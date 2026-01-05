# 📚 MatterGen Training: Complete Explanation

## Overview

MatterGen is a **diffusion-based generative model** that learns to generate novel inorganic crystal structures with desired material properties. The training process teaches the model to gradually denoise random structures into realistic crystalline materials.

---

## High-Level Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                         │
└─────────────────────────────────────────────────────────────────┘

Step 1: DATA PREPARATION
├─ Raw CSV (materials database)
├─ Convert to preprocessed dataset
└─ Create train/validation splits

Step 2: MODEL INITIALIZATION
├─ Load GemNetT backbone
├─ Add property embeddings
└─ Set up diffusion schedules

Step 3: TRAINING LOOP (360+ epochs)
├─ Add noise to structures (corruption)
├─ Train denoiser to predict original structure
├─ Apply classifier-free guidance training
└─ Validate on held-out set

Step 4: SAVE CHECKPOINTS
├─ Best model weights
├─ Configuration file
└─ Training state

Step 5: OPTIONAL: FINE-TUNING
├─ Load base model
├─ Add new property embeddings (adapter)
└─ Train on new properties only
```

---

## 1. Data Preparation Phase

### Input: Raw Materials Data
```
datasets/mp_20/train.csv
├── Columns: atomic_numbers, lattice, coordinates, formula, ...
├── Properties: dft_band_gap, dft_mag_density, energy_above_hull, ...
└── ~20,000 structures (MP database)
```

### Conversion to Dataset
```bash
csv-to-dataset --csv-folder datasets/mp_20/ \
               --output-folder datasets/mp_20/processed/ \
               --train-fraction 0.8 \
               --val-fraction 0.1 \
               --test-fraction 0.1
```

**What happens:**
- Convert CSV rows → `ChemGraph` objects
- Split into train (80%), validation (10%), test (10%)
- Precompute graph structures
- Save as PyTorch DataLoader-compatible format

### Output Structure
```
datasets/mp_20/processed/
├── train/
│   ├── chunk_0000.pt  (batched ChemGraph tensors)
│   ├── chunk_0001.pt
│   └── ...
├── val/
│   └── chunk_0000.pt
└── test/
    └── chunk_0000.pt
```

---

## 2. Core Data Structure: ChemGraph

All structures flow through the **immutable `ChemGraph`** class (extends PyTorch Geometric's `Data`).

### Required Fields
```python
ChemGraph:
├── atomic_numbers: LongTensor[num_atoms]        # Element type (3→Li, 8→O, 26→Fe)
├── pos: Tensor[num_atoms, 3]                    # Fractional coordinates (0-1)
├── cell: Tensor[1, 3, 3]                        # Lattice vectors (3×3 matrix)
├── num_atoms: LongTensor[num_structures]        # Atoms per structure
└── batch: LongTensor[num_atoms]                 # Graph index (auto-added)

Property Fields (optional):
├── dft_band_gap: Tensor[num_graphs]             # Energy gap (eV)
├── dft_mag_density: Tensor[num_graphs]          # Magnetic moment (μB/cell)
├── energy_above_hull: Tensor[num_graphs]        # Stability (meV/atom)
└── ... (any registered property)

Internal Flags:
└── _USE_UNCONDITIONAL_EMBEDDING: Dict            # Which properties to drop during training
```

### Critical Convention: Immutable Modification
```python
# ✓ CORRECT: Use .replace() method
x_modified = x.replace(pos=new_pos, atomic_numbers=new_atoms)

# ✗ WRONG: In-place assignment forbidden
x.pos = new_pos  # Raises AttributeError - data is frozen!
```

---

## 3. Model Architecture: Three-Layer System

### Layer 1: GemNetT Denoiser
**File:** `mattergen/denoiser.py`

**Purpose:** The backbone "score network" that predicts denoising direction

```
Input: (noisy_structure, time_step, properties)
  ↓
GemNetT Graph Neural Network
  ├─ Message passing on crystal graph
  ├─ Equivariant geometric operations
  └─ Property-conditioned attention
  ↓
Output: (predicted_pos, predicted_lattice, predicted_atoms)
```

**Architecture:**
```
- Node features: GemNetT message passing (atom type + position embedding)
- Edge features: Distance, vector direction (SE(3) equivariant)
- Graph pooling: Aggregates to graph-level features
- Output head: Predicts denoising for pos, cell, atomic_numbers separately

Key: SE(3) equivariance ensures predictions are rotation/translation invariant
```

### Layer 2: Property Embeddings
**File:** `mattergen/property_embeddings.py`

**Purpose:** Condition the model on target properties (band gap, magnetic moment, etc.)

```
Property Value (e.g., 2.5 eV)
  ↓
Embedding Network (MLP)
  ↓
Conditioning Vector → Injected into GemNetT attention
```

**Classifier-Free Guidance Training:**
```
During training, randomly set embeddings to ZERO with probability p:
├─ With prob (1-p): Use actual property embedding → conditional score
├─ With prob p:    Use zero embedding → unconditional score
└─ Result: Model learns BOTH conditional AND unconditional distributions

At inference:
├─ guidance_factor = 0.0  → Pure unconditional generation
├─ guidance_factor = 1.0  → Pure conditional (property aware)
└─ guidance_factor = 2.0  → Strong conditioning (adhere to property)
```

**Registered Properties** (in `mattergen/common/utils/globals.py`):
```python
PROPERTY_SOURCE_IDS = {
    'chemical_system',           # e.g., "Li-Fe-P-O"
    'dft_band_gap',              # eV
    'dft_mag_density',           # μB/cell
    'energy_above_hull',         # meV/atom
    'dft_bulk_modulus',          # GPa
    'space_group',               # 1-230
    ...
}
```

### Layer 3: Adapter (Optional Fine-tuning)
**File:** `mattergen/adapter.py`

**Purpose:** Add new properties without retraining the entire model

```
Base Model (frozen or fine-tuned slightly)
  ↓
Adapter Layer (small trainable module)
  ├─ New property embeddings
  └─ Output → injected into base model
```

**Key Insight:** Adapter unconditional embeddings always return **zero**, preserving the base model's unconditional behavior.

---

## 4. Diffusion Process: Training

### What is Diffusion?

**Forward Process (Corruption):** Add noise gradually to real structures
```
Real Structure → Add noise → More noise → ... → Pure Noise
(time=0)                                        (time=T)
```

**Reverse Process (Denoising):** Learn to remove noise step by step
```
Pure Noise → Denoise → Less noise → ... → Real Structure
(time=T)                                   (time=0)
```

**Training Objective:** Predict the structure at time $t$ given noisy version at time $t+\Delta t$

### Multi-Field Diffusion

MatterGen handles **three fields separately** with different corruption schedules:

#### 1. Atom Positions (Continuous)
```
Corruption: Add Gaussian noise to fractional coordinates
Schedule:  σ(t) = √(σ_min² + t·(σ_max² - σ_min²))
Loss:      L_pos = ||predicted_pos - original_pos||²
```

#### 2. Lattice Vectors (Continuous)
```
Corruption: Add noise to 3×3 cell matrix (log-space for numerical stability)
Schedule:  Different σ(t) for lattice vs positions
Loss:      L_cell = ||predicted_cell - original_cell||²
```

#### 3. Atom Types (Discrete - D3PM)
```
Corruption: Randomly flip atom types (e.g., 1→25 or stay same)
Schedule:  Corruption rate increases with time
Loss:      L_atoms = Cross-entropy(predicted_types, original_types)
```

### Loss Computation

**Total Loss:**
```
L_total = λ_pos · L_pos(t) + λ_cell · L_cell(t) + λ_atoms · L_atoms(t)

Where:
- λ_pos, λ_cell, λ_atoms = learnable or fixed weights per field
- Loss is accumulated over all timesteps
- Weighting controls which field to emphasize
```

**File:** `mattergen/diffusion/losses.py`

### Training Loop Algorithm

```python
for epoch in range(360):
    for batch in train_dataloader:
        # 1. Sample random timestep
        t = random_timestep()  # t ∈ [0, T]
        
        # 2. Corrupt structures (forward process)
        x_noisy = corrupt(x, t)
        
        # 3. Optional: dropout conditioning (classifier-free guidance)
        if random() < dropout_prob:
            properties = None  # Unconditional
        else:
            properties = batch.properties  # Conditional
        
        # 4. Forward through denoiser
        x_pred = denoiser(x_noisy, t, properties)
        
        # 5. Compute loss
        loss = compute_loss(x_pred, x, t)
        
        # 6. Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 7. Validation
    val_loss = evaluate(model, val_dataloader)
    if val_loss < best_loss:
        save_checkpoint(model)
```

---

## 5. Configuration System (Hydra)

### Hierarchical Config Structure

```
mattergen/conf/
├── default.yaml                          (top-level)
│   ├── data_module: mp_20
│   ├── trainer: default
│   └── lightning_module: default
│
├── data_module/
│   ├── mp_20.yaml               ← 20K materials, 10 properties
│   └── alex_mp_20.yaml          ← Alternative dataset
│
├── trainer/
│   └── default.yaml             ← PyTorch Lightning config
│       ├── max_epochs: 360
│       ├── batch_size: 32
│       └── accumulate_grad_batches: 4
│
└── lightning_module/
    ├── diffusion_module/
    │   ├── default.yaml         ← Diffusion params
    │   └── model/
    │       ├── mattergen.yaml   ← GemNetT config
    │       └── property_embeddings/
    │           ├── dft_band_gap.yaml
    │           ├── dft_mag_density.yaml
    │           └── ...
    │
    └── adapter/
        └── default.yaml         ← Fine-tuning config
```

### Command-Line Overrides (Hydra Syntax)

```bash
# Train with custom dataset
mattergen-train data_module=alex_mp_20

# Adjust batch size and accumulation
mattergen-train trainer.batch_size=64 trainer.accumulate_grad_batches=2

# Add new configuration
mattergen-train +model.new_param=value

# Remove configuration
mattergen-train ~trainer.logger

# Override nested values
mattergen-train lightning_module.diffusion_module.noise_schedule=cosine
```

---

## 6. Training Workflow: Commands

### Training Base Model (From Scratch)
```bash
mattergen-train data_module=mp_20 \
                 trainer.max_epochs=360 \
                 ~trainer.logger
```

**Expected Results:**
- **Runtime:** 7-14 days on single GPU (80K training steps)
- **Validation Loss:** ≈ 0.4 after 360 epochs
- **Output:** `outputs/singlerun/${date}/${time}/`
  - Checkpoint files (.pt)
  - Configuration file (config.yaml)
  - Metrics log

### Fine-tuning on New Property
```bash
mattergen-finetune adapter.pretrained_name=mattergen_base \
                    data_module=mp_20 \
                    data_module.properties=["my_property"] \
                    +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.my_property=my_property \
                    ~trainer.logger
```

**What happens:**
1. Load base model weights (mattergen_base checkpoint)
2. Freeze most of the model (except adapter)
3. Train ONLY the new property embeddings
4. Runtime: ~1-2 days (much faster than base training)

---

## 7. Key Training Parameters

### Model Hyperparameters
```yaml
Model (GemNetT):
  num_layers: 4                    # Message passing layers
  hidden_dim: 256                  # Channel dimension
  use_position_encoding: true      # Fourier features
  dropout: 0.1                     # Regularization

Diffusion:
  noise_schedule: 'linear'         # or 'cosine', 'sqrt'
  sigma_min: 0.001                 # Min noise level
  sigma_max: 1.0                   # Max noise level
  num_timesteps: 1000              # Discretization steps
  
Conditioning:
  property_embedding_dim: 128      # Property vector size
  unconditional_prob: 0.1          # Classifier-free guidance dropout
```

### Training Hyperparameters
```yaml
Optimizer:
  learning_rate: 1e-3
  weight_decay: 1e-5
  warmup_steps: 1000
  
Batch:
  batch_size: 32                   # Per GPU
  accumulate_grad_batches: 4       # Gradient accumulation
  num_workers: 8                   # Data loading parallelism
  
Validation:
  val_check_interval: 0.5          # Check every 50% of epoch
  patience: 20                      # Early stopping
```

---

## 8. Monitoring Training

### Key Metrics
```
Logged during training:
├── loss/train_total              (combined loss)
├── loss/train_pos                (position prediction)
├── loss/train_cell               (lattice prediction)
├── loss/train_atoms              (atom type prediction)
├── loss/val_total                (validation loss)
└── learning_rate                 (optimizer LR)
```

### Tensorboard Visualization
```bash
tensorboard --logdir outputs/singlerun/
```

### Expected Learning Curve
```
Epoch 0:    loss ≈ 2.5  (random predictions)
Epoch 50:   loss ≈ 1.2  (structure learning)
Epoch 150:  loss ≈ 0.6  (fine details)
Epoch 360:  loss ≈ 0.4  (converged)
```

---

## 9. Troubleshooting Training

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| **OOM (Out of Memory)** | Batch too large | Increase `accumulate_grad_batches`, reduce `batch_size` |
| **Loss not decreasing** | Learning rate too low | Increase `learning_rate` (1e-3 to 1e-2) |
| **Overfitting** | Model too large/dropout too low | Increase `dropout`, reduce `hidden_dim` |
| **Slow convergence** | Dataset too small | Use data augmentation or larger dataset |
| **NaN loss** | Exploding gradients | Use gradient clipping, reduce `learning_rate` |

---

## 10. Next Steps After Training

### Generate Structures (Unconditional)
```bash
mattergen-generate results/ \
  --pretrained-name=mattergen_base \
  --batch_size=16 --num_batches=10
```

### Generate with Property Conditioning
```bash
mattergen-generate results/ \
  --pretrained-name=dft_band_gap \
  --properties_to_condition_on="{'dft_band_gap': 2.5}" \
  --diffusion_guidance_factor=2.0
```

### Evaluate Generated Structures
```bash
mattergen-evaluate results/generated_crystals.extxyz \
  --compute-metrics=True \
  --mp-api-key=${MP_API_KEY}
```

---

## Summary

**MatterGen Training = Learning to Denoise Structures**

1. **Data:** Load crystal structures from Materials Project
2. **Model:** GemNetT denoiser + property embeddings
3. **Process:** Add noise → train to remove it
4. **Conditioning:** Classifier-free guidance for property control
5. **Result:** Model learns to generate diverse, realistic structures

**Training Timeline:** 7-14 days → produces a model that can generate millions of novel crystal structures!

