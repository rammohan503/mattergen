# MatterGen: Quick Technology Reference Card

**Use this when you need to find what technology solves which problem.**

---

## 🎯 Problem → Technology (Quick Lookup)

| Need | Technology | File | Use |
|------|-----------|------|-----|
| **Graph representation** | ChemGraph | `chemgraph.py` | Store atoms/bonds, immutable pattern |
| **Message passing** | PyTorch Geometric + scatter | `gemnet.py` | Aggregate neighbor info to nodes |
| **GNN backbone** | GemNetT | `gemnet.py` | Learn from geometry, predict denoised |
| **Training orchestration** | PyTorch Lightning | `lightning_module.py` | Train loops, checkpoints, multi-GPU |
| **Noise corruption** | Continuous diffusion (Gaussian) | `diffusion_module.py` | Add noise to structures for training |
| **Atom type changes** | D3PM (discrete diffusion) | `diffusion_module.py` | Corrupt atom types, predict types |
| **Sampling structures** | Predictor-Corrector | `pc_sampler.py` | Generate structures iteratively |
| **Conditional generation** | Classifier-free guidance | `pc_sampler.py` | Steer generation to properties |
| **Property control** | Property embeddings | `property_embeddings.py` | Create embeddings for dft_mag_density, etc |
| **Config management** | Hydra | `conf/` | CLI args, hierarchical config |
| **Structure I/O** | pymatgen | `eval_utils.py` | Load/save CIF, check validity |
| **Extended XYZ** | ASE | `eval_utils.py` | Save `.extxyz` trajectories |
| **Optimization** | Adam/AdamW | `conf/trainer/` | Update model weights |
| **Loss computation** | L2 + cross-entropy | `losses.py` | Supervise training |
| **Structure validation** | pymatgen Matcher | `evaluate.py` | Detect novelty, match structures |
| **Relaxation** | MatterSim MLFF | `evaluate.py` | Minimize energy, find stable state |
| **Metrics** | Custom + pymatgen | `metrics/` | Validity, novelty, stability, AH |
| **Data loading** | PyTorch Dataset/Loader | `data/` | Iterate batches, parallel I/O |
| **Distributed training** | DDP (PyTorch Lightning) | `lightning_module.py` | Multi-GPU sync |
| **Checkpointing** | PyTorch Lightning | `scripts/` | Save/load model state |

---

## 📚 Technology → Files (Where to look)

### Foundational
- **PyTorch**: everywhere (tensors, .backward(), optim)
- **PyTorch Geometric**: `common/gemnet/`, `common/data/chemgraph.py`
- **PyTorch Lightning**: `diffusion/lightning_module.py`, `scripts/run.py`

### Architecture
- **GemNetT**: `common/gemnet/gemnet.py`, `denoiser.py`
- **Message Passing**: `common/gemnet/layers/interaction_block.py`, `atom_update_block.py`
- **Property Embeddings**: `property_embeddings.py`, `conf/lightning_module/diffusion_module/model/property_embeddings/`

### Diffusion & Generation
- **Continuous Diffusion**: `diffusion/diffusion_module.py`
- **D3PM (Discrete)**: `diffusion/diffusion_module.py`
- **Predictor-Corrector**: `diffusion/sampling/pc_sampler.py`
- **Classifier-Free Guidance**: `diffusion/sampling/pc_sampler.py`
- **Loss Functions**: `diffusion/losses.py`

### Data
- **ChemGraph**: `common/data/chemgraph.py`
- **Dataset**: `common/data/` (loaders)
- **Preprocessing**: `scripts/csv_to_dataset.py`

### Evaluation
- **Structure Matching**: `evaluation/evaluate.py` (pymatgen Matcher)
- **Metrics**: `evaluation/metrics/`, `evaluation/evaluate.py`
- **Relaxation**: `evaluation/evaluate.py` (calls MatterSim)

### Config & Scripts
- **Hydra**: `conf/` (all YAML files)
- **Train Script**: `scripts/run.py`
- **Generate Script**: `scripts/generate.py`
- **Evaluate Script**: `scripts/evaluate.py`
- **Finetune Script**: `scripts/finetune.py`

---

## 🔄 Pipeline: Technology Order

```
1. Load CSV
   └─ Pandas + Hydra

2. Create ChemGraph
   └─ ChemGraph class + PyTorch Geometric

3. Batch structures
   └─ DataLoader + Batch collation

4. Add noise (training)
   ├─ Continuous diffusion (pos, cell)
   └─ D3PM (atom types)

5. Forward denoiser
   └─ GemNetT + message passing

6. Compute loss
   ├─ L2 (continuous fields)
   ├─ Cross-entropy (types)
   └─ Weighted sum

7. Backward & optimize
   ├─ Adam/AdamW
   ├─ Gradient accumulation
   └─ DDP (multi-GPU)

8. Sample structures
   ├─ Predictor-Corrector
   ├─ Classifier-free guidance
   └─ Noise schedule

9. Convert & save
   ├─ pymatgen Structure
   ├─ ASE (extxyz)
   └─ CIF format

10. Evaluate
    ├─ Structure matching (novelty)
    ├─ MatterSim (relax)
    └─ Metrics (validity, stability)
```

---

## 🚀 Technology by Execution Stage

### Dataset Preparation
```
CSV → Pandas → ChemGraph (PyTorch + PyTorch Geometric) → Pickle dataset
```

### Training (mattergen-train)
```
Hydra config
  ↓
PyTorch Lightning Trainer + Distributed (DDP)
  ↓
DataLoader + PyTorch Dataset
  ↓
Sample batch → ChemGraph
  ↓
Noise (diffusion) + corrupt atoms (D3PM)
  ↓
GemNetT forward (message passing)
  ↓
Loss (L2 + cross-entropy)
  ↓
Adam/AdamW update
  ↓
Checkpoint (PyTorch Lightning)
```

### Generation (mattergen-generate)
```
Hydra config
  ↓
Load checkpoint
  ↓
GemNetT in eval mode
  ↓
Property embeddings (if conditioning)
  ↓
Predictor-Corrector sampler
  ├─ Noise schedule
  ├─ Classifier-free guidance (if needed)
  └─ Timestep loop
  ↓
ChemGraph samples
  ↓
Convert (pymatgen Structure)
  ↓
Save (ASE .extxyz + pymatgen .cif)
```

### Evaluation (mattergen-evaluate)
```
Load structures (pymatgen + ASE)
  ↓
Relax (MatterSim, optional)
  ↓
Matcher (pymatgen OrderedStructureMatcher or Disordered)
  ↓
Metrics
  ├─ Validity
  ├─ Novelty (structure matching)
  ├─ Stability (energy, AH)
  └─ Property scores
  ↓
Output summary (CSV/JSON)
```

---

## 📊 Technology Dependencies (What needs what)

```
PyTorch
  ├─ PyTorch Geometric
  │   ├─ Message passing (scatter_add, etc.)
  │   └─ Data/Batch classes
  ├─ PyTorch Lightning
  │   ├─ Trainer (orchestration)
  │   ├─ Callbacks (checkpointing)
  │   └─ DDP (distributed)
  ├─ GemNetT (denoiser)
  │   ├─ Radial basis functions
  │   ├─ Angle embeddings
  │   └─ Message passing layers
  └─ Optimizers (Adam, AdamW)
  
Diffusion
  ├─ Noise schedules
  ├─ Corruptions (Gaussian + D3PM)
  ├─ Score prediction (GemNetT)
  ├─ Predictor-Corrector sampler
  └─ Classifier-free guidance

Data Processing
  ├─ Pandas (CSV → tables)
  ├─ ChemGraph (struct → graph)
  ├─ DataLoader (batching)
  └─ Hydra (config)

Evaluation
  ├─ pymatgen (Structure I/O, matching)
  ├─ ASE (EXTXYZ format)
  ├─ MatterSim (relaxation)
  ├─ Novelty metrics
  └─ Stability metrics
```

---

## 💡 Common Use Cases (Which technologies?)

### "I want to understand how structures are represented"
→ Read: ChemGraph, PyTorch Geometric `Data`/`Batch`  
→ Files: `chemgraph.py`, understand shapes and `.replace()` immutability

### "I want to understand how the model learns from geometry"
→ Read: Message passing, GemNetT, radial basis functions, angle features  
→ Files: `gemnet.py`, `interaction_block.py`, `atom_update_block.py`

### "I want to understand how structures are generated"
→ Read: Diffusion (continuous + D3PM), predictor-corrector, classifier-free guidance  
→ Files: `diffusion_module.py`, `pc_sampler.py`, `losses.py`

### "I want to condition on properties"
→ Read: Property embeddings, classifier-free guidance, Hydra config  
→ Files: `property_embeddings.py`, `pc_sampler.py`, `conf/`

### "I want to train the model"
→ Read: PyTorch Lightning, DataLoader, optimizers, checkpointing, DDP  
→ Files: `lightning_module.py`, `run.py`, `conf/trainer/`

### "I want to generate structures"
→ Read: Predictor-corrector sampling, pymatgen conversion, ASE I/O  
→ Files: `generator.py`, `pc_sampler.py`, `eval_utils.py`

### "I want to evaluate structures"
→ Read: Structure matching, MatterSim, metrics computation  
→ Files: `evaluate.py`, `metrics/`

### "I want to add a new property"
→ Read: Property embeddings, Hydra config, dataset CSV  
→ Files: `property_embeddings.py`, `conf/lightning_module/diffusion_module/model/property_embeddings/`

---

## 🎓 Learning Path (Recommended Order)

1. **Understand the data representation** (1 hour)
   - ChemGraph, PyTorch Geometric Data/Batch
   - Read: TECHNOLOGIES_AND_TECHNIQUES.md § 4, chemgraph.py

2. **Understand message passing & GNN** (2 hours)
   - How information flows in graphs
   - GemNetT architecture
   - Read: § 2, § 3, gemnet.py, interaction_block.py

3. **Understand diffusion & generation** (2 hours)
   - Continuous + discrete diffusion
   - Sampling algorithm
   - Read: § 3, § 10, diffusion_module.py, pc_sampler.py

4. **Understand training pipeline** (1 hour)
   - PyTorch Lightning, DataLoader, losses
   - Read: § 9, lightning_module.py, losses.py

5. **Understand conditioning & guidance** (1 hour)
   - Property embeddings, classifier-free guidance
   - Read: § 3, property_embeddings.py, pc_sampler.py

6. **Understand evaluation & metrics** (1 hour)
   - Structure validation, novelty, stability
   - Read: § 11, evaluate.py, metrics/

7. **Run end-to-end example** (30 min)
   - Run generate → evaluate
   - Trace with DEBUGGING_*.md files
   - Observe outputs

---

## 🔍 Debug Checklist (Which technology to check)

| Issue | Check Technology |
|-------|------------------|
| Generation produces invalid structures | pymatgen validity, geometry checks |
| Properties not conditioned properly | Property embeddings, classifier-free guidance |
| Model not improving during training | Loss functions, optimizer, learning rate schedule |
| GPU out of memory | Batch size, gradient accumulation, model params |
| Sampling very slow | Predictor-corrector steps, noise schedule |
| Novelty always 0% | Structure matching, Matcher type |
| Relaxation fails | MatterSim installation, energy prediction issues |
| Config won't load | Hydra syntax, YAML indentation, instantiate() |
| Multi-GPU training hangs | DDP, gradient sync, device assignment |

---

## 📖 Technologies at a Glance

| Tech | Why | Learn by | Time |
|------|-----|----------|------|
| **PyTorch** | Core tensor computation | Tutorials, operations | 1-2 days |
| **PyTorch Geometric** | Graph operations | Example code | 4-6 hours |
| **GemNetT** | Geometry-aware GNN | Paper + code | 8-10 hours |
| **Diffusion** | Generative model | DDPM/DPM papers | 6-8 hours |
| **Predictor-Corrector** | High-quality sampling | Code + tuning | 4-6 hours |
| **pymatgen** | Structure I/O & tools | Docs + examples | 3-4 hours |
| **Hydra** | Config management | Docs + examples | 2-3 hours |
| **PyTorch Lightning** | Training boilerplate | Tutorials | 3-4 hours |

---

**Version:** 1.0  
**Created:** 31 December 2025  
**For:** Quick reference and learning prioritization
