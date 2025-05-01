# scDiffSim
A Framework for Simulating Heterogeneous Single-Cell and Pseudobulk Expression Data

**scDiffSim** is a Python framework for simulating realistic single-cell and pseudobulk gene expression data through **classifier-guided diffusion**, extended from the original [**scDiffusion**](https://github.com/YosefLab/scdiffusion) model. It integrates **distribution-aware latent representations from scVI** and supports user-defined condition labels and cell-type–specific variability, enabling biologically grounded benchmarking for deconvolution and other downstream tasks.

---

## 🔍 Key Features

- **Conditional generation** of single-cell profiles via guided diffusion  
- **scVI-derived latent space** for capturing distribution-aware expression variation  
- **User-defined heterogeneity classifiers** to model condition effects per cell type  
- **Annotation-guided filtering** using transfer learning to filter out low quanlity cells
- **Pseudobulk profile generation** with realistic sample- and cell-type–level variability  
- Visualization tools for exploring simulation outputs

---

## 🚀 Getting Started

### Installation

```bash
conda create -n scDiffusion
conda activate scDiffusion
conda install pytorch scvi-tools torchvision torchaudio pytorch-cuda jupyter scanpy python-igraph leidenalg mpi4py openmpi jax=0.4.34 jaxlib=0.4.34 -c conda-forge -c pytorch -c nvidia 

git clone https://github.com/humengying0907/scDiffSim.git
```

### Example Usage

```
cd scDiffSim/module1.0
python main.py # adjust to your datapath, will generate a commend.txt file with all commands required 
```

## 📘 Use Cases

- Benchmarking **cell-type-specific expression (CTSE)** deconvolution methods  
- Simulating **heterogeneous pseudobulk datasets**  
- Modeling **cell-type–condition interactions**  
- Evaluating algorithms under realistic **bulk and single-cell variability**

---

## 🧠 Background & Credits

This framework builds on:
- [**scDiffusion**](https://github.com/YosefLab/scdiffusion), a generative diffusion model for single-cell simulation developed by the Yosef Lab.
- [**scVI**](https://github.com/scverse/scvi-tools), a deep generative model for single-cell transcriptomics that provides **distribution-aware latent representations** of scRNA-seq data.

**scDiffSim** extends these tools by:
- Introducing **classifier-guided diffusion** for condition-specific generation  
- Incorporating **scVI-based latent spaces** to better model count-distributed gene expression  
- Enabling **cell-type–specific simulation** of bulk-like pseudobulk profiles  
- Supporting **annotation-informed cell filtering** via transfer learning
---

