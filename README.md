---
title: MOSAIC
emoji: 🧬
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

<div align="center">
  <h1>🧬 MOSAIC</h1>
  <h3>Multi-Omic Single-cell Attractor Circuit Inference</h3>
  <p><i>A proprietary enterprise platform for continuous thermodynamic cell fate modeling and inverse drug design.</i></p>
</div>

---

## 🌌 The Paradigm Shift: Thermodynamics vs. Clustering
Legacy bioinformatics pipelines (e.g., Scanpy, Seurat) rely on static, discrete clustering algorithms (Louvain, Leiden) to categorize cell types. However, biology is not discrete—it is a continuous physical process governed by energy gradients. 

**MOSAIC** discards clustering entirely. Instead, it models cellular differentiation as a dynamic journey across a continuous thermodynamic energy landscape. Using principles from statistical physics, MOSAIC maps the high-dimensional omic vector of a cell into a basin of attraction, allowing researchers to physically simulate, predict, and reverse-engineer cell fate transitions.

---

## 🔬 Scientific Methodology

MOSAIC is powered by three mathematical engines:

### 1. Thermodynamic State Modeling (RBM Free Energy)
We capture the joint multi-omic distribution of a cell using a Restricted Boltzmann Machine (RBM). The scalar Free Energy $F(v)$ of any cell vector $v$ defines its stability (basin of attraction).
```math
F(v) = -b^T v - \sum_{j} \log(1 + \exp(c_j + W_{:, j}^T v))
```
*Where $W$ is the interaction weight matrix, and $b, c$ are the visible and hidden biases, respectively.*

### 2. Time-Series Trajectory (Langevin Dynamics)
To predict how a cell will physically respond over time to a multi-gene perturbation, we simulate its descent down the energy gradient using the Euler-Maruyama discretization of Langevin dynamics:
```math
v_{t+\Delta t} = v_t - \nabla_v F(v_t) \Delta t + \sqrt{2T \Delta t} \cdot \mathcal{N}(0, 1)
```
*Where $T$ is the thermodynamic temperature (biological stochasticity) and $\nabla_v F(v)$ is the analytical gradient of the Free Energy landscape.*

### 3. Autonomous Target Discovery (Inverse Design)
To discover the optimal perturbation required to push a cell into a Target basin safely, MOSAIC utilizes gradient descent on the landscape itself. We optimize a continuous dosage vector $\Delta v$ using the following loss function:
```math
\mathcal{L}(\Delta v) = F(v_{target} + \Delta v) - \lambda \sum_{a \in A} F(v_{avoid, a} + \Delta v) + \alpha \|\Delta v\|_1
```
*Where $A$ is a set of dangerous Avoidance states (e.g., malignant attractors). The $\lambda$ parameter enforces Collateral Pleiotropy limits, while the L1 penalty ($\alpha$) ensures sparse, highly specific drug targets.*

---

## 🏗️ Architecture & Zero-Dependency Stack

MOSAIC was built from the ground up to be ultra-fast and strictly mathematical. 
**Zero legacy bioinformatics libraries** (Scanpy, AnnData, Mudata, NetworkX) are used.

**Core Stack:**
*   `PyTorch` (RBM, Inverse Design Autograd, Langevin Dynamics)
*   `NumPy` & `SciPy` (Loopy Belief Propagation Matrix Operations)
*   `Streamlit` & `Plotly` (Enterprise Telemetry & 3D Visualization)

### Deterministic Production Versions (Strict Pinning)
To guarantee zero pip backtracking and a 100% reproducible build environment (especially for Docker/Hugging Face deployments), **every single transitive dependency (150+ packages) is strictly pinned.** 

This includes all core packages as well as sub-dependencies that commonly cause version resolution glitches (such as `babel`, `wcwidth`, `async-lru`, and `argon2-cffi`). 
If you need to add upcoming tools or new libraries in the future, please ensure you use `pip freeze` locally and hard-pin their exact resolved versions in `requirements.txt` before deploying.
*(See `requirements.txt` for the full locked dependency graph)*
---

## 🚀 Quick Start

### Hugging Face / Docker Deployment
MOSAIC is packaged with an optimized CPU-wheel `Dockerfile` for instantaneous deployment on Hugging Face Spaces.
```bash
# Build the container
docker build -t mosaic-engine .

# Run the UI on port 7860
docker run -p 7860:7860 mosaic-engine
```

### Local Virtual Environment
If you prefer standard python execution:
```bash
# Install strict requirements
pip install -r requirements.txt

# Launch the God-Tier Laboratory
streamlit run app/streamlit_app.py
```
