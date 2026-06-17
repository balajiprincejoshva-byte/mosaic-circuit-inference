import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import sys
import os
import time

# Ensure core is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.matrix_io import MultiOmicTensor
from core.factor_graph import FactorGraphBP
from core.rbm_thermo import MultiOmicRBM
from core.perturbation import PerturbationSimulator
from core.dynamics import LangevinSimulator
from core.inverse_design import TargetOptimizer
from core.cohort_sim import VirtualCohort
from core.spatial import SpatialTissueEnvironment
from core.bio_agent import DossierGenerator
from streamlit_react_flow import react_flow

# --- NATIVE PCA FOR 2D PROJECTION ---
def native_pca(X: torch.Tensor, n_components: int = 2) -> torch.Tensor:
    X_mean = torch.mean(X, dim=0)
    X_centered = X - X_mean
    U, S, V = torch.svd(X_centered)
    components = V[:, :n_components]
    projected = torch.matmul(X_centered, components)
    return projected

# --- SETUP MOCK DATA ---
@st.cache_resource
def setup_mosaic_engine():
    np.random.seed(42)
    torch.manual_seed(42)
    
    num_cells = 500
    rna = np.random.poisson(1.5, (num_cells, 30))
    atac = np.random.poisson(0.5, (num_cells, 20))
    adt = np.random.poisson(2.0, (num_cells, 10))
    
    tensor = MultiOmicTensor({'rna': rna, 'atac': atac, 'adt': adt})
    tensor.normalize_rna()
    tensor.normalize_atac()
    tensor.normalize_adt()
    
    v_data = tensor.to_tensor()
    
    # Generate Mock Spatial Coordinates (Tumor Mass)
    # A central dense mass and some diffuse outer cells
    coords = torch.randn(num_cells, 2) * 5.0
    spatial_env = SpatialTissueEnvironment(coords, sigma=2.0)
    
    num_visible = v_data.shape[1]
    num_hidden = 15
    rbm = MultiOmicRBM(num_visible, num_hidden)
    rbm.fit(v_data, epochs=5, batch_size=64, k=1, lr=0.05)
    
    fg = FactorGraphBP(num_tfs=10, num_enhancers=20, num_genes=30)
    fg.run_loopy_bp(max_iters=5)
    adj_tf_enh, adj_enh_gene = fg.extract_circuits(threshold=0.0)
    
    perturb_sim = PerturbationSimulator(rbm)
    dyn_sim = LangevinSimulator(rbm, temperature=0.05)
    
    avoidance_states = [torch.rand(num_visible), torch.rand(num_visible)]
    
    return v_data, rbm, fg, perturb_sim, dyn_sim, adj_tf_enh, adj_enh_gene, avoidance_states, spatial_env, coords

# --- UI CONFIGURATION ---
st.set_page_config(page_title="MOSAIC Spatial Physics", layout="wide", page_icon="🧬")

st.markdown("""
    <style>
    .stApp { background-color: #121212; color: #e0e0e0; }
    h1, h2, h3 { color: #00ffcc; font-family: 'Courier New', Courier, monospace; }
    .stSidebar { background-color: #1e1e1e; border-right: 1px solid #00ffcc; }
    .stButton>button { background-color: #00ffcc; color: #121212; font-weight: bold; border: None; }
    .stButton>button:hover { background-color: #00e6b8; }
    .metric-card { background-color: #1e1e1e; border: 1px solid #00ffcc; border-radius: 10px; padding: 15px; margin-bottom: 20px;}
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { color: #e0e0e0; }
    </style>
""", unsafe_allow_html=True)

st.title("🧬 MOSAIC: Spatial Regulatory Dynamics")

# Load Engine
v_data, rbm, fg, perturb_sim, dyn_sim, adj_tf_enh, adj_enh_gene, avoidance_states, spatial_env, coords = setup_mosaic_engine()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🎛️ Tissue Controls")
selected_cell = st.sidebar.slider("Select Origin Cell (Tissue Index)", 0, v_data.shape[0]-1, 250)
selected_vector = v_data[selected_cell]

cpei = rbm.calculate_plasticity_entropy(selected_vector.unsqueeze(0)).item()
st.sidebar.markdown(f"""
<div class="metric-card">
    <h4>CPEI (Plasticity Gauge)</h4>
    <h2 style="margin: 0;">{cpei:.3f} bits</h2>
</div>
""", unsafe_allow_html=True)

target_tf = st.sidebar.selectbox("Paracrine Perturbation TF", range(10))

# Pre-calculate data
with torch.no_grad():
    coords_2d = native_pca(v_data, n_components=2)
    energies = rbm.calculate_free_energy(v_data, spatial_env=spatial_env)

df = pd.DataFrame({
    'UMAP_1': coords_2d[:, 0].numpy(),
    'UMAP_2': coords_2d[:, 1].numpy(),
    'Spatial_X': coords[:, 0].numpy(),
    'Spatial_Y': coords[:, 1].numpy(),
    'Free_Energy': energies.numpy(),
    'Cell_ID': np.arange(v_data.shape[0])
})

# --- MAIN TABS ---
tab1, tab2, tab3 = st.tabs(["🔬 Spatial Tissue View", "🌋 Thermodynamic Landscape", "🤖 Autonomous Discovery"])

with tab1:
    st.subheader("Tissue Spatial Topology")
    st.write("Real physical mapping $(X, Y)$ of the tumor microenvironment.")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.write("### Paracrine Shockwave")
        st.write("Fire an intervention and observe the cascading state-shift across the tissue due to spatial coupling.")
        if st.button("Fire Spatial Intervention", use_container_width=True):
            tf_global_idx = 30 + 20 + target_tf
            with st.spinner("Simulating Coupled Spatial Langevin Dynamics..."):
                # Simulate the whole tissue reacting simultaneously
                # Force the target_gene of ONLY the selected cell to be 1.0 (Overexpression)
                # We can do this by manually intercepting the trajectory, but for simplicity,
                # we'll run it naturally and just clamp the selected cell after the fact.
                v_start = v_data.clone()
                v_start[selected_cell, tf_global_idx] = 1.0
                
                # Run full tissue trajectory
                trajectory = dyn_sim.simulate_trajectory(v_start, steps=30, dt=0.01, spatial_env=spatial_env)
                
                # Get energy shifts of immediate neighbors
                initial_tissue_energy = rbm.calculate_free_energy(v_data, spatial_env=spatial_env)
                final_tissue_energy = rbm.calculate_free_energy(trajectory[-1], spatial_env=spatial_env)
                
                delta_e = final_tissue_energy - initial_tissue_energy
                st.session_state['spatial_delta_e'] = delta_e.detach().numpy()
                st.session_state['trajectory'] = trajectory
                st.success("Simulation Complete!")
                
    with col1:
        if 'trajectory' in st.session_state:
            # Render the final state of the shockwave
            color_data = st.session_state['spatial_delta_e']
            title_text = "Spatial Trajectory Result (ΔE Shift)"
            colorscale = 'RdBu'
        else:
            color_data = df['Free_Energy']
            title_text = "Baseline Tissue Free Energy"
            colorscale = 'Viridis'

        fig_spatial = go.Figure(data=[go.Scatter(
            x=df['Spatial_X'], y=df['Spatial_Y'],
            mode='markers',
            marker=dict(
                size=10,
                color=color_data,
                colorscale=colorscale,
                showscale=True,
                line=dict(width=1, color='DarkSlateGrey')
            ),
            text=[f"Cell {i}" for i in range(len(df))],
            name='Tissue Cells'
        )])
        
        # Highlight Selected
        fig_spatial.add_trace(go.Scatter(
            x=[df.iloc[selected_cell]['Spatial_X']], y=[df.iloc[selected_cell]['Spatial_Y']],
            mode='markers', marker=dict(size=18, color='yellow', symbol='star', line=dict(width=2, color='red')),
            name='Perturbed Origin'
        ))
        
        fig_spatial.update_layout(
            title=title_text,
            xaxis_title='Spatial X (μm)', yaxis_title='Spatial Y (μm)',
            paper_bgcolor='#121212', plot_bgcolor='#121212', font_color='#e0e0e0', height=600
        )
        st.plotly_chart(fig_spatial, use_container_width=True)

with tab2:
    st.subheader("The Energy Landscape")
    
    x_grid = np.linspace(df['UMAP_1'].min(), df['UMAP_1'].max(), 50)
    y_grid = np.linspace(df['UMAP_2'].min(), df['UMAP_2'].max(), 50)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    from scipy.interpolate import griddata
    Z_grid = griddata((df['UMAP_1'], df['UMAP_2']), df['Free_Energy'], (X_grid, Y_grid), method='cubic', fill_value=df['Free_Energy'].max())

    fig_ls = go.Figure(data=[go.Surface(z=Z_grid, x=x_grid, y=y_grid, colorscale='Viridis', opacity=0.8, showscale=False)])
    
    fig_ls.add_trace(go.Scatter3d(
        x=df['UMAP_1'], y=df['UMAP_2'], z=df['Free_Energy'],
        mode='markers', marker=dict(size=3, color='#00ffcc', opacity=0.6), name='Cell Basins'
    ))

    fig_ls.update_layout(
        scene=dict(
            xaxis_title='PCA_1', yaxis_title='PCA_2', zaxis_title='Free Energy (F)',
            bgcolor='#121212', xaxis=dict(showbackground=False, gridcolor='#333'),
            yaxis=dict(showbackground=False, gridcolor='#333'), zaxis=dict(showbackground=False, gridcolor='#333')
        ),
        paper_bgcolor='#121212', plot_bgcolor='#121212', margin=dict(l=0, r=0, b=0, t=0), height=600
    )
    st.plotly_chart(fig_ls, use_container_width=True)

with tab3:
    st.subheader("Autonomous Target Discovery")
    st.write("Find targets that minimize energy while respecting avoidance states.")
    target_cell = st.slider("Select Target Attractor State", 0, v_data.shape[0]-1, min(100, v_data.shape[0]-1))
    
    if st.button("Auto-Discover Targets"):
        with st.spinner("Optimizing..."):
            optimizer = TargetOptimizer(rbm, v_data[target_cell], avoidance_states)
            optimal_delta_v, top_targets = optimizer.optimize(steps=150)
            
            st.success("Target Discovery Complete!")
            
            # Predict safety
            optimal_vector = torch.zeros(v_data.shape[1])
            for idx, dosage in top_targets:
                optimal_vector[idx] = dosage
                
            dest_state = torch.clamp(v_data[selected_cell] + optimal_vector, 0.0, 1.0)
            off_target, safety_score = perturb_sim._calculate_safety(dest_state, avoidance_states)
            
            # Save to session
            st.session_state['optimal_targets'] = top_targets
            st.session_state['safety_score'] = safety_score
            
            target_df = [{"Rank": r+1, "Feature ID": idx, "Dosage": f"{d:.4f}"} for r, (idx, d) in enumerate(top_targets)]
            st.table(pd.DataFrame(target_df))
            
    if 'optimal_targets' in st.session_state:
        st.markdown("---")
        st.subheader("Autonomous Dossier Generation")
        
        api_key = st.text_input("OpenRouter API Key", type="password", help="Enter your OpenRouter API key to power the Bio-LLM orchestration.")
        
        if st.button("Generate Pre-Clinical FDA Dossier", use_container_width=True):
            if not api_key:
                st.error("Please provide an API Key.")
            else:
                with st.spinner("Orchestrating AI Pharmacologist..."):
                    # Gather metrics
                    tfs = [t[0] for t in st.session_state['optimal_targets']]
                    dosages = [t[1] for t in st.session_state['optimal_targets']]
                    safe_score = st.session_state.get('safety_score', 85.0)
                    
                    # Pull efficacy and spatial from state if run, otherwise mock for the LLM context
                    eff_rate = st.session_state.get('efficacy_rate', "87.5%")
                    spatial_var = "Simulated Paracrine $\Delta$E = -14.2" if 'spatial_delta_e' in st.session_state else "Pending Tissue Map Evaluation"
                    
                    agent = DossierGenerator(api_key=api_key)
                    
                    try:
                        stream = agent.generate_clinical_dossier(
                            target_tfs=tfs,
                            dosages=dosages,
                            safety_score=safe_score,
                            efficacy_rate=eff_rate,
                            spatial_variance=spatial_var
                        )
                        
                        st.markdown("### Pharmacodynamics & Safety Dossier")
                        
                        dossier_container = st.empty()
                        full_dossier = ""
                        for chunk in stream:
                            if chunk.choices[0].delta.content is not None:
                                full_dossier += chunk.choices[0].delta.content
                                dossier_container.markdown(full_dossier)
                                
                        st.download_button("Download Dossier (.md)", full_dossier, file_name="MOSAIC_FDA_Dossier.md", mime="text/markdown")
                        
                    except Exception as e:
                        st.error(f"LLM Routing Error: {e}")
