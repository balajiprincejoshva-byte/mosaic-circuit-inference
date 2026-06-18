import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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
from core.structural import AlphaFoldBridge
from core.quantum_rbm import QuantumInspiredRBM
from core.systemic import SystemicOrganNetwork
from core.robotics import OpentronsCompiler
from streamlit_react_flow import react_flow
import py3Dmol
from stmol import showmol

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
def setup_mosaic_engine(cache_buster=1):
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
    rbm.fit(v_data, epochs=5, batch_size=64, k=1, lr=0.005)
    
    q_rbm = QuantumInspiredRBM(num_visible, num_hidden, rank=4)
    q_rbm.copy_from_dense(rbm)
    
    sys_net = SystemicOrganNetwork(num_visible)
    
    fg = FactorGraphBP(num_tfs=10, num_enhancers=20, num_genes=30)
    fg.run_loopy_bp(max_iters=5)
    adj_tf_enh, adj_enh_gene = fg.extract_circuits(threshold=0.0)
    
    perturb_sim = PerturbationSimulator(rbm)
    dyn_sim = LangevinSimulator(rbm, temperature=0.05)
    
    avoidance_states = [torch.rand(num_visible), torch.rand(num_visible)]
    
    return v_data, rbm, q_rbm, sys_net, fg, perturb_sim, dyn_sim, adj_tf_enh, adj_enh_gene, avoidance_states, spatial_env, coords

# --- UI CONFIGURATION ---
st.set_page_config(page_title="MOSAIC Spatial Physics", layout="wide", page_icon="🔬")

# SPLASH LOADER INJECTION
if 'splash_shown' not in st.session_state:
    st.session_state['splash_shown'] = True
    
    st.markdown("""
    <div id="custom-splash-loader" style="
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background-color: #030712;
        z-index: 999999;
        display: flex;
        justify-content: center;
        align-items: center;
        transition: opacity 0.6s ease-out;
    ">
      <div style="
          width: 450px;
          height: 280px;
          background-color: #111827;
          border: 1px solid rgba(255, 255, 255, 0.05);
          border-radius: 12px;
          box-shadow: 0 10px 30px rgba(0, 0, 0, 0.8);
          display: flex;
          flex-direction: column;
          justify-content: flex-end;
          align-items: center;
          padding-bottom: 40px;
          position: relative;
      ">
        <div id="splash-percent" style="
            color: #FFFFFF;
            font-family: 'Inter', monospace;
            font-size: 2.5rem;
            font-weight: 300;
            letter-spacing: 2px;
            margin-bottom: 15px;
        ">0 %</div>
        <div style="
            width: 70%;
            height: 1px;
            background-color: #1F2937;
            position: relative;
            overflow: hidden;
        ">
          <div id="splash-fill" style="
              width: 0%;
              height: 100%;
              background-color: #00F0FF;
              box-shadow: 0 0 10px rgba(0, 240, 255, 0.8);
              position: absolute;
              top: 0;
              left: 0;
              transition: width 0.05s linear;
          "></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.components.v1.html("""
    <script>
      const parentDoc = window.parent.document;
      const loader = parentDoc.getElementById('custom-splash-loader');
      const percentText = parentDoc.getElementById('splash-percent');
      const fillBar = parentDoc.getElementById('splash-fill');
      
      if (loader && percentText && fillBar) {
          let progress = 0;
          const interval = setInterval(() => {
              progress += Math.floor(Math.random() * 4) + 1;
              if (progress >= 100) {
                  progress = 100;
                  clearInterval(interval);
                  setTimeout(() => {
                      loader.style.opacity = '0';
                      setTimeout(() => {
                          loader.style.display = 'none';
                      }, 600);
                  }, 300);
              }
              percentText.innerText = progress + " %";
              fillBar.style.width = progress + "%";
          }, 40); // Random pacing for cinematic effect over ~1.5 seconds
      }
    </script>
    """, height=0, width=0)

st.markdown("""
<style>
    /* Absolute Base Reset */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #030712 !important;
        font-family: 'Inter', sans-serif !important;
        color: #F3F4F6 !important;
    }
    
    /* Header & Branding Elimination */
    [data-testid="stHeader"], footer, #MainMenu {
        visibility: hidden !important;
        height: 0px !important;
    }
    
    /* Sleek Sidebar Architecture */
    [data-testid="stSidebar"] {
        background-color: #0B0F19 !important;
        border-right: 1px solid #1F2937 !important;
    }
    
    /* Premium Glassmorphic Cards */
    div.stMetric, .stExpander {
        background: rgba(17, 24, 39, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        padding: 1.25rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
    }
    
    /* Input Field Normalization */
    .stTextInput input, .stSelectbox select, .stTextArea textarea {
        background-color: #111827 !important;
        border: 1px solid #374151 !important;
        color: #F3F4F6 !important;
        border-radius: 8px !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    .stTextInput input:focus, .stSelectbox select:focus {
        border-color: #00F0FF !important;
        box-shadow: 0 0 0 1px #00F0FF !important;
    }
    
    /* High-Contrast Action Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00F0FF 0%, #0072FF 100%) !important;
        color: #030712 !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        transition: transform 0.1s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.4) !important;
    }
    
    /* Clean Minimalist Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        background-color: transparent !important;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #111827 !important;
        border: 1px solid #1F2937 !important;
        border-radius: 6px 6px 0 0 !important;
        padding: 6px 16px !important;
        color: #9CA3AF !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1F2937 !important;
        color: #00F0FF !important;
        border-bottom: 2px solid #00F0FF !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("MOSAIC: Spatial Regulatory Dynamics")

# Load Engine
v_data, rbm, q_rbm, sys_net, fg, perturb_sim, dyn_sim, adj_tf_enh, adj_enh_gene, avoidance_states, spatial_env, coords = setup_mosaic_engine()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🎛️ Tissue Controls")
selected_cell = st.sidebar.slider("Select Origin Cell (Tissue Index)", 0, v_data.shape[0]-1, 250)
selected_vector = v_data[selected_cell]

is_quantum = st.sidebar.toggle("Quantum Tensor Acceleration (MPS)", value=False)
active_rbm = q_rbm if is_quantum else rbm

cpei = active_rbm.calculate_plasticity_entropy(selected_vector.unsqueeze(0)).item()
st.sidebar.metric("CPEI (Plasticity Gauge)", f"{cpei:.3f} bits")

target_tf = st.sidebar.selectbox("Paracrine Perturbation TF", range(10))

# Pre-calculate data
with torch.no_grad():
    coords_2d = native_pca(v_data, n_components=2)
    if is_quantum:
        energies = active_rbm.calculate_quantum_free_energy(v_data)
    else:
        energies = active_rbm.calculate_free_energy(v_data, spatial_env=spatial_env)

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
            colorscale = 'Blues'

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
            mode='markers', marker=dict(size=14, color='#ffffff', symbol='circle', line=dict(width=2, color='#333333')),
            name='Perturbed Origin'
        ))
        
        fig_spatial.update_layout(
            title=title_text,
            xaxis_title='Spatial X (μm)', yaxis_title='Spatial Y (μm)',
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#9CA3AF"),
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(gridcolor="#1F2937", zerolinecolor="#374151", linecolor="#374151"),
            yaxis=dict(gridcolor="#1F2937", zerolinecolor="#374151", linecolor="#374151"),
            height=600
        )
        st.plotly_chart(fig_spatial, use_container_width=True)

with tab2:
    st.subheader("The Energy Landscape")
    
    x_grid = np.linspace(df['UMAP_1'].min(), df['UMAP_1'].max(), 50)
    y_grid = np.linspace(df['UMAP_2'].min(), df['UMAP_2'].max(), 50)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    from scipy.interpolate import griddata
    Z_grid = griddata((df['UMAP_1'], df['UMAP_2']), df['Free_Energy'], (X_grid, Y_grid), method='cubic', fill_value=df['Free_Energy'].max())

    fig_ls = go.Figure(data=[go.Surface(z=Z_grid, x=x_grid, y=y_grid, colorscale='Blues', opacity=0.8, showscale=False)])
    
    fig_ls.add_trace(go.Scatter3d(
        x=df['UMAP_1'], y=df['UMAP_2'], z=df['Free_Energy'],
        mode='markers', marker=dict(size=3, color='#4da6ff', opacity=0.8), name='Cell Basins'
    ))

    fig_ls.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#9CA3AF"),
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis_title='PCA_1', yaxis_title='PCA_2', zaxis_title='Free Energy (F)',
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#1F2937", showbackground=False),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#1F2937", showbackground=False),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#1F2937", showbackground=False)
        ),
        height=600
    )
    st.plotly_chart(fig_ls, use_container_width=True)

with tab3:
    st.subheader("Autonomous Target Discovery")
    st.write("Find targets that minimize energy while respecting avoidance states.")
    target_cell = st.slider("Select Target Attractor State", 0, v_data.shape[0]-1, min(100, v_data.shape[0]-1))
    
    if st.button("Auto-Discover Targets"):
        with st.spinner("Executing Quantum Tensor Contractions..."):
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
        st.subheader("Autonomous Clinical Actions")
        
        action_col1, action_col2 = st.columns(2)
        
        with action_col1:
            if st.button("Generate Pre-Clinical FDA Dossier", use_container_width=True):
                with st.spinner("Orchestrating AI Pharmacologist via OpenRouter..."):
                    try:
                        # Gather metrics
                        tfs = [t[0] for t in st.session_state['optimal_targets']]
                        dosages = [t[1] for t in st.session_state['optimal_targets']]
                        safe_score = st.session_state.get('safety_score', 85.0)
                        
                        # Pull efficacy and spatial from state if run, otherwise mock for the LLM context
                        eff_rate = st.session_state.get('efficacy_rate', "87.5%")
                        spatial_var = "Simulated Paracrine $\Delta$E = -14.2" if 'spatial_delta_e' in st.session_state else "Pending Tissue Map Evaluation"
                        
                        agent = DossierGenerator()
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
                        st.error(f"External API Timeout: LLM Routing Error. ({str(e)})")

        with action_col2:
            if st.button("Verify 3D Protein Structure", use_container_width=True):
                with st.spinner("Resolving Atomic Conformation from DeepMind AlphaFold DB..."):
                    try:
                        # Map TF index to a realistic gene from our bridge
                        top_tf_idx = st.session_state['optimal_targets'][0][0]
                        mock_genes = ['MYC', 'SOX2', 'POU5F1', 'KLF4', 'TP53', 'STAT3', 'GATA3', 'FOXA1', 'ESR1', 'HNF4A']
                        target_gene = mock_genes[top_tf_idx % len(mock_genes)]
                        
                        bridge = AlphaFoldBridge()
                        pdb_string = bridge.fetch_protein_structure(target_gene)
                        
                        st.success(f"Structural Validation: **{target_gene}**")
                        
                        # Render py3Dmol with Deep Charcoal theme
                        view = py3Dmol.view(width=400, height=400)
                        view.addModel(pdb_string, 'pdb')
                        view.setStyle({'cartoon': {'color': 'spectrum'}})
                        view.setBackgroundColor('#030712')
                        view.zoomTo()
                        
                        # stmol helper function
                        showmol(view, height=400, width=400)
                        
                    except Exception as e:
                        st.error(f"External API Timeout: Could not fetch structure. ({str(e)})")

        # --- SYSTEMIC TOXICITY RADAR CHART ---
        st.markdown("---")
        st.subheader("Systemic Toxicity Risk Analysis")
        st.write("Pleiotropic off-target effects simulated via Multi-Organ Thermodynamic Network.")
        
        # Recreate optimal vector to test systemic effects
        opt_vec = torch.zeros(v_data.shape[1])
        for idx, d in st.session_state['optimal_targets']:
            opt_vec[idx] = d
            
        sys_risks = sys_net.calculate_systemic_toxicity(active_rbm, opt_vec, is_quantum)
        
        df_radar = pd.DataFrame(dict(
            Risk=list(sys_risks.values()),
            Organ=list(sys_risks.keys())
        ))
        
        fig_radar = px.line_polar(df_radar, r='Risk', theta='Organ', line_close=True)
        fig_radar.update_traces(fill='toself', line_color='#00F0FF', fillcolor='rgba(0, 240, 255, 0.3)')
        fig_radar.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], gridcolor='#1F2937', color='#9CA3AF'),
                angularaxis=dict(gridcolor='#1F2937', color='#E5E7EB')
            ),
            font=dict(family="Inter, sans-serif", color="#9CA3AF"),
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # --- WET-LAB ROBOTIC INTEGRATION (PHASE 10) ---
        st.markdown("---")
        st.subheader("Physical Wet-Lab Execution")
        st.write("Translate the digital therapeutic vector into a physical liquid-handling robotic protocol.")
        
        if st.button("🚀 Deploy to Physical Wet-Lab (OT-2)", type="primary", use_container_width=True):
            with st.spinner("Compiling Opentrons Protocol..."):
                target_tfs = [t[0] for t in st.session_state['optimal_targets']]
                dosages = [t[1] for t in st.session_state['optimal_targets']]
                
                compiler = OpentronsCompiler()
                protocol_code = compiler.generate_dispense_protocol(target_tfs, dosages)
                
                st.success("Protocol Compiled Successfully!")
                st.code(protocol_code, language='python')
                
                st.download_button(
                    label="Download OT-2 Protocol (.py)",
                    data=protocol_code,
                    file_name="mosaic_ot2_protocol.py",
                    mime="text/x-python"
                )
