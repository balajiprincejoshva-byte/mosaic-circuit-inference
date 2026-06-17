import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import sys
import os

# Ensure core is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.matrix_io import MultiOmicTensor
from core.factor_graph import FactorGraphBP
from core.rbm_thermo import MultiOmicRBM
from core.perturbation import PerturbationSimulator
from core.dynamics import LangevinSimulator
from core.inverse_design import TargetOptimizer
from core.cohort_sim import VirtualCohort
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
    
    rna = np.random.poisson(1.5, (500, 30))
    atac = np.random.poisson(0.5, (500, 20))
    adt = np.random.poisson(2.0, (500, 10))
    
    tensor = MultiOmicTensor({'rna': rna, 'atac': atac, 'adt': adt})
    tensor.normalize_rna()
    tensor.normalize_atac()
    tensor.normalize_adt()
    
    v_data = tensor.to_tensor()
    
    num_visible = v_data.shape[1]
    num_hidden = 15
    rbm = MultiOmicRBM(num_visible, num_hidden)
    # Cache weights will be called inside fit()
    rbm.fit(v_data, epochs=5, batch_size=64, k=1, lr=0.05)
    
    fg = FactorGraphBP(num_tfs=10, num_enhancers=20, num_genes=30)
    fg.run_loopy_bp(max_iters=5)
    adj_tf_enh, adj_enh_gene = fg.extract_circuits(threshold=0.0)
    
    perturb_sim = PerturbationSimulator(rbm)
    dyn_sim = LangevinSimulator(rbm, temperature=0.5)
    
    # Generate avoidance states
    avoidance_states = [torch.rand(num_visible), torch.rand(num_visible)]
    
    return v_data, rbm, fg, perturb_sim, dyn_sim, adj_tf_enh, adj_enh_gene, avoidance_states

# --- UI CONFIGURATION ---
st.set_page_config(page_title="MOSAIC Pharma", layout="wide", page_icon="🧬")

st.markdown("""
    <style>
    .stApp { background-color: #121212; color: #e0e0e0; }
    h1, h2, h3 { color: #00ffcc; font-family: 'Courier New', Courier, monospace; }
    .stSidebar { background-color: #1e1e1e; border-right: 1px solid #00ffcc; }
    .stButton>button { background-color: #00ffcc; color: #121212; font-weight: bold; border: None; }
    .stButton>button:hover { background-color: #00e6b8; }
    .metric-card { background-color: #1e1e1e; border: 1px solid #00ffcc; border-radius: 10px; padding: 15px; margin-bottom: 20px;}
    .safety-safe { background-color: #1e3320; border: 1px solid #00ff00; border-radius: 10px; padding: 15px; color: #00ff00; }
    .safety-danger { background-color: #331e1e; border: 1px solid #ff0000; border-radius: 10px; padding: 15px; color: #ff0000; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { color: #e0e0e0; }
    </style>
""", unsafe_allow_html=True)

st.title("🧬 MOSAIC: Autonomous Clinical Engine")

# Load Engine
v_data, rbm, fg, perturb_sim, dyn_sim, adj_tf_enh, adj_enh_gene, avoidance_states = setup_mosaic_engine()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🎛️ Global Controls")
selected_cell = st.sidebar.slider("Select Origin Cell Basin", 0, v_data.shape[0]-1, 0)
target_cell = st.sidebar.slider("Select Target Cell Basin (for Inverse Design)", 0, v_data.shape[0]-1, min(100, v_data.shape[0]-1))

selected_vector = v_data[selected_cell]
target_vector = v_data[target_cell]

# Epigenetic Aging
st.sidebar.markdown("---")
st.sidebar.subheader("⏳ Thermodynamic Aging")
erosion_factor = st.sidebar.slider("Epigenetic Age (Erosion)", 0.0, 1.0, 0.0, 0.05)
# Apply the erosion factor globally to the RBM
rbm.apply_epigenetic_erosion(erosion_factor)

cpei = rbm.calculate_plasticity_entropy(selected_vector.unsqueeze(0)).item()
st.sidebar.markdown(f"""
<div class="metric-card">
    <h4>CPEI (Plasticity Gauge)</h4>
    <h2 style="margin: 0;">{cpei:.3f} bits</h2>
</div>
""", unsafe_allow_html=True)

# Project Data to 2D
with torch.no_grad():
    coords_2d = native_pca(v_data, n_components=2)
    energies = rbm.calculate_free_energy(v_data)

df = pd.DataFrame({
    'UMAP_1': coords_2d[:, 0].numpy(),
    'UMAP_2': coords_2d[:, 1].numpy(),
    'Free_Energy': energies.numpy(),
    'Cell_ID': np.arange(v_data.shape[0])
})

# --- MAIN TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["🌋 Landscape & Aging", "🤖 Autonomous Discovery", "🧪 Virtual Clinical Trials", "🔌 Causal Circuits"])

with tab1:
    st.subheader("The Energy Landscape")
    st.markdown("Observe how the Epigenetic Age slider actively flattens the thermodynamic barriers.")
    
    x_grid = np.linspace(df['UMAP_1'].min(), df['UMAP_1'].max(), 50)
    y_grid = np.linspace(df['UMAP_2'].min(), df['UMAP_2'].max(), 50)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    from scipy.interpolate import griddata
    Z_grid = griddata((df['UMAP_1'], df['UMAP_2']), df['Free_Energy'], (X_grid, Y_grid), method='cubic', fill_value=df['Free_Energy'].max())

    fig = go.Figure(data=[go.Surface(z=Z_grid, x=x_grid, y=y_grid, colorscale='Viridis', opacity=0.8, showscale=False)])
    
    fig.add_trace(go.Scatter3d(
        x=df['UMAP_1'], y=df['UMAP_2'], z=df['Free_Energy'],
        mode='markers', marker=dict(size=2, color='#00ffcc', opacity=0.5), name='Cell Basins'
    ))
    
    sel_x, sel_y, sel_z = df.iloc[selected_cell]['UMAP_1'], df.iloc[selected_cell]['UMAP_2'], df.iloc[selected_cell]['Free_Energy']
    fig.add_trace(go.Scatter3d(
        x=[sel_x], y=[sel_y], z=[sel_z],
        mode='markers', marker=dict(size=8, color='red', symbol='diamond'), name='Origin'
    ))
    
    tar_x, tar_y, tar_z = df.iloc[target_cell]['UMAP_1'], df.iloc[target_cell]['UMAP_2'], df.iloc[target_cell]['Free_Energy']
    fig.add_trace(go.Scatter3d(
        x=[tar_x], y=[tar_y], z=[tar_z],
        mode='markers', marker=dict(size=8, color='orange', symbol='star'), name='Target Basin'
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='PCA_1', yaxis_title='PCA_2', zaxis_title='Free Energy (F)',
            bgcolor='#121212', xaxis=dict(showbackground=False, gridcolor='#333'),
            yaxis=dict(showbackground=False, gridcolor='#333'), zaxis=dict(showbackground=False, gridcolor='#333'),
            zaxis_range=[-20, max(0, Z_grid.max()+5)] # Lock Z to see erosion effects
        ),
        paper_bgcolor='#121212', plot_bgcolor='#121212', margin=dict(l=0, r=0, b=0, t=0), height=600
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Autonomous Inverse Target Discovery")
    st.write("Reverse-engineer the optimal multi-gene perturbation required to push a cell into the Target Basin safely.")
    
    if st.button("Auto-Discover Optimal Targets"):
        with st.spinner("Running Adam Optimizer on Landscape..."):
            optimizer = TargetOptimizer(rbm, target_vector, avoidance_states)
            optimal_delta_v, top_targets = optimizer.optimize(steps=150, lr=0.1)
            
            # Predict safety
            dest_state = torch.clamp(selected_vector + optimal_delta_v, 0.0, 1.0)
            off_target, safety_score = perturb_sim._calculate_safety(dest_state, avoidance_states)
            
            # Save to session for use in Clinical Trials Tab
            st.session_state['optimal_targets'] = top_targets
            st.session_state['safety_score'] = safety_score
            
            st.success("Target Discovery Complete!")
            
    if 'optimal_targets' in st.session_state:
        score = st.session_state['safety_score']
        safe_class = "safety-safe" if score > 50 else "safety-danger"
        st.markdown(f'<div class="{safe_class}"><h4>Safety Score: {score:.1f}/100</h4></div>', unsafe_allow_html=True)
        st.write("### Target Leaderboard")
        
        target_df = []
        for rank, (idx, dosage) in enumerate(st.session_state['optimal_targets']):
            target_df.append({"Rank": rank+1, "Global Feature Index": idx, "Dosage (Δv)": f"{dosage:.4f}"})
        
        st.table(pd.DataFrame(target_df))

with tab3:
    st.subheader("Virtual Clinical Trials")
    st.write("Simulate the robustness of the top discovered perturbation across a population with biological variance.")
    
    if st.button("Run Virtual Cohort (1,000 Patients)"):
        if 'optimal_targets' not in st.session_state or len(st.session_state['optimal_targets']) == 0:
            st.warning("Please run Auto-Discover Optimal Targets first!")
        else:
            with st.spinner("Simulating Population Dynamics..."):
                top_target_idx = st.session_state['optimal_targets'][0][0]
                top_target_dosage = st.session_state['optimal_targets'][0][1]
                target_val = torch.clamp(selected_vector[top_target_idx] + top_target_dosage, 0.0, 1.0).item()
                
                cohort = VirtualCohort(selected_vector, n_patients=1000, variance=0.15)
                efficacy_rate, final_energies = cohort.run_trial(
                    dyn_sim, rbm, target_gene_idx=top_target_idx, target_gene_value=target_val, steps=50
                )
                
                initial_energies = rbm.calculate_free_energy(cohort.cohort_matrix).numpy()
                final_e_np = final_energies.numpy()
                
                success_threshold = np.mean(initial_energies) - 0.5
                
                # Plotly Histogram
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(x=initial_energies, name='Initial Cohort Energy', opacity=0.5, marker_color='gray'))
                
                # Split final energies into success and fail
                success_e = final_e_np[final_e_np < success_threshold]
                fail_e = final_e_np[final_e_np >= success_threshold]
                
                fig_hist.add_trace(go.Histogram(x=success_e, name='Successfully Reprogrammed', opacity=0.8, marker_color='#00ffcc'))
                fig_hist.add_trace(go.Histogram(x=fail_e, name='Failed (Off-Target)', opacity=0.8, marker_color='#ff4d4d'))
                
                fig_hist.update_layout(
                    barmode='overlay', title=f"Clinical Efficacy Rate: {efficacy_rate:.1f}%",
                    xaxis_title="Free Energy Basin", yaxis_title="Patient Count",
                    paper_bgcolor='#121212', plot_bgcolor='#121212', font_color='#e0e0e0'
                )
                
                st.plotly_chart(fig_hist, use_container_width=True)

with tab4:
    st.subheader("🔌 Causal Circuit Explorer")
    st.write("Belief Propagation Graph for Top Inferred TF Connections")
    target_tf = st.selectbox("Select TF Node", range(10))
    
    elements = []
    elements.append({"id": f"TF_{target_tf}", "data": {"label": f"TF {target_tf}"}, "position": {"x": 250, "y": 50}, "style": {"background": "#ff4d4d", "color": "#fff", "border": "1px solid #ff4d4d"}})
    connected_enh = np.where(adj_tf_enh[target_tf, :] > 0)[0]
    
    for i, enh in enumerate(connected_enh[:3]):
        enh_id = f"Enh_{enh}"
        elements.append({"id": enh_id, "data": {"label": f"Enh {enh}"}, "position": {"x": 100 + i*150, "y": 200}, "style": {"background": "#00ffcc", "color": "#000", "border": "1px solid #00ffcc"}})
        elements.append({"id": f"e_{target_tf}_{enh}", "source": f"TF_{target_tf}", "target": enh_id, "animated": True, "style": {"stroke": "#00ffcc"}})
        
        connected_genes = np.where(adj_enh_gene[enh, :] > 0)[0]
        for j, gene in enumerate(connected_genes[:2]):
            gene_id = f"Gene_{gene}"
            elements.append({"id": gene_id, "data": {"label": f"Gene {gene}"}, "position": {"x": 50 + i*150 + j*100, "y": 350}, "style": {"background": "#4da6ff", "color": "#fff", "border": "1px solid #4da6ff"}})
            elements.append({"id": f"e_{enh}_{gene}", "source": enh_id, "target": gene_id, "animated": True, "style": {"stroke": "#4da6ff"}})

    if len(elements) <= 1:
        st.info("No connections found.")
    else:
        react_flow("circuit_flow", elements=elements, flow_styles={"height": 500, "width": "100%", "background": "#1e1e1e"})
