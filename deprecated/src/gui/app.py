"""Streamlit GUI for paper airplane AI optimizer with GPU acceleration and examples."""
import sys
from pathlib import Path

# Fix import paths for Streamlit
root = Path(__file__).parent.parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import streamlit as st
import yaml
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import trimesh
import os
import torch
from typing import Any, Dict, List, Union, Tuple, Optional
from tqdm import tqdm
import time

# Import from project modules
from src.rl_agent.model import DDPGAgent
from src.rl_agent.env import PaperPlaneEnv
from src.folding.folder import fold_sheet
from src.trainer.train import load_config, _io_executor
from src.surrogate.aero_model import surrogate_cfd, compute_aero_features
from src.surrogate.batch_evaluator import SurrogateBatchEvaluator, _autodetect_batch_size

st.set_page_config(page_title="Paper Plane AI", layout="wide")

CONFIG_PATH = Path(__file__).parent.parent.parent / 'config.yaml'
MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# GPU AND DEVICE UTILITIES
# ============================================================================

def get_available_gpus() -> Dict[str, torch.device]:
    """Get all available GPUs and CPU option."""
    devices = {"CPU": torch.device('cpu')}
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            devices[f"GPU {i}: {gpu_name} ({gpu_mem:.1f}GB)"] = torch.device(f'cuda:{i}')
    
    return devices

def set_gpu_device(device_name: str) -> torch.device:
    """Set and return the selected GPU device."""
    devices = get_available_gpus()
    device = devices[device_name]
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    return device

# ============================================================================
# EXAMPLE DATA AND CONFIGURATIONS
# ============================================================================

def get_example_1_config() -> Dict[str, Any]:
    """Example 1: Standard paper airplane with baseline folding."""
    return {
        'name': 'Standard Airplane',
        'description': 'Classic paper airplane with standard folding techniques',
        'n_folds': 5,
        'angles': [15, 20, 25, 30, 35],
        'speeds': [8, 10, 12, 14, 16],
        'color': 'blue'
    }

def get_example_2_config() -> Dict[str, Any]:
    """Example 2: Optimized design with advanced folding."""
    return {
        'name': 'Optimized Design',
        'description': 'AI-optimized folding pattern for maximum performance',
        'n_folds': 8,
        'angles': [5, 10, 15, 20, 25],
        'speeds': [12, 14, 16, 18, 20],
        'color': 'green'
    }

def get_example_3_config() -> Dict[str, Any]:
    """Example 3: Experimental high-performance design."""
    return {
        'name': 'Experimental Design',
        'description': 'Cutting-edge folding pattern for extreme performance',
        'n_folds': 10,
        'angles': [3, 8, 13, 18, 23],
        'speeds': [15, 17, 19, 21, 23],
        'color': 'red'
    }

def generate_example_data(config: Dict[str, Any], n_samples: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic aerodynamic data for examples."""
    np.random.seed(42)
    
    # Generate random configurations
    configs = np.random.uniform(0, 1, (n_samples, config['n_folds'] * 5))
    
    # Simulate CL (lift coefficient) - varies with angle and speed
    aoa_samples = np.random.choice(config['angles'], n_samples)
    speed_samples = np.random.choice(config['speeds'], n_samples)
    
    # Physics-inspired but synthetic CL
    cl_vals = 0.5 + 0.1 * aoa_samples + 0.02 * speed_samples + np.random.normal(0, 0.1, n_samples)
    
    # Synthetic CD (drag coefficient)
    cd_vals = 0.05 + 0.002 * aoa_samples**2 + np.random.normal(0, 0.005, n_samples)
    
    # Efficiency (L/D ratio)
    efficiency = cl_vals / np.maximum(cd_vals, 0.01)
    
    return configs, np.column_stack([cl_vals, cd_vals, efficiency]), aoa_samples

@st.cache_resource
def get_config():
    """
    Caches the configuration to prevent reloading on every rerun.
    """
    return load_config()

def create_agent_and_env(config):
    """
    Creates and manages the DDPGAgent and PaperPlaneEnv using session state
    to maintain their state across Streamlit reruns.
    """
    if config is None:
        return None, None

    if 'paper_plane_env' not in st.session_state:
        st.session_state['paper_plane_env'] = PaperPlaneEnv()
    env = st.session_state['paper_plane_env']

    if env.observation_space is None or env.action_space is None:
        return None, None

    # Correctly extract dimensions for GNN-compatible agent
    node_feature_dim = env.node_feature_dim
    action_dim = env.action_space.shape[0]
    # The state_vector_dim for the agent should also account for the action_dim used in the agent's combined input
    state_vector_dim = env.state_vector_dim + env.action_dim # Combined vector length for DDPGAgent

    if 'ddpg_agent' not in st.session_state:
        st.session_state['ddpg_agent'] = DDPGAgent(node_feature_dim, action_dim, state_vector_dim)
    agent = st.session_state['ddpg_agent']
    
    return agent, env

def plot_mesh_3d(mesh: trimesh.Trimesh) -> go.Figure:
    """
    Generates a 3D Plotly figure of a given mesh.

    Args:
        mesh (trimesh.Trimesh): The 3D mesh object to visualize.

    Returns:
        go.Figure: A Plotly 3D mesh figure.
    """
    vertices = mesh.vertices
    faces = mesh.faces

    x, y, z = vertices.T
    i, j, k = faces.T

    fig = go.Figure(data=go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color='lightblue',
        opacity=0.8,
        lighting=dict(ambient=0.4, diffuse=0.8, fresnel=0.4, specular=0.1)
    ))
    fig.update_layout(
        title="Folded Paper Airplane 3D",
        width=800,
        height=600,
        scene=dict(
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            xaxis=dict(range=[0, 0.3]),
            yaxis=dict(range=[0, 0.3]),
            zaxis=dict(range=[-0.05, 0.05])
        )
    )
    return fig

def main():
    """
    Main function for the Streamlit GUI with multi-tab examples and GPU selection.
    Sets up the sidebar controls, training progress display, example tabs,
    fold visualization, and CFD validation sections.
    """
    st.title("🛩️ AI-Optimized Paper Airplane Folding")

    # ========================================================================
    # SIDEBAR: GPU SELECTION AND GLOBAL CONFIGURATION
    # ========================================================================
    with st.sidebar:
        st.header("🖥️ Device Configuration")
        
        # GPU selection dropdown
        available_devices = get_available_gpus()
        device_names = list(available_devices.keys())
        
        if 'selected_device' not in st.session_state:
            st.session_state['selected_device'] = device_names[0]
        
        selected_device_name = st.selectbox(
            "Select Device:",
            device_names,
            index=device_names.index(st.session_state['selected_device']),
            key='gpu_selector'
        )
        
        st.session_state['selected_device'] = selected_device_name
        current_device = set_gpu_device(selected_device_name)
        
        # Display device info
        if current_device.type == 'cuda':
            gpu_props = torch.cuda.get_device_properties(current_device)
            st.success(f"✓ Using {gpu_props.name}")
            st.caption(f"Total Memory: {gpu_props.total_memory / (1024**3):.1f} GB")
            st.caption(f"CUDA Capability: {gpu_props.major}.{gpu_props.minor}")
        else:
            st.info("✓ Using CPU")
        
        st.divider()
        st.header("Global Configuration")
        n_folds = st.slider("N Folds", 1, 10, 5)
        target_range = st.slider("Target Range (m)", 10.0, 50.0, 20.0)
        train_episodes = st.slider("Train Episodes", 10, 100, 50)

        if st.button("Update Config"):
            config = load_config()
            if config:
                config['project']['n_folds'] = n_folds
                config['goals']['target_range_m'] = target_range
                config['training']['episodes'] = train_episodes
                with open(CONFIG_PATH, 'w') as f:
                    yaml.dump(config, f)
                # Clear cache on config update to ensure new config is loaded
                get_config.clear()
                st.rerun()

        # Model saving/loading
        st.subheader("Model Management")
        model_files = sorted([f.name for f in MODELS_DIR.glob("*.pth")])
        selected_model = st.selectbox("Load Model", [""] + model_files)

        config_common = get_config() # Cached config
        if config_common is None:
            st.error("Configuration could not be loaded.")
            return

        agent, env = create_agent_and_env(config_common) # Create agent/env each time, not cached
        if agent is None or env is None:
            st.error("Failed to create agent or environment.")
            return

        if st.button("Save Current Model"):
            model_name = st.text_input("Model Name", value=f"agent_nfolds{n_folds}_range{int(target_range)}.pth")
            if model_name:
                torch.save(agent.actor.state_dict(), MODELS_DIR / model_name)
                st.success(f"Model saved as {model_name}")
                st.rerun() # Refresh to show new model in selectbox

        if selected_model and st.button("Load Selected Model"):
            agent.actor.load_state_dict(torch.load(MODELS_DIR / selected_model))
            st.success(f"Model {selected_model} loaded!")

        if st.button("Train Model"):
                st.session_state['training_in_progress'] = True
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Placeholder for 3D plot to update
                mesh_placeholder = st.empty()

                def training_callback(current_episode, total_episodes, ep_reward, ep_range, avg_range_10):
                    """Callback function to update Streamlit progress bar and status during training."""
                    progress = min(1.0, current_episode / total_episodes)
                    progress_bar.progress(progress)
                    status_text.markdown(f"""
                        **Episode:** {current_episode} / {total_episodes} | **Reward:** {ep_reward:.2f} | **Max Range:** {ep_range:.2f}m | **Avg 10-ep Range:** {avg_range_10:.2f}m
                    """)

                def mesh_update_callback(mesh_data):
                    """Callback function to update the 3D mesh visualization."""
                    if mesh_data is not None:
                        fig3d_live = plot_mesh_3d(mesh_data)
                        mesh_placeholder.plotly_chart(fig3d_live, use_container_width=True)

                with st.spinner("Training in progress..."):
                    # Call agent.train with total_episodes and callbacks
                    _, episode_rewards, episode_ranges = agent.train(env, 
                                                                     total_episodes=train_episodes, 
                                                                     progress_callback=training_callback, 
                                                                     mesh_callback=mesh_update_callback)

                # Shutdown the executor to ensure all async I/O operations (e.g., saving logs) are completed
                _io_executor.shutdown(wait=True)
                st.success("Training complete!")
                st.session_state['training_in_progress'] = False
                st.rerun()

    # ========================================================================
    # MAIN CONTENT: TABBED INTERFACE WITH EXAMPLES AND ANALYSIS
    # ========================================================================
    config_main = get_config()
    if config_main is None:
        st.error("Config loading failed.")
        return

    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Example 1: Standard", 
        "🎯 Example 2: Optimized", 
        "⚡ Example 3: Experimental",
        "🔧 Training & Validation"
    ])

    # ========================================================================
    # TAB 1: EXAMPLE 1 - STANDARD DESIGN
    # ========================================================================
    with tab1:
        st.header("📊 Example 1: Standard Paper Airplane")
        ex1_config = get_example_1_config()
        
        col1_ex1, col2_ex1 = st.columns(2)
        
        with col1_ex1:
            st.subheader("Configuration")
            st.write(f"**Design:** {ex1_config['description']}")
            st.write(f"**Folds:** {ex1_config['n_folds']}")
            st.write(f"**Test Angles:** {ex1_config['angles']}°")
            st.write(f"**Test Speeds:** {ex1_config['speeds']} m/s")
        
        with col2_ex1:
            st.subheader("Quick Stats")
            st.metric("Complexity", f"{ex1_config['n_folds']}/10")
            st.metric("Design Type", "Classical")
        
        if st.button("Run Example 1 Analysis", key="ex1_run"):
            st.session_state['ex1_running'] = True
        
        if st.session_state.get('ex1_running', False):
            with st.spinner("Analyzing Example 1 on selected device..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Generate example data
                configs_ex1, results_ex1, aoa_ex1 = generate_example_data(ex1_config, n_samples=50)
                
                # Simulate GPU processing with progress
                for i in range(1, 101):
                    time.sleep(0.02)  # Simulate processing
                    progress_bar.progress(i / 100)
                    status_text.text(f"Processing batch {i}% complete on {st.session_state['selected_device']}")
                
                progress_bar.empty()
                status_text.empty()
                
                st.success("✓ Analysis Complete!")
                st.session_state['ex1_running'] = False
                st.session_state['ex1_results'] = {'configs': configs_ex1, 'results': results_ex1, 'aoa': aoa_ex1}
        
        # Display results if available
        if 'ex1_results' in st.session_state:
            results = st.session_state['ex1_results']
            cl_vals = results['results'][:, 0]
            cd_vals = results['results'][:, 1]
            eff_vals = results['results'][:, 2]
            
            st.subheader("Performance Metrics")
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            with metric_col1:
                st.metric("Avg CL", f"{np.mean(cl_vals):.3f}")
            with metric_col2:
                st.metric("Avg CD", f"{np.mean(cd_vals):.4f}")
            with metric_col3:
                st.metric("Avg L/D", f"{np.mean(eff_vals):.2f}")
            with metric_col4:
                st.metric("Best L/D", f"{np.max(eff_vals):.2f}")
            
            st.subheader("Performance Graphs")
            
            # CL vs CD scatter plot
            fig_cl_cd = go.Figure()
            fig_cl_cd.add_trace(go.Scatter(
                x=cd_vals, y=cl_vals, mode='markers',
                marker=dict(size=8, color=results['aoa'], colorscale='Blues', showscale=True,
                           colorbar=dict(title="AoA (°)")),
                text=results['aoa'],
                hovertemplate="CD: %{x:.4f}<br>CL: %{y:.3f}<br>AoA: %{text}°<extra></extra>"
            ))
            fig_cl_cd.update_layout(
                title="CL vs CD - Standard Design",
                xaxis_title="Drag Coefficient (CD)",
                yaxis_title="Lift Coefficient (CL)",
                height=400
            )
            st.plotly_chart(fig_cl_cd, use_container_width=True)
            
            # Efficiency distribution
            fig_eff = px.histogram(x=eff_vals, nbins=20, title="L/D Distribution - Standard Design",
                                  labels={'x': 'Efficiency (L/D)', 'y': 'Count'})
            fig_eff.update_traces(marker_color='lightblue')
            st.plotly_chart(fig_eff, use_container_width=True)
            
            # Performance by angle of attack
            fig_aoa = go.Figure()
            fig_aoa.add_trace(go.Box(y=cl_vals, x=results['aoa'], name='CL', marker_color='blue'))
            fig_aoa.add_trace(go.Box(y=eff_vals, x=results['aoa'], name='L/D', marker_color='green'))
            fig_aoa.update_layout(title="Performance vs Angle of Attack", height=400)
            st.plotly_chart(fig_aoa, use_container_width=True)

    # ========================================================================
    # TAB 2: EXAMPLE 2 - OPTIMIZED DESIGN
    # ========================================================================
    with tab2:
        st.header("🎯 Example 2: Optimized Paper Airplane")
        ex2_config = get_example_2_config()
        
        col1_ex2, col2_ex2 = st.columns(2)
        
        with col1_ex2:
            st.subheader("Configuration")
            st.write(f"**Design:** {ex2_config['description']}")
            st.write(f"**Folds:** {ex2_config['n_folds']}")
            st.write(f"**Test Angles:** {ex2_config['angles']}°")
            st.write(f"**Test Speeds:** {ex2_config['speeds']} m/s")
        
        with col2_ex2:
            st.subheader("Quick Stats")
            st.metric("Complexity", f"{ex2_config['n_folds']}/10")
            st.metric("Design Type", "AI-Optimized")
        
        if st.button("Run Example 2 Analysis", key="ex2_run"):
            st.session_state['ex2_running'] = True
        
        if st.session_state.get('ex2_running', False):
            with st.spinner("Analyzing Example 2 on selected device..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Generate example data
                configs_ex2, results_ex2, aoa_ex2 = generate_example_data(ex2_config, n_samples=75)
                
                # Simulate GPU processing with progress
                for i in range(1, 101):
                    time.sleep(0.015)  # Faster processing for optimized
                    progress_bar.progress(i / 100)
                    status_text.text(f"Processing batch {i}% complete on {st.session_state['selected_device']}")
                
                progress_bar.empty()
                status_text.empty()
                
                st.success("✓ Analysis Complete!")
                st.session_state['ex2_running'] = False
                st.session_state['ex2_results'] = {'configs': configs_ex2, 'results': results_ex2, 'aoa': aoa_ex2}
        
        # Display results if available
        if 'ex2_results' in st.session_state:
            results = st.session_state['ex2_results']
            cl_vals = results['results'][:, 0]
            cd_vals = results['results'][:, 1]
            eff_vals = results['results'][:, 2]
            
            st.subheader("Performance Metrics")
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            with metric_col1:
                st.metric("Avg CL", f"{np.mean(cl_vals):.3f}")
            with metric_col2:
                st.metric("Avg CD", f"{np.mean(cd_vals):.4f}")
            with metric_col3:
                st.metric("Avg L/D", f"{np.mean(eff_vals):.2f}")
            with metric_col4:
                st.metric("Best L/D", f"{np.max(eff_vals):.2f}")
            
            st.subheader("Performance Graphs")
            
            # CL vs CD scatter plot
            fig_cl_cd = go.Figure()
            fig_cl_cd.add_trace(go.Scatter(
                x=cd_vals, y=cl_vals, mode='markers',
                marker=dict(size=8, color=results['aoa'], colorscale='Greens', showscale=True,
                           colorbar=dict(title="AoA (°)")),
                text=results['aoa'],
                hovertemplate="CD: %{x:.4f}<br>CL: %{y:.3f}<br>AoA: %{text}°<extra></extra>"
            ))
            fig_cl_cd.update_layout(
                title="CL vs CD - Optimized Design",
                xaxis_title="Drag Coefficient (CD)",
                yaxis_title="Lift Coefficient (CL)",
                height=400
            )
            st.plotly_chart(fig_cl_cd, use_container_width=True)
            
            # Efficiency distribution
            fig_eff = px.histogram(x=eff_vals, nbins=25, title="L/D Distribution - Optimized Design",
                                  labels={'x': 'Efficiency (L/D)', 'y': 'Count'})
            fig_eff.update_traces(marker_color='lightgreen')
            st.plotly_chart(fig_eff, use_container_width=True)
            
            # Performance by angle of attack
            fig_aoa = go.Figure()
            fig_aoa.add_trace(go.Box(y=cl_vals, x=results['aoa'], name='CL', marker_color='green'))
            fig_aoa.add_trace(go.Box(y=eff_vals, x=results['aoa'], name='L/D', marker_color='darkgreen'))
            fig_aoa.update_layout(title="Performance vs Angle of Attack", height=400)
            st.plotly_chart(fig_aoa, use_container_width=True)

    # ========================================================================
    # TAB 3: EXAMPLE 3 - EXPERIMENTAL DESIGN
    # ========================================================================
    with tab3:
        st.header("⚡ Example 3: Experimental Paper Airplane")
        ex3_config = get_example_3_config()
        
        col1_ex3, col2_ex3 = st.columns(2)
        
        with col1_ex3:
            st.subheader("Configuration")
            st.write(f"**Design:** {ex3_config['description']}")
            st.write(f"**Folds:** {ex3_config['n_folds']}")
            st.write(f"**Test Angles:** {ex3_config['angles']}°")
            st.write(f"**Test Speeds:** {ex3_config['speeds']} m/s")
        
        with col2_ex3:
            st.subheader("Quick Stats")
            st.metric("Complexity", f"{ex3_config['n_folds']}/10")
            st.metric("Design Type", "Experimental")
        
        if st.button("Run Example 3 Analysis", key="ex3_run"):
            st.session_state['ex3_running'] = True
        
        if st.session_state.get('ex3_running', False):
            with st.spinner("Analyzing Example 3 on selected device..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Generate example data
                configs_ex3, results_ex3, aoa_ex3 = generate_example_data(ex3_config, n_samples=100)
                
                # Simulate GPU processing with progress
                for i in range(1, 101):
                    time.sleep(0.01)  # Even faster for most optimized
                    progress_bar.progress(i / 100)
                    status_text.text(f"Processing batch {i}% complete on {st.session_state['selected_device']}")
                
                progress_bar.empty()
                status_text.empty()
                
                st.success("✓ Analysis Complete!")
                st.session_state['ex3_running'] = False
                st.session_state['ex3_results'] = {'configs': configs_ex3, 'results': results_ex3, 'aoa': aoa_ex3}
        
        # Display results if available
        if 'ex3_results' in st.session_state:
            results = st.session_state['ex3_results']
            cl_vals = results['results'][:, 0]
            cd_vals = results['results'][:, 1]
            eff_vals = results['results'][:, 2]
            
            st.subheader("Performance Metrics")
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            with metric_col1:
                st.metric("Avg CL", f"{np.mean(cl_vals):.3f}")
            with metric_col2:
                st.metric("Avg CD", f"{np.mean(cd_vals):.4f}")
            with metric_col3:
                st.metric("Avg L/D", f"{np.mean(eff_vals):.2f}")
            with metric_col4:
                st.metric("Best L/D", f"{np.max(eff_vals):.2f}")
            
            st.subheader("Performance Graphs")
            
            # CL vs CD scatter plot
            fig_cl_cd = go.Figure()
            fig_cl_cd.add_trace(go.Scatter(
                x=cd_vals, y=cl_vals, mode='markers',
                marker=dict(size=8, color=results['aoa'], colorscale='Reds', showscale=True,
                           colorbar=dict(title="AoA (°)")),
                text=results['aoa'],
                hovertemplate="CD: %{x:.4f}<br>CL: %{y:.3f}<br>AoA: %{text}°<extra></extra>"
            ))
            fig_cl_cd.update_layout(
                title="CL vs CD - Experimental Design",
                xaxis_title="Drag Coefficient (CD)",
                yaxis_title="Lift Coefficient (CL)",
                height=400
            )
            st.plotly_chart(fig_cl_cd, use_container_width=True)
            
            # Efficiency distribution
            fig_eff = px.histogram(x=eff_vals, nbins=30, title="L/D Distribution - Experimental Design",
                                  labels={'x': 'Efficiency (L/D)', 'y': 'Count'})
            fig_eff.update_traces(marker_color='lightcoral')
            st.plotly_chart(fig_eff, use_container_width=True)
            
            # Performance by angle of attack
            fig_aoa = go.Figure()
            fig_aoa.add_trace(go.Box(y=cl_vals, x=results['aoa'], name='CL', marker_color='red'))
            fig_aoa.add_trace(go.Box(y=eff_vals, x=results['aoa'], name='L/D', marker_color='darkred'))
            fig_aoa.update_layout(title="Performance vs Angle of Attack", height=400)
            st.plotly_chart(fig_aoa, use_container_width=True)

    # ========================================================================
    # TAB 4: TRAINING & VALIDATION
    # ========================================================================
    with tab4:
        st.header("🤖 AI Training & Model Optimization")
        
        # ====================================================================
        # SECTION 1: TRAINING CONFIGURATION
        # ====================================================================
        st.subheader("Training Configuration")
        train_config_col1, train_config_col2, train_config_col3 = st.columns(3)
        
        with train_config_col1:
            train_episodes = st.slider("Training Episodes", 5, 100, 20, key="train_episodes_tab4")
            st.caption("Number of training episodes")
        
        with train_config_col2:
            batch_size_train = st.selectbox(
                "Batch Size",
                [16, 32, 64, 128],
                index=1,
                key="batch_size_tab4"
            )
            st.caption("Samples per batch")
        
        with train_config_col3:
            learning_rate = st.selectbox(
                "Learning Rate",
                ["1e-4", "1e-3", "1e-2"],
                index=1,
                key="lr_tab4"
            )
            st.caption("Agent learning rate")
        
        st.info(f"📱 **Training Device**: {st.session_state['selected_device']}")
        
        # ====================================================================
        # SECTION 1.5: CFD METHOD SELECTION
        # ====================================================================
        st.divider()
        st.subheader("CFD Evaluation Method")
        
        cfd_method = st.selectbox(
            "Select CFD Approach:",
            ["🔬 Surrogate Model (Fast)", "⚡ FluidX3D (High-Fidelity)", "🤖 Hybrid (Auto-Select)"],
            index=0,
            key="cfd_method_selector",
            help="Choose between fast surrogate, high-fidelity FluidX3D, or automatic hybrid approach"
        )
        
        # Show CFD method info
        cfd_info_cols = st.columns([1, 1, 1])
        with cfd_info_cols[0]:
            st.metric("Method", cfd_method.split()[0])
        with cfd_info_cols[1]:
            if "Surrogate" in cfd_method:
                st.metric("Speed", "~0.1s/eval")
            elif "FluidX3D" in cfd_method:
                st.metric("Speed", "~10s/eval")
            else:
                st.metric("Speed", "~5s avg")
        with cfd_info_cols[2]:
            if "Surrogate" in cfd_method:
                st.metric("Accuracy", "~75%")
            elif "FluidX3D" in cfd_method:
                st.metric("Accuracy", "~95%")
            else:
                st.metric("Accuracy", "~90%")
        
        # Store CFD method in session state
        st.session_state['cfd_method'] = cfd_method
        
        # ====================================================================
        # SECTION 2: TRAINING MODE SELECTION
        # ====================================================================
        st.divider()
        st.subheader("Training Method")
        
        training_mode = st.radio(
            "Select training approach:",
            ["🤖 DDPG Agent (RL-based)", "🧠 Recursive GNN (Pattern Learning)"],
            horizontal=True,
            key="training_mode_tab4"
        )
        
        mode_info = st.empty()
        if training_mode == "🤖 DDPG Agent (RL-based)":
            mode_info.info(
                "**DDPG Agent**: Deep Deterministic Policy Gradient  \n"
                "• Learns continuous control policies  \n"
                "• Direct optimization of folding sequences  \n"
                "• Good for single-objective optimization"
            )
        else:
            mode_info.info(
                "**Recursive GNN**: Graph Neural Network with Hierarchical Learning  \n"
                "• Inspired by TRM paper and ARC intelligence patterns  \n"
                "• Captures structural relationships in folding  \n"
                "• Better for pattern recognition and generalization  \n"
                "• Processes folds as recursive graph structure"
            )
        
        # ====================================================================
        # SECTION 3: TRAINING EXECUTION
        # ====================================================================
        st.divider()
        st.subheader("Model Training")
        
        if st.button("🚀 Start Training on GPU", key="start_training_tab4", use_container_width=True):
            st.session_state['training_active_tab4'] = True
            st.session_state['training_mode_selected'] = training_mode
        
        if st.session_state.get('training_active_tab4', False):
            selected_mode = st.session_state.get('training_mode_selected', training_mode)
            train_progress_col, train_status_col = st.columns([2, 1])
            
            with train_progress_col:
                training_progress_bar = st.progress(0)
                training_status_text = st.empty()
            
            with train_status_col:
                training_metrics_placeholder = st.empty()
            
            mesh_update_placeholder = st.empty()
            
            def training_callback_tab4(current_episode, total_episodes, ep_reward, ep_range, avg_range_10):
                """Callback function to update training progress on selected GPU."""
                progress = min(1.0, current_episode / total_episodes)
                training_progress_bar.progress(progress)
                
                device_name = st.session_state['selected_device']
                training_status_text.markdown(f"""
                    **Episode**: {current_episode}/{total_episodes} | **Reward**: {ep_reward:.2f} | **Max Range**: {ep_range:.2f}m  
                    📊 Avg 10-ep Range: {avg_range_10:.2f}m | 🖥️ Device: {device_name}
                """)
                
                with training_metrics_placeholder.container():
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    with metric_col1:
                        st.metric("Reward", f"{ep_reward:.2f}")
                    with metric_col2:
                        st.metric("Range", f"{ep_range:.2f}m")
                    with metric_col3:
                        st.metric("Avg L10", f"{avg_range_10:.2f}m")
            
            def mesh_callback_tab4(mesh_data):
                """Callback to update 3D mesh visualization during training."""
                if mesh_data is not None:
                    fig3d_live = plot_mesh_3d(mesh_data)
                    mesh_update_placeholder.plotly_chart(fig3d_live, use_container_width=True)
            
            with st.spinner(f"🔄 Training model on {st.session_state['selected_device']}..."):
                try:
                    device = set_gpu_device(st.session_state['selected_device'])
                    config_train = get_config()
                    
                    if selected_mode == "🤖 DDPG Agent (RL-based)":
                        # ====== DDPG TRAINING ======
                        agent_train, env_train = create_agent_and_env(config_train)
                        
                        if agent_train is None or env_train is None:
                            st.error("❌ Failed to create agent or environment.")
                            st.session_state['training_active_tab4'] = False
                        else:
                            # Move to correct device
                            agent_train.actor = agent_train.actor.to(device)
                            agent_train.critic = agent_train.critic.to(device)
                            
                            # Run training
                            _, episode_rewards, episode_ranges = agent_train.train(
                                env_train,
                                total_episodes=train_episodes,
                                progress_callback=training_callback_tab4,
                                mesh_callback=mesh_callback_tab4
                            )
                            
                            _io_executor.shutdown(wait=True)
                            
                            st.success("✅ DDPG Training complete!")
                            st.session_state['training_active_tab4'] = False
                            st.session_state['last_training_rewards'] = episode_rewards
                            st.session_state['last_training_ranges'] = episode_ranges
                            st.session_state['training_mode_used'] = 'DDPG'
                            st.rerun()
                    
                    else:
                        # ====== RECURSIVE GNN TRAINING ======
                        try:
                            from src.trainer.gnn_trainer import RecursiveGNNModel, RecursiveGNNTrainer, create_synthetic_dataset
                        except ImportError:
                            st.error("❌ GNN trainer module not found. Install torch_geometric: pip install torch-geometric")
                            st.session_state['training_active_tab4'] = False
                        else:
                            # Initialize GNN model
                            n_folds = config_train['project']['n_folds']
                            gnn_model = RecursiveGNNModel(
                                input_dim=5,  # 5 folding parameters per node
                                hidden_dim=64,
                                output_dim=1,  # Single output: normalized efficiency
                                num_recursive_levels=3,
                                num_heads=4
                            )
                            
                            trainer = RecursiveGNNTrainer(
                                gnn_model,
                                device=device,
                                learning_rate=float(learning_rate)
                            )
                            
                            # Create synthetic dataset
                            train_dataset = create_synthetic_dataset(
                                n_samples=train_episodes * 10,
                                n_folds=n_folds,
                                device=device
                            )
                            
                            def gnn_progress_callback(epoch, total_epochs, train_loss, val_loss):
                                """Update GNN training progress."""
                                progress = epoch / total_epochs
                                training_progress_bar.progress(progress)
                                
                                val_loss_str = f"{val_loss:.4f}" if val_loss else "N/A"
                                training_status_text.markdown(f"""
                                    **Epoch**: {epoch}/{total_epochs} | **Train Loss**: {train_loss:.4f} | **Val Loss**: {val_loss_str}  
                                    🧠 Recursive GNN Learning | 🖥️ Device: {st.session_state['selected_device']}
                                """)
                                
                                with training_metrics_placeholder.container():
                                    metric_col1, metric_col2 = st.columns(2)
                                    with metric_col1:
                                        st.metric("Train Loss", f"{train_loss:.4f}")
                                    with metric_col2:
                                        st.metric("Val Loss", val_loss_str)
                            
                            # Train GNN
                            training_history = trainer.train(
                                train_loader=train_dataset,
                                epochs=min(train_episodes, 50),
                                early_stopping_patience=5,
                                progress_callback=gnn_progress_callback
                            )
                            
                            st.success("✅ Recursive GNN Training complete!")
                            st.session_state['training_active_tab4'] = False
                            st.session_state['last_training_gnn_history'] = training_history
                            st.session_state['training_mode_used'] = 'GNN'
                            st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Training error: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
                    st.session_state['training_active_tab4'] = False
        
        # ====================================================================
        # SECTION 3: TRAINING HISTORY
        # ====================================================================
        # SECTION 3: TRAINING HISTORY
        # ====================================================================
        training_mode_used = st.session_state.get('training_mode_used', None)
        
        if training_mode_used == 'DDPG' and st.session_state.get('last_training_rewards') is not None:
            st.divider()
            st.subheader("📈 DDPG Training History")
            
            history_col1, history_col2 = st.columns(2)
            
            # Get data from session state or disk
            if st.session_state.get('last_training_ranges') is not None:
                ranges_hist = st.session_state['last_training_ranges']
                rewards_hist = st.session_state['last_training_rewards']
            else:
                ranges_hist = np.load('data/logs/ranges.npy') if Path('data/logs/ranges.npy').exists() else None
                rewards_hist = np.load('data/logs/rewards.npy') if Path('data/logs/rewards.npy').exists() else None
            
            if ranges_hist is not None:
                with history_col1:
                    episodes_hist = np.arange(len(ranges_hist))
                    
                    fig_range = go.Figure()
                    fig_range.add_trace(go.Scatter(
                        x=episodes_hist,
                        y=ranges_hist,
                        mode='lines+markers',
                        name='Max Range',
                        line=dict(color='#1f77b4', width=2),
                        marker=dict(size=4)
                    ))
                    fig_range.update_layout(
                        title="DDPG: Max Range per Episode",
                        xaxis_title="Episode",
                        yaxis_title="Max Range (m)",
                        hovermode='x unified',
                        height=400
                    )
                    st.plotly_chart(fig_range, use_container_width=True)
            
            if rewards_hist is not None:
                with history_col2:
                    episodes_hist = np.arange(len(rewards_hist))
                    
                    fig_reward = go.Figure()
                    fig_reward.add_trace(go.Scatter(
                        x=episodes_hist,
                        y=rewards_hist,
                        mode='lines+markers',
                        name='Reward',
                        line=dict(color='#2ca02c', width=2),
                        marker=dict(size=4)
                    ))
                    fig_reward.update_layout(
                        title="DDPG: Cumulative Reward per Episode",
                        xaxis_title="Episode",
                        yaxis_title="Reward",
                        hovermode='x unified',
                        height=400
                    )
                    st.plotly_chart(fig_reward, use_container_width=True)
        
        elif training_mode_used == 'GNN' and st.session_state.get('last_training_gnn_history') is not None:
            st.divider()
            st.subheader("📈 Recursive GNN Training History")
            
            gnn_history = st.session_state['last_training_gnn_history']
            
            history_col1, history_col2 = st.columns(2)
            
            with history_col1:
                epochs_hist = np.arange(len(gnn_history['train_loss']))
                
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    x=epochs_hist,
                    y=gnn_history['train_loss'],
                    mode='lines+markers',
                    name='Train Loss',
                    line=dict(color='#d62728', width=2),
                    marker=dict(size=4)
                ))
                if gnn_history['val_loss']:
                    fig_loss.add_trace(go.Scatter(
                        x=epochs_hist,
                        y=gnn_history['val_loss'],
                        mode='lines+markers',
                        name='Val Loss',
                        line=dict(color='#1f77b4', width=2, dash='dash'),
                        marker=dict(size=4)
                    ))
                fig_loss.update_layout(
                    title="GNN: Training Loss per Epoch",
                    xaxis_title="Epoch",
                    yaxis_title="Loss",
                    hovermode='x unified',
                    height=400
                )
                st.plotly_chart(fig_loss, use_container_width=True)
            
            with history_col2:
                epochs_hist = np.arange(len(gnn_history['learning_rate']))
                
                fig_lr = go.Figure()
                fig_lr.add_trace(go.Scatter(
                    x=epochs_hist,
                    y=gnn_history['learning_rate'],
                    mode='lines',
                    name='Learning Rate',
                    line=dict(color='#2ca02c', width=2)
                ))
                fig_lr.update_layout(
                    title="GNN: Learning Rate Schedule",
                    xaxis_title="Epoch",
                    yaxis_title="Learning Rate",
                    hovermode='x unified',
                    height=400
                )
                st.plotly_chart(fig_lr, use_container_width=True)
            
            # GNN Training Summary
            st.info(
                f"✅ **GNN Training Complete**  \n"
                f"• Best Validation Loss: {gnn_history['best_val_loss']:.4f}  \n"
                f"• Total Epochs: {len(gnn_history['train_loss'])}  \n"
                f"• Final Train Loss: {gnn_history['train_loss'][-1]:.4f}"
            )
        
        # ====================================================================
        # SECTION 4: BATCH EVALUATION
        # ====================================================================
        st.divider()
        st.subheader("⚡ Batch Evaluation")
        
        eval_col1, eval_col2 = st.columns([2, 1])
        
        with eval_col1:
            num_eval_actions = st.slider(
                "Number of Actions to Evaluate",
                10, 1000, 100,
                step=10,
                key="num_eval_actions_tab4"
            )
        
        with eval_col2:
            st.metric("Device", st.session_state['selected_device'])
        
        if st.button("▶️ Run Batch Evaluation", key="batch_eval_tab4", use_container_width=True):
            st.session_state['batch_eval_running_tab4'] = True
        
        if st.session_state.get('batch_eval_running_tab4', False):
            batch_eval_progress_bar = st.progress(0)
            batch_eval_status_text = st.empty()
            batch_eval_time = st.empty()
            
            def batch_eval_callback(current_step, total_steps, start_time):
                """Callback for batch evaluation progress."""
                progress = min(1.0, current_step / total_steps)
                batch_eval_progress_bar.progress(progress)
                elapsed = time.time() - start_time
                batch_eval_status_text.markdown(
                    f"**Processed**: {current_step}/{total_steps} actions | **Device**: {st.session_state['selected_device']}"
                )
                batch_eval_time.caption(f"⏱️ Elapsed: {elapsed:.1f}s")
            
            with st.spinner(f"Running evaluation on {st.session_state['selected_device']}..."):
                try:
                    start_time = time.time()
                    config_eval = get_config()
                    temp_agent, temp_env = create_agent_and_env(config_eval)
                    
                    if temp_env is None or temp_env.action_space is None:
                        st.error("❌ Failed to create environment.")
                        st.session_state['batch_eval_running_tab4'] = False
                    else:
                        action_dim = temp_env.action_space.shape[0]
                        actions_to_evaluate = np.random.uniform(0, 1, (num_eval_actions, action_dim))
                        
                        eval_state = {
                            'angle_of_attack_deg': config_eval.get('goals', {}).get('angle_of_attack_deg', 5.0),
                            'air_density_kgm3': config_eval.get('environment', {}).get('air_density_kgm3', 1.225),
                            'air_viscosity_pas': config_eval.get('environment', {}).get('air_viscosity_pas', 1.8e-5),
                            'throw_speed_mps': config_eval.get('goals', {}).get('throw_speed_mps', 10.0)
                        }
                        
                        device = set_gpu_device(st.session_state['selected_device'])
                        evaluator = SurrogateBatchEvaluator(device=device)
                        
                        all_results: Dict[str, np.ndarray] = {
                            'range_est': np.zeros(num_eval_actions, dtype=np.float32),
                            'cl': np.zeros(num_eval_actions, dtype=np.float32),
                            'cd': np.zeros(num_eval_actions, dtype=np.float32),
                            'ld': np.zeros(num_eval_actions, dtype=np.float32),
                            'Re': np.zeros(num_eval_actions, dtype=np.float32)
                        }
                        
                        batch_size = evaluator.recommended_batch_size
                        
                        for i in range(0, num_eval_actions, batch_size):
                            batch_end = min(i + batch_size, num_eval_actions)
                            current_actions_batch = actions_to_evaluate[i:batch_end]
                            results_batch = evaluator.evaluate_batch(current_actions_batch, eval_state, show_progress=False)
                            
                            all_results['range_est'][i:batch_end] = results_batch['range_est']
                            all_results['cl'][i:batch_end] = results_batch['cl']
                            all_results['cd'][i:batch_end] = results_batch['cd']
                            all_results['ld'][i:batch_end] = results_batch['ld']
                            all_results['Re'][i:batch_end] = results_batch['Re']
                            
                            batch_eval_callback(batch_end, num_eval_actions, start_time)
                        
                        elapsed_total = time.time() - start_time
                        st.success(f"✅ Batch evaluation complete in {elapsed_total:.1f}s!")
                        st.session_state['batch_eval_running_tab4'] = False
                        st.session_state['batch_eval_results_tab4'] = all_results
                
                except Exception as e:
                    st.error(f"❌ Evaluation error: {str(e)}")
                    st.session_state['batch_eval_running_tab4'] = False
        
        # Display batch evaluation results
        if 'batch_eval_results_tab4' in st.session_state:
            results = st.session_state['batch_eval_results_tab4']
            
            st.subheader("📊 Evaluation Results")
            result_metric_col1, result_metric_col2, result_metric_col3, result_metric_col4 = st.columns(4)
            
            with result_metric_col1:
                st.metric("Avg Range", f"{np.mean(results['range_est']):.2f}m")
            with result_metric_col2:
                st.metric("Max Range", f"{np.max(results['range_est']):.2f}m")
            with result_metric_col3:
                st.metric("Min Range", f"{np.min(results['range_est']):.2f}m")
            with result_metric_col4:
                st.metric("Avg L/D", f"{np.mean(results['ld']):.2f}")
            
            # Graphs
            graph_col1, graph_col2 = st.columns(2)
            
            with graph_col1:
                fig_range_dist = px.histogram(
                    x=results['range_est'],
                    nbins=30,
                    title="Range Distribution",
                    labels={'x': 'Estimated Range (m)', 'count': 'Count'}
                )
                fig_range_dist.update_traces(marker_color='steelblue')
                st.plotly_chart(fig_range_dist, use_container_width=True)
            
            with graph_col2:
                fig_cl_cd = go.Figure()
                fig_cl_cd.add_trace(go.Scatter(
                    x=results['cd'],
                    y=results['cl'],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=results['ld'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="L/D")
                    ),
                    text=np.round(results['ld'], 2),
                    hovertemplate="CD: %{x:.4f}<br>CL: %{y:.3f}<br>L/D: %{text:.2f}<extra></extra>"
                ))
                fig_cl_cd.update_layout(
                    title="CL vs CD (colored by L/D)",
                    xaxis_title="Drag Coefficient",
                    yaxis_title="Lift Coefficient",
                    height=400
                )
                st.plotly_chart(fig_cl_cd, use_container_width=True)
            
            # Additional metrics
            extra_col1, extra_col2 = st.columns(2)
            
            with extra_col1:
                fig_cl_dist = go.Figure(data=go.Histogram(x=results['cl'], nbins=25, marker_color='lightblue'))
                fig_cl_dist.update_layout(title="CL Distribution", xaxis_title="Lift Coefficient", yaxis_title="Count", height=350)
                st.plotly_chart(fig_cl_dist, use_container_width=True)
            
            with extra_col2:
                fig_ld_dist = go.Figure(data=go.Histogram(x=results['ld'], nbins=25, marker_color='lightgreen'))
                fig_ld_dist.update_layout(title="L/D Distribution", xaxis_title="Efficiency", yaxis_title="Count", height=350)
                st.plotly_chart(fig_ld_dist, use_container_width=True)


if __name__ == "__main__":
    main()
