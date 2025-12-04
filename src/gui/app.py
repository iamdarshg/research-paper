"""Streamlit GUI for paper airplane AI optimizer."""
import sys
from pathlib import Path
root = Path(__file__).parent.parent.parent
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

from src.rl_agent.model import DDPGAgent
from src.rl_agent.env import PaperPlaneEnv
from src.folding.folder import fold_sheet
from src.trainer.train import load_config, _io_executor # Reuse, and import for async ops
from src.surrogate.aero_model import surrogate_cfd
from src.surrogate.batch_evaluator import SurrogateBatchEvaluator, _autodetect_batch_size # New imports

st.set_page_config(page_title="Paper Plane AI", layout="wide")

CONFIG_PATH = Path(__file__).parent.parent.parent / 'config.yaml'
MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

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
    Main function for the Streamlit GUI.
    Sets up the sidebar controls, training progress display,
    fold visualization, and CFD validation sections.
    """
    st.title("🛩️ AI-Optimized Paper Airplane Folding")

    # Sidebar config for global parameters and model management
    with st.sidebar:
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

    # Main content area - use cached config
    config_main = get_config()
    if config_main is None:
        st.error("Config loading failed.")
        return

    col1, col2, col3 = st.columns([1, 1, 0.8]) # Add a third column for batch evaluation

    with col1:
        st.header("Training Progress")
        # Display live progress for "Train Model" button
        if st.session_state.get('training_in_progress', False):
            st.subheader("Live Training Metrics")
            st.info("Training is in progress. Metrics will update above.") # The status_text and progress_bar handle the live updates
        else: # Only show historical if no training is active
            st.subheader("Historical Training Progress")
            if Path('data/logs/ranges.npy').exists():
                ranges = np.load('data/logs/ranges.npy')
                episodes_hist = np.arange(len(ranges)) # Renamed to avoid conflict with `episodes` from config
                zoom_ep = min(50, len(ranges))

                # Create subplot with plotly
                from plotly.subplots import make_subplots
                fig = make_subplots(rows=2, cols=1,
                                    subplot_titles=("Full Training Progress", f"First {zoom_ep} Episodes"))

                # Full plot
                fig.add_trace(go.Scatter(x=episodes_hist, y=ranges, mode='lines', name='Full'), row=1, col=1)

                # Zoomed plot
                fig.add_trace(go.Scatter(x=episodes_hist[:zoom_ep], y=ranges[:zoom_ep], mode='lines', name='Zoomed'), row=2, col=1)

                fig.update_layout(height=600, showlegend=False)
                fig.update_xaxes(title_text="Episode", row=1, col=1)
                fig.update_xaxes(title_text="Episode", row=2, col=1)
                fig.update_yaxes(title_text="Max Range (m)", row=1, col=1)
                fig.update_yaxes(title_text="Max Range (m)", row=2, col=1)

                st.plotly_chart(fig)
            else:
                st.info("No historical training data found. Run training to generate logs.")


    with col2:
        st.header("Fold Visualization")
        # Placeholder for the 3D visualization, will be updated by mesh_update_callback
        static_mesh_placeholder = st.empty()
        
        # Standard paper airplane fold: nose, body wrinkles, tail with angles
        n_folds_vis = config_main['project']['n_folds']
        # Create standard action if matches, else default 0.5
        if n_folds_vis == 5:
            standard_action = np.array([0.3,0.5,0.7,0.5, 0.5, 0.5,0.3,0.5,0.7, 0.5, 0.2,0.2,0.8,0.2, 0.5, 0.1,0.8,0.9,0.8, 0.5, 0.4,0.6,0.6,0.6, 0.5])  # 5 folds
        else:
            standard_action = np.full(n_folds_vis * 5, 0.5)
        mesh = fold_sheet(standard_action, resolution=50)
        fig3d = plot_mesh_3d(mesh)
        static_mesh_placeholder.plotly_chart(fig3d, use_container_width=True)

        aero = surrogate_cfd(mesh, config_main)
        st.metric("Est Range", f"{aero['range_est']:.2f}m")
        st.metric("L/D", f"{aero['ld']:.2f}")

        # This "Run Training" button is separate from the sidebar's "Train Model" button.
        # It's here for direct use if needed, with a fixed number of steps for demonstration.
        if st.button("Run Training"):
            st.session_state['training_in_progress_main'] = True # Use a different session state key to distinguish from sidebar training
            progress_bar_main = st.progress(0)
            status_text_main = st.empty()
            mesh_placeholder_main = st.empty() # Placeholder for 3D plot to update

            def training_callback_main(current_episode, total_episodes, ep_reward, ep_range, avg_range_10):
                """Callback function to update Streamlit progress bar and status for main training button."""
                progress = min(1.0, current_episode / total_episodes)
                progress_bar_main.progress(progress)
                status_text_main.markdown(f"""
                    **Episode:** {current_episode} / {total_episodes} | **Reward:** {ep_reward:.2f} | **Max Range:** {ep_range:.2f}m | **Avg 10-ep Range:** {avg_range_10:.2f}m
                """)

            def mesh_update_callback_main(mesh_data):
                """Callback function to update the 3D mesh visualization for main training button."""
                if mesh_data is not None:
                    fig3d_live_main = plot_mesh_3d(mesh_data)
                    mesh_placeholder_main.plotly_chart(fig3d_live_main, use_container_width=True)

            with st.spinner("Training in progress..."):
                agent_train, env_train = create_agent_and_env(config_main) # Create fresh for training
                if agent_train is None or env_train is None:
                    st.error("Failed to create agent or environment for training.")
                    st.session_state['training_in_progress_main'] = False
                    return
                # Use a smaller, fixed number of episodes for this demo button
                _, rewards_main, ranges_main = agent_train.train(env_train, 
                                                                 total_episodes=5, # Fixed 5 episodes for this demo button
                                                                 progress_callback=training_callback_main,
                                                                 mesh_callback=mesh_update_callback_main)

            # Shutdown the executor to ensure all async I/O operations (e.g., saving logs) are completed
            _io_executor.shutdown(wait=True)
            st.success("Training complete!")
            st.session_state['training_in_progress_main'] = False
            st.rerun()

    with col3:
        st.header("Batch Evaluation")
        num_eval_actions = st.slider("Number of Actions for Batch Eval", 10, 1000, 100)

        if st.button("Run Batch Evaluation"):
            st.session_state['batch_eval_in_progress'] = True
            batch_eval_progress_bar = st.progress(0)
            batch_eval_status_text = st.empty()

            def batch_eval_callback(current_step, total_steps):
                """Callback for batch evaluation progress."""
                progress = min(1.0, current_step / total_steps)
                batch_eval_progress_bar.progress(progress)
                batch_eval_status_text.text(f"Processed {current_step}/{total_steps} actions.")

            with st.spinner("Running batch evaluation..."):
                # Create env for action_dim, using config_main
                temp_agent, temp_env = create_agent_and_env(config_main)
                if temp_env is None or temp_env.action_space is None: # Added check for temp_env.action_space
                    st.error("Failed to create environment for batch evaluation. Action space not found.")
                    st.session_state['batch_eval_in_progress'] = False
                    return
                action_dim = temp_env.action_space.shape[0]
                actions_to_evaluate = np.random.uniform(0, 1, (num_eval_actions, action_dim))

                # Create dummy state for evaluation
                # Safely access config values, providing defaults if keys are missing
                eval_state = {
                    'angle_of_attack_deg': config_main.get('goals', {}).get('angle_of_attack_deg', 5.0),
                    'air_density_kgm3': config_main.get('environment', {}).get('air_density_kgm3', 1.225),
                    'air_viscosity_pas': config_main.get('environment', {}).get('air_viscosity_pas', 1.8e-5),
                    'throw_speed_mps': config_main.get('goals', {}).get('throw_speed_mps', 10.0)
                }

                evaluator = SurrogateBatchEvaluator(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

                # Manual progress tracking for batch evaluation
                all_results: Dict[str, np.ndarray] = {
                    'range_est': np.zeros(num_eval_actions, dtype=np.float32),
                    'cl': np.zeros(num_eval_actions, dtype=np.float32),
                    'cd': np.zeros(num_eval_actions, dtype=np.float32),
                    'ld': np.zeros(num_eval_actions, dtype=np.float32),
                    'Re': np.zeros(num_eval_actions, dtype=np.float32)
                }

                batch_size = evaluator.recommended_batch_size # Use auto-detected batch size

                for i in range(0, num_eval_actions, batch_size):
                    batch_end = min(i + batch_size, num_eval_actions)
                    current_actions_batch = actions_to_evaluate[i:batch_end]

                    results_batch = evaluator.evaluate_batch(current_actions_batch, eval_state, show_progress=False)

                    # Store results
                    all_results['range_est'][i:batch_end] = results_batch['range_est']
                    all_results['cl'][i:batch_end] = results_batch['cl']
                    all_results['cd'][i:batch_end] = results_batch['cd']
                    all_results['ld'][i:batch_end] = results_batch['ld']
                    all_results['Re'][i:batch_end] = results_batch['Re']

                    batch_eval_callback(batch_end, num_eval_actions)

                st.success("Batch evaluation complete!")
                st.session_state['batch_eval_in_progress'] = False
                st.session_state['last_batch_eval_results'] = all_results # Store results in session state

                st.subheader("Batch Evaluation Results")
                st.metric("Avg Range", f"{np.mean(all_results['range_est']):.2f}m")
                st.metric("Max Range", f"{np.max(all_results['range_est']):.2f}m")
                st.metric("Min Range", f"{np.min(all_results['range_est']):.2f}m")

        st.subheader("Interactive Performance Metrics")
        if 'last_batch_eval_results' in st.session_state:
            results_to_plot = st.session_state['last_batch_eval_results']

            fig_hist = px.histogram(x=results_to_plot['range_est'], nbins=30,
                                    title='Distribution of Estimated Ranges from Batch Evaluation')
            fig_hist.update_xaxes(title_text="Estimated Range (m)")
            fig_hist.update_yaxes(title_text="Count")
            st.plotly_chart(fig_hist)
        else:
            st.info("Run a batch evaluation to see interactive performance metrics here.")


    st.header("Live CFD Validation")
    left_col, right_col = st.columns(2)
    with left_col:
        if st.button("Run CFD Validation"):
            if 'project' not in config_main or 'n_folds' not in config_main['project']:
                st.error("Config missing project.fold information.")
                return
            n_folds_val = config_main['project']['n_folds']
            if n_folds_val == 5:
                default_action = np.array([0.3,0.5,0.7,0.5, 0.5, 0.5,0.3,0.5,0.7, 0.5, 0.2,0.2,0.8,0.2, 0.5, 0.1,0.8,0.9,0.8, 0.5, 0.4,0.6,0.6,0.6, 0.5])
            else:
                default_action = np.full(n_folds_val * 5, 0.5)
            state = {
                'throw_speed_mps': config_main['goals']['throw_speed_mps'],
                'angle_of_attack_deg': config_main['goals']['angle_of_attack_deg'],
                'air_density_kgm3': config_main['environment']['air_density_kgm3'],
                'air_viscosity_pas': config_main['environment']['air_viscosity_pas']
            }
            from src.cfd.runner import run_openfoam_cfd
            try:
                aero_cfd = run_openfoam_cfd(default_action, state, 'low')
                st.metric("CFD CL", f"{aero_cfd['cl']:.3f}")
                st.metric("CFD CD", f"{aero_cfd['cd']:.4f}")
                st.metric("CFD Range Est", f"{aero_cfd['range_est']:.2f}m")
            except Exception as e:
                st.error(f"CFD failed: {e}")

    with right_col:
        # Use the same mesh from Fold Visualization
        aero_sur = surrogate_cfd(mesh, config_main)
        st.metric("Surrogate CL", f"{aero_sur['cl']:.3f}")
        st.metric("Surrogate CD", f"{aero_sur['cd']:.4f}")
        st.metric("Surrogate Range Est", f"{aero_sur['range_est']:.2f}m")
        if st.button("Compare with CFD"):
            st.subheader("Comparison Notes")
            st.info("CFD provides ground truth for validation. Surrogate uses approximate NS.")

if __name__ == "__main__":
    main()
