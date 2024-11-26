import streamlit as st
from ui.main_view import render_data_input_section
from ui.visualization import TimeSeriesVisualization, plot_time_series
from analysis.time_series import TimeSeriesAnalysis
import numpy as np



def main():
    st.set_page_config(
        page_title="Time Series TDA Analysis",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Time Series Analysis with TDA")
    
    # Initialize session state for storing data
    if 'time_series_data' not in st.session_state:
        st.session_state.time_series_data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'persistence_data' not in st.session_state:
        st.session_state.persistence_data = None

    # Get input data
    if st.button("Generate Data"):
        st.session_state.time_series_data = None
        st.session_state.processed_data = None
        st.session_state.persistence_data = None
        st.success("Data generated successfully!")

    if st.session_state.time_series_data is None:
        time_series_data, error = render_data_input_section()
    else:
        _, error = render_data_input_section()
        time_series_data = st.session_state.time_series_data

    if error:
        st.error(error)
    
    if time_series_data is not None:
        st.session_state.time_series_data = time_series_data
        
    # Display time series data
    if st.session_state.time_series_data is not None:
        st.subheader("Time Series Data")
        fig = plot_time_series(time_series_data)
        st.pyplot(fig)
        try:
            # Basic visualization of raw data
            st.subheader("Analysis Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                window_size = st.slider(
                    "Window Size",
                    min_value=2,
                    max_value=len(time_series_data)//2,
                    value=min(50, len(time_series_data)//4),
                    key="window_size"
                )
                
                embedding_dim = st.slider(
                    "Embedding Dimension",
                    min_value=2,
                    max_value=10,
                    value=3,
                    key="embedding_dim"
                )
            
            with col2:
                stride = st.slider(
                    "Stride",
                    min_value=1,
                    max_value=window_size,
                    value=window_size//2,
                    key="stride"
                )
                
                time_delay = st.slider(
                    "Time Delay",
                    min_value=1,
                    max_value=20,
                    value=1,
                    key="time_delay"
                )
            
            projection_dim = st.radio(
                "Projection Dimension",
                options=[2, 3],
                horizontal=True,
                key="projection_dim"
            )
            
            # Process button
            if st.button("Process Data"):
                with st.spinner("Processing..."):
                    # Create analyzer
                    analyzer = TimeSeriesAnalysis(time_series_data)
                    
                    # Compute windows
                    windows = analyzer.create_sliding_windows(window_size, stride)
                    
                    # Create and project embedding
                    embedding = analyzer.create_takens_embedding(embedding_dim, time_delay)
                    projected_embedding = analyzer.project_embedding(embedding, projection_dim)
                    
                    # Store in session state
                    st.session_state.processed_data = {
                        'windows': windows,
                        'embedding': embedding,
                        'projected_embedding': projected_embedding,
                        'stats': {
                            'length': len(time_series_data),
                            'mean': float(np.mean(time_series_data)),
                            'std': float(np.std(time_series_data)),
                            'range': float(np.ptp(time_series_data))
                        }
                    }
                    
                st.success(f"Processing complete")
            
            # Display results if data has been processed
            if st.session_state.processed_data is not None:
                st.subheader("Analysis Results")
                
                # Display basic statistics
                stats = st.session_state.processed_data['stats']
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Length", stats['length'])
                with col2:
                    st.metric("Mean", f"{stats['mean']:.2f}")
                with col3:
                    st.metric("Std Dev", f"{stats['std']:.2f}")
                with col4:
                    st.metric("Range", f"{stats['range']:.2f}")

                st.subheader("Takens Embedding Visualization")
                fig = TimeSeriesVisualization.plot_takens_embedding(
                    st.session_state.processed_data['projected_embedding'],
                    time_series_data,
                    f"{projection_dim}D Takens Embedding"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Persistence Analysis Section
                st.subheader("TDA Analysis")
                
                max_homology_dim = st.slider(
                    "Maximum Homology Dimension",
                    min_value=0,
                    max_value=2,
                    value=1,
                    key="max_homology_dim"
                )
                
                if st.button("Compute Persistence"):
                    with st.spinner("Computing persistence diagrams..."):
                        analyzer = TimeSeriesAnalysis(time_series_data)
                        # Create analyzer and compute persistence
                        persistence_results = analyzer.compute_persistence(
                            st.session_state.processed_data['embedding'], 
                            max_dim=max_homology_dim
                        )
                        
                        st.session_state.persistence_data = persistence_results
                        st.success("Persistence computation complete")
                
                # Display persistence results if available
                if st.session_state.persistence_data is not None:
                    

                    vis_type = st.selectbox(
                        "Select Visualization",
                        ["Persistence Diagram", "Persistence Landscape"]
                    )
                    
                    if vis_type == "Persistence Diagram":
                        fig = TimeSeriesVisualization.plot_persistence_diagrams(
                            st.session_state.persistence_data['diagrams']
                        )
                        st.pyplot(fig)
                        
                    elif vis_type == "Persistence Landscape":
                        n_layers = st.slider(
                            "Number of Landscape Layers", 
                            min_value=1,
                            max_value=10,
                            value=5
                        )
                        fig = TimeSeriesVisualization.plot_persistence_landscape(
                            st.session_state.persistence_data['diagrams'],
                            n_layers=n_layers
                        )
                        st.pyplot(fig)


                    
                    # Statistics in a separate section
                    with st.expander("Persistence Statistics"):
                        for stat in st.session_state.persistence_data['stats']:
                            st.write(f"### Dimension {stat.dimension}")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Number of Features", stat.num_features)
                            with col2:
                                st.metric("Total Persistence", f"{stat.total_persistence:.3f}")
                            with col3:
                                st.metric("Max Persistence", f"{stat.max_persistence:.3f}")
                
        except Exception as e:
            raise e

if __name__ == "__main__":
    main()