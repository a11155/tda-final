import streamlit as st
from ui.sidebar import render_data_input_section
from ui.visualization import TimeSeriesVisualization, plot_time_series
from analysis.time_series import TimeSeriesAnalysis
from analysis.critical_points import TopologicalCriticalPoints
import numpy as np
import streamlit.components.v1 as components



def main():
    st.set_page_config(
        page_title="Time Series TDA Analysis",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Time Series Analysis with TDA")
    
    with st.expander("Explanation"):
        components.iframe("https://docs.google.com/presentation/d/e/2PACX-1vTsGHcRt8J49QrVtKM8IDboOy8dX-AFLz3LUKgvi1NITECu7kGd9sLCLFwDadZlLY4SqpYXoYyKm-6g/embed?start=false&loop=false&delayms=3000", height=500)


    if 'time_series_data' not in st.session_state:
        st.session_state.time_series_data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'persistence_data' not in st.session_state:
        st.session_state.persistence_data = None
    
    if 'critical_point_detector' not in st.session_state:
        st.session_state.critical_point_detector = None
    if 'critical_points' not in st.session_state:
        st.session_state.critical_points = None

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
            embedding_dim = st.slider(
                    "Embedding Dimension",
                    min_value=2,
                    max_value=100,
                    value=15,
                    key="embedding_dim"
            )
            
            col1, col2 = st.columns(2)
            

                

            
            with col1:
                stride = st.slider(
                    "Stride",
                    min_value=1,
                    max_value=embedding_dim,
                    value=1,
                    key="stride"
                )
            with col2:
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
                    analyzer = TimeSeriesAnalysis(time_series_data)
                    
                    
                    embedding = analyzer.create_takens_embedding(embedding_dim, time_delay, stride=stride)
                    projected_embedding = analyzer.project_embedding(embedding, projection_dim)
                    
                    st.session_state.processed_data = {
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
            
            if st.session_state.processed_data is not None:
                st.subheader("Analysis Results")
                
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
                
                if st.session_state.persistence_data is not None:
                    

                    vis_type = st.selectbox(
                        "Select Visualization",
                        ["Persistence Diagram", "Barcode", "Persistence Landscape", "Betti Curves", "Persistence Images"]
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

                    elif vis_type == "Betti Curves":
                        fig = TimeSeriesVisualization.plot_betti_curves(
                            st.session_state.persistence_data['diagrams']
                        )
                        st.pyplot(fig)

                    elif vis_type == "Persistence Images":
                        pixel_size = st.slider(
                            "Pixel Size",
                            min_value=0.001,
                            max_value=1.0,
                            value=0.1
                        )
                        fig = TimeSeriesVisualization.plot_persistence_images(
                            st.session_state.persistence_data['diagrams'],
                            pixel_size=pixel_size
                        )
                        st.pyplot(fig)
                    
                    elif vis_type == "Barcode":
                        fig = TimeSeriesVisualization.plot_persistence_barcode(
                            st.session_state.persistence_data['diagrams']
                        )
                        st.pyplot(fig)


                    
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
                

            st.subheader("Critical Point Detection")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                detection_window = st.slider(
                    "Detection Window Size",
                    min_value=10,
                    max_value=len(time_series_data)//4,
                    value=min(50, len(time_series_data)//8)
                )
            with col2:
                detection_threshold = st.slider(
                    "Detection Threshold",
                    min_value=0.8,
                    max_value=0.99,
                    value=0.95,
                    step=0.01
                )
            with col3:
                metric = st.selectbox(
                    "Distance Metric",
                    ["wasserstein", "bottleneck"]
                )


            col4, col5, col6 = st.columns(3)

            with col4:
                critical_points_embedding_dim = st.slider(
                    "Embedding Dimension",
                    min_value=2,
                    max_value=100,
                    value=15,
                    key="critical_embedding_dim"
            )
            
            with col5:
                critical_points_stride = st.slider(
                    "Stride",
                    min_value=1,
                    max_value=embedding_dim,
                    value=1,
                    key="critical_stride"
                )
            with col6:
                critical_points_time_delay = st.slider(
                    "Time Delay",
                    min_value=1,
                    max_value=20,
                    value=1,
                    key="critical_time_delay"
                )
            

            if st.button("Detect Critical Points"):
                with st.spinner("Detecting critical points..."):
                    if st.session_state.critical_point_detector is None:
                        st.session_state.critical_point_detector = TopologicalCriticalPoints(
                        window_size=detection_window,
                        stride=critical_points_stride,
                        embedding_dim=critical_points_embedding_dim,
                        time_delay=critical_points_time_delay
                    )
                        
                    detector = st.session_state.critical_point_detector
                    
                    critical_points = detector.find_critical_points(
                        time_series_data,
                        threshold=detection_threshold,
                        metric=metric
                    )
                    st.session_state.critical_points = critical_points
                    
            
            
            if st.session_state.critical_points is not None:
                if len(st.session_state.critical_points) > 0:
                        critical_points = st.session_state.critical_points
                        detector = st.session_state.critical_point_detector

                        fig = TimeSeriesVisualization.plot_critical_points(
                            time_series_data,
                            critical_points
                        )
                        st.pyplot(fig)
                        st.success(f"Found {len(critical_points)} critical points")
                        
                        selected_cp = st.selectbox(
                            "Select critical point to examine:",
                            range(len(critical_points)),
                            format_func=lambda x: f"Critical Point {x+1} (t={critical_points[x]})"
                        )
                        
                        if selected_cp is not None:
                            cp = critical_points[selected_cp]
                            diagrams_before, diagrams_after = detector.get_critical_point_diagrams(
                                time_series_data,
                                cp,
                                detection_window
                            )
                            
                            fig = TimeSeriesVisualization.plot_critical_point_topology(
                                time_series_data,
                                cp,
                                detection_window,
                                diagrams_before,
                                diagrams_after,
                                f"Topological Changes at Critical Point {selected_cp+1}"
                            )
                            st.pyplot(fig)
                else:
                        st.info("No critical points detected with current parameters")

        except Exception as e:
            raise e

if __name__ == "__main__":
    main()