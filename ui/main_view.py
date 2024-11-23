import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from pathlib import Path

def render_data_input_section() -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Render the data input section with multiple input methods.
    
    Returns:
        Tuple of (time_series_data, error_message)
    """
    st.sidebar.header("Data Input")
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Upload CSV", "Paste Data", "Sample Time Series"]
    )
    
    if input_method == "Upload CSV":
        return handle_csv_upload()
    elif input_method == "Paste Data":
        return handle_pasted_data()
    else:  # Sample Time Series
        return handle_sample_data()

def handle_csv_upload() -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Handle CSV file upload."""
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Let user select the column if multiple columns exist
            if len(df.columns) > 1:
                selected_column = st.sidebar.selectbox(
                    "Select time series column",
                    df.columns
                )
                time_series = df[selected_column].values
            else:
                time_series = df.iloc[:, 0].values
                
            return time_series, None
            
        except Exception as e:
            return None, f"Error loading CSV: {str(e)}"
    
    return None, None

def handle_pasted_data() -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Handle pasted data input."""
    data_format = st.sidebar.selectbox(
        "Data format",
        ["Single Column", "CSV Format"]
    )
    
    data = st.sidebar.text_area(
        "Paste your time series data",
        help=("For single column: One value per line\n"
              "For CSV: Comma-separated values with header")
    )
    
    if data:
        try:
            if data_format == "Single Column":
                values = [float(x.strip()) for x in data.split('\n') if x.strip()]
                return np.array(values), None
            else:
                df = pd.read_csv(pd.StringIO(data))
                if len(df.columns) > 1:
                    selected_column = st.sidebar.selectbox(
                        "Select time series column",
                        df.columns
                    )
                    return df[selected_column].values, None
                return df.iloc[:, 0].values, None
                
        except Exception as e:
            return None, f"Error parsing data: {str(e)}"
    
    return None, None

def handle_sample_data() -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Handle sample time series generation."""
    sample_type = st.sidebar.selectbox(
        "Select sample type",
        ["Sine Wave", "Random Walk", "Composite Signal"]
    )
    
    n_points = st.sidebar.slider("Number of points", 100, 1000, 500)
    
    try:
        if sample_type == "Sine Wave":
            t = np.linspace(0, 10, n_points)
            frequency = st.sidebar.slider("Frequency", 0.1, 2.0, 1.0)
            return np.sin(2 * np.pi * frequency * t), None
            
        elif sample_type == "Random Walk":
            return np.cumsum(np.random.randn(n_points)), None
            
        else:  # Composite Signal
            t = np.linspace(0, 10, n_points)
            f1 = st.sidebar.slider("Frequency 1", 0.1, 2.0, 0.5)
            f2 = st.sidebar.slider("Frequency 2", 0.1, 2.0, 1.0)
            return (np.sin(2 * np.pi * f1 * t) + 
                   0.5 * np.sin(2 * np.pi * f2 * t)), None
                   
    except Exception as e:
        return None, f"Error generating sample data: {str(e)}"