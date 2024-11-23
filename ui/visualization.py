import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def plot_time_series(time_series: np.ndarray, title: str = "Time Series") -> plt.Figure:
    """Create a basic time series plot."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_series, '-b', alpha=0.7)
    ax.scatter(range(len(time_series)), time_series, c='b', alpha=0.2, s=10)
    
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    
    return fig


class TimeSeriesVisualization:
    @staticmethod
    def plot_takens_embedding(projected_embedding: np.ndarray, 
                            original_time_series: np.ndarray,
                            title: str = "Takens Embedding") -> go.Figure:
        """
        Create an interactive plot of the Takens embedding.
        
        Args:
            projected_embedding: 2D or 3D projected embedding
            original_time_series: Original time series for coloring
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        is_3d = projected_embedding.shape[1] == 3
        
        # Create color gradient based on time
        colors = original_time_series[:projected_embedding.shape[0]]
        
        # Create figure
        fig = go.Figure()
        
        if is_3d:
            fig.add_trace(
                go.Scatter3d(
                    x=projected_embedding[:, 0],
                    y=projected_embedding[:, 1],
                    z=projected_embedding[:, 2],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=colors,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Time Series Value")
                    ),
                    name='Embedding'
                )
            )
            
            fig.update_layout(
                scene=dict(
                    xaxis_title="Component 1",
                    yaxis_title="Component 2",
                    zaxis_title="Component 3",
            
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=projected_embedding[:, 0],
                    y=projected_embedding[:, 1],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=colors,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Time Series Value")
                    ),
                    name='Embedding'
                )
            )
            
            fig.update_layout(
                xaxis_title="Component 1",
                yaxis_title="Component 2",
                plot_bgcolor='rgba(240, 240, 240, 0.9)'
            )
        
        fig.update_layout(
            title=title,
            hovermode='closest',
            showlegend=False,
        )
        
        return fig