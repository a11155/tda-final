import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from persim.landscapes import PersLandscapeExact
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List
import plotly.graph_objects as go
import numpy as np
from typing import List
import gudhi.representations
from persim.landscapes import plot_landscape_simple
import gudhi
import plotly.express as px
import matplotlib.pyplot as plt
from persim import plot_diagrams
from analysis.time_series import PersistenceStats
from persim import plot_diagrams, persistent_entropy
from persim.landscapes import PersistenceLandscaper
from persim import PersistenceImager
from typing import Tuple

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
    
    @staticmethod
    def plot_persistence_diagrams(diagrams: List[np.ndarray], 
                                    title: str = "Persistence Diagram") -> plt.Figure:
            """
            Create persistence diagram visualization using persim.
            
            Args:
                diagrams: List of persistence diagrams from ripser
                title: Plot title
                
            Returns:
                Matplotlib figure
            """
            fig = plt.figure(figsize=(10, 10))
            plot_diagrams(diagrams, show=False)
            plt.title(title, size=16)
            plt.tight_layout()
            return fig 


    @staticmethod
    def plot_betti_curves(diagrams: List[np.ndarray]) -> plt.Figure:
        """Plot Betti curves using gtda"""
        from gtda.plotting import plot_betti_curves
        
        fig = plt.figure(figsize=(12, 6))
        plot_betti_curves(diagrams)
        plt.title("Betti Curves")
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_persistence_images(diagrams: List[np.ndarray], pixel_size: float= 0.1) -> plt.Figure:
        """Plot persistence images using gtda"""
        pimgr = PersistenceImager(pixel_size=pixel_size)
        
        finite_diagrams = []
        for diagram in diagrams:
            finite_mask = np.isfinite(diagram).all(axis=1)
            finite_diagrams.append(diagram[finite_mask])
        diagrams = finite_diagrams
        pimgr.fit(diagrams, skew=True)
        pimgs = pimgr.transform(diagrams, skew=True)
        fig = plt.figure(figsize=(12, 6))
        
        fig, axs = plt.subplots(1, 3, figsize=(10,5))

        axs[0].set_title("Original Diagram")
        pimgr.plot_diagram(diagrams[1], skew=False, ax=axs[0])

        axs[1].set_title("Birth-Persistence\nCoordinates")
        pimgr.plot_diagram(diagrams[1], skew=True, ax=axs[1])

        axs[2].set_title("Persistence Image")

        # print(len(pimgs))
        # print(pimgs)
        pimgr.plot_image(pimgs[1], ax=axs[2])

        plt.tight_layout()
        return fig
   

    @staticmethod
    def plot_persistence_barcode(diagrams: List[np.ndarray],
                    title: str = "Persistence Barcode") -> plt.Figure:
        """
        Create a traditional persistence barcode visualization with validation.
        
        Args:
            diagrams: List of persistence diagrams from ripser
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Set up the figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Colors for different homology dimensions
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
        
        # Function to validate and clean diagram
        def clean_diagram(diag):
            if len(diag) == 0:
                return diag
            # Remove infinite values and NaN
            valid_mask = np.all(np.isfinite(diag), axis=1)
            return diag[valid_mask]
        
        # Clean diagrams and find valid ranges
        cleaned_diagrams = [clean_diagram(diag) for diag in diagrams]
        
        # Calculate global min and max for x-axis
        all_valid_points = []
        for diag in cleaned_diagrams:
            if len(diag) > 0:
                all_valid_points.extend(diag.flatten())
                
        if all_valid_points:
            global_min = np.min(all_valid_points)
            global_max = np.max(all_valid_points)
        else:
            global_min, global_max = 0, 1
            
        current_y = 0
        dimension_labels = []
        dimension_positions = []
        
        # Plot bars for each dimension
        for dim, diagram in enumerate(cleaned_diagrams):
            if len(diagram) > 0:
                # Sort by persistence
                persistence = diagram[:, 1] - diagram[:, 0]
                sorted_idx = np.argsort(persistence)[::-1]
                diagram = diagram[sorted_idx]
                
                dim_start_y = current_y
                
                # Plot each bar
                for birth, death in diagram:
                    if np.isfinite(birth) and np.isfinite(death):
                        ax.plot([birth, death], [current_y, current_y], 
                               color=colors[dim], 
                               linewidth=2,
                               solid_capstyle='butt')
                        current_y += 1
                
                if current_y > dim_start_y:  # Only add label if dimension has valid bars
                    dimension_labels.append(f"$H_{dim}$")
                    dimension_positions.append((dim_start_y + (current_y - dim_start_y)/2))
                
                # Add small gap between dimensions
                current_y += 1
        
        # Only proceed with plot styling if we have valid data
        if current_y > 0:
            # Set axis labels and title
            ax.set_title(title, pad=20, size=16)
            ax.set_xlabel('Time', size=12)
            
            # Set y-axis properties if we have valid dimension positions
            if dimension_positions:
                ax.set_yticks(dimension_positions)
                ax.set_yticklabels(dimension_labels)
            
            # Set x-axis limits with small padding
            padding = (global_max - global_min) * 0.05
            ax.set_xlim(global_min - padding, global_max + padding)
            
            # Set y-axis limits with padding
            ax.set_ylim(-1, current_y)
            
            # Style the plot
            ax.grid(True, axis='x', linestyle='--', alpha=0.7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
            # Add legend for dimensions that have valid bars
            valid_dims = [i for i, diag in enumerate(cleaned_diagrams) if len(diag) > 0]
            if valid_dims:
                legend_elements = [plt.Line2D([0], [0], color=colors[i], 
                                            label=f'$H_{i}$', linewidth=2)
                                 for i in valid_dims]
                ax.legend(handles=legend_elements, loc='center left', 
                         bbox_to_anchor=(1, 0.5))
        else:
            ax.text(0.5, 0.5, 'No valid persistence intervals found',
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_betti_curves(diagrams: List[np.ndarray],
                        title: str = "Betti Curves") -> plt.Figure:
        """
        Create Betti curves visualization.
        
        Args:
            diagrams: List of persistence diagrams from ripser
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Clean diagrams and get valid ranges
        def clean_diagram(diag):
            if len(diag) == 0:
                return diag
            valid_mask = np.all(np.isfinite(diag), axis=1)
            return diag[valid_mask]
        
        cleaned_diagrams = [clean_diagram(diag) for diag in diagrams]
        
        # Get global min/max for time range
        births_deaths = []
        for diag in cleaned_diagrams:
            if len(diag) > 0:
                births_deaths.extend(diag.flatten())
        
        if births_deaths:
            t_min, t_max = np.min(births_deaths), np.max(births_deaths)
            t = np.linspace(t_min, t_max, 100)
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
            
            for dim, diagram in enumerate(cleaned_diagrams):
                if len(diagram) > 0:
                    betti = np.zeros_like(t)
                    for birth, death in diagram:
                        if np.isfinite(birth) and np.isfinite(death):
                            betti += ((t >= birth) & (t < death)).astype(int)
                            
                    ax.plot(t, betti, color=colors[dim], 
                            label=f'$\\beta_{dim}$', linewidth=2)
            
            ax.set_title(title, pad=20, size=16)
            ax.set_xlabel('Time', size=12)
            ax.set_ylabel('Betti Number', size=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No valid persistence intervals found',
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes)
        
        plt.tight_layout()
        return fig


    @staticmethod
    def plot_persistence_landscape(diagrams: List[np.ndarray],
                                title: str = "Persistence Landscape",
                                n_layers: int = 5) -> plt.Figure:
    
        fig = plt.figure(figsize=(12, 6))
        plot_landscape_simple(PersLandscapeExact(diagrams, hom_deg=1),
                                    title=title)

        fig.tight_layout()    

        return fig


    @staticmethod
    def plot_critical_points(time_series: np.ndarray, 
                           critical_points: List[int],
                           title: str = "Critical Points Detection") -> plt.Figure:
        """
        Visualize critical points on the time series.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot time series
        ax.plot(time_series, alpha=0.7, label='Time Series')
        
        # Plot critical points
        ax.scatter(critical_points, 
                  time_series[critical_points],
                  color='red',
                  s=100,
                  label='Critical Points')
        
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig 
    


    @staticmethod
    def plot_critical_point_topology(time_series: np.ndarray,
                                   critical_point: int,
                                   window_size: int,
                                   diagrams_before: List[np.ndarray],
                                   diagrams_after: List[np.ndarray],
                                   title: str = "Topological Changes at Critical Point") -> plt.Figure:
        """
        Visualize topological changes around a critical point.
        """
        # Create subplots: time series window and persistence diagrams
        fig = plt.figure(figsize=(15, 5))
        gs = plt.GridSpec(1, 3, width_ratios=[2, 1, 1])
        
        # Plot time series window
        ax_ts = fig.add_subplot(gs[0])
        start = max(0, critical_point - window_size)
        end = min(len(time_series), critical_point + window_size)
        window = time_series[start:end]
        
        ax_ts.plot(range(start, end), window, 'b-', alpha=0.7)
        ax_ts.axvline(x=critical_point, color='r', linestyle='--', label='Critical Point')
        ax_ts.set_title('Time Series around Critical Point')
        ax_ts.set_xlabel('Time')
        ax_ts.set_ylabel('Value')
        ax_ts.legend()
        
        # Plot persistence diagram before
        ax_before = fig.add_subplot(gs[1])
        plot_diagrams(diagrams_before, show=False, ax=ax_before)
        ax_before.set_title('Persistence Diagram Before')
        
        # Plot persistence diagram after
        ax_after = fig.add_subplot(gs[2])
        plot_diagrams(diagrams_after, show=False, ax=ax_after)
        ax_after.set_title('Persistence Diagram After')
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
