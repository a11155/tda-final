import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from sklearn.decomposition import PCA
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from ripser import ripser
from persim import plot_diagrams
from typing import Dict, List, Tuple
from dataclasses import dataclass
@dataclass
class TimeSeriesWindow:
    """Represents a window of time series data."""
    start_idx: int
    end_idx: int
    values: np.ndarray




@dataclass
class PersistenceStats:
    """Statistics for persistence diagrams."""
    total_persistence: float
    max_persistence: float
    avg_persistence: float
    num_features: int
    birth_range: Tuple[float, float]
    death_range: Tuple[float, float]
    dimension: int

class TimeSeriesAnalysis:
    """Core class for time series analysis and TDA preparation."""
    
    def __init__(self, time_series: np.ndarray, sampling_rate: Optional[float] = None):
        """
        Initialize with a time series.
        
        Args:
            time_series: 1D numpy array containing the time series values
            sampling_rate: Optional sampling rate of the time series
        """
        if not isinstance(time_series, np.ndarray):
            raise TypeError("Time series must be a numpy array")
        
        if time_series.ndim != 1:
            raise ValueError("Time series must be 1-dimensional")
            
        self.time_series = time_series
        self.sampling_rate = sampling_rate
        self.length = len(time_series)

    def compute_stats(self, diagram: np.ndarray, dim: int) -> PersistenceStats:
        """Compute statistics for a persistence diagram after removing infinite values."""
        # Remove infinite values
        finite_mask = np.isfinite(diagram).all(axis=1)
        finite_diagram = diagram[finite_mask]
        
        if len(finite_diagram) == 0:
            return PersistenceStats(
                total_persistence=0.0,
                max_persistence=0.0,
                avg_persistence=0.0,
                num_features=0,
                birth_range=(0.0, 0.0),
                death_range=(0.0, 0.0),
                dimension=dim
            )
        
        
        persistence = finite_diagram[:, 1] - finite_diagram[:, 0]
        
        return PersistenceStats(
            total_persistence=float(np.sum(persistence)),
            max_persistence=float(np.max(persistence)) if len(persistence) > 0 else 0.0,
            avg_persistence=float(np.mean(persistence)) if len(persistence) > 0 else 0.0,
            num_features=len(persistence),
            birth_range=(float(np.min(finite_diagram[:, 0])), float(np.max(finite_diagram[:, 0]))),
            death_range=(float(np.min(finite_diagram[:, 1])), float(np.max(finite_diagram[:, 1]))),
            dimension=dim
        )

    def compute_persistence(self, embedding, max_dim: int = 1) -> Dict:
        """
        Compute persistence diagrams from Takens embedding.
        
        Args:
            max_dim: Maximum homology dimension to compute
            
        Returns:
            Dictionary containing persistence data and statistics
        """
            
        # Compute persistence diagrams
        diagrams = ripser(embedding, maxdim=max_dim)['dgms']
        
        # Compute statistics for each dimension
        stats = []
        for dim, diagram in enumerate(diagrams):
            if len(diagram) > 0:
                stats.append(self.compute_stats(diagram, dim))
            else:
                stats.append(PersistenceStats(
                    total_persistence=0.0,
                    max_persistence=0.0,
                    avg_persistence=0.0,
                    num_features=0,
                    birth_range=(0.0, 0.0),
                    death_range=(0.0, 0.0),
                    dimension=dim
                ))
        
        return {
            'diagrams': diagrams,
            'stats': stats
        }
    
    def create_sliding_windows(self, 
                             window_size: int, 
                             stride: Optional[int] = None) -> List[TimeSeriesWindow]:
        """
        Create sliding windows from the time series.
        
        Args:
            window_size: Size of each window
            stride: Number of points to move between windows (default: window_size/2)
            
        Returns:
            List of TimeSeriesWindow objects
        """
        if window_size > self.length:
            raise ValueError("Window size cannot be larger than time series length")
            
        if window_size < 2:
            raise ValueError("Window size must be at least 2")
            
        stride = stride or window_size // 2
        
        windows = []
        start_idx = 0
        
        while start_idx + window_size <= self.length:
            end_idx = start_idx + window_size
            window_data = self.time_series[start_idx:end_idx]
            
            windows.append(TimeSeriesWindow(
                start_idx=start_idx,
                end_idx=end_idx,
                values=window_data
            ))
            
            start_idx += stride
            
        return windows
    

    def create_takens_embedding(self, 
                              embedding_dimension: int, 
                              time_delay: int, stride = 1) -> np.ndarray:
        """
        Create Takens embedding of the time series.
        
        Args:
            embedding_dimension: Number of embedding dimensions
            time_delay: Time delay (tau) for the embedding
            
        Returns:
            Array of shape (n_points, embedding_dimension) containing the embedding
        """
        if embedding_dimension < 2:
            raise ValueError("Embedding dimension must be at least 2")
        
        if time_delay < 1:
            raise ValueError("Time delay must be at least 1")
            
        # Calculate number of points considering stride
        n_points = (len(self.time_series) - (embedding_dimension - 1) * time_delay - 1) // stride + 1
        
        if n_points < 1:
            raise ValueError("Time series too short for these parameters")
        
        embedding = np.zeros((n_points, embedding_dimension))
        
        for i in range(embedding_dimension):
            start_idx = i * time_delay
            indices = np.arange(start_idx, start_idx + n_points * stride, stride)
            embedding[:, i] = self.time_series[indices]
        
        self.embedding = embedding
        return embedding
    def project_embedding(self, 
                         embedding: np.ndarray, 
                         n_components: int = 2) -> np.ndarray:
        """
        Project embedding to lower dimension using PCA.
        
        Args:
            embedding: The high-dimensional embedding
            n_components: Number of components for projection (2 or 3)
            
        Returns:
            Array of shape (n_points, n_components)
        """
        if n_components not in [2, 3]:
            raise ValueError("Number of components must be 2 or 3")
            
        pca = PCA(n_components=n_components)
        return pca.fit_transform(embedding)



    def normalize_windows(self, windows: List[TimeSeriesWindow]) -> List[TimeSeriesWindow]:
        """
        Normalize each window to zero mean and unit variance.
        
        Args:
            windows: List of TimeSeriesWindow objects
            
        Returns:
            List of normalized TimeSeriesWindow objects
        """
        normalized_windows = []
        
        for window in windows:
            values = window.values
            normalized_values = (values - np.mean(values)) / (np.std(values) + 1e-8)
            
            normalized_windows.append(TimeSeriesWindow(
                start_idx=window.start_idx,
                end_idx=window.end_idx,
                values=normalized_values
            ))
            
        return normalized_windows

