import numpy as np
from typing import List, Tuple
from persim import bottleneck, wasserstein
from scipy.signal import find_peaks
from ripser import ripser
from analysis.time_series import TimeSeriesAnalysis


class TopologicalCriticalPoints:
    def __init__(self, window_size: int, stride: int, embedding_dim: int = 2, time_delay: int = 1):
        """
        Initialize the detector.
        
        Args:
            window_size: Size of sliding window
            stride: Step size between windows
            embedding_dim: Dimension of the Takens embedding
            time_delay: Time delay for the embedding
        """
        self.window_size = window_size
        self.stride = stride
        self.embedding_dim = embedding_dim
        self.time_delay = time_delay

    def _create_takens_embedding(self, window: np.ndarray) -> np.ndarray:
        """
        Create Takens embedding from time series window.
        
        Args:
            window: Time series window
            
        Returns:
            Takens embedding of the window
        """
        N = len(window)
        if N < self.embedding_dim * self.time_delay:
            raise ValueError("Window size too small for selected embedding parameters")
            
        num_points = N - (self.embedding_dim - 1) * self.time_delay
        embedding = np.zeros((num_points, self.embedding_dim))
        
        for i in range(num_points):
            for j in range(self.embedding_dim):
                embedding[i, j] = window[i + j * self.time_delay]
                
        return embedding


    def _create_takens_embedding(self, 
                              window: np.ndarray) -> np.ndarray:
        """
        Create Takens embedding of the time series.
        
        Args:
            embedding_dimension: Number of embedding dimensions
            time_delay: Time delay (tau) for the embedding
            
        Returns:
            Array of shape (n_points, embedding_dimension) containing the embedding
        """
        if self.embedding_dim < 2:
            raise ValueError("Embedding dimension must be at least 2")
        
        if self.time_delay < 1:
            raise ValueError("Time delay must be at least 1")
            
        n_points = (len(window) - (self.embedding_dim - 1) * self.time_delay - 1) // self.stride + 1
        
        if n_points < 1:
            raise ValueError("Time series too short for these parameters")
        
        embedding = np.zeros((n_points, self.embedding_dim))
        
        for i in range(self.embedding_dim):
            start_idx = i * self.time_delay
            indices = np.arange(start_idx, start_idx + n_points * self.stride, self.stride)
            embedding[:, i] = window[indices]
        
        return embedding

    def _compute_diagram_distances(self, 
                                 diagrams: List[List[np.ndarray]],
                                 metric: str = 'wasserstein') -> np.ndarray:
        """
        Compute distances between consecutive persistence diagrams.
        
        Args:
            diagrams: List of persistence diagrams
            metric: Distance metric ('wasserstein' or 'bottleneck')
            
        Returns:
            Array of distances
        """
        if not diagrams or len(diagrams) < 2:
            return np.array([])
            
        distances = []
        for i in range(len(diagrams) - 1):
            total_dist = 0
            for dim in range(len(diagrams[0])):
                diag1 = diagrams[i][dim]
                diag2 = diagrams[i + 1][dim]
                
                if len(diag1) == 0 and len(diag2) == 0:
                    continue
                elif len(diag1) == 0 or len(diag2) == 0:
                    total_dist += 1.0
                else:
                    if metric == 'wasserstein':
                        total_dist += wasserstein(diag1, diag2)
                    else:
                        total_dist += bottleneck(diag1, diag2)
                        
            distances.append(total_dist)
            
        return np.array(distances)

    def find_critical_points(self, 
                           time_series: np.ndarray, 
                           threshold: float = 0.95,
                           metric: str = 'wasserstein') -> List[int]:
        """
        Find critical points using topological features.
        
        Args:
            time_series: Input time series
            threshold: Percentile threshold for peak detection
            metric: Distance metric to use
            
        Returns:
            List of indices where critical points occur
        """
        if len(time_series) < self.window_size:
            return []
            
        windows = []
        for i in range(0, len(time_series) - self.window_size + 1, self.stride):
            windows.append(time_series[i:i + self.window_size])
        
        diagrams = []
        for window in windows:
            try:
                #print("WINDOW:")
                #print(window)
                embedding = self._create_takens_embedding(window)
                diagram = ripser(embedding)['dgms']
                diagrams.append(diagram)
            except Exception as e:
                print(f"Warning: Failed to compute diagram: {e}")
                return []
        
        if not diagrams:
            return []
            
        distances = self._compute_diagram_distances(diagrams, metric)
        
        if len(distances) == 0:
            return []
            
        height = np.percentile(distances, threshold * 100)
        peaks, _ = find_peaks(distances, height=height)
        
        # Convert window indices to time series indices
        return [p * self.stride + self.window_size // 2 for p in peaks]

    def get_critical_point_diagrams(self, 
                                  time_series: np.ndarray, 
                                  critical_point: int,
                                  window_size: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Get persistence diagrams before and after a critical point.
        """
    
            
        # Might have issues with the boundary cases
        start_before = critical_point - window_size
        end_before = critical_point
        start_after = critical_point
        end_after = critical_point + window_size
        
        window_before = time_series[max(0, start_before):end_before]
        window_after = time_series[start_after:min(end_after, len(time_series) - window_size)]
        
        embedding_before = self._create_takens_embedding(window_before)
        embedding_after = self._create_takens_embedding(window_after)
        
        # Compute persistence diagrams
        diagrams_before = ripser(embedding_before)['dgms']
        diagrams_after = ripser(embedding_after)['dgms']
        
        return diagrams_before, diagrams_after