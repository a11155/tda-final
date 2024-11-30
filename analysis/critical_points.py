import numpy as np
from typing import List, Tuple
from persim import bottleneck, wasserstein
from scipy.signal import find_peaks
from ripser import ripser
from analysis.time_series import TimeSeriesAnalysis


class TopologicalCriticalPoints:
    def __init__(self, window_size: int, stride: int):
        self.window_size = window_size
        self.stride = stride
        
    def _compute_diagram_distances(self, 
                                 diagrams: List[List[np.ndarray]],
                                 metric: str = 'wasserstein') -> np.ndarray:
        """
        Compute distances between consecutive persistence diagrams.
        
        Args:
            diagrams: List of persistence diagrams, where each diagram is a list of
                     arrays for different homology dimensions
            metric: Either 'wasserstein' or 'bottleneck'
            
        Returns:
            Array of distances
        """
        distances = []
        for i in range(len(diagrams) - 1):
            # Sum distances across all dimensions
            total_dist = 0
            for dim in range(len(diagrams[0])):  # For each homology dimension
                if metric == 'wasserstein':
                    total_dist += wasserstein(diagrams[i][dim], diagrams[i + 1][dim])
                else:
                    total_dist += bottleneck(diagrams[i][dim], diagrams[i + 1][dim])
            distances.append(total_dist)
        return np.array(distances)
    
    def find_critical_points(self, time_series: np.ndarray, 
                           threshold: float = 0.95,
                           metric: str = 'wasserstein') -> List[int]:
        """
        Find critical points using topological features.
        
        Args:
            time_series: Input time series
            threshold: Percentile threshold for peak detection
            metric: Distance metric to use ('wasserstein' or 'bottleneck')
            
        Returns:
            List of indices where critical points occur
        """
        # Get sliding windows
        windows = []
        for i in range(0, len(time_series) - self.window_size + 1, self.stride):
            windows.append(time_series[i:i + self.window_size])
        
        # Compute persistence diagrams for each window
        diagrams = []
        for window in windows:
            # Create Takens embedding
            embedding = np.array([
                window[i:i + 2] for i in range(len(window) - 1)
            ])
            diagram = ripser(embedding)['dgms']
            diagrams.append(diagram)
        
        # Compute distances between consecutive diagrams
        distances = self._compute_diagram_distances(diagrams, metric)
        
        # Find peaks in distances
        height = np.percentile(distances, threshold * 100)
        peaks, _ = find_peaks(distances, height=height)
        
        # Convert window indices to time series indices
        return [p * self.stride + self.window_size // 2 for p in peaks]


    def get_critical_point_diagrams(self, time_series: np.ndarray, 
                                  critical_point: int,
                                  window_size: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Get persistence diagrams before and after a critical point.
        """
        # Get windows before and after critical point
        start_before = max(0, critical_point - window_size)
        end_before = critical_point
        start_after = critical_point
        end_after = min(len(time_series), critical_point + window_size)
        
        window_before = time_series[start_before:end_before]
        window_after = time_series[start_after:end_after]
        
        # Create embeddings
        embedding_before = np.array([
            window_before[i:i + 2] for i in range(len(window_before) - 1)
        ])
        embedding_after = np.array([
            window_after[i:i + 2] for i in range(len(window_after) - 1)
        ])
        
        # Compute persistence diagrams
        diagrams_before = ripser(embedding_before)['dgms']
        diagrams_after = ripser(embedding_after)['dgms']
        
        return diagrams_before, diagrams_after

