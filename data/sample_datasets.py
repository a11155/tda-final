import numpy as np
from typing import List, Tuple, Optional
class SampleData:
   @staticmethod
   def get_sine_wave(n_points: int, frequency: float = 1.0, t_max: float = 10.0) -> np.ndarray:
       t = np.linspace(0, t_max, n_points)
       return np.sin(2 * np.pi * frequency * t)

   @staticmethod
   def get_random_walk(n_points: int, seed: Optional[int] = None) -> np.ndarray:
       if seed is not None:
           np.random.seed(seed)
       return np.cumsum(np.random.randn(n_points))

   @staticmethod
   def get_composite_signal(n_points: int, 
                          f1: float = 0.5, 
                          f2: float = 1.0, 
                          noise_level: float = 0.0,
                          t_max: float = 10.0) -> np.ndarray:
       t = np.linspace(0, t_max, n_points)
       signal = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)
       if noise_level > 0:
           signal += np.random.normal(0, noise_level, n_points)
       return signal