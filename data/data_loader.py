import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import urllib.request
import os
import zipfile
import tempfile

class MusicDataLoader:
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the music data loader.
        
        Args:
            cache_dir: Directory to cache downloaded datasets. If None, uses system temp dir.
        """
        self.cache_dir = cache_dir or tempfile.gettempdir()
        self.sampling_rate = 22050  # Standard sampling rate for music analysis
        
    def load_audio_file(self, file_path: str) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Load an audio file and compute its MFCCs.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (raw_audio, mfccs)
        """
        # Load the audio file
        y, sr = librosa.load(file_path, sr=self.sampling_rate)
        
        # Compute MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        return y, mfccs
    
    def batch_load_directory(self, dir_path: str) -> Dict[str, Tuple[np.ndarray, List[np.ndarray]]]:
        """
        Load all audio files from a directory.
        
        Args:
            dir_path: Path to directory containing audio files
            
        Returns:
            Dictionary mapping filenames to their (audio, mfccs) tuples
        """
        results = {}
        path = Path(dir_path)
        
        for audio_file in path.glob("*.wav"):
            try:
                results[audio_file.name] = self.load_audio_file(str(audio_file))
            except Exception as e:
                print(f"Error loading {audio_file}: {e}")
                
        return results