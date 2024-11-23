from typing import Dict, List, Tuple
import os
import tempfile
import urllib.request
import zipfile
import numpy as np

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pkg_resources
import json

class SampleDatasets:
    """Handles sample music datasets for TDA analysis."""
    
    def __init__(self):
        # Define paths relative to package
        self.data_dir = Path(__file__).parent / "samples"
        self.metadata_path = self.data_dir / "metadata.json"
        
        # Load metadata if exists
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict:
        """Load metadata for samples."""
        try:
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._create_default_metadata()
    
    def _create_default_metadata(self) -> Dict:
        """Create default metadata structure."""
        return {
            'gtzan_subset': {
                'name': 'GTZAN Genre Subset',
                'description': 'Small subset of the GTZAN genre dataset featuring 3 genres with 2 samples each',
                'genres': ['classical', 'jazz', 'rock'],
                'samples': {
                    'classical': [
                        {'file': 'classical.00000.wav', 'duration': 30},
                        {'file': 'classical.00001.wav', 'duration': 30}
                    ],
                    'jazz': [
                        {'file': 'jazz.00000.wav', 'duration': 30},
                        {'file': 'jazz.00001.wav', 'duration': 30}
                    ],
                    'rock': [
                        {'file': 'rock.00000.wav', 'duration': 30},
                        {'file': 'rock.00001.wav', 'duration': 30}
                    ]
                }
            }
        }
    
    def list_datasets(self) -> Dict:
        """Return information about available datasets."""
        return self.metadata
    
    def get_dataset_path(self, dataset_name: str) -> Optional[Path]:
        """Get path to dataset directory."""
        dataset_path = self.data_dir / dataset_name
        return dataset_path if dataset_path.exists() else None
    
    def get_sample_path(self, dataset_name: str, genre: str, sample_idx: int) -> Optional[Path]:
        """Get path to specific sample file."""
        try:
            dataset = self.metadata[dataset_name]
            sample = dataset['samples'][genre][sample_idx]
            sample_path = self.data_dir / dataset_name / genre / sample['file']
            return sample_path if sample_path.exists() else None
        except (KeyError, IndexError):
            return None