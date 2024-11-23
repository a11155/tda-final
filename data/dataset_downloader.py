import requests
import os
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import urllib.request
import tarfile
import shutil
import json


class GTZANDownloader:
    """Downloads and processes GTZAN dataset samples."""
    
    # GTZAN is typically hosted at several locations
    GTZAN_URLS = [
        "http://opihi.cs.uvic.ca/sound/genres.tar.gz",
        "https://librosa.org/data/audio/genres.tar.gz"  # Backup source
    ]
    
    def __init__(self, target_dir: Path):
        self.target_dir = target_dir
        self.genres = ['classical', 'jazz', 'rock']
        self.samples_per_genre = 2
        self.duration = 30  # seconds
        self.sr = 22050  # sampling rate
        
    def download_dataset(self) -> bool:
        """
        Download and extract GTZAN dataset.
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Create temporary directory for downloads
        temp_dir = Path(self.target_dir) / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Try each URL until one works
            for url in self.GTZAN_URLS:
                try:
                    print(f"Attempting to download from {url}")
                    tar_path = temp_dir / "genres.tar.gz"
                    
                    # Download with progress bar
                    with tqdm(unit='B', unit_scale=True, desc="Downloading") as pbar:
                        urllib.request.urlretrieve(
                            url, 
                            filename=str(tar_path),
                            reporthook=lambda count, block_size, total_size: pbar.update(block_size)
                        )
                    
                    # If download successful, break the loop
                    break
                except Exception as e:
                    print(f"Failed to download from {url}: {e}")
                    continue
            else:
                raise Exception("Failed to download from all sources")
            
            # Extract the tar file
            print("Extracting files...")
            with tarfile.open(tar_path) as tar:
                tar.extractall(path=temp_dir)
            
            # Process and organize samples
            self._process_samples(temp_dir / "genres")
            
            return True
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return False
        
        finally:
            # Cleanup
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def _process_samples(self, genres_dir: Path):
        """Process and organize the downloaded samples."""
        # Create metadata structure
        metadata = {
            "gtzan_subset": {
                "name": "GTZAN Genre Subset",
                "description": "Small subset of the GTZAN genre dataset featuring 3 genres with 2 samples each",
                "genres": self.genres,
                "samples": {genre: [] for genre in self.genres}
            }
        }
        
        # Process each genre
        for genre in self.genres:
            genre_dir = self.target_dir / "gtzan_subset" / genre
            genre_dir.mkdir(parents=True, exist_ok=True)
            
            source_dir = genres_dir / genre
            if not source_dir.exists():
                print(f"Warning: Genre directory {genre} not found")
                continue
            
            # Process samples for this genre
            wav_files = list(source_dir.glob("*.wav"))[:self.samples_per_genre]
            
            for idx, wav_file in enumerate(wav_files):
                try:
                    # Load and resample if necessary
                    y, sr = librosa.load(wav_file, sr=self.sr, duration=self.duration)
                    
                    # Ensure exact duration
                    if len(y) < self.sr * self.duration:
                        y = np.pad(y, (0, self.sr * self.duration - len(y)))
                    else:
                        y = y[:self.sr * self.duration]
                    
                    # Save processed file
                    output_filename = f"{genre}.{idx:05d}.wav"
                    output_path = genre_dir / output_filename
                    sf.write(output_path, y, self.sr)
                    
                    # Add to metadata
                    metadata["gtzan_subset"]["samples"][genre].append({
                        "file": output_filename,
                        "duration": self.duration
                    })
                    
                except Exception as e:
                    print(f"Error processing {wav_file}: {e}")
                    continue
        
        # Save metadata
        metadata_path = self.target_dir / "samples" / "metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

def main():
    # Specify target directory (relative to script location)
    target_dir = Path(__file__).parent
    
    # Create and run downloader
    downloader = GTZANDownloader(target_dir)
    
    print("Starting GTZAN subset download and processing...")
    if downloader.download_dataset():
        print("Successfully created GTZAN subset!")
    else:
        print("Failed to create GTZAN subset")

if __name__ == "__main__":
    main()