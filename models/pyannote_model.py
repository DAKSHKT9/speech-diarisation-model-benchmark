"""
PyAnnote speaker diarization model wrapper.
Supports both pyannote/speaker-diarization-3.0 and 3.1.
"""

import os
from typing import List, Optional
import torch

from .base import DiarizationModel

from utils import RTTMSegment


class PyAnnoteModel(DiarizationModel):
    """
    Wrapper for pyannote speaker diarization models.
    
    Supports:
    - pyannote/speaker-diarization-3.1 (latest)
    - pyannote/speaker-diarization-3.0
    """
    
    SUPPORTED_VERSIONS = {
        'pyannote-3.1': 'pyannote/speaker-diarization-3.1',
        'pyannote-3.0': 'pyannote/speaker-diarization-3.0',
    }
    
    def __init__(self, version: str = 'pyannote-3.1', hf_token: Optional[str] = None):
        """
        Initialize PyAnnote model.
        
        Args:
            version: Model version key ('pyannote-3.1' or 'pyannote-3.0')
            hf_token: HuggingFace token for model access
        """
        if version not in self.SUPPORTED_VERSIONS:
            raise ValueError(f"Unsupported version: {version}. Choose from {list(self.SUPPORTED_VERSIONS.keys())}")
        
        super().__init__(name=version)
        self.version = version
        self.model_id = self.SUPPORTED_VERSIONS[version]
        self.hf_token = hf_token or os.environ.get('HF_TOKEN')
        self.pipeline = None
        self.device = self._get_device()
    
    def _get_device(self) -> torch.device:
        """Get the best available device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    
    def load(self) -> None:
        """Load the pyannote pipeline."""
        from pyannote.audio import Pipeline
        
        if not self.hf_token:
            raise ValueError(
                "HuggingFace token required for pyannote models. "
                "Set HF_TOKEN environment variable or pass hf_token parameter."
            )
        
        print(f"Loading {self.name} on {self.device}...")
        self.pipeline = Pipeline.from_pretrained(
            self.model_id,
            use_auth_token=self.hf_token
        )
        
        # Check if pipeline loaded successfully
        if self.pipeline is None:
            raise RuntimeError(
                f"Failed to load {self.model_id}. "
                f"Make sure you've accepted the model terms at https://huggingface.co/{self.model_id}"
            )
        
        # Move to appropriate device
        self.pipeline.to(self.device)
        print(f"  ✓ {self.name} loaded successfully")
    
    def _diarize(self, audio_path: str, file_id: str) -> List[RTTMSegment]:
        """
        Run diarization on audio file.
        
        Args:
            audio_path: Path to audio file
            file_id: File identifier for RTTM output
            
        Returns:
            List of RTTMSegment objects
        """
        # Run the pipeline
        diarization = self.pipeline(audio_path)
        
        # Convert pyannote Annotation to RTTMSegment list
        segments = []
        
        # pyannote uses its own speaker labels (SPEAKER_00, etc.)
        # We'll map them to spk00, spk01, etc.
        speaker_map = {}
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speaker_map:
                speaker_map[speaker] = f"spk{len(speaker_map):02d}"
            
            segment = RTTMSegment(
                file_id=file_id,
                channel=1,
                start=turn.start,
                duration=turn.duration,
                speaker=speaker_map[speaker]
            )
            segments.append(segment)
        
        return segments
    
    def unload(self) -> None:
        """Unload the pipeline from memory."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            
            # Clear GPU/MPS cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
        
        super().unload()

