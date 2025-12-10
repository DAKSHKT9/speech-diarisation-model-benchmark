"""
Abstract base class for diarization models.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass
import time

from utils import RTTMSegment


@dataclass
class DiarizationResult:
    """Result from a diarization model."""
    segments: List[RTTMSegment]
    processing_time: float  # seconds
    model_name: str
    file_id: str
    error: Optional[str] = None


class DiarizationModel(ABC):
    """Abstract base class for all diarization models."""
    
    def __init__(self, name: str):
        self.name = name
        self._is_loaded = False
    
    @abstractmethod
    def load(self) -> None:
        """Load the model into memory. Called once before processing."""
        pass
    
    @abstractmethod
    def _diarize(self, audio_path: str, file_id: str) -> List[RTTMSegment]:
        """
        Perform diarization on an audio file.
        
        Args:
            audio_path: Path to the audio file
            file_id: Identifier for the file (used in RTTM output)
            
        Returns:
            List of RTTMSegment objects
        """
        pass
    
    def diarize(self, audio_path: str, file_id: str) -> DiarizationResult:
        """
        Public method to perform diarization with timing.
        
        Args:
            audio_path: Path to the audio file
            file_id: Identifier for the file
            
        Returns:
            DiarizationResult with segments and timing info
        """
        if not self._is_loaded:
            self.load()
            self._is_loaded = True
        
        start_time = time.time()
        try:
            segments = self._diarize(audio_path, file_id)
            processing_time = time.time() - start_time
            return DiarizationResult(
                segments=segments,
                processing_time=processing_time,
                model_name=self.name,
                file_id=file_id
            )
        except Exception as e:
            processing_time = time.time() - start_time
            return DiarizationResult(
                segments=[],
                processing_time=processing_time,
                model_name=self.name,
                file_id=file_id,
                error=str(e)
            )
    
    def unload(self) -> None:
        """Unload the model from memory. Override if needed."""
        self._is_loaded = False
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

