"""
AssemblyAI speaker diarization API wrapper.
"""

import os
from typing import List, Optional
import time as time_module

from .base import DiarizationModel

from utils import RTTMSegment


class AssemblyAIModel(DiarizationModel):
    """
    Speaker diarization using AssemblyAI API.
    
    Note: This is a cloud-based API, so processing time includes
    upload and API latency.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize AssemblyAI model.
        
        Args:
            api_key: AssemblyAI API key (or set ASSEMBLYAI_API_KEY env var)
        """
        super().__init__(name='assemblyai')
        self.api_key = api_key or os.environ.get('ASSEMBLYAI_API_KEY')
        self.client = None
    
    def load(self) -> None:
        """Initialize the AssemblyAI client."""
        import assemblyai as aai
        
        if not self.api_key:
            raise ValueError(
                "AssemblyAI API key required. "
                "Set ASSEMBLYAI_API_KEY environment variable or pass api_key parameter."
            )
        
        print(f"Initializing {self.name} client...")
        aai.settings.api_key = self.api_key
        self.client = aai.Transcriber()
        print(f"  ✓ {self.name} client initialized")
    
    def _diarize(self, audio_path: str, file_id: str) -> List[RTTMSegment]:
        """
        Run diarization on audio file using AssemblyAI API.
        
        Args:
            audio_path: Path to audio file
            file_id: File identifier for RTTM output
            
        Returns:
            List of RTTMSegment objects
        """
        import assemblyai as aai
        
        # Configure transcription with speaker diarization
        config = aai.TranscriptionConfig(
            speaker_labels=True,
            # Don't need full transcription, but API requires it
            # We'll just use the speaker segments
        )
        
        # Submit and wait for transcription
        transcript = self.client.transcribe(audio_path, config=config)
        
        if transcript.status == aai.TranscriptStatus.error:
            raise RuntimeError(f"AssemblyAI transcription failed: {transcript.error}")
        
        # Extract speaker segments from utterances
        segments = []
        speaker_map = {}
        
        if transcript.utterances:
            for utterance in transcript.utterances:
                speaker = utterance.speaker
                
                # Map speaker letters (A, B, C...) to spk00, spk01, etc.
                if speaker not in speaker_map:
                    speaker_map[speaker] = f"spk{len(speaker_map):02d}"
                
                # Convert milliseconds to seconds
                start_sec = utterance.start / 1000.0
                end_sec = utterance.end / 1000.0
                duration = end_sec - start_sec
                
                segment = RTTMSegment(
                    file_id=file_id,
                    channel=1,
                    start=start_sec,
                    duration=duration,
                    speaker=speaker_map[speaker]
                )
                segments.append(segment)
        
        return segments
    
    def unload(self) -> None:
        """Clean up client."""
        self.client = None
        super().unload()

