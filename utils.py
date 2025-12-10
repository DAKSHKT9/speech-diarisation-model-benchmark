"""
Utility functions for RTTM parsing, conversion, and result formatting.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json


@dataclass
class RTTMSegment:
    """Represents a single RTTM segment."""
    file_id: str
    channel: int
    start: float
    duration: float
    speaker: str
    
    @property
    def end(self) -> float:
        return self.start + self.duration
    
    def to_rttm_line(self) -> str:
        """Convert segment to RTTM format line."""
        return f"SPEAKER {self.file_id} {self.channel} {self.start:.6f} {self.duration:.6f} <NA> <NA> {self.speaker} <NA> <NA>"


def parse_rttm_file(rttm_path: str) -> List[RTTMSegment]:
    """
    Parse an RTTM file and return a list of RTTMSegment objects.
    
    RTTM format:
    SPEAKER <file_id> <channel> <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>
    """
    segments = []
    with open(rttm_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith('SPEAKER'):
                continue
            
            parts = line.split()
            if len(parts) < 8:
                continue
                
            segment = RTTMSegment(
                file_id=parts[1],
                channel=int(parts[2]),
                start=float(parts[3]),
                duration=float(parts[4]),
                speaker=parts[7]
            )
            segments.append(segment)
    
    return segments


def segments_to_rttm_string(segments: List[RTTMSegment]) -> str:
    """Convert a list of segments to RTTM format string."""
    return '\n'.join(seg.to_rttm_line() for seg in segments)


def write_rttm_file(segments: List[RTTMSegment], output_path: str) -> None:
    """Write segments to an RTTM file."""
    with open(output_path, 'w') as f:
        f.write(segments_to_rttm_string(segments))
        f.write('\n')


def convert_assemblyai_to_segments(
    utterances: List[Dict],
    file_id: str,
    channel: int = 1
) -> List[RTTMSegment]:
    """
    Convert AssemblyAI utterances to RTTMSegment objects.
    
    AssemblyAI format: [{"speaker": "A", "start": 1234, "end": 5678}, ...]
    Times are in milliseconds.
    """
    # Map speaker letters to spk00, spk01, etc.
    speaker_map = {}
    segments = []
    
    for utt in utterances:
        speaker_letter = utt.get('speaker', 'A')
        if speaker_letter not in speaker_map:
            speaker_map[speaker_letter] = f"spk{len(speaker_map):02d}"
        
        start_sec = utt['start'] / 1000.0
        end_sec = utt['end'] / 1000.0
        duration = end_sec - start_sec
        
        segment = RTTMSegment(
            file_id=file_id,
            channel=channel,
            start=start_sec,
            duration=duration,
            speaker=speaker_map[speaker_letter]
        )
        segments.append(segment)
    
    return segments


def convert_speechbrain_to_segments(
    diarization_output: List[Tuple[float, float, int]],
    file_id: str,
    channel: int = 1
) -> List[RTTMSegment]:
    """
    Convert SpeechBrain diarization output to RTTMSegment objects.
    
    SpeechBrain format: [(start, end, cluster_id), ...]
    """
    segments = []
    
    for start, end, cluster_id in diarization_output:
        duration = end - start
        speaker = f"spk{int(cluster_id):02d}"
        
        segment = RTTMSegment(
            file_id=file_id,
            channel=channel,
            start=start,
            duration=duration,
            speaker=speaker
        )
        segments.append(segment)
    
    return segments


def load_dataset_info(dataset_path: str) -> Dict:
    """
    Load dataset information from stats.json if available,
    otherwise scan the directory.
    """
    dataset_path = Path(dataset_path)
    stats_file = dataset_path / 'stats.json'
    
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            return json.load(f)
    
    # Scan directory for audio files
    audio_dir = dataset_path / 'audio'
    labels_dir = dataset_path / 'labels'
    
    files = []
    for audio_file in audio_dir.glob('*.wav'):
        file_id = audio_file.stem
        label_file = labels_dir / f'{file_id}.rttm'
        
        if label_file.exists():
            files.append({
                'file_id': file_id,
                'audio_path': str(audio_file),
                'label_path': str(label_file)
            })
    
    return {
        'total_files': len(files),
        'files': files
    }


def get_audio_label_pairs(dataset_path: str) -> List[Dict]:
    """
    Get list of (audio_path, label_path, file_id) for all files in dataset.
    """
    dataset_path = Path(dataset_path)
    audio_dir = dataset_path / 'audio'
    labels_dir = dataset_path / 'labels'
    
    pairs = []
    for audio_file in sorted(audio_dir.glob('*.wav')):
        file_id = audio_file.stem
        label_file = labels_dir / f'{file_id}.rttm'
        
        if label_file.exists():
            pairs.append({
                'file_id': file_id,
                'audio_path': str(audio_file),
                'label_path': str(label_file)
            })
    
    return pairs


def ensure_cache_dir(cache_dir: str = '.benchmark_cache') -> Path:
    """Ensure cache directory exists and return path."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)
    return cache_path


def get_cache_path(cache_dir: str, model_name: str, file_id: str) -> Path:
    """Get cache file path for a specific model and audio file."""
    cache_path = ensure_cache_dir(cache_dir)
    model_dir = cache_path / model_name.replace('/', '_').replace(' ', '_')
    model_dir.mkdir(exist_ok=True)
    return model_dir / f'{file_id}.rttm'


def load_cached_result(cache_path: Path) -> Optional[List[RTTMSegment]]:
    """Load cached RTTM result if exists."""
    if cache_path.exists():
        return parse_rttm_file(str(cache_path))
    return None


def save_cached_result(cache_path: Path, segments: List[RTTMSegment]) -> None:
    """Save RTTM result to cache."""
    write_rttm_file(segments, str(cache_path))

