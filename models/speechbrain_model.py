"""
SpeechBrain speaker diarization model wrapper.
Uses ECAPA-TDNN embeddings with spectral clustering.
"""

import os
from typing import List, Tuple
import torch
import numpy as np

from .base import DiarizationModel

from utils import RTTMSegment


class SpeechBrainModel(DiarizationModel):
    """
    Speaker diarization using SpeechBrain ECAPA-TDNN embeddings
    with spectral clustering.
    """
    
    def __init__(self, embedding_model: str = 'speechbrain/spkrec-ecapa-voxceleb'):
        """
        Initialize SpeechBrain diarization model.
        
        Args:
            embedding_model: HuggingFace model ID for speaker embeddings
        """
        super().__init__(name='speechbrain-ecapa')
        self.embedding_model_id = embedding_model
        self.encoder = None
        self.device = self._get_device()
        
        # VAD and segmentation parameters
        self.segment_duration = 1.5  # seconds
        self.segment_step = 0.75  # seconds (50% overlap)
        self.min_segment_duration = 0.5  # minimum speech segment
        
    def _get_device(self) -> torch.device:
        """Get the best available device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    
    def load(self) -> None:
        """Load the SpeechBrain encoder model."""
        from speechbrain.inference.speaker import EncoderClassifier
        
        print(f"Loading {self.name} on {self.device}...")
        
        self.encoder = EncoderClassifier.from_hparams(
            source=self.embedding_model_id,
            savedir=f".model_cache/{self.embedding_model_id.replace('/', '_')}",
            run_opts={"device": str(self.device)}
        )
        
        print(f"  ✓ {self.name} loaded successfully")
    
    def _load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """Load audio file and return waveform and sample rate."""
        import torchaudio
        
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        return waveform, sample_rate
    
    def _extract_segments_with_vad(
        self, 
        waveform: torch.Tensor, 
        sample_rate: int
    ) -> List[Tuple[float, float, torch.Tensor]]:
        """
        Extract audio segments using energy-based VAD.
        
        Returns list of (start_time, end_time, segment_waveform) tuples.
        """
        # Simple energy-based VAD
        waveform_np = waveform.squeeze().numpy()
        
        # Frame-based energy computation
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        hop_length = int(0.010 * sample_rate)    # 10ms hop
        
        # Compute energy per frame
        num_frames = (len(waveform_np) - frame_length) // hop_length + 1
        energy = np.zeros(num_frames)
        
        for i in range(num_frames):
            start = i * hop_length
            end = start + frame_length
            frame = waveform_np[start:end]
            energy[i] = np.sum(frame ** 2)
        
        # Dynamic threshold based on energy distribution
        threshold = np.percentile(energy, 30)  # Bottom 30% considered silence
        
        # Find speech regions
        is_speech = energy > threshold
        
        # Smooth the VAD decisions (median filter)
        from scipy.ndimage import median_filter
        is_speech = median_filter(is_speech.astype(float), size=15) > 0.5
        
        # Extract contiguous speech segments
        segments = []
        in_speech = False
        speech_start = 0
        
        for i, speech in enumerate(is_speech):
            time = i * hop_length / sample_rate
            
            if speech and not in_speech:
                speech_start = time
                in_speech = True
            elif not speech and in_speech:
                speech_end = time
                if speech_end - speech_start >= self.min_segment_duration:
                    segments.append((speech_start, speech_end))
                in_speech = False
        
        # Handle case where speech continues to end
        if in_speech:
            speech_end = len(waveform_np) / sample_rate
            if speech_end - speech_start >= self.min_segment_duration:
                segments.append((speech_start, speech_end))
        
        # Split long segments and extract waveforms
        final_segments = []
        for start, end in segments:
            duration = end - start
            
            if duration <= self.segment_duration:
                start_sample = int(start * sample_rate)
                end_sample = int(end * sample_rate)
                segment_wave = waveform[:, start_sample:end_sample]
                final_segments.append((start, end, segment_wave))
            else:
                # Split into overlapping chunks
                current = start
                while current < end:
                    chunk_end = min(current + self.segment_duration, end)
                    start_sample = int(current * sample_rate)
                    end_sample = int(chunk_end * sample_rate)
                    segment_wave = waveform[:, start_sample:end_sample]
                    
                    if segment_wave.shape[1] > int(0.3 * sample_rate):  # At least 0.3s
                        final_segments.append((current, chunk_end, segment_wave))
                    
                    current += self.segment_step
        
        return final_segments
    
    def _extract_embeddings(
        self, 
        segments: List[Tuple[float, float, torch.Tensor]]
    ) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """
        Extract speaker embeddings for each segment.
        
        Returns (embeddings array, list of (start, end) times).
        """
        embeddings = []
        times = []
        
        for start, end, segment_wave in segments:
            # Move to device
            segment_wave = segment_wave.to(self.device)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.encoder.encode_batch(segment_wave)
                embeddings.append(embedding.squeeze().cpu().numpy())
            
            times.append((start, end))
        
        return np.array(embeddings), times
    
    def _cluster_embeddings(
        self, 
        embeddings: np.ndarray,
        max_speakers: int = 10
    ) -> np.ndarray:
        """
        Cluster embeddings using spectral clustering.
        Automatically estimates number of speakers.
        
        Returns array of cluster labels.
        """
        from sklearn.cluster import SpectralClustering
        from sklearn.metrics import silhouette_score
        from scipy.spatial.distance import cosine
        
        n_samples = len(embeddings)
        
        if n_samples <= 1:
            return np.zeros(n_samples, dtype=int)
        
        # Compute affinity matrix (cosine similarity)
        affinity = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                if i == j:
                    affinity[i, j] = 1.0
                else:
                    sim = 1 - cosine(embeddings[i], embeddings[j])
                    affinity[i, j] = max(0, sim)  # Ensure non-negative
        
        # Estimate number of speakers using silhouette score
        max_speakers = min(max_speakers, n_samples - 1)
        best_n_clusters = 2
        best_score = -1
        
        for n_clusters in range(2, max_speakers + 1):
            try:
                clustering = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity='precomputed',
                    random_state=42,
                    assign_labels='kmeans'
                )
                labels = clustering.fit_predict(affinity)
                
                if len(set(labels)) > 1:
                    score = silhouette_score(embeddings, labels, metric='cosine')
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters
            except Exception:
                continue
        
        # Final clustering with best number of clusters
        clustering = SpectralClustering(
            n_clusters=best_n_clusters,
            affinity='precomputed',
            random_state=42,
            assign_labels='kmeans'
        )
        labels = clustering.fit_predict(affinity)
        
        return labels
    
    def _merge_adjacent_segments(
        self,
        times: List[Tuple[float, float]],
        labels: np.ndarray,
        gap_threshold: float = 0.5
    ) -> List[Tuple[float, float, int]]:
        """
        Merge adjacent segments with same speaker label.
        
        Args:
            times: List of (start, end) times
            labels: Cluster labels for each segment
            gap_threshold: Maximum gap (seconds) to merge across
            
        Returns:
            List of (start, end, speaker_id) tuples
        """
        if len(times) == 0:
            return []
        
        # Sort by start time
        sorted_indices = np.argsort([t[0] for t in times])
        sorted_times = [times[i] for i in sorted_indices]
        sorted_labels = labels[sorted_indices]
        
        merged = []
        current_start, current_end = sorted_times[0]
        current_label = sorted_labels[0]
        
        for i in range(1, len(sorted_times)):
            start, end = sorted_times[i]
            label = sorted_labels[i]
            
            # Check if should merge
            if label == current_label and start - current_end <= gap_threshold:
                current_end = max(current_end, end)
            else:
                merged.append((current_start, current_end, int(current_label)))
                current_start, current_end = start, end
                current_label = label
        
        # Add last segment
        merged.append((current_start, current_end, int(current_label)))
        
        return merged
    
    def _diarize(self, audio_path: str, file_id: str) -> List[RTTMSegment]:
        """
        Run diarization on audio file.
        
        Args:
            audio_path: Path to audio file
            file_id: File identifier for RTTM output
            
        Returns:
            List of RTTMSegment objects
        """
        # Load audio
        waveform, sample_rate = self._load_audio(audio_path)
        
        # Extract speech segments with VAD
        segments = self._extract_segments_with_vad(waveform, sample_rate)
        
        if len(segments) == 0:
            return []
        
        # Extract embeddings
        embeddings, times = self._extract_embeddings(segments)
        
        if len(embeddings) == 0:
            return []
        
        # Cluster embeddings
        labels = self._cluster_embeddings(embeddings)
        
        # Merge adjacent segments
        merged_segments = self._merge_adjacent_segments(times, labels)
        
        # Convert to RTTMSegment
        rttm_segments = []
        for start, end, speaker_id in merged_segments:
            segment = RTTMSegment(
                file_id=file_id,
                channel=1,
                start=start,
                duration=end - start,
                speaker=f"spk{speaker_id:02d}"
            )
            rttm_segments.append(segment)
        
        return rttm_segments
    
    def unload(self) -> None:
        """Unload the encoder from memory."""
        if self.encoder is not None:
            del self.encoder
            self.encoder = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
        
        super().unload()

