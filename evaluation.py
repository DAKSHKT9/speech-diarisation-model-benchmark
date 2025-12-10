"""
Evaluation metrics for speaker diarization using pyannote.metrics.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
import numpy as np

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import (
    DiarizationErrorRate,
    JaccardErrorRate,
    DiarizationPurity,
    DiarizationCoverage
)

from utils import RTTMSegment, parse_rttm_file


@dataclass
class DiarizationMetrics:
    """Container for diarization evaluation metrics."""
    der: float  # Diarization Error Rate
    jer: float  # Jaccard Error Rate
    confusion: float  # Speaker confusion rate
    missed: float  # Missed detection rate
    false_alarm: float  # False alarm rate
    total_speech: float  # Total speech duration in reference
    
    def to_dict(self) -> Dict:
        return {
            'der': round(self.der, 4),
            'jer': round(self.jer, 4),
            'confusion': round(self.confusion, 4),
            'missed': round(self.missed, 4),
            'false_alarm': round(self.false_alarm, 4),
            'total_speech_sec': round(self.total_speech, 2)
        }


@dataclass 
class BenchmarkResult:
    """Result for a single file evaluation."""
    file_id: str
    model_name: str
    metrics: Optional[DiarizationMetrics]
    processing_time: float
    num_speakers_ref: int
    num_speakers_hyp: int
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        result = {
            'file_id': self.file_id,
            'model_name': self.model_name,
            'processing_time_sec': round(self.processing_time, 2),
            'num_speakers_ref': self.num_speakers_ref,
            'num_speakers_hyp': self.num_speakers_hyp,
        }
        if self.metrics:
            result['metrics'] = self.metrics.to_dict()
        if self.error:
            result['error'] = self.error
        return result


@dataclass
class ModelBenchmarkSummary:
    """Summary of benchmark results for a single model."""
    model_name: str
    num_files: int
    num_successful: int
    num_failed: int
    avg_der: float
    avg_jer: float
    avg_confusion: float
    avg_missed: float
    avg_false_alarm: float
    total_processing_time: float
    per_file_results: List[BenchmarkResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'num_files': self.num_files,
            'num_successful': self.num_successful,
            'num_failed': self.num_failed,
            'avg_der': round(self.avg_der, 4),
            'avg_jer': round(self.avg_jer, 4),
            'avg_confusion': round(self.avg_confusion, 4),
            'avg_missed': round(self.avg_missed, 4),
            'avg_false_alarm': round(self.avg_false_alarm, 4),
            'total_processing_time_sec': round(self.total_processing_time, 2),
            'per_file': [r.to_dict() for r in self.per_file_results]
        }


def segments_to_annotation(segments: List[RTTMSegment], uri: str = '') -> Annotation:
    """
    Convert list of RTTMSegment to pyannote Annotation.
    
    Args:
        segments: List of RTTMSegment objects
        uri: Uniform Resource Identifier for the annotation
        
    Returns:
        pyannote.core.Annotation object
    """
    annotation = Annotation(uri=uri)
    
    for seg in segments:
        segment = Segment(seg.start, seg.end)
        annotation[segment] = seg.speaker
    
    return annotation


def load_reference_annotation(rttm_path: str) -> Annotation:
    """
    Load reference annotation from RTTM file.
    
    Args:
        rttm_path: Path to RTTM file
        
    Returns:
        pyannote.core.Annotation object
    """
    segments = parse_rttm_file(rttm_path)
    if segments:
        uri = segments[0].file_id
    else:
        uri = ''
    return segments_to_annotation(segments, uri=uri)


def compute_metrics(
    reference: Annotation,
    hypothesis: Annotation,
    collar: float = 0.25,
    skip_overlap: bool = False
) -> DiarizationMetrics:
    """
    Compute diarization metrics between reference and hypothesis.
    
    Args:
        reference: Ground truth annotation
        hypothesis: Predicted annotation
        collar: Forgiveness collar in seconds (default 0.25s)
        skip_overlap: Whether to skip overlapping speech regions
        
    Returns:
        DiarizationMetrics object with all computed metrics
    """
    # Initialize metric calculators
    der_metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    jer_metric = JaccardErrorRate(collar=collar, skip_overlap=skip_overlap)
    
    # Compute DER and its components
    der_result = der_metric(reference, hypothesis, detailed=True)
    
    # Extract components
    total = der_result['total']
    if total > 0:
        confusion = der_result['confusion'] / total
        missed = der_result['missed detection'] / total
        false_alarm = der_result['false alarm'] / total
        der = confusion + missed + false_alarm
    else:
        confusion = missed = false_alarm = der = 0.0
    
    # Compute JER
    jer = jer_metric(reference, hypothesis)
    
    return DiarizationMetrics(
        der=der,
        jer=jer,
        confusion=confusion,
        missed=missed,
        false_alarm=false_alarm,
        total_speech=total
    )


def evaluate_single_file(
    reference_path: str,
    hypothesis_segments: List[RTTMSegment],
    file_id: str,
    model_name: str,
    processing_time: float,
    collar: float = 0.25
) -> BenchmarkResult:
    """
    Evaluate diarization for a single file.
    
    Args:
        reference_path: Path to reference RTTM file
        hypothesis_segments: Predicted segments
        file_id: File identifier
        model_name: Name of the model
        processing_time: Time taken to process
        collar: Forgiveness collar in seconds
        
    Returns:
        BenchmarkResult object
    """
    try:
        # Load reference
        ref_segments = parse_rttm_file(reference_path)
        reference = segments_to_annotation(ref_segments, uri=file_id)
        
        # Convert hypothesis
        hypothesis = segments_to_annotation(hypothesis_segments, uri=file_id)
        
        # Count speakers
        num_speakers_ref = len(set(seg.speaker for seg in ref_segments))
        num_speakers_hyp = len(set(seg.speaker for seg in hypothesis_segments)) if hypothesis_segments else 0
        
        # Compute metrics
        metrics = compute_metrics(reference, hypothesis, collar=collar)
        
        return BenchmarkResult(
            file_id=file_id,
            model_name=model_name,
            metrics=metrics,
            processing_time=processing_time,
            num_speakers_ref=num_speakers_ref,
            num_speakers_hyp=num_speakers_hyp
        )
        
    except Exception as e:
        return BenchmarkResult(
            file_id=file_id,
            model_name=model_name,
            metrics=None,
            processing_time=processing_time,
            num_speakers_ref=0,
            num_speakers_hyp=0,
            error=str(e)
        )


def aggregate_results(results: List[BenchmarkResult], model_name: str) -> ModelBenchmarkSummary:
    """
    Aggregate results for a model across all files.
    
    Args:
        results: List of BenchmarkResult objects
        model_name: Name of the model
        
    Returns:
        ModelBenchmarkSummary object
    """
    successful = [r for r in results if r.metrics is not None]
    failed = [r for r in results if r.metrics is None]
    
    if successful:
        avg_der = np.mean([r.metrics.der for r in successful])
        avg_jer = np.mean([r.metrics.jer for r in successful])
        avg_confusion = np.mean([r.metrics.confusion for r in successful])
        avg_missed = np.mean([r.metrics.missed for r in successful])
        avg_false_alarm = np.mean([r.metrics.false_alarm for r in successful])
    else:
        avg_der = avg_jer = avg_confusion = avg_missed = avg_false_alarm = float('nan')
    
    total_time = sum(r.processing_time for r in results)
    
    return ModelBenchmarkSummary(
        model_name=model_name,
        num_files=len(results),
        num_successful=len(successful),
        num_failed=len(failed),
        avg_der=avg_der,
        avg_jer=avg_jer,
        avg_confusion=avg_confusion,
        avg_missed=avg_missed,
        avg_false_alarm=avg_false_alarm,
        total_processing_time=total_time,
        per_file_results=results
    )

