#!/usr/bin/env python3
"""
Speaker Diarization Benchmark Script

Benchmarks multiple speaker diarization models against a dataset
and reports comprehensive metrics.

Usage:
    python benchmark.py --dataset ./benchmark_subset --output results.json
    python benchmark.py --dataset ./benchmark_subset --models pyannote-3.1 assemblyai
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich import box

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from utils import (
    get_audio_label_pairs,
    load_dataset_info,
    get_cache_path,
    load_cached_result,
    save_cached_result,
    parse_rttm_file
)
from evaluation import (
    evaluate_single_file,
    aggregate_results,
    ModelBenchmarkSummary
)
from models import PyAnnoteModel, SpeechBrainModel, AssemblyAIModel
from models.base import DiarizationModel

console = Console()


# Available models registry
AVAILABLE_MODELS = {
    'pyannote-3.1': lambda: PyAnnoteModel(version='pyannote-3.1'),
    'pyannote-3.0': lambda: PyAnnoteModel(version='pyannote-3.0'),
    'speechbrain': lambda: SpeechBrainModel(),
    'assemblyai': lambda: AssemblyAIModel(),
}


def get_models(model_names: Optional[List[str]] = None) -> List[DiarizationModel]:
    """
    Get list of model instances to benchmark.
    
    Args:
        model_names: List of model names to use, or None for all
        
    Returns:
        List of DiarizationModel instances
    """
    if model_names is None:
        model_names = list(AVAILABLE_MODELS.keys())
    
    models = []
    for name in model_names:
        if name not in AVAILABLE_MODELS:
            console.print(f"[yellow]Warning: Unknown model '{name}', skipping[/yellow]")
            continue
        models.append(AVAILABLE_MODELS[name]())
    
    return models


def run_benchmark(
    dataset_path: str,
    models: List[DiarizationModel],
    collar: float = 0.25,
    use_cache: bool = True,
    cache_dir: str = '.benchmark_cache'
) -> Dict[str, ModelBenchmarkSummary]:
    """
    Run benchmark on all models.
    
    Args:
        dataset_path: Path to dataset directory
        models: List of models to benchmark
        collar: Forgiveness collar for DER computation
        use_cache: Whether to use cached results
        cache_dir: Directory for caching results
        
    Returns:
        Dictionary mapping model names to their benchmark summaries
    """
    # Get audio-label pairs
    pairs = get_audio_label_pairs(dataset_path)
    
    if not pairs:
        console.print("[red]Error: No audio files found in dataset[/red]")
        return {}
    
    console.print(f"\n[bold]Dataset:[/bold] {dataset_path}")
    console.print(f"[bold]Files:[/bold] {len(pairs)}")
    console.print(f"[bold]Models:[/bold] {', '.join(m.name for m in models)}")
    console.print(f"[bold]Collar:[/bold] {collar}s")
    console.print()
    
    results = {}
    
    for model in models:
        console.print(Panel(f"[bold blue]Benchmarking: {model.name}[/bold blue]", expand=False))
        
        file_results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"Processing files...", total=len(pairs))
            
            for pair in pairs:
                file_id = pair['file_id']
                audio_path = pair['audio_path']
                label_path = pair['label_path']
                
                progress.update(task, description=f"Processing {file_id}...")
                
                # Check cache
                cache_path = get_cache_path(cache_dir, model.name, file_id)
                cached_segments = None
                
                if use_cache:
                    cached_segments = load_cached_result(cache_path)
                
                if cached_segments is not None:
                    # Use cached result
                    hyp_segments = cached_segments
                    processing_time = 0.0  # Cached, no processing time
                else:
                    # Run diarization
                    try:
                        result = model.diarize(audio_path, file_id)
                        hyp_segments = result.segments
                        processing_time = result.processing_time
                        
                        if result.error:
                            console.print(f"  [yellow]Warning: {file_id}: {result.error}[/yellow]")
                        elif use_cache and hyp_segments:
                            # Cache successful result
                            save_cached_result(cache_path, hyp_segments)
                            
                    except Exception as e:
                        console.print(f"  [red]Error processing {file_id}: {e}[/red]")
                        hyp_segments = []
                        processing_time = 0.0
                
                # Evaluate
                eval_result = evaluate_single_file(
                    reference_path=label_path,
                    hypothesis_segments=hyp_segments,
                    file_id=file_id,
                    model_name=model.name,
                    processing_time=processing_time,
                    collar=collar
                )
                file_results.append(eval_result)
                
                progress.update(task, advance=1)
        
        # Aggregate results for this model
        summary = aggregate_results(file_results, model.name)
        results[model.name] = summary
        
        # Print summary for this model
        print_model_summary(summary)
        
        # Unload model to free memory
        model.unload()
    
    return results


def print_model_summary(summary: ModelBenchmarkSummary) -> None:
    """Print summary table for a single model."""
    table = Table(title=f"Results: {summary.model_name}", box=box.ROUNDED)
    
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Files Processed", str(summary.num_files))
    table.add_row("Successful", str(summary.num_successful))
    table.add_row("Failed", str(summary.num_failed))
    table.add_row("─" * 20, "─" * 15)
    table.add_row("Avg DER", f"{summary.avg_der:.2%}")
    table.add_row("Avg JER", f"{summary.avg_jer:.2%}")
    table.add_row("Avg Confusion", f"{summary.avg_confusion:.2%}")
    table.add_row("Avg Missed", f"{summary.avg_missed:.2%}")
    table.add_row("Avg False Alarm", f"{summary.avg_false_alarm:.2%}")
    table.add_row("─" * 20, "─" * 15)
    table.add_row("Total Time", f"{summary.total_processing_time:.1f}s")
    
    console.print(table)
    console.print()


def print_comparison_table(results: Dict[str, ModelBenchmarkSummary]) -> None:
    """Print comparison table across all models."""
    if not results:
        return
    
    table = Table(title="[bold]Model Comparison[/bold]", box=box.DOUBLE_EDGE)
    
    table.add_column("Model", style="bold cyan")
    table.add_column("DER ↓", justify="right")
    table.add_column("JER ↓", justify="right")
    table.add_column("Confusion", justify="right")
    table.add_column("Missed", justify="right")
    table.add_column("False Alarm", justify="right")
    table.add_column("Time (s)", justify="right")
    
    # Sort by DER (lower is better)
    sorted_results = sorted(results.values(), key=lambda x: x.avg_der if not np.isnan(x.avg_der) else float('inf'))
    
    for i, summary in enumerate(sorted_results):
        # Highlight best (first) row
        style = "bold green" if i == 0 else ""
        
        der_str = f"{summary.avg_der:.2%}" if not np.isnan(summary.avg_der) else "N/A"
        jer_str = f"{summary.avg_jer:.2%}" if not np.isnan(summary.avg_jer) else "N/A"
        conf_str = f"{summary.avg_confusion:.2%}" if not np.isnan(summary.avg_confusion) else "N/A"
        miss_str = f"{summary.avg_missed:.2%}" if not np.isnan(summary.avg_missed) else "N/A"
        fa_str = f"{summary.avg_false_alarm:.2%}" if not np.isnan(summary.avg_false_alarm) else "N/A"
        
        table.add_row(
            summary.model_name,
            der_str,
            jer_str,
            conf_str,
            miss_str,
            fa_str,
            f"{summary.total_processing_time:.1f}",
            style=style
        )
    
    console.print()
    console.print(table)
    console.print()
    
    # Print winner
    if sorted_results and not np.isnan(sorted_results[0].avg_der):
        winner = sorted_results[0].model_name
        console.print(f"[bold green]🏆 Best Model (lowest DER): {winner}[/bold green]")
        console.print()


def save_results(
    results: Dict[str, ModelBenchmarkSummary],
    output_path: str,
    dataset_path: str
) -> None:
    """Save results to JSON file."""
    # Load dataset info for metadata
    dataset_info = load_dataset_info(dataset_path)
    
    output = {
        'benchmark_info': {
            'dataset': str(Path(dataset_path).resolve()),
            'num_files': dataset_info.get('total_files', len(results)),
            'total_duration_sec': dataset_info.get('total_duration_sec', 0),
            'timestamp': datetime.now().isoformat(),
            'collar': 0.25
        },
        'results': {
            name: summary.to_dict() 
            for name, summary in results.items()
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    console.print(f"[bold]Results saved to:[/bold] {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark speaker diarization models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py --dataset ./benchmark_subset
  python benchmark.py --dataset ./benchmark_subset --models pyannote-3.1 assemblyai
  python benchmark.py --dataset ./benchmark_subset --output results.json --no-cache

Available models:
  - pyannote-3.1  : PyAnnote Speaker Diarization 3.1 (requires HF_TOKEN)
  - pyannote-3.0  : PyAnnote Speaker Diarization 3.0 (requires HF_TOKEN)
  - speechbrain   : SpeechBrain ECAPA-TDNN + Spectral Clustering
  - assemblyai    : AssemblyAI API (requires ASSEMBLYAI_API_KEY)

Environment variables:
  HF_TOKEN            : HuggingFace token for pyannote models
  ASSEMBLYAI_API_KEY  : API key for AssemblyAI
        """
    )
    
    parser.add_argument(
        '--dataset', '-d',
        required=True,
        help='Path to dataset directory (with audio/ and labels/ subdirs)'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='results.json',
        help='Output JSON file path (default: results.json)'
    )
    
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        choices=list(AVAILABLE_MODELS.keys()),
        help='Models to benchmark (default: all)'
    )
    
    parser.add_argument(
        '--collar',
        type=float,
        default=0.25,
        help='Collar for DER computation in seconds (default: 0.25)'
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable result caching'
    )
    
    parser.add_argument(
        '--cache-dir',
        default='.benchmark_cache',
        help='Cache directory (default: .benchmark_cache)'
    )
    
    args = parser.parse_args()
    
    # Validate dataset path
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        console.print(f"[red]Error: Dataset path does not exist: {dataset_path}[/red]")
        sys.exit(1)
    
    if not (dataset_path / 'audio').exists() or not (dataset_path / 'labels').exists():
        console.print("[red]Error: Dataset must have 'audio/' and 'labels/' subdirectories[/red]")
        sys.exit(1)
    
    # Print header
    console.print()
    console.print(Panel.fit(
        "[bold blue]Speaker Diarization Benchmark[/bold blue]",
        border_style="blue"
    ))
    
    # Get models
    models = get_models(args.models)
    
    if not models:
        console.print("[red]Error: No valid models selected[/red]")
        sys.exit(1)
    
    # Run benchmark
    try:
        results = run_benchmark(
            dataset_path=str(dataset_path),
            models=models,
            collar=args.collar,
            use_cache=not args.no_cache,
            cache_dir=args.cache_dir
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark interrupted by user[/yellow]")
        sys.exit(1)
    
    # Print comparison table
    print_comparison_table(results)
    
    # Save results
    if results:
        save_results(results, args.output, str(dataset_path))


if __name__ == '__main__':
    main()

