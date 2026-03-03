"""
Batch Runner for Open-XDDx Benchmarking on Colab (vLLM + A100).

Loads Open-XDDx.xlsx, runs each case through v10's DDxSystem with
vLLM backend, saves per-case JSON with checkpoint/resume support.

Usage:
    python batch_runner.py --config benchmark_config.yaml
    python batch_runner.py --config benchmark_config.yaml --cases 0-9
    python batch_runner.py --config benchmark_config.yaml --resume
"""

import os
import sys
import ast
import json
import time
import yaml
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# Add v10 and v9 modules to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
V10_MODULES = os.path.join(PROJECT_ROOT, 'v10_full_pipeline', 'Modules')
V9_MODULES = os.path.join(PROJECT_ROOT, 'v9_ollama_ui', 'Modules')

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, V10_MODULES)
sys.path.insert(0, V9_MODULES)


def load_benchmark_config(config_path: str) -> Dict[str, Any]:
    """Load benchmark configuration YAML"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """
    Load Open-XDDx dataset and parse ground truth.

    Returns list of dicts with keys: index, patient_info, ground_truth, specialty
    """
    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas required. pip install pandas openpyxl")
        sys.exit(1)

    print(f"Loading dataset: {dataset_path}")
    df = pd.read_excel(dataset_path)
    print(f"  {len(df)} cases, columns: {list(df.columns)}")

    cases = []
    parse_errors = 0

    for _, row in df.iterrows():
        idx = int(row['Index'])
        patient_info = str(row['patient_info'])

        # Parse interpretation column -> {diagnosis: [evidence]}
        interpretation = row['interpretation']
        gt = None
        try:
            gt = ast.literal_eval(interpretation) if isinstance(interpretation, str) else interpretation
        except (ValueError, SyntaxError):
            try:
                gt = json.loads(interpretation) if isinstance(interpretation, str) else None
            except (json.JSONDecodeError, TypeError):
                pass

        if gt is None or not isinstance(gt, dict):
            parse_errors += 1
            continue

        cases.append({
            'index': idx,
            'patient_info': patient_info,
            'ground_truth': gt,
            'specialty': str(row.get('specialty', 'Unknown')),
            'disease_num': int(row.get('disease_num', len(gt))),
        })

    print(f"  Parsed {len(cases)} cases ({parse_errors} parse errors)")

    # Validate first 3 cases
    for c in cases[:3]:
        gt_diags = list(c['ground_truth'].keys())
        print(f"    Case {c['index']}: {len(gt_diags)} GT diagnoses, "
              f"specialty={c['specialty']}")
        print(f"      GT: {gt_diags[:3]}{'...' if len(gt_diags) > 3 else ''}")

    return cases


def load_checkpoint(results_dir: str) -> set:
    """Load checkpoint to find completed case indices"""
    checkpoint_path = os.path.join(results_dir, 'checkpoint.json')
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            data = json.load(f)
        completed = set(data.get('completed_indices', []))
        print(f"  Checkpoint: {len(completed)} cases already completed")
        return completed
    return set()


def save_checkpoint(results_dir: str, completed_indices: set,
                    config: Dict[str, Any]):
    """Save checkpoint with completed case indices"""
    checkpoint_path = os.path.join(results_dir, 'checkpoint.json')
    data = {
        'completed_indices': sorted(completed_indices),
        'total_completed': len(completed_indices),
        'last_updated': datetime.now().isoformat(),
        'config': {
            'model': config.get('vllm', {}).get('model_name', 'unknown'),
            'pipeline_mode': config.get('benchmark', {}).get('pipeline_mode', 'full'),
        }
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(data, f, indent=2)


def parse_case_range(cases_arg: str, total_cases: int) -> List[int]:
    """Parse case range argument (e.g., '0-9', '5', '10-19')"""
    if '-' in cases_arg:
        start, end = cases_arg.split('-', 1)
        return list(range(int(start), min(int(end) + 1, total_cases)))
    else:
        idx = int(cases_arg)
        return [idx] if idx < total_cases else []


def initialize_system(config: Dict[str, Any]) -> Any:
    """
    Initialize DDxSystem with vLLM backend.

    Manually configures model manager to use vLLM instead of Ollama.
    """
    from ddx_core import OllamaModelManager, ModelConfig, load_system_config
    from ddx_sliding_context import TranscriptManager
    from ddx_core import DynamicAgentGenerator

    # Import DDxSystem
    from ddx_runner import DDxSystem

    system = DDxSystem()

    # Build model configs from benchmark config
    model_configs = {}
    for key in ['conservative_model', 'innovative_model']:
        mc = config[key]
        model_configs[key] = ModelConfig(
            name=mc['name'],
            model_name=mc['model_name'],
            temperature=mc.get('temperature', 0.7),
            top_p=mc.get('top_p', 0.9),
            max_tokens=mc.get('max_tokens', 1024),
            role=mc.get('role', 'balanced'),
        )

    # Create model manager with vLLM backend
    vllm_kwargs = {
        'model_name': config['vllm']['model_name'],
        'gpu_memory_utilization': config['vllm'].get('gpu_memory_utilization', 0.90),
        'max_model_len': config['vllm'].get('max_model_len', 4096),
        'quantization': config['vllm'].get('quantization', 'gptq'),
        'dtype': config['vllm'].get('dtype', 'float16'),
    }

    print("\n" + "=" * 60)
    print("Initializing DDxSystem with vLLM backend")
    print("=" * 60)
    print(f"  Model: {vllm_kwargs['model_name']}")
    print(f"  GPU utilization: {vllm_kwargs['gpu_memory_utilization']}")
    print(f"  Max model len: {vllm_kwargs['max_model_len']}")

    system.model_manager = OllamaModelManager(
        model_configs, backend_type="vllm", **vllm_kwargs
    )

    if not system.model_manager.initialize():
        print("ERROR: Failed to initialize vLLM backend")
        return None

    # Load model (single load — both configs use same model)
    print("\nLoading model...")
    for model_id in system.model_manager.get_available_models():
        print(f"  Loading {model_id}...")
        system.model_manager.load_model(model_id)

    # Initialize transcript + agent generator
    system.transcript = TranscriptManager()
    system.agent_generator = DynamicAgentGenerator(
        system.model_manager, system.transcript
    )

    print("\nSystem ready!")
    return system


def run_single_case(system: Any, case: Dict[str, Any],
                    config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Run a single case through the pipeline"""
    idx = case['index']
    patient_info = case['patient_info']
    max_specialists = config.get('benchmark', {}).get('max_specialists', 5)
    pipeline_mode = config.get('benchmark', {}).get('pipeline_mode', 'full')

    print(f"\n{'='*60}")
    print(f"CASE {idx} ({case['specialty']})")
    print(f"{'='*60}")
    print(f"  GT diagnoses: {list(case['ground_truth'].keys())}")

    start_time = time.time()

    try:
        # Analyze case (generate specialist team)
        analysis = system.analyze_case(
            patient_info, case_name=f"case_{idx:03d}",
            max_specialists=max_specialists
        )

        if not analysis.get('success'):
            print(f"  ERROR: Case analysis failed: {analysis.get('error')}")
            return None

        print(f"  Team: {len(analysis['specialists'])} specialists")

        # Run diagnosis pipeline
        if pipeline_mode == 'full':
            result = system.run_full_diagnosis()
        else:
            result = system.run_quick_diagnosis()

        elapsed = time.time() - start_time

        # Build output
        output = {
            'case_index': idx,
            'case': {
                'name': f"case_{idx:03d}",
                'description': patient_info[:500],
                'specialty': case['specialty'],
            },
            'ground_truth': case['ground_truth'],
            'specialists': analysis['specialists'],
            'results': result,
            'duration_seconds': elapsed,
            'config': {
                'model': config['vllm']['model_name'],
                'pipeline_mode': pipeline_mode,
                'max_specialists': max_specialists,
                'conservative_temp': config['conservative_model']['temperature'],
                'innovative_temp': config['innovative_model']['temperature'],
            },
            'timestamp': datetime.now().isoformat(),
        }

        # Print summary
        final_diags = result.get('final_diagnoses', [])
        if final_diags:
            print(f"  Final diagnoses ({len(final_diags)}):")
            for d in final_diags[:5]:
                diag_name = d[0] if isinstance(d, (list, tuple)) else d
                print(f"    - {diag_name}")

        print(f"  Duration: {elapsed:.1f}s")
        return output

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  ERROR after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Run v10 pipeline benchmark on Open-XDDx via vLLM"
    )
    parser.add_argument('--config', default='benchmark/benchmark_config.yaml',
                        help='Path to benchmark config YAML')
    parser.add_argument('--cases', default=None,
                        help='Case range to run (e.g., "0-9", "5", "100-199")')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint, skip completed cases')
    parser.add_argument('--dry-run', action='store_true',
                        help='Load data and config but do not run pipeline')
    args = parser.parse_args()

    # Load config
    config = load_benchmark_config(args.config)
    benchmark_cfg = config.get('benchmark', {})

    # Resolve dataset path relative to project root
    dataset_path = os.path.join(PROJECT_ROOT, benchmark_cfg['dataset'])
    results_dir = os.path.join(PROJECT_ROOT, benchmark_cfg.get('results_dir', 'benchmark/results'))

    os.makedirs(results_dir, exist_ok=True)

    # Load dataset
    cases = load_dataset(dataset_path)
    if not cases:
        print("ERROR: No valid cases loaded")
        return

    # Determine which cases to run
    if args.cases:
        indices = parse_case_range(args.cases, len(cases))
        cases = [c for c in cases if c['index'] in indices]
        print(f"\nRunning {len(cases)} cases (range: {args.cases})")
    else:
        pilot_count = benchmark_cfg.get('pilot_count', 0)
        if pilot_count > 0:
            cases = cases[:pilot_count]
            print(f"\nPilot run: {len(cases)} cases")

    # Check for resume
    completed_indices = set()
    if args.resume:
        completed_indices = load_checkpoint(results_dir)
        remaining = [c for c in cases if c['index'] not in completed_indices]
        print(f"  Skipping {len(cases) - len(remaining)} already completed")
        cases = remaining

    if not cases:
        print("All cases already completed!")
        return

    if args.dry_run:
        print(f"\n[DRY RUN] Would run {len(cases)} cases")
        for c in cases[:5]:
            gt_keys = list(c['ground_truth'].keys())
            print(f"  Case {c['index']}: {len(gt_keys)} GT, {c['specialty']}")
        return

    # Initialize system
    system = initialize_system(config)
    if system is None:
        return

    # Save run config
    run_config_path = os.path.join(results_dir, 'run_config.json')
    with open(run_config_path, 'w') as f:
        json.dump({
            'config': config,
            'cases_to_run': [c['index'] for c in cases],
            'started_at': datetime.now().isoformat(),
        }, f, indent=2)

    # Run cases
    checkpoint_interval = benchmark_cfg.get('checkpoint_interval', 1)
    success_count = 0
    error_count = 0

    print(f"\n{'='*60}")
    print(f"STARTING BENCHMARK: {len(cases)} cases")
    print(f"{'='*60}")
    batch_start = time.time()

    for i, case in enumerate(cases):
        print(f"\n[{i+1}/{len(cases)}] ", end="")

        result = run_single_case(system, case, config)

        if result is not None:
            # Save per-case JSON
            case_file = os.path.join(results_dir, f"case_{case['index']:03d}.json")
            with open(case_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)

            completed_indices.add(case['index'])
            success_count += 1
        else:
            error_count += 1

        # Checkpoint
        if (i + 1) % checkpoint_interval == 0:
            save_checkpoint(results_dir, completed_indices, config)

        # Progress
        elapsed = time.time() - batch_start
        avg_time = elapsed / (i + 1)
        remaining = avg_time * (len(cases) - i - 1)
        print(f"  Progress: {i+1}/{len(cases)} "
              f"({success_count} ok, {error_count} err) "
              f"ETA: {remaining/60:.0f}min")

    # Final checkpoint
    save_checkpoint(results_dir, completed_indices, config)

    # Summary
    total_elapsed = time.time() - batch_start
    print(f"\n{'='*60}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"  Successful: {success_count}/{len(cases)}")
    print(f"  Errors:     {error_count}")
    print(f"  Total time: {total_elapsed/60:.1f} minutes")
    print(f"  Avg/case:   {total_elapsed/max(success_count,1):.1f}s")
    print(f"  Results in: {results_dir}")

    # Unload model
    if system.model_manager:
        system.model_manager.backend.unload()
        print("  GPU memory released")


if __name__ == '__main__':
    main()
