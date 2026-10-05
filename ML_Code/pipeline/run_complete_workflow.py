from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print('\n>', ' '.join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    p=argparse.ArgumentParser(description='Run the complete 5-classifier × 4-strategy MLV experiment.')
    p.add_argument('--probcli', required=True)
    p.add_argument('--baseline_export', required=True)
    p.add_argument('--baseline_eventb_name', default='Ventilator_Composed_mch.eventb')
    p.add_argument('--timeout_ms', type=int, default=8000)
    p.add_argument('--no_clpfd', action='store_true')
    p.add_argument('--execute_all_missing', action='store_true', help='Execute every candidate lacking traces instead of generating only the strategy-selected plan.')
    p.add_argument('--max_passes', type=int, default=5)
    args=p.parse_args()

    root=Path(__file__).resolve().parents[2]
    code=root/'code'/'full_experiment_runner'
    outputs=root/'run_outputs'/'complete_real_run'
    cache=outputs/'execution_cache'
    outputs.mkdir(parents=True,exist_ok=True); cache.mkdir(parents=True,exist_ok=True)

    if args.execute_all_missing:
        requests=root/'data'/'manifests'/'execution_requests_all_missing_traces.csv'
    else:
        plan=outputs/'strategy_execution_plan'
        run([
            sys.executable,str(code/'generate_strategy_execution_plan.py'),
            '--candidate_combinations',str(root/'data'/'manifests'/'candidate_combinations.csv'),
            '--initial_sets',str(root/'data'/'manifests'/'initial_sets.csv'),
            '--metadata',str(root/'data'/'base_features_with_aliases'/'trace_metadata.csv'),
            '--config',str(root/'config'/'study_config.json'),
            '--out_dir',str(plan),
        ])
        requests=plan/'execution_requests.csv'

    execute_command=[
        sys.executable,str(code/'execute_candidates.py'),
        '--legacy_pipeline',str(code/'legacy_pipeline.py'),
        '--candidate_combinations',str(root/'data'/'manifests'/'candidate_combinations.csv'),
        '--execution_requests',str(requests),
        '--tests',str(root/'inputs'/'tests.xlsx'),
        '--baseline_export',str(Path(args.baseline_export).resolve()),
        '--baseline_eventb_name',args.baseline_eventb_name,
        '--probcli',str(Path(args.probcli).resolve()),
        '--cache_dir',str(cache),
        '--timeout_ms',str(args.timeout_ms),
    ]
    if args.no_clpfd: execute_command.append('--no_clpfd')
    run(execute_command)

    for pass_number in range(1,args.max_passes+1):
        raw_features=outputs/'features_raw'
        features=outputs/'features_with_aliases'
        study=outputs/'all_classifier_strategy_results'
        for folder in [raw_features,features,study]:
            if folder.exists(): shutil.rmtree(folder)

        run([
            sys.executable,str(code/'build_trace_features.py'),
            '--base_states',str(root/'inputs'/'expanded_states.csv'),
            '--execution_cache',str(cache),
            '--legacy_pipeline',str(code/'legacy_pipeline.py'),
            '--out_dir',str(raw_features),
        ])
        run([
            sys.executable,str(code/'add_existing_candidate_aliases.py'),
            '--features',str(raw_features/'trace_features.npz'),
            '--metadata',str(raw_features/'trace_metadata.csv'),
            '--candidate_combinations',str(root/'data'/'manifests'/'candidate_combinations.csv'),
            '--out_dir',str(features),
        ])
        run([
            sys.executable,str(code/'run_study.py'),
            '--features',str(features/'trace_features.npz'),
            '--metadata',str(features/'trace_metadata.csv'),
            '--candidate_combinations',str(root/'data'/'manifests'/'candidate_combinations.csv'),
            '--initial_sets',str(root/'data'/'manifests'/'initial_sets.csv'),
            '--config',str(root/'config'/'study_config.json'),
            '--out_dir',str(study),
            '--all_classifiers',
        ])
        status=json.loads((study/'run_status.json').read_text(encoding='utf-8'))
        if status.get('status')=='COMPLETE':
            aggregate=outputs/'complete_results_by_checkpoint.csv'
            run([sys.executable,str(code/'aggregate_complete_results.py'),'--raw_results',str(study/'internal_comparison.csv'),'--out',str(aggregate)])
            validation=outputs/'coverage_validation'
            run([sys.executable,str(code/'validate_completion.py'),'--results',str(aggregate),'--expected_grid',str(root/'data'/'manifests'/'full_experiment_grid_900.csv'),'--out_dir',str(validation)])
            run([sys.executable,str(code/'create_complete_internal_figures.py'),'--results',str(aggregate),'--out_dir',str(outputs/'figures')])
            print('\nCOMPLETE. Final outputs:',outputs)
            return

        extra=study/'execution_requests.csv'
        print(f"\nPass {pass_number}: additional trace requests were found. Executing them now.")
        execute_command[execute_command.index('--execution_requests')+1]=str(extra)
        run(execute_command)

    raise RuntimeError('The workflow still has missing traces after the allowed passes. Check execution logs and failed ProB runs.')

if __name__=='__main__': main()
