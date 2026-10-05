from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
out = root / 'rerun_output'
cmd = [
    sys.executable,
    str(Path(__file__).with_name('generate_study_analysis.py')),
    '--classifier-results', str(root / 'Datasets_Traces' / 'results' / 'experiment_results' / 'classifier_sensitivity.csv'),
    '--strategy-results', str(root / 'Datasets_Traces' / 'results' / 'experiment_results' / 'results_by_checkpoint.csv'),
    '--output-dir', str(out),
]
subprocess.run(cmd, check=True)
print(f'Regenerated analysis in: {out}')
