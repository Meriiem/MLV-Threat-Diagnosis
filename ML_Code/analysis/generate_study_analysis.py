from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INITIALS = ['5-5-5', '10-10-10', '30-30-30']
TASKS = ['Task A: Component', 'Task B: Threat Type', 'Task C: Joint']
MODELS = ['LightGBM', 'XGBoost', 'Gradient Boosting', 'Random Forest', 'Logistic Regression']
STRATEGIES = ['Random', 'Entropy', 'Margin', 'Jaccard Diversity']
XMAP = {'Initial set only': 0.0, '1%': 1.0, '2%': 2.0, '4%': 4.0, '8%': 8.0, '16%': 16.0}
NORMALIZE_INITIAL = {'5/5/2005': '5-5-5', '10/10/2010': '10-10-10', '30/30/2030': '30-30-30'}


def read_csv(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if 'initial_set' in row:
            row['initial_set'] = NORMALIZE_INITIAL.get(row['initial_set'], row['initial_set'])
    return rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def saturation(x, gain, rate, y0):
    x = np.asarray(x, dtype=float)
    return y0 + gain * (1.0 - np.exp(-rate * x))


def fit_curve(points):
    points = sorted(points)
    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    y0 = float(y[np.where(x == 0.0)[0][0]])

    def model(xx, gain, rate):
        return saturation(xx, gain, rate, y0)

    popt, pcov = curve_fit(
        model, x, y,
        p0=[max(0.05, float(y.max() - y0)), 0.5],
        bounds=([0.0, 1e-6], [100.0, 20.0]),
        maxfev=50000,
    )
    gain, rate = map(float, popt)
    fitted = model(x, gain, rate)
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    rmse = float(np.sqrt(np.mean((y - fitted) ** 2)))
    se = np.sqrt(np.diag(pcov)) if pcov.shape == (2, 2) else np.array([np.nan, np.nan])
    best = float(y.max())
    practical = next((px, py) for px, py in points if py >= best - 0.25)
    return {
        'initial_f1': y0,
        'gain': gain,
        'gain_se': float(se[0]),
        'rate_k': rate,
        'rate_k_se': float(se[1]),
        'plateau_f1': y0 + gain,
        'x95_percent': math.log(20.0) / rate,
        'r_squared': r2,
        'rmse': rmse,
        'best_observed_f1': best,
        'best_observed_percent': float(points[int(np.argmax(y))][0]),
        'practical_percent': float(practical[0]),
        'practical_f1': float(practical[1]),
        'equation': f'y = {y0:.2f} + {gain:.2f}(1 - exp(-{rate:.4f}x))',
    }


def safe_name(text):
    return text.replace(':', '').replace(' ', '_').replace('/', '-').replace('%', 'pct')


def setup_axis(ax, ylabel):
    ax.set_xlabel('Added candidate set (%)')
    ax.set_ylabel(ylabel)
    ax.set_xticks([0, 1, 2, 4, 8, 16])
    ax.set_xticklabels(['Init', '1%', '2%', '4%', '8%', '16%'])
    ax.set_xlim(-0.35, 16.4)
    ax.grid(True, alpha=0.25)


def save_figure(fig, out_root: Path, stem: str):
    for ext in ('png', 'pdf'):
        target = out_root / ext / f'{stem}.{ext}'
        target.parent.mkdir(parents=True, exist_ok=True)
        kwargs = {'dpi': 240} if ext == 'png' else {}
        fig.savefig(target, bbox_inches='tight', **kwargs)
    plt.close(fig)


def plot_fit(points, fit, title, ylabel, out_root, stem):
    points = sorted(points)
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    xx = np.linspace(0, 16, 250)
    yy = saturation(xx, fit['gain'], fit['rate_k'], fit['initial_f1'])
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.scatter(x, y, s=48, label='Observed values', zorder=3)
    ax.plot(xx, yy, linewidth=2.2, label='Saturating exponential fit')
    setup_axis(ax, ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, loc='upper left')
    note = (
        f"{fit['equation']}\n"
        f"Plateau = {fit['plateau_f1']:.2f}%   |   k = {fit['rate_k']:.4f}\n"
        f"95% saturation = {fit['x95_percent']:.2f}%   |   R² = {fit['r_squared']:.4f}"
    )
    ax.text(0.98, 0.06, note, transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            bbox={'boxstyle': 'round,pad=0.45', 'facecolor': 'white', 'alpha': 0.9, 'edgecolor': '0.75'})
    fig.tight_layout(); save_figure(fig, out_root, stem)


def plot_comparison(series, title, ylabel, out_root, stem):
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    for label, points in series.items():
        points = sorted(points)
        ax.plot([p[0] for p in points], [p[1] for p in points], marker='o', linewidth=2, label=label)
    setup_axis(ax, ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout(); save_figure(fig, out_root, stem)


def main():
    parser = argparse.ArgumentParser(description='Fit and plot the complete MLV active-learning learning-curve analysis.')
    parser.add_argument('--classifier-results', required=True, help='classifier_sensitivity.csv')
    parser.add_argument('--strategy-results', required=True, help='results_by_checkpoint.csv')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    figures = output / 'figures'
    tables = output / 'tables'
    classifier_rows = read_csv(Path(args.classifier_results))
    strategy_rows = read_csv(Path(args.strategy_results))

    classifier_groups = defaultdict(list)
    for row in classifier_rows:
        classifier_groups[(row['initial_set'], row['task'], row['diagnostic_classifier'])].append(
            (XMAP[row['checkpoint']], float(row['internal_validation_f1_mean']))
        )

    baselines = {}
    strategy_groups = defaultdict(list)
    for row in strategy_rows:
        key = (row['initial_set'], row['task'])
        if row['selection_strategy'] == 'Initial set only':
            baselines[key] = (0.0, float(row['hidden_test_f1_macro']))
        else:
            strategy_groups[(row['initial_set'], row['task'], row['selection_strategy'])].append(
                (XMAP[row['checkpoint']], float(row['hidden_test_f1_macro']))
            )

    classifier_fits = []
    for initial in INITIALS:
        for task in TASKS:
            for model in MODELS:
                points = sorted(classifier_groups[(initial, task, model)])
                fit = fit_curve(points)
                classifier_fits.append({'initial_set': initial, 'task': task, 'diagnostic_classifier': model, **fit})
                plot_fit(points, fit, f'{model} — {task} — Initial set {initial}',
                         'Internal validation F1-macro (%)', figures,
                         f'classifier_fits/{safe_name(model)}__{safe_name(task)}__{initial}')

    strategy_fits = []
    for initial in INITIALS:
        for task in TASKS:
            for strategy in STRATEGIES:
                points = sorted([baselines[(initial, task)], *strategy_groups[(initial, task, strategy)]])
                strategy_groups[(initial, task, strategy)] = points
                fit = fit_curve(points)
                strategy_fits.append({'initial_set': initial, 'task': task, 'diagnostic_classifier': 'LightGBM',
                                      'selection_strategy': strategy, **fit})
                plot_fit(points, fit, f'{strategy} selection — {task} — Initial set {initial}',
                         'Hidden-test F1-macro (%)', figures,
                         f'strategy_fits/{safe_name(strategy)}__{safe_name(task)}__{initial}')

    for task in TASKS:
        for initial in INITIALS:
            plot_comparison({m: classifier_groups[(initial, task, m)] for m in MODELS},
                            f'Classifier comparison — {task} — Initial set {initial}',
                            'Internal validation F1-macro (%)', figures,
                            f'comparisons/Classifiers__{safe_name(task)}__{initial}')
            plot_comparison({s: strategy_groups[(initial, task, s)] for s in STRATEGIES},
                            f'Selection-strategy comparison — {task} — Initial set {initial}',
                            'Hidden-test F1-macro (%)', figures,
                            f'comparisons/Strategies__{safe_name(task)}__{initial}')

    write_csv(tables / 'classifier_curve_fit_parameters.csv', classifier_fits)
    write_csv(tables / 'strategy_curve_fit_parameters.csv', strategy_fits)
    print(f'Wrote {len(classifier_fits)} classifier fits and {len(strategy_fits)} strategy fits to {output}')


if __name__ == '__main__':
    main()
