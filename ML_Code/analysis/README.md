# Reusable analysis code

Run the complete analysis from the package root:

```bash
python ML_Code/analysis/run_all.py
```

Or use your own CSV files:

```bash
python ML_Code/analysis/generate_study_analysis.py \
  --classifier-results path/to/classifier_sensitivity.csv \
  --strategy-results path/to/results_by_checkpoint.csv \
  --output-dir path/to/output
```

The fitted model is:

```text
y(x) = y0 + G(1 - exp(-k x))
```

`fit_curve_table.py` is a smaller generic utility for fitting any grouped checkpoint table. `plot_one_curve.py` plots a single curve from `x,y` points and one fitted-parameter row.
