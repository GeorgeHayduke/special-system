# Data Profiler CLI

A command-line data profiler for fraud-detection-style tabular datasets,
adapted from `aws-samples/aws-fraud-detector-samples` (the notebook-only
`profiler/ManualNotebookSolution/afd_profile.py`). This repo is the starting
point for converting that notebook script into a proper CLI tool.

## Setup

```
pip install -r requirements.txt
```

## Test

```
pytest -v
```

Two tests fail on a fresh checkout — that's expected. See "Known issues"
below.

## Run

There is no CLI entrypoint yet. Today the only way to generate a report is
to import the module directly:

```python
from profiler.afd_profile import get_config, profile_report

config = get_config("config.sample.json")
html = profile_report(config)
open("report.html", "w").write(html)
```

Building `cli.py` (argparse: `--input`, `--label`, `--timestamp`,
`--output`) is the first exercise — see `specs/json_output_mode.md` for the
follow-up feature once the CLI exists.

## Structure

```
profiler/           → type inference, stats, correlation, report rendering
templates/           → Jinja2 report template (profile.html)
sample_data/         → synthetic transactions.csv for local testing
tests/                → pytest suite (two tests currently fail — see below)
specs/                → feature specs, written before implementation
config.sample.json    → legacy config format afd_profile.py expects
.cursor/rules/         → team conventions Cursor should follow in this repo
```

## Known issues

1. `check_if_datetime_as_object_feature()` and `get_stats()` both call
   `DataFrame.append()`, which was removed in pandas 2.0. Both raise
   `AttributeError` on any modern pandas install. Fix: replace with
   `pd.concat()`. Regression tests already exist in
   `tests/test_profiler.py` (currently failing) — get them green.
2. No CLI entrypoint. Everything is only reachable by importing the module.
3. `check_if_nlp_feature()` misclassifies three-or-more-word categorical
   labels as free text (see `merchant_category` in the sample data). Not
   yet fixed — flagged as a follow-up, not blocking the CLI conversion.

## Conventions

See `.cursor/rules/profiler.mdc` for the enforced conventions (type hints,
no `DataFrame.append()`, argparse over config dicts, tests required, no AWS
SDK imports in the local path).
