# Feature spec: `--format json` output mode

## Problem

`profile_report()` only renders an HTML report. Teams that want to gate CI
on data-quality thresholds (e.g. "fail the build if more than 20% of a
column is missing") have to scrape numbers out of the rendered HTML, which
is brittle and not meant to be machine-read.

## Goals

- Add a `--format json` flag to the CLI (once `cli.py` exists — see
  `AGENTS.md`), alongside the default `--format html`.
- JSON output should carry the same underlying stats as the HTML report:
  overview, per-column type/quality stats, label distribution, and
  feature-label correlation.
- Output goes to stdout by default, or to the path given by `--output`.

## Non-goals

- Not deprecating or changing the HTML report.
- Not adding a `--format csv` or other formats in this pass.
- Not changing the underlying stats functions (`get_overview`,
  `get_stats`, etc.) — this is a serialization change only.

## Design sketch

Add a `profile_stats(config) -> dict` function that runs the same pipeline
as `profile_report()` but returns the raw dict instead of rendering it
through Jinja2. `profile_report()` becomes a thin wrapper: call
`profile_stats()`, then render HTML from the dict. The CLI's `--format`
flag picks `json.dumps(profile_stats(config))` or the HTML render.

## Acceptance criteria

- `pytest -k json_output` passes.
- `python cli.py --input sample_data/transactions.csv --label label
  --timestamp event_timestamp --format json` prints valid JSON containing
  at least `overview`, `column_stats`, `label_distribution`, and
  `correlation` keys.
- Existing HTML output is byte-for-byte unchanged for the same input.
