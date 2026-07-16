# Cursor Tutorial — Data Profiler Demo

Hands-on companion project for the "Cursor for Engineers" deck. Everyone
opens this same starting point in Cursor and works the exercises live,
using the exact prompts shown in the slides.

## What this is

A trimmed, faithful adaptation of `afd_profile.py` from
[`aws-samples/aws-fraud-detector-samples`](https://github.com/aws-samples/aws-fraud-detector-samples/tree/master/profiler/ManualNotebookSolution)
— a data profiler that infers column types (numeric, category, datetime,
email, IP, free text) and generates an HTML data-quality report for a
fraud-detection-style CSV. Upstream, it only runs from a Jupyter notebook.
This repo is the "before" state for turning it into a real CLI tool.

Two real bugs from the original code are intentionally left unfixed (see
`AGENTS.md` → Known issues), plus the missing CLI entrypoint and missing
tests. Nothing here is fabricated — the type-inference heuristics, the
10,000-row minimum check, and the `DataFrame.append()` calls are the actual
upstream logic.

## Layout

| Path | Purpose |
|---|---|
| `profiler/afd_profile.py` | The profiler logic (type inference, stats, correlation, report rendering) |
| `templates/profile.html` | Jinja2 report template (simplified from upstream's chart-heavy version) |
| `sample_data/transactions.csv` | 480 synthetic rows — under the 10K minimum on purpose, includes an email column and 3-word category labels |
| `config.sample.json` | Legacy config format `afd_profile.py` expects |
| `tests/test_profiler.py` | Starter pytest suite — two tests fail today, on purpose |
| `specs/json_output_mode.md` | Spec for the next feature after the CLI exists |
| `.cursor/rules/profiler.mdc` | Team conventions for this repo (scoped via `globs`) |
| `AGENTS.md` | Onboarding doc — setup/build/test/run commands, structure, known issues |

## Quick start

```bash
pip install -r requirements.txt
pytest -v          # 2 of 4 tests currently fail — expected
```

## How this maps to the deck

Work through these roughly in deck order — each one is a real, reproducible
task against this exact codebase:

1. **Ask mode** — `@afd_profile.py how does check_if_datetime_as_object_feature decide if a column is a date stored as text?`
2. **Plan mode** — "Convert afd_profile.py to a CLI" (clarifying questions, step checklist, then hand off to Agent)
3. **Agent mode** — have it add `cli.py` with argparse (`--input`, `--label`, `--timestamp`, `--output`) and fix the CLI-blocking issues it hits along the way
4. **TDD** — `pytest -k get_overview` already passes; use it as the pattern for writing the next test before the next feature
5. **Multi-file refactor** — "replace every `DataFrame.append()` call with `pd.concat()`, pin `pandas>=2.0` in requirements.txt, get `test_datetime_as_object_detection` and `test_get_stats_handles_email_column` passing"
6. **Debugging** — `@afd_profile.py why is merchant_category classified as TEXT instead of CATEGORY?` (real bug, see `test_merchant_category_misclassified_as_text`)
7. **Rules / AGENTS.md** — already in place; extend `.cursor/rules/profiler.mdc` as new conventions come up
8. **Writing a spec** — `specs/json_output_mode.md` is ready to implement once the CLI exists
9. **Git workflow** — stage the `pandas.append()` fix, generate the commit message, practice a conflict if two people touch `afd_profile.py` at once

## Known issues

See `AGENTS.md` → Known issues for the full list (two `DataFrame.append()`
bugs, no CLI entrypoint, the NLP-misclassification heuristic).
