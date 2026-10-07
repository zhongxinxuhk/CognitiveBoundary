# Cognitive Boundary Experiment Scripts

## Release status

| Item | Status |
| --- | --- |
| Work stage | **Pre-publication experimental code release** |
| Snapshot date | 2026-10-07 |
| Manuscript | **Unpublished and not included** |
| Research data | **Not included** |
| Model outputs and annotations | **Not included** |
| Numerical results and figures | **Not included** |
| Reproduction status | Partial: scripts are available, but private/local inputs are required |

This repository is a deliberately narrow, scripts-only release. It exposes the
experiment execution, task-specific scoring, and statistical-analysis code
without disclosing the unpublished manuscript or any research records.

No scientific conclusion or performance claim should be inferred from the
presence of these scripts. The repository is not yet a complete reproduction
package.

## Work stages and contents

| Stage | Directory | Public content | Current status |
| --- | --- | --- | --- |
| 1. Experiment execution | `stage_1_experiment_execution/` | Four-arm runner with plan-only default, preflight checks, prompt hashing, metadata capture, and resume support | Pre-publication; requires user-supplied tasks and API access |
| 2. Scoring and review | `stage_2_scoring/` | Task-specific scoring plus bilingual TruthfulQA and trip-planning review helpers | Provisional; heuristics require independent validation and human adjudication |
| 3. Statistical analysis | `stage_3_statistical_analysis/` | Paired tests, confidence intervals, bootstrap analysis, effect sizes, and confidence-parsing sensitivity analysis | Pre-publication; requires user-supplied scored records |

## Explicitly excluded

The public history must not contain any of the following:

- manuscript files, manuscript excerpts, tracked revisions, reviewer material,
  or submission metadata;
- benchmark rows, ground-truth records, source-dataset copies, manual labels,
  or any other research data;
- prompts or model outputs exported from completed runs, apart from prompt
  templates that are part of the execution source code;
- aggregate results, statistical result files, tables, figures, dashboards,
  spreadsheets, logs, caches, or backups;
- API keys, access tokens, cookies, private endpoints, `.env` files, personal
  email addresses, hostnames, or local absolute paths.

See [`RELEASE_SCOPE.md`](RELEASE_SCOPE.md) for the full allowlist and review
boundary.

## Local input layout

The scripts expect the user to provide inputs locally. These locations are
ignored by Git and must remain untracked:

```text
data/
├── raw_outputs.jsonl
├── source_datasets/
│   ├── TruthfulQA.csv
│   ├── natural_plan_calendar_scheduling.json
│   ├── natural_plan_meeting_planning.json
│   └── natural_plan_trip_planning.json
└── scoring_records/
    └── scored_outputs.jsonl
```

Obtain datasets independently from their official repositories and comply with
their current licenses and terms. Dataset copies are intentionally not vendored
here.

## Environment

- Python 3.10 or later is recommended.
- Most scripts use only the Python standard library.
- `scipy` is used when available for statistical distribution functions.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Safe execution

The Stage 1 runner defaults to planning and does not send model requests unless
`--execute` is supplied. Provide the task file explicitly:

```bash
python stage_1_experiment_execution/exp_four_arm_runner.py \
  --task-file /private/path/scored_outputs.jsonl
```

For a real run, provide the credential only through the process environment.
Never place a credential in a source file, command history, issue, or commit:

```bash
P2_API_KEY='REDACTED' \
python stage_1_experiment_execution/exp_four_arm_runner.py \
  --execute \
  --task-file /private/path/scored_outputs.jsonl \
  --model YOUR_MODEL
```

`REDACTED` above is a literal documentation placeholder, not a usable key.

For the Stage 2 scorer, point `CB_RAW_FILE` to the private/local model-output
file:

```bash
CB_RAW_FILE=/private/path/raw_outputs.jsonl \
python stage_2_scoring/score_experiment.py
```

## Scientific-use boundary

- The bilingual scoring rules are provisional heuristics. Validate them against
  a blinded, independently adjudicated sample before using them for a claim.
- Do not interpret a mean difference as superiority without the paired test,
  confidence interval, effect size, and multiplicity policy defined in the
  analysis protocol.
- Preserve random seeds, model identifiers, sampling settings, dataset versions,
  and immutable input hashes outside this public repository.
- Negative results, missing records, parsing failures, and manual-review cases
  must remain visible in any later complete reproduction package.

## Security

Do not commit `.env` files or research inputs. If a secret is ever committed,
removing it from the latest revision is not sufficient: revoke or rotate it and
clean the complete Git history before publication. See [`SECURITY.md`](SECURITY.md).

## License

The scripts are released under the Apache License 2.0. See [`LICENSE`](LICENSE).
