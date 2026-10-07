# Security Policy

## Credentials

Use environment variables or a local secret manager. Never store credentials
in source code, configuration committed to Git, example commands, logs, or test
fixtures. `.env` files are ignored and are not part of this release.

If a secret is committed at any point:

1. Revoke or rotate it at the provider immediately.
2. Remove it from the complete Git history, not only the current file tree.
3. Re-run a history-aware secret scan before publishing again.

## Sensitive research material

Do not attach the unpublished manuscript, datasets, model outputs, annotations,
results, figures, or local logs to a public issue or pull request.

## Reporting

Use GitHub private vulnerability reporting when available. Do not include live
credentials or private research records in a report.
