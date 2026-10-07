# Public Release Scope

## Purpose

This document defines the disclosure boundary for the pre-publication release.
The repository is intended to contain experiment scripts and minimal public
documentation only.

## Allowed files

- Python source files in the three stage directories;
- `README.md`, `RELEASE_SCOPE.md`, `SECURITY.md`, and `CHANGELOG.md`;
- `.gitignore`, `requirements.txt`, and the Apache-2.0 `LICENSE`;
- Git metadata created for this standalone repository.

## Prohibited content

- Manuscripts, abstracts, unpublished narrative claims, submission files, and
  tracked-change documents;
- raw or processed research data, benchmark items, ground truth, annotations,
  prompts exported from completed runs, model outputs, and derived results;
- figures, tables, spreadsheets, PDFs, slide decks, archives, logs, caches, and
  backups;
- credentials, tokens, cookies, private URLs, personal email addresses, machine
  names, and local absolute paths;
- files copied from the parent research repository unless they appear in the
  explicit allowlist above.

## Review gates

Before every public push:

1. Enumerate the exact tracked file list.
2. Compile every Python file.
3. Scan tracked content and Git history for credentials and personal paths.
4. Reject prohibited extensions and generated artifacts.
5. Confirm that the commit history begins with the clean public-release commit
   and has no ancestry from the private research repository.
6. Run `git diff --cached` or its equivalent and manually inspect the complete
   disclosure set.
7. Push only the standalone repository; never push the parent research branch.

## Current boundary

This snapshot does not disclose data or results and therefore cannot, by
itself, reproduce any scientific claim. That limitation is intentional and
must remain visible until a separately reviewed reproduction package is ready.
