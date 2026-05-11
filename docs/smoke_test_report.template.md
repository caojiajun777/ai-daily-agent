# Smoke Test Report — report-agent → juya-ai-daily

> Copy this template to `docs/smoke_tests/<test_date>.md` and fill it in
> *during* the smoke test. Reports are append-only history; do not edit a
> filed report afterward without a clear `**Edit (date):**` note.

---

## Run metadata

| Field             | Value |
| ----------------- | ----- |
| `test_date`       | <!-- e.g. 2026-05-09 (the date of the test, not the report's logical date) --> |
| `target_repo`     | <!-- e.g. imjuya/juya-ai-daily-fork --> |
| `run_id`          | <!-- e.g. 2026-05-09-49af7eff (from artifacts/reports/<date>.json -> run_id) --> |
| `provider`        | <!-- mock | deepseek --> |
| `model`           | <!-- e.g. deepseek-v4-pro --> |
| `report_date`     | <!-- the YYYY-MM-DD passed as --date / inputs.date --> |
| `tester`          | <!-- GitHub handle of the human running the test --> |
| `pat_id`          | <!-- last 4 chars of the fine-grained PAT, NEVER the full token --> |
| `pat_expires_at`  | <!-- ISO date; remember to rotate --> |

---

## Step 6 — `dry_run = true`

| Field              | Value |
| ------------------ | ----- |
| `dry_run_result`   | <!-- "PASS" if gate_ok=true and would_publish=true, else "BLOCKED" --> |
| `gate_ok`          | <!-- true | false (from publish_preview_<date>.json -> gate_result.ok) --> |
| `blocked_reasons`  | <!-- copy gate_result.blocked_reasons[] verbatim, even if empty --> |
| `body_length`      | <!-- bytes (sanity check the draft isn't empty) --> |
| `duplicates_seen`  | <!-- number from publish_preview_<date>.json -> duplicates --> |
| `preview_artifact` | <!-- URL of the workflow artifact zip --> |

Notes:

```text
<!-- anything noteworthy from the dry-run preview -->
```

---

## Step 7 — `dry_run = false` (confirm)

| Field                   | Value |
| ----------------------- | ----- |
| `confirm_result`        | <!-- "PUBLISHED" | "BLOCKED_BY_GATE" | "BLOCKED_BY_DUPLICATE" | "PUBLISHER_ERROR" --> |
| `issue_url`             | <!-- e.g. https://github.com/<owner>/<repo>/issues/84 --> |
| `issue_number`          | <!-- integer --> |
| `issue_author_login`    | <!-- MUST equal the repo owner; otherwise the test FAILED --> |
| `issue_created_at`      | <!-- ISO 8601 from publish_result_<date>.json --> |
| `forced_over_duplicate` | <!-- true | false; if true, justify in Notes --> |
| `result_artifact`       | <!-- URL of the workflow artifact zip --> |

Notes:

```text
<!-- anything noteworthy from the confirm step -->
```

---

## Step 8 — Downstream pipeline (`generate_readme.yml` + `generate_site.yml`)

| Field                            | Value |
| -------------------------------- | ----- |
| `github_actions_url`             | <!-- URL of the generate_readme.yml run that the issue triggered --> |
| `generate_readme_status`         | <!-- success | failure | cancelled --> |
| `generate_readme_finished_at`    | <!-- ISO 8601 --> |
| `generate_site_status`           | <!-- success | failure | cancelled --> |
| `pages_url`                      | <!-- e.g. https://<owner>.github.io/<repo>/ --> |
| `README_updated`                 | <!-- true | false --> |
| `RSS_updated`                    | <!-- true | false --> |
| `BACKUP_updated`                 | <!-- true | false; expected file: BACKUP/<issue_number>_*.md --> |
| `Pages_updated`                  | <!-- true | false; was the site rebuilt with the new entry? --> |

---

## Step 9 — `verify-gitblog --json` report

```json
<!-- paste the full JSON output of:
     python -m agent.cli verify-gitblog --repo <repo> --issue-number <N> --date <date> --json
     here, verbatim. Do not summarize. -->
```

Per-check summary:

| Check                                  | ok    | detail (truncated) |
| -------------------------------------- | ----- | ------------------ |
| `issue_exists`                         |       |                    |
| `author_is_owner`                      |       |                    |
| `title_or_body_contains_date`          |       |                    |
| `generate_readme_workflow_triggered`   |       |                    |
| `readme_contains_issue_title`          |       |                    |
| `backup_has_issue_file`                |       |                    |
| `rss_contains_issue_title`             |       |                    |

---

## `failures`

```text
<!--
For each check that did not pass, write one bullet:
  - <check name>: <one-line root cause hypothesis>

If everything passed, write "none" verbatim.
-->
```

---

## `fixes`

```text
<!--
For each failure above, write one bullet:
  - <check name>: <what you actually changed to make it pass>
        (commit hash / setting changed / re-run timestamp)

If everything passed first try, write "none" verbatim.
-->
```

---

## `screenshots`

Attach screenshots to a `docs/smoke_tests/<test_date>/` folder and link
them here.

| File                              | Caption |
| --------------------------------- | ------- |
| `01_workflow_dry_run.png`         | Manual workflow dry-run summary in Actions tab |
| `02_workflow_confirm.png`         | Manual workflow confirm summary in Actions tab |
| `03_issue_authored_by_owner.png`  | Issue page showing owner (NOT bot) as author   |
| `04_generate_readme_run.png`      | `generate_readme.yml` run triggered by `issues` |
| `05_readme_diff.png`              | git diff of `README.md` showing new entry     |
| `06_backup_file.png`              | `BACKUP/<N>_*.md` listing                     |
| `07_rss_entry.png`                | `rss.xml` showing the new `<item>`            |
| `08_pages_homepage.png`           | Pages site rendering the new entry            |
| `09_verify_gitblog_output.png`    | Terminal output of `verify-gitblog`           |

---

## Sign-off

- [ ] All seven `verify-gitblog` checks passed (or every failure has an
      entry under `fixes`).
- [ ] Issue author is the repo **owner**, not `github-actions[bot]`.
- [ ] PAT will be rotated by `pat_expires_at`.
- [ ] No modifications were made to `main.py`, `generate_readme.yml`, or
      `generate_site.yml`.

Tested by: <!-- name --> on <!-- test_date -->
