# Smoke Test Guide — report-agent end-to-end publish test

End-to-end test plan for verifying that an agent-generated draft can be
published to **your own** GitHub repo as an Issue via
`publish-issue --confirm`.

This guide is meant to be followed **once per smoke test**, with the final
state captured in [smoke_test_report.template.md](smoke_test_report.template.md).

---

## 0. What this test validates

`report-agent` ends one stage upstream of a published issue: it produces a
local draft, runs critic + eval gate checks, then creates a GitHub issue via
the `GitHubIssuePublisher` tool. The smoke test answers:

1. Does the gate correctly pass or block the draft?
2. Does the issue actually appear in the target repo with **your account** as
   author (not `github-actions[bot]`)?
3. Do the artifacts (draft, report, trace, publish result) land on disk as
   expected?

---

## 1. Prerequisites

- A GitHub account.
- Python 3.10+ with `pip install -r requirements.txt` done.
- A working DeepSeek API key (or `--provider mock` to skip the LLM).

You will create:

- A **target repo** under your own account (can be an existing repo, or a
  new empty one created for testing).
- A **Personal Access Token (PAT)** owned by the *same account* that owns
  the repo.

> ⚠️ **Never commit the PAT to git.** Use `.env` locally and repo Secrets
> for CI.

---

## 2. Create the target repo

You can test against any repo you own. For a clean test with no risk of
polluting an existing project, create a fresh one:

1. Open <https://github.com/new>.
2. Name it anything (e.g. `ai-daily-test`).
3. Set it to **Private** or **Public** — either works.
4. Click **Create repository**.

Your target is now `<your-handle>/ai-daily-test`. From here on we'll call
this `<owner>/<repo>`.

---

## 3. Create the owner PAT

### Why not `GITHUB_TOKEN`

The default `GITHUB_TOKEN` authenticates as `github-actions[bot]`. Issues
created by a bot lack proper owner authorship. We refuse it by design to
maintain audit attribution and resume/demo consistency.

### Creating the PAT

1. Go to **GitHub → Settings → Developer settings → Personal access
   tokens → Fine-grained tokens → Generate new token**.
2. **Resource owner**: your own account.
3. **Repository access**: only `<owner>/<repo>` (least privilege).
4. **Repository permissions**:
   - **Issues**: Read and write — needed to create the issue.
   - **Contents**: Read-only — needed by `verify-gitblog` to read files.
   - **Actions**: Read-only — needed by `verify-gitblog` to list workflow runs.
   - **Metadata**: Read-only — auto-included.
5. **Expiration**: 30 days (rotate after the test).
6. Click **Generate token** and copy the value.

---

## 4. Configure local environment

```bash
cp .env.example .env
# Edit .env and set:
#   GITHUB_PUBLISH_TOKEN=<your-fine-grained-PAT>
#   PUBLISH_REPO=<owner>/<repo>
#   DEEPSEEK_API_KEY=<your-key>   # skip if using --provider mock
```

Load the env:

```bash
# PowerShell
Get-Content .env | ForEach-Object { if ($_ -match '^(\w+)=(.*)$') { [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2]) } }

# bash / zsh
set -a && source .env && set +a
```

---

## 5. Generate a draft

If you don't have artifacts from a prior run, generate them now:

```bash
# Mock provider (offline, fast)
python -m agent.cli run --provider mock

# Or DeepSeek (real LLM)
python -m agent.cli run --provider deepseek
```

Note the `date` printed in the run summary (e.g. `2026-05-09`). All
subsequent commands use this date.

---

## 6. Trigger a dry-run

```bash
python -m agent.cli publish-issue --run-id <date> --dry-run
```

### Expected outcome

- Exit code 0.
- Prints a JSON summary with `mode: dry-run`, `gate_ok: true|false`,
  `would_publish: true|false`.
- Writes `artifacts/reports/publish_preview_<date>.json`.

### Inspect the preview file

```bash
cat artifacts/reports/publish_preview_<date>.json
```

Check:

- `target_repo` matches your `PUBLISH_REPO`.
- `gate_result.ok` is `true`. If `false`, read `blocked_reasons`.
- `body_preview` contains the 6-section draft, not boilerplate.
- `duplicates` is `[]` on first attempt.

If `gate_ok=false`, **stop**. Fix the upstream draft or re-run with a
fresh date. The gate exists for a reason; don't override it.

---

## 7. Trigger confirm (real publish)

```bash
python -m agent.cli publish-issue --run-id <date> --confirm
```

### Expected result

- Exit code 0.
- Prints `mode: confirm`, `status: published`, `issue_number: <N>`,
  `issue_url: https://github.com/<owner>/<repo>/issues/<N>`.
- A new GitHub issue appears at the URL.

### Verify issue authorship

Open the issue URL. The author must be **your GitHub account**, not
`github-actions`. If it shows a bot, the PAT is wrong — recreate it (§3)
and retry.

---

## 8. Check artifacts on disk

```text
artifacts/
├── drafts/<date>.md                     # 6-section Markdown draft
├── drafts/<date>.json                   # same in structured JSON
├── traces/<date>.jsonl                  # append-only event log
├── reports/<date>.json                  # RunState snapshot
├── reports/publish_preview_<date>.json  # dry-run preview
└── reports/publish_result_<date>.json   # confirm result
```

Open `publish_result_<date>.json` and confirm:

- `status` = `"published"`
- `issue_number` and `issue_url` are populated.
- `forced_over_duplicate` is absent or `false`.

---

## 9. Optional: `verify-gitblog` (legacy downstream check)

If your target repo has a `generate_readme.yml` workflow and a `BACKUP/`
directory that follows the juya-ai-daily convention, you can run the
legacy verifier:

```bash
python -m agent.cli verify-gitblog \
  --repo $PUBLISH_REPO \
  --issue-number <N> \
  --date <date> \
  --json
```

For a plain test repo without those downstream artifacts, most checks will
fail — that is expected. The important check is `issue_exists` and
`author_is_owner`.

---

## 10. Common failure modes

| Symptom                                              | Likely cause                              | Fix                                                      |
| ---------------------------------------------------- | ----------------------------------------- | -------------------------------------------------------- |
| `GITHUB_PUBLISH_TOKEN is not set`                    | Env var missing or wrong name             | Check `.env`; reload env                                 |
| `Refusing to use GITHUB_TOKEN`                       | PAT value equals `GITHUB_TOKEN`           | The wrong token was set; use a fine-grained PAT          |
| `repo must be 'owner/name'`                          | `PUBLISH_REPO` not set or malformed       | Set `PUBLISH_REPO=<owner>/<repo>` in `.env`              |
| `gate_ok=false`, critic FAIL                         | LLM draft has too few sections/items      | Re-run with new date, or use `--provider deepseek`       |
| `gate_ok=false`, "draft has N items, below minimum"  | Mock/real feeds returned very few items   | Check source config; mock always produces ≥6 items       |
| Issue created but author is `github-actions[bot]`    | Wrong token (bot token used)              | Recreate PAT under your own account (§3)                 |
| `blocked_by_duplicate`                               | Issue for this date already exists        | Close/delete the old issue, or use `--force`             |

---

## 11. Cleanup

- **Rotate or revoke** `GITHUB_PUBLISH_TOKEN`. Fine-grained PATs with
  Issues:write are persistent attack surface.
- Optionally close the test issue with a comment "smoke test — safe to delete".
- Save the publish artifacts for your audit folder.
- Fill in [smoke_test_report.template.md](smoke_test_report.template.md)
  and commit it under `docs/smoke_tests/<date>.md`.
