"""Publisher (MVP): write the draft to artifacts/drafts/YYYY-MM-DD.md.

GitHub Issue publishing lives in a future phase. When we add it the publisher
MUST authenticate with the repo OWNER's PAT, not the default GITHUB_TOKEN:
original ``main.py`` filters issues by ``is_me(issue, me)``, so issues created
by ``github-actions[bot]`` would silently be skipped downstream.
"""

from __future__ import annotations

import os
from typing import Tuple

from agent.agents.writer import render_markdown
from agent.schemas import Draft


def publish_local(draft: Draft, out_dir: str = "artifacts/drafts") -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    md = render_markdown(draft)
    md_path = os.path.join(out_dir, f"{draft.date}.md")
    json_path = os.path.join(out_dir, f"{draft.date}.json")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(draft.model_dump_json(indent=2))
    return md_path, json_path
