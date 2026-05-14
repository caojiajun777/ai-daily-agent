"""PDF generation and email delivery for AI daily reports.

Converts the draft Markdown to a styled HTML page, renders it to PDF
via Playwright (already available), and emails it to subscribers.

Environment variables required:
  SMTP_HOST      — default smtp.qq.com
  SMTP_PORT      — default 465
  SMTP_USER      — sender email, e.g. 905562805@qq.com
  SMTP_PASSWORD  — QQ邮箱 SMTP 授权码 (NOT QQ password)
"""

from __future__ import annotations

import json
import os
import smtplib
import tempfile
from datetime import datetime
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional


def md_to_html(md_text: str, title: str = "AI 日报") -> str:
    """Convert draft Markdown to a styled HTML page."""
    import markdown as md_lib

    body_html = md_lib.markdown(
        md_text,
        extensions=["extra", "codehilite", "tables"],
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{
    max-width: 720px; margin: 0 auto; padding: 32px 16px;
    font-family: "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", "WenQuanYi Micro Hei", sans-serif;
    font-size: 15px; line-height: 1.8; color: #222; background: #fff;
  }}
  h1 {{ font-size: 22px; border-bottom: 2px solid #1a1a2e; padding-bottom: 8px; }}
  h2 {{ font-size: 18px; margin-top: 28px; color: #1a1a2e; }}
  h3 {{ font-size: 16px; margin-top: 20px; }}
  h2 em {{ font-size: 14px; color: #888; font-weight: normal; }}
  blockquote {{ border-left: 3px solid #e0e0e0; margin: 8px 0; padding: 4px 16px; color: #555; }}
  img {{ max-width: 100%; border-radius: 4px; margin: 8px 0; }}
  a {{ color: #1a56db; text-decoration: none; }}
  hr {{ border: none; border-top: 1px solid #eee; margin: 24px 0; }}
  ul {{ padding-left: 20px; }}
  li {{ margin: 4px 0; }}
  @media print {{
    body {{ font-size: 13px; }}
    hr {{ page-break-after: avoid; }}
    h3 {{ page-break-before: always; }}
  }}
</style>
</head>
<body>
{body_html}
</body>
</html>"""


def generate_pdf(md_text: str, title: str = "AI 日报", output_path: Optional[str] = None) -> str:
    """Convert markdown to a PDF file using Playwright.

    Returns the path to the generated PDF.
    """
    html = md_to_html(md_text, title)
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".pdf")

    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser = None
        for channel in ["msedge", "chrome", None]:
            try:
                kw = {"headless": True, "args": ["--no-sandbox"]}
                if channel:
                    kw["channel"] = channel
                browser = pw.chromium.launch(**kw)
                break
            except Exception:
                continue
        if not browser:
            raise RuntimeError("No browser available for PDF generation")

        page = browser.new_page()
        page.set_content(html, wait_until="load")
        page.pdf(
            path=output_path,
            format="A4",
            margin={"top": "16mm", "bottom": "16mm", "left": "14mm", "right": "14mm"},
            print_background=True,
        )
        page.close()
        browser.close()

    return output_path


def send_email(
    *,
    subject: str,
    md_text: str,
    pdf_path: Optional[str] = None,
    recipients: Optional[List[str]] = None,
) -> Dict:
    """Send the daily report as HTML email with optional PDF attachment.

    Returns a dict with keys: ok, recipients, error (if any).
    """
    smtp_host = os.getenv("SMTP_HOST", "smtp.qq.com")
    smtp_port = int(os.getenv("SMTP_PORT", "465"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_password = os.getenv("SMTP_PASSWORD", "")

    if not smtp_user or not smtp_password:
        return {"ok": False, "error": "SMTP_USER or SMTP_PASSWORD not set"}

    if recipients is None:
        recipients = _load_subscribers()
    if not recipients:
        return {"ok": False, "error": "no recipients configured"}

    html_body = md_to_html(md_text, subject)

    msg = MIMEMultipart()
    msg["From"] = smtp_user
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    if pdf_path and os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            part = MIMEBase("application", "pdf")
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f'attachment; filename="{os.path.basename(pdf_path)}"',
            )
            msg.attach(part)

    try:
        with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=30) as server:
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, recipients, msg.as_string())
        return {"ok": True, "recipients": recipients}
    except Exception as e:
        return {"ok": False, "error": str(e), "recipients": recipients}


def _load_subscribers() -> List[str]:
    """Load subscriber emails from multiple sources. Priority:

    1. EMAIL_SUBSCRIBERS env var (comma-separated)
    2. GitHub Issue comments (label: subscribe)
    3. subscribers.txt in project root (one email per line)
    4. publish.subscribers in default.yaml
    """
    import re as _re

    env_val = os.getenv("EMAIL_SUBSCRIBERS", "")
    if env_val:
        return [e.strip() for e in env_val.split(",") if e.strip()]

    all_subs: List[str] = []

    # GitHub Issue comments (label: subscribe).
    gh_token = os.getenv("GITHUB_PUBLISH_TOKEN", "") or os.getenv("GITHUB_TOKEN", "")
    gh_repo = os.getenv("PUBLISH_REPO", "") or os.getenv("GITHUB_REPOSITORY", "")
    if gh_token and gh_repo:
        try:
            import httpx as _httpx
            issues_url = f"https://api.github.com/repos/{gh_repo}/issues"
            client = _httpx.Client(
                headers={
                    "Authorization": f"Bearer {gh_token}",
                    "Accept": "application/vnd.github.v3+json",
                    "User-Agent": "report-agent",
                },
                timeout=15.0,
            )
            # Find the first open issue with label "subscribe".
            resp = client.get(issues_url, params={"labels": "subscribe", "state": "open", "per_page": 1})
            issues = resp.json() if resp.status_code == 200 else []
            if issues:
                issue_number = issues[0]["number"]
                # Get comments.
                comments_url = f"{issues_url}/{issue_number}/comments"
                resp2 = client.get(comments_url, params={"per_page": 100})
                comments = resp2.json() if resp2.status_code == 200 else []
                for c in comments:
                    body = c.get("body", "")
                    # Extract email addresses from the comment.
                    emails = _re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", body)
                    for e in emails:
                        if e not in all_subs:
                            all_subs.append(e)
            client.close()
        except Exception:
            pass

    # subscribers.txt in project root.
    root = os.path.join(os.path.dirname(__file__), "..", "..")
    txt_path = os.path.join(root, "subscribers.txt")
    if os.path.exists(txt_path):
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        emails = _re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", line)
                        for e in emails:
                            if e not in all_subs:
                                all_subs.append(e)
        except Exception:
            pass

    # default.yaml fallback.
    config_path = os.path.join(root, "agent", "configs", "default.yaml")
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        subs = cfg.get("publish", {}).get("subscribers", [])
        for s in subs:
            s = s.strip()
            if s and s not in all_subs:
                all_subs.append(s)
    except Exception:
        pass

    return all_subs


def send_daily_report(
    *,
    date: Optional[str] = None,
    md_path: Optional[str] = None,
    pdf_dir: str = "artifacts/pdf",
) -> Dict:
    """One-stop: generate PDF from draft, email to subscribers.

    Usage:
      python -m agent.cli send --run-id 2026-05-10
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    if md_path is None:
        md_path = f"artifacts/drafts/{date}.md"

    if not os.path.exists(md_path):
        return {"ok": False, "error": f"draft not found: {md_path}"}

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    title = f"AI 日报 {date}"
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, f"ai-daily-{date}.pdf")

    # Generate PDF.
    try:
        generate_pdf(md_text, title, pdf_path)
    except Exception as e:
        pdf_path = None  # PDF generation failed, send HTML only.

    # Send email.
    result = send_email(subject=title, md_text=md_text, pdf_path=pdf_path)

    if pdf_path and os.path.exists(pdf_path):
        result["pdf_path"] = pdf_path
    return result
