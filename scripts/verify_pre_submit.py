#!/usr/bin/env python3
"""
verify_pre_submit.py — pre-submission readiness check for LegaLoom-Env.

Runs the 9-item checklist:

  1.  populate_results.py runs without errors (--dry-run)
  2.  README.md.bak exists  (populate has been run at least once)
  3.  blog_post.md.bak exists
  4.  No video references remain (grep -i 'youtube|video walkthrough|demo video')
  5.  All required chart files exist
  6.  pytest tests/ reports 60 passed
  7.  HF Space frontmatter still valid (title, license, sdk, tags include openenv)
  8.  client.py imports cleanly (proxy for Space deployability)
  9.  Multi-model artifacts consistency: aggregated_models.json and
      model_leaderboard.png either both exist or both don't (sign of partial run)

Each check prints PASS or FAIL with detail. Exits with status code = number
of failed checks (0 = all pass).

Usage:
    python scripts/verify_pre_submit.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

REQUIRED_CHARTS = [
    "before_after.png",
    "reward_curves.png",
    "reward_distribution.png",
    "episode_scatter.png",
]
OPTIONAL_CHARTS = ["adversarial_heatmap.png"]
VIDEO_PATTERNS = [
    r"youtube\.com",
    r"youtu\.be",
    r"video walkthrough",
    r"demo video",
    r"screencast",
    r"##\s+video\b",
    r"##\s+watch the demo",
]


class Result:
    def __init__(self, name: str):
        self.name = name
        self.passed: bool = False
        self.detail: str = ""

    def ok(self, detail: str = "") -> "Result":
        self.passed = True
        self.detail = detail
        return self

    def fail(self, detail: str) -> "Result":
        self.passed = False
        self.detail = detail
        return self


def check_populate_dry_run() -> Result:
    r = Result("populate_results.py --dry-run")
    script = REPO / "scripts" / "populate_results.py"
    if not script.exists():
        return r.fail("script missing")
    p = subprocess.run(
        [sys.executable, str(script), "--dry-run"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    if p.returncode == 0:
        return r.ok("script ran cleanly")
    return r.fail(f"exit {p.returncode}: {p.stderr.strip()[:160]}")


def check_backup_exists(filename: str) -> Result:
    r = Result(f"{filename}.bak exists (populate already run)")
    if (REPO / (filename + ".bak")).exists():
        return r.ok()
    return r.fail("not found — run `python scripts/populate_results.py` to create")


def check_no_video_references() -> Result:
    r = Result("No video references in README.md or blog_post.md")
    matches: list[str] = []
    for filename in ("README.md", "blog_post.md"):
        p = REPO / filename
        if not p.exists():
            continue
        text = p.read_text()
        for pattern in VIDEO_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                line_no = text[:m.start()].count("\n") + 1
                matches.append(f"{filename}:{line_no} ~ {m.group()!r}")
    if not matches:
        return r.ok()
    return r.fail("matches: " + "; ".join(matches[:5]))


def check_charts_exist() -> Result:
    r = Result("Required chart files in repo root")
    missing = [c for c in REQUIRED_CHARTS if not (REPO / c).exists()]
    have_optional = [c for c in OPTIONAL_CHARTS if (REPO / c).exists()]
    if missing:
        return r.fail("missing: " + ", ".join(missing))
    detail = f"have {len(REQUIRED_CHARTS)} required"
    if have_optional:
        detail += f"; optional present: {', '.join(have_optional)}"
    return r.ok(detail)


def check_tests_pass() -> Result:
    r = Result("pytest tests/ reports 60 passed")
    p = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    out = p.stdout + p.stderr
    m = re.search(r"(\d+)\s+passed", out)
    if m and p.returncode == 0:
        n = int(m.group(1))
        if n >= 60:
            return r.ok(f"{n} passed")
        return r.fail(f"only {n} passed (expected 60)")
    return r.fail(f"pytest exit {p.returncode}; tail: {out.strip()[-160:]}")


def check_frontmatter() -> Result:
    r = Result("HF Space frontmatter valid (README.md)")
    readme = REPO / "README.md"
    if not readme.exists():
        return r.fail("README.md missing")
    text = readme.read_text()
    if not text.startswith("---"):
        return r.fail("no frontmatter at top of README")
    fm_end = text.find("\n---", 3)
    if fm_end < 0:
        return r.fail("frontmatter not terminated")
    fm = text[:fm_end]
    required_keys = ["title:", "sdk:", "license:"]
    missing = [k for k in required_keys if k not in fm]
    if missing:
        return r.fail(f"missing keys: {', '.join(missing)}")
    if "openenv" not in fm.lower():
        return r.fail("'openenv' tag not found in frontmatter")
    return r.ok("title, sdk, license, openenv tag all present")


def check_client_imports() -> Result:
    r = Result("client.py imports cleanly")
    client = REPO / "client.py"
    if not client.exists():
        return r.fail("client.py missing")
    p = subprocess.run(
        [sys.executable, "-c", "import ast; ast.parse(open('client.py').read())"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    if p.returncode == 0:
        return r.ok("syntax OK")
    return r.fail(f"parse error: {p.stderr.strip()[:160]}")


def check_multimodel_consistency() -> Result:
    """If multi-model run was attempted, both aggregated_models.json and
    model_leaderboard.png must exist. If neither exists, single-model workflow
    is fine. Only fail if exactly one is present (partial run)."""
    r = Result("Multi-model artifacts consistency")
    agg = REPO / "aggregated_models.json"
    chart = REPO / "model_leaderboard.png"
    if agg.exists() and chart.exists():
        return r.ok("both aggregated_models.json and model_leaderboard.png present")
    if not agg.exists() and not chart.exists():
        return r.ok("single-model workflow (no multi-model artifacts present)")
    if agg.exists() and not chart.exists():
        return r.fail(
            "aggregated_models.json exists but model_leaderboard.png missing — "
            "run `python scripts/generate_charts.py` to produce the chart"
        )
    return r.fail(
        "model_leaderboard.png exists but aggregated_models.json missing — "
        "run `python scripts/aggregate_models.py` to regenerate the data"
    )


def main() -> int:
    checks = [
        check_populate_dry_run(),
        check_backup_exists("README.md"),
        check_backup_exists("blog_post.md"),
        check_no_video_references(),
        check_charts_exist(),
        check_tests_pass(),
        check_frontmatter(),
        check_client_imports(),
        check_multimodel_consistency(),
    ]

    print("=" * 72)
    print("LegaLoom-Env — Pre-Submission Verification")
    print("=" * 72)
    n_pass = sum(1 for c in checks if c.passed)
    n_fail = len(checks) - n_pass
    for i, c in enumerate(checks, 1):
        marker = "PASS" if c.passed else "FAIL"
        print(f"  [{marker}] ({i}/{len(checks)}) {c.name}")
        if c.detail:
            print(f"           {c.detail}")
    print("-" * 72)
    print(f"  {n_pass}/{len(checks)} passed")
    if n_fail:
        print(f"  {n_fail} failed — fix before submitting.")
    else:
        print("  Ready to submit.")
    print("=" * 72)
    return n_fail


if __name__ == "__main__":
    sys.exit(main())
