#!/usr/bin/env python3
"""Restore README.md and blog_post.md from their .bak backups."""
import shutil, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
restored = []
for name in ("README.md", "blog_post.md"):
    src = REPO / (name + ".bak")
    if src.exists():
        shutil.copyfile(src, REPO / name)
        restored.append(name)
print(f"Restored: {restored}" if restored else "No .bak files found.")
sys.exit(0 if restored else 1)
