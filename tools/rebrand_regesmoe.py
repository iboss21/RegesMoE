"""One-shot rebrand of the user-visible strings in the RegesMoE tree.

WHAT IT CHANGES, AND WHAT IT DELIBERATELY DOES NOT.

CHANGES (what a user SEES):
  the web app's title, brand, GitHub link and API-monitor header; the compliance modal's brand; setup's
  banner; the README headings and links; the licence copyright; the ready-made-engine release URL.

DOES NOT CHANGE (the wire contract - renaming these breaks real installs):
  the engine binary names (strata.exe), every STRATA_* environment variable, the `strata.*` localStorage keys,
  the C++ namespace and every `strata_*` CMake target, the `strata-<model>.json` config filename, and the
  `strata mtp:` / `strata serve:` engine log prefixes the server parses.  A user who upgrades keeps their
  config, their theme, their engine and their env vars; a rename here would silently reset all of them.

So this is a SURFACE rename, not a fork of the internals.  `RegesMoE` is a single repo whose engine is still
the strata engine under the hood - deliberately, because the alternative is breaking every existing install.

    python tools/rebrand_regesmoe.py --check    # what would change
    python tools/rebrand_regesmoe.py --apply
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

BRAND = "RegesMoE"
ORG = "RegesAi"
COMPANY = "LIKE A KING INC."
SITE = "https://regescore.pro"
DEV_SITE = "https://davidio.dev"
GITHUB = "https://github.com/iboss21/RegesMoE"
REPO = "https://github.com/iboss21/RegesMoE/releases/latest/download/"

# (file, pattern, replacement, why)
EDITS: list[tuple[str, str, str, str]] = [
    # ---- the web app
    ("serve/web/index.html", r"<title>Strata</title>", f"<title>{BRAND}</title>", "browser tab"),
    ("serve/web/index.html", r'<span class="st-header__brand">Strata</span>',
     f'<span class="st-header__brand">{BRAND}</span>', "header wordmark"),
    ("serve/web/index.html", r'<a href="https://github\.com/Niko1221/Strata"[^>]*>Strata on GitHub</a>',
     f'<a href="{GITHUB}" target="_blank" rel="noopener noreferrer">{BRAND} on GitHub</a>',
     "repo link -> the user's own repo"),
    ("serve/web/index.html", r'close Strata\'s window', f"close {BRAND}'s window", "help text"),
    ("serve/web/index.html", r'<div class="st-modal__brand">Strata</div>',
     f'<div class="st-modal__brand">{BRAND}</div>', "compliance modal wordmark"),
    ("serve/web/monitor.html", r"<title>Strata API Monitor</title>", f"<title>{BRAND} API Monitor</title>",
     "monitor tab"),
    ("serve/web/monitor.html", r'<div class="brand">Strata <span>',
     f'<div class="brand">{BRAND} <span>', "monitor header"),
    # ---- the installer
    ("setup.py", r'PREBUILT_URL = "https://github\.com/Niko1221/Strata/releases/latest/download/"',
     f'PREBUILT_URL = "{REPO}"', "ready-made engine download folder -> the user's repo"),
    ("setup.py", r'PREBUILT_TAG_URL = "https://github\.com/Niko1221/Strata/releases/download/v\{version\}/"',
     f'PREBUILT_TAG_URL = "https://github.com/iboss21/RegesMoE/releases/download/v{{version}}/"',
     "version-pinned engine folder"),
    ("setup.py", r'say\("Strata - Qwen3\.8-Flash-Next on a normal PC \(a GPU \+ system RAM \+ CPU\)"\)',
     f'say("{BRAND} - your AI models on a normal PC (a GPU + system RAM + CPU)")', "installer banner"),
    ("setup.py", r'"https://github\.com/<you>/Strata/releases/latest/download/"',
     f'"{REPO}"', "the comment telling a forker where to point this"),
    # ---- the licence
    ("LICENSE", r"Copyright \(c\) 2026 Niko1221 and the Strata contributors",
     f"Copyright (c) 2026 {ORG} / {COMPANY}, and the {BRAND} contributors", "copyright holder"),
    # ---- the README
    ("README.md", r"<h1 align=\"center\">Strata</h1>", f'<h1 align="center">{BRAND}</h1>', "title"),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--apply", action="store_true", help="write the changes (default: report only)")
    ap.add_argument("--check", action="store_true", help="report only (the default)")
    args = ap.parse_args(argv)

    total = 0
    for rel, pattern, repl, why in EDITS:
        path = ROOT / rel
        if not path.is_file():
            print(f"  [skip] {rel}: not found")
            continue
        src = path.read_text(encoding="utf-8")
        new, n = re.subn(pattern, repl, src)
        if n == 0:
            print(f"  [no match] {rel}: {why}")
            continue
        total += n
        print(f"  [{n}] {rel}: {why}")
        if args.apply:
            path.write_text(new, encoding="utf-8")
    print(f"\n{total} replacement(s)" + ("" if args.apply else "  (--check only, nothing written)"))
    if not args.apply:
        print("run again with --apply to write them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())