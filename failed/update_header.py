#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Git Pre‑Commit Hook — Auto‑Generate / Update Header in Python Files
-------------------------------------------------------------------------------
Author      : Taken from local git config
Email       : Taken from local git config
Edited      : 2025‑05‑05
Description : Scans every staged *.py file, calls an OpenAI chat model to
              obtain a complete header block (Title, Description, Author, Email,
              Edited) and inserts/replaces that block at the top of the file.
              Modified files are automatically re‑added to the Git index so the
              header update is part of the commit.
===============================================================================
"""
from __future__ import annotations

import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import textwrap

import openai  # requires:  pip install openai


# ---------------------------------------------------------------------------#
# Basic shell helpers
# ---------------------------------------------------------------------------#
def run(cmd: list[str]) -> str:
    """Run a shell command and return its stdout (stripped)."""
    return subprocess.check_output(cmd, text=True).strip()


def get_git_user() -> tuple[str, str]:
    """Return (author, email) from local git config."""
    return (
        run(["git", "config", "--get", "user.name"]),
        run(["git", "config", "--get", "user.email"]),
    )


# ---------------------------------------------------------------------------#
# Ask GPT for a ready‑made header block
# ---------------------------------------------------------------------------#
MODEL = "gpt-4o-mini"  # or another chat‑completion model
TEMPERATURE = 0.3
MAX_CODE_CHARS = 4_000  # truncate long files to keep the prompt small


def generate_header_block(source_code: str, *, author: str, email: str) -> str:
    """
    Send the code plus metadata to OpenAI and get back a header block.

    Returns the header with a single trailing newline so it can be concatenated
    safely in front of the file contents.
    """
    today = datetime.now().strftime("%Y-%m-%d")

    prompt = textwrap.dedent(
        f"""
        You are a coding assistant that writes concise **Python file headers**.

        • Input you receive:
            – The author's full name and e‑mail
            – Today's date
            – A snippet of Python source code
            – The exact header layout to follow

        • What you must do:
            1. Generate a Title (≤ 60 chars, Title Case, no period)
            2. Generate a one‑sentence Description (≤ 120 chars, ends with '.')
            3. Fill Author, Email, Edited fields with the supplied data
            4. Return ONLY the finished header block (no extra text)

        ----  HEADER FORMAT  ----
        Title       : <title>
        Description : <description>
        Author      : {author}
        Email       : {email}
        Edited      : {today}
        --------------------------

        ----  PYTHON CODE SNIPPET ----
        {source_code[:MAX_CODE_CHARS]}
        --------------------------------
    """
    ).strip()

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You write file headers."},
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURE,
    )

    header = response.choices[0].message["content"].strip()
    return header + ("\n" if not header.endswith("\n") else "")


# ---------------------------------------------------------------------------#
# Regex to find any existing header (after optional top‑of‑file comments)
# ---------------------------------------------------------------------------#
HEADER_RE = re.compile(
    r"""
    ^(?:\"\"\"[\s\S]*?\"\"\"\s*\n)?      # optional module docstring
    (?:^\#.*\n)*?                        # leading comment lines
    (?:^Title\s*:.*\n                   # existing Title line
       ^Description\s*:.*\n
       ^Author\s*:.*\n
       ^Email\s*:.*\n
       ^Edited\s*:.*\n)?
    """,
    re.VERBOSE | re.MULTILINE,
)


# ---------------------------------------------------------------------------#
# Main workflow
# ---------------------------------------------------------------------------#
def main() -> None:
    author, email = get_git_user()

    # list of files currently staged for commit
    staged = run(["git", "diff", "--cached", "--name-only"]).splitlines()
    py_files = [Path(p) for p in staged if p.endswith(".py") and Path(p).is_file()]

    if not py_files:
        print("No Python files staged — nothing to do.")
        return

    for path in py_files:
        text = path.read_text(encoding="utf-8")

        # ask GPT for a header tailored to this file
        header_block = generate_header_block(text, author=author, email=email)

        # ------------------------------------------------------#
        # Insert or replace the header at the top of the file
        # ------------------------------------------------------#
        if HEADER_RE.match(text):
            # keep the (optional) docstring or comments that appear
            # before the header, then inject the new header
            def _inject(match: re.Match) -> str:  # noqa: D401
                first_line = match.group(0).splitlines()[0]
                return f"{first_line}\n{header_block}"

            new_text = HEADER_RE.sub(_inject, text, count=1)
        else:
            # no previous header found — prepend one
            new_text = header_block + text

        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            run(["git", "add", str(path)])

    print("Pre‑commit header update complete.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(f"Git command failed:\n{exc}\n")
        sys.exit(1)
    except KeyboardInterrupt:
        sys.stderr.write("Interrupted by user.\n")
        sys.exit(1)
