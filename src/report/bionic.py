"""
DeltaPace :: Bionic Markdown Formatting
=======================================

``to_bionic`` applies a fan-magazine style where the first ~half of each word is
bolded (``Verstappen`` → ``**Ver**stappen``). Used optionally on report output
for a distinctive blog aesthetic.

Protected regions (left untouched):
* Markdown headers (lines starting with ``#``)
* Fenced code blocks (``` ... ```)
* Inline code (`` `...` ``)
* Markdown links (``[text](url)`` — only the bracket text is transformed)
"""

from __future__ import annotations

import re

# Split text into tokens we must not transform.
_PROTECTED_PATTERN = re.compile(
    r"(```[\s\S]*?```|`[^`]+`|\[[^\]]+\]\([^)]+\))"
)

# Word characters plus trailing/leading punctuation attached to a word.
_WORD_PATTERN = re.compile(r"([A-Za-z0-9]+)([^A-Za-z0-9]*)")


def _bionic_word(word: str) -> str:
    """Bold the first ~half of *word* (letter/digit run only)."""
    if not word:
        return word
    split_at = max(1, len(word) // 2)
    return f"**{word[:split_at]}**{word[split_at:]}"


def _bionic_segment(text: str) -> str:
    """Apply bionic formatting to a plain-text segment."""

    def _replace(match: re.Match[str]) -> str:
        letters = match.group(1)
        trailing = match.group(2)
        return _bionic_word(letters) + trailing

    return _WORD_PATTERN.sub(_replace, text)


def _bionic_link(match: re.Match[str]) -> str:
    """Transform visible link text only; preserve the URL."""
    inner = match.group(0)
    bracket_match = re.match(r"\[([^\]]+)\]\(([^)]+)\)", inner)
    if not bracket_match:
        return inner
    label, url = bracket_match.groups()
    return f"[{_bionic_segment(label)}]({url})"


def to_bionic(text: str) -> str:
    """Return *text* with bionic bolding applied to eligible words."""
    if not text:
        return text

    lines: list[str] = []
    in_fence = False

    for line in text.splitlines(keepends=True):
        # Preserve newline-only lines.
        if line in ("\n", "\r\n"):
            lines.append(line)
            continue

        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            lines.append(line)
            continue

        if in_fence or stripped.startswith("#"):
            lines.append(line)
            continue

        # Process line in protected / plain chunks.
        pos = 0
        chunks: list[str] = []
        for match in _PROTECTED_PATTERN.finditer(line):
            if match.start() > pos:
                chunks.append(_bionic_segment(line[pos : match.start()]))
            token = match.group(0)
            if token.startswith("["):
                chunks.append(_bionic_link(match))
            else:
                chunks.append(token)  # code — leave as-is
            pos = match.end()
        if pos < len(line):
            chunks.append(_bionic_segment(line[pos:]))
        lines.append("".join(chunks))

    return "".join(lines)
