#!/usr/bin/env python3
"""render_doc.py -- build-time documentation tool for muzero-tictactoe.

Converts docs/algorithm-explained.md into a styled, self-contained
docs/algorithm-explained.html.

This is a documentation build tool, NOT a runtime dependency of the
MuZero C++ project. The project itself remains C++17 with zero external
dependencies; this script is only ever run by hand, offline, to
regenerate the HTML walkthrough after docs/algorithm-explained.md
changes. It uses nothing but the Python 3 standard library -- no
`markdown`, no `mistune`, no pip install -- in keeping with the
project's zero-dependency policy.

Usage:
    python3 tools/render_doc.py

It reads docs/algorithm-explained.md, borrows the CSS/theme <style>
block verbatim from the sibling alphazero-tictactoe project's
docs/algorithm-explained.html (so the two documents render as
siblings), and writes docs/algorithm-explained.html.

Markdown subset supported
--------------------------
This is a small parser tailored to what algorithm-explained.md actually
uses. It is not a general CommonMark implementation.

  - ATX headings `#` .. `######`, with GitHub-style anchor slugs that
    match the hand-written links in the document's own "## Contents"
    section (lowercase, drop anything that isn't a letter/digit/space/
    hyphen/underscore -- including em dashes -- then turn each space
    into a hyphen; duplicate slugs get -1, -2, ... suffixes).
  - Paragraphs (soft-wrapped lines joined with a single space).
  - Fenced code blocks, ``` or ```lang, rendered verbatim with HTML
    metacharacters escaped and the language kept as a CSS class.
  - Tables: leading/trailing `|` optional, a `---`/`:--`/`--:`/`:-:`
    alignment row, and a backslash-escaped pipe as a literal pipe inside
    a cell.
  - Blockquotes, including ones with more than one paragraph (a bare
    `>` line separates paragraphs within the same blockquote).
  - Single-level ordered (`1.`) and unordered (`-`) lists, with
    indented continuation lines folded into the same item.
  - Horizontal rules: a line that is exactly `---`.
  - Inline: `code`, **bold**, *italic*, [text](url), with HTML
    metacharacters escaped everywhere outside of tags this script
    itself generates.

Diagrams
--------
Two inline SVG diagrams are spliced into the rendered HTML at fixed
points, identified by matching a unique, literal sentence from the
markdown source (see DIAGRAM_ANCHORS below). They use the borrowed
stylesheet's CSS custom properties for every colour, so they theme
with the rest of the page.
"""

import html
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MD_PATH = REPO_ROOT / "docs" / "algorithm-explained.md"
HTML_PATH = REPO_ROOT / "docs" / "algorithm-explained.html"
SIBLING_HTML_PATH = REPO_ROOT.parent / "alphazero-tictactoe" / "docs" / "algorithm-explained.html"
# The sibling's stylesheet is located by content rather than by line
# number: it is the font <link> plus the <style> block. A hardcoded line
# count silently breaks the moment that file gains or loses a line above
# the stylesheet, which is exactly what happened when it grew a <head>.


# ---------------------------------------------------------------------------
# Inline formatting
# ---------------------------------------------------------------------------

_CODE_SPAN_RE = re.compile(r"`([^`]+)`")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"\*(.+?)\*")


def render_inline(text: str) -> str:
    """Render inline markdown (code, links, bold, italic) to HTML.

    Code spans are protected behind placeholders before anything else
    runs, so `**` or `*` inside `` `code` `` is never mistaken for
    emphasis, and so HTML metacharacters inside code spans get escaped
    exactly once.
    """
    placeholders = []

    def stash_code(m: re.Match) -> str:
        escaped = html.escape(m.group(1), quote=False)
        placeholders.append(f"<code>{escaped}</code>")
        return f"\x00{len(placeholders) - 1}\x00"

    protected = _CODE_SPAN_RE.sub(stash_code, text)

    # Escape remaining HTML metacharacters in the non-code text.
    escaped = html.escape(protected, quote=False)

    # Links: [text](url). Render the link label through bold/italic too.
    def render_link(m: re.Match) -> str:
        label = _BOLD_RE.sub(r"<strong>\1</strong>", m.group(1))
        label = _ITALIC_RE.sub(r"<em>\1</em>", label)
        url = html.escape(m.group(2), quote=True)
        return f'<a href="{url}">{label}</a>'

    escaped = _LINK_RE.sub(render_link, escaped)
    escaped = _BOLD_RE.sub(r"<strong>\1</strong>", escaped)
    escaped = _ITALIC_RE.sub(r"<em>\1</em>", escaped)

    # Restore code spans.
    def restore(m: re.Match) -> str:
        return placeholders[int(m.group(1))]

    return re.sub(r"\x00(\d+)\x00", restore, escaped)


# ---------------------------------------------------------------------------
# Heading anchors
# ---------------------------------------------------------------------------

_SLUG_STRIP_RE = re.compile(r"[^a-z0-9 _-]")

_used_slugs = {}


def slugify(text: str) -> str:
    """GitHub-style heading anchor: lowercase, drop everything that
    isn't a letter/digit/space/hyphen/underscore, then turn each
    remaining space into a hyphen. Deleting (not replacing) stripped
    characters is what makes an em dash surrounded by spaces collapse
    a single space on each side into a double hyphen, matching the
    links already written in this document's "## Contents" section.
    """
    lowered = text.lower()
    stripped = _SLUG_STRIP_RE.sub("", lowered)
    slug = stripped.replace(" ", "-")
    if slug in _used_slugs:
        _used_slugs[slug] += 1
        slug = f"{slug}-{_used_slugs[slug]}"
    else:
        _used_slugs[slug] = 0
    return slug


# ---------------------------------------------------------------------------
# Block-level parser
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_UL_RE = re.compile(r"^([-*+])\s+(.*)$")
_OL_RE = re.compile(r"^(\d+)\.\s+(.*)$")
_HR_RE = re.compile(r"^-{3,}$")
_TABLE_SEP_RE = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$")


def split_table_row(line: str):
    """Split a table row on unescaped `|`, honouring `\\|` as a literal
    pipe inside a cell, and dropping a leading/trailing empty cell
    caused by leading/trailing `|`.
    """
    cells = []
    current = []
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "\\" and i + 1 < len(line) and line[i + 1] == "|":
            current.append("|")
            i += 2
            continue
        if ch == "|":
            cells.append("".join(current))
            current = []
            i += 1
            continue
        current.append(ch)
        i += 1
    cells.append("".join(current))
    if cells and cells[0].strip() == "":
        cells = cells[1:]
    if cells and cells[-1].strip() == "":
        cells = cells[:-1]
    return [c.strip() for c in cells]


def parse_alignment(sep_cells):
    aligns = []
    for cell in sep_cells:
        left = cell.startswith(":")
        right = cell.endswith(":")
        if left and right:
            aligns.append("center")
        elif right:
            aligns.append("right")
        elif left:
            aligns.append("left")
        else:
            aligns.append(None)
    return aligns


def render_table(header_cells, aligns, body_rows):
    def style_for(i):
        if i < len(aligns) and aligns[i]:
            return f' style="text-align:{aligns[i]}"'
        return ""

    # Wrapped in the copied stylesheet's own .wrap-table (overflow-x:auto)
    # so a wide table scrolls within itself instead of widening the page.
    out = ['<div class="wrap-table">', "<table>", "<thead>", "<tr>"]
    for i, cell in enumerate(header_cells):
        out.append(f"<th{style_for(i)}>{render_inline(cell)}</th>")
    out.append("</tr>")
    out.append("</thead>")
    out.append("<tbody>")
    for row in body_rows:
        out.append("<tr>")
        for i, cell in enumerate(row):
            out.append(f"<td{style_for(i)}>{render_inline(cell)}</td>")
        out.append("</tr>")
    out.append("</tbody>")
    out.append("</table>")
    out.append("</div>")
    return "\n".join(out)


def render_blockquote(raw_lines):
    stripped = []
    for line in raw_lines:
        if line == ">":
            stripped.append("")
        elif line.startswith("> "):
            stripped.append(line[2:])
        elif line.startswith(">"):
            stripped.append(line[1:])
        else:
            stripped.append(line)
    # split into paragraphs on blank lines
    paragraphs = []
    current = []
    for line in stripped:
        if line.strip() == "":
            if current:
                paragraphs.append(" ".join(current))
                current = []
        else:
            current.append(line)
    if current:
        paragraphs.append(" ".join(current))
    inner = "\n".join(f"<p>{render_inline(p)}</p>" for p in paragraphs)
    return f"<blockquote>\n{inner}\n</blockquote>"


def render_code_block(lang, code_lines):
    escaped = html.escape("\n".join(code_lines), quote=False)
    cls = f' class="language-{html.escape(lang)}"' if lang else ""
    return f"<pre><code{cls}>{escaped}</code></pre>"


def render_list(items, ordered):
    tag = "ol" if ordered else "ul"
    out = [f"<{tag}>"]
    for item_text in items:
        out.append(f"<li>{render_inline(item_text)}</li>")
    out.append(f"</{tag}>")
    return "\n".join(out)


def parse_markdown(text: str):
    """Parse the markdown subset into a list of (raw_source, html) block
    tuples. raw_source is kept so the diagram-splicing step can locate
    insertion points by matching literal sentences from the original
    document.
    """
    lines = text.split("\n")
    n = len(lines)
    i = 0
    blocks = []

    while i < n:
        line = lines[i]

        if line.strip() == "":
            i += 1
            continue

        # Fenced code block
        if line.startswith("```"):
            lang = line[3:].strip()
            start = i
            i += 1
            code_lines = []
            while i < n and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            raw = "\n".join(lines[start:i])
            blocks.append((raw, render_code_block(lang, code_lines)))
            continue

        # Heading
        m = _HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            text_content = m.group(2).strip()
            slug = slugify(text_content)
            blocks.append(
                (line, f'<h{level} id="{slug}">{render_inline(text_content)}</h{level}>')
            )
            i += 1
            continue

        # Horizontal rule
        if _HR_RE.match(line.strip()):
            blocks.append((line, "<hr>"))
            i += 1
            continue

        # Table: a line starting with '|' followed by a separator line
        if line.lstrip().startswith("|") and i + 1 < n and _TABLE_SEP_RE.match(lines[i + 1].strip()):
            start = i
            header_cells = split_table_row(line)
            sep_cells = split_table_row(lines[i + 1])
            aligns = parse_alignment(sep_cells)
            i += 2
            body_rows = []
            while i < n and lines[i].lstrip().startswith("|"):
                body_rows.append(split_table_row(lines[i]))
                i += 1
            raw = "\n".join(lines[start:i])
            blocks.append((raw, render_table(header_cells, aligns, body_rows)))
            continue

        # Blockquote
        if line.startswith(">"):
            start = i
            while i < n and (lines[i].startswith(">") or lines[i].strip() == ""):
                if lines[i].strip() == "" and (i + 1 >= n or not lines[i + 1].startswith(">")):
                    break
                i += 1
            raw_lines = lines[start:i]
            raw = "\n".join(raw_lines)
            blocks.append((raw, render_blockquote(raw_lines)))
            continue

        # List (unordered or ordered), tight, single-level, with
        # indented continuation lines folded into the current item.
        ul_m = _UL_RE.match(line)
        ol_m = _OL_RE.match(line)
        if ul_m or ol_m:
            ordered = ol_m is not None
            start = i
            items = []
            marker_re = _OL_RE if ordered else _UL_RE
            current_item = (ul_m or ol_m).group(2)
            i += 1
            while i < n:
                if lines[i].strip() == "":
                    break
                mm = marker_re.match(lines[i])
                if mm:
                    items.append(current_item)
                    current_item = mm.group(2)
                    i += 1
                elif lines[i].startswith("  ") or lines[i].startswith("\t"):
                    current_item += " " + lines[i].strip()
                    i += 1
                else:
                    break
            items.append(current_item)
            raw = "\n".join(lines[start:i])
            blocks.append((raw, render_list(items, ordered)))
            continue

        # Paragraph: soft-wrapped lines until a blank line or a new block.
        start = i
        para_lines = [line]
        i += 1
        while i < n and lines[i].strip() != "" and not (
            lines[i].startswith("```")
            or _HEADING_RE.match(lines[i])
            or _HR_RE.match(lines[i].strip())
            or lines[i].startswith(">")
            or _UL_RE.match(lines[i])
            or _OL_RE.match(lines[i])
            or lines[i].lstrip().startswith("|")
        ):
            para_lines.append(lines[i])
            i += 1
        raw = "\n".join(lines[start:i])
        joined = " ".join(l.strip() for l in para_lines)
        blocks.append((raw, f"<p>{render_inline(joined)}</p>"))

    return blocks


# ---------------------------------------------------------------------------
# Diagrams
# ---------------------------------------------------------------------------

def diagram_search_tree() -> str:
    """Diagram 1: the three functions and how search uses them.

    A real board at the root feeds `h` (representation) exactly once.
    From the resulting latent, `g` (dynamics) is applied repeatedly
    going down the tree, and `f` (prediction) hangs off every latent,
    including the root's.
    """
    return """
<figure class="diagram">
  <div class="fig-frame">
    <svg viewBox="-50 0 810 430" role="img" aria-labelledby="diag1-title diag1-desc"
         style="max-width:100%; height:auto; font-family:'IBM Plex Mono',monospace;">
      <title id="diag1-title">The three functions and how search uses them</title>
      <desc id="diag1-desc">A real board at the root feeds the representation
        function h exactly once. Dynamics g is applied repeatedly going down
        the search tree from the resulting latent, and prediction f hangs off
        every latent, at the root and below.</desc>

      <!-- real board, root -->
      <rect x="16" y="8" width="108" height="52" rx="6" class="board-box" />
      <text x="70" y="30" text-anchor="middle" class="lbl">real board</text>
      <text x="70" y="46" text-anchor="middle" class="lbl faint" font-size="11">observation enters here, once</text>

      <line x1="70" y1="60" x2="70" y2="88" class="rule-line" marker-end="url(#arrow)"/>
      <text x="86" y="78" class="lbl x-fill" font-size="12">h</text>

      <!-- root latent s0 -->
      <circle cx="70" cy="112" r="22" class="latent-node root-node"/>
      <text x="70" y="117" text-anchor="middle" class="lbl on-node">s⁰</text>

      <!-- f off the root -->
      <line x1="92" y1="112" x2="150" y2="112" class="rule-line" marker-end="url(#arrow)"/>
      <text x="112" y="105" class="lbl o-fill" font-size="12">f</text>
      <rect x="152" y="94" width="118" height="36" rx="5" class="predict-box"/>
      <text x="211" y="117" text-anchor="middle" class="lbl" font-size="12">policy, value</text>

      <!-- g down to depth 1, three children -->
      <line x1="58" y1="132" x2="20" y2="200" class="rule-line" marker-end="url(#arrow)"/>
      <line x1="70" y1="134" x2="70" y2="200" class="rule-line" marker-end="url(#arrow)"/>
      <line x1="82" y1="132" x2="120" y2="200" class="rule-line" marker-end="url(#arrow)"/>
      <text x="14" y="168" class="lbl x-fill" font-size="12">g</text>
      <text x="74" y="172" class="lbl x-fill" font-size="12">g</text>
      <text x="112" y="168" class="lbl x-fill" font-size="12">g</text>

      <circle cx="20" cy="222" r="18" class="latent-node"/>
      <text x="20" y="227" text-anchor="middle" class="lbl on-node" font-size="12">s</text>
      <circle cx="70" cy="222" r="18" class="latent-node"/>
      <text x="70" y="227" text-anchor="middle" class="lbl on-node" font-size="12">s</text>
      <circle cx="120" cy="222" r="18" class="latent-node"/>
      <text x="120" y="227" text-anchor="middle" class="lbl on-node" font-size="12">s</text>

      <!-- f off each depth-1 latent -->
      <line x1="20" y1="240" x2="20" y2="270" class="rule-line" marker-end="url(#arrow)"/>
      <line x1="70" y1="240" x2="70" y2="270" class="rule-line" marker-end="url(#arrow)"/>
      <line x1="120" y1="240" x2="120" y2="270" class="rule-line" marker-end="url(#arrow)"/>
      <text x="4" y="258" class="lbl o-fill" font-size="11">f</text>
      <text x="54" y="258" class="lbl o-fill" font-size="11">f</text>
      <text x="104" y="258" class="lbl o-fill" font-size="11">f</text>
      <rect x="0" y="272" width="40" height="26" rx="4" class="predict-box"/>
      <rect x="50" y="272" width="40" height="26" rx="4" class="predict-box"/>
      <rect x="100" y="272" width="40" height="26" rx="4" class="predict-box"/>
      <text x="20" y="289" text-anchor="middle" class="lbl" font-size="9">π, v</text>
      <text x="70" y="289" text-anchor="middle" class="lbl" font-size="9">π, v</text>
      <text x="120" y="289" text-anchor="middle" class="lbl" font-size="9">π, v</text>

      <!-- one more level of g, expand the middle child only, to depth 2 -->
      <line x1="60" y1="238" x2="30" y2="330" class="rule-line" marker-end="url(#arrow)"/>
      <line x1="80" y1="238" x2="110" y2="330" class="rule-line" marker-end="url(#arrow)"/>
      <text x="42" y="267" class="lbl x-fill" font-size="11">g</text>
      <text x="90" y="267" class="lbl x-fill" font-size="11">g</text>

      <circle cx="30" cy="350" r="16" class="latent-node"/>
      <text x="30" y="355" text-anchor="middle" class="lbl on-node" font-size="11">s</text>
      <circle cx="110" cy="350" r="16" class="latent-node"/>
      <text x="110" y="355" text-anchor="middle" class="lbl on-node" font-size="11">s</text>

      <line x1="30" y1="366" x2="30" y2="392" class="rule-line" marker-end="url(#arrow)"/>
      <line x1="110" y1="366" x2="110" y2="392" class="rule-line" marker-end="url(#arrow)"/>
      <text x="16" y="382" class="lbl o-fill" font-size="10">f</text>
      <text x="96" y="382" class="lbl o-fill" font-size="10">f</text>
      <rect x="12" y="394" width="36" height="22" rx="4" class="predict-box"/>
      <rect x="92" y="394" width="36" height="22" rx="4" class="predict-box"/>
      <text x="30" y="409" text-anchor="middle" class="lbl" font-size="8">π, v</text>
      <text x="110" y="409" text-anchor="middle" class="lbl" font-size="8">π, v</text>

      <!-- explanatory column on the right -->
      <line x1="300" y1="8" x2="300" y2="420" class="rule-line-static"/>
      <text x="322" y="26" class="lbl faint" font-size="12">h runs once: real board → s⁰, at the root only.</text>
      <text x="322" y="50" class="lbl faint" font-size="12">g runs at every edge below the root, latent → latent,</text>
      <text x="322" y="68" class="lbl faint" font-size="12">predicting a reward and never touching the board again.</text>
      <text x="322" y="92" class="lbl faint" font-size="12">f runs at every node, root included: latent → (policy, value).</text>
      <text x="322" y="116" class="lbl faint" font-size="12">Below the root there is no legality oracle and no terminal</text>
      <text x="322" y="134" class="lbl faint" font-size="12">check — every one of the nine actions gets a child, and the</text>
      <text x="322" y="152" class="lbl faint" font-size="12">search leans entirely on f's value head to discover which</text>
      <text x="322" y="170" class="lbl faint" font-size="12">branches are worthless.</text>
      <rect x="322" y="196" width="330" height="52" rx="6" class="board-box" style="fill:none;"/>
      <text x="336" y="216" class="lbl x-fill" font-size="12">The observation enters exactly once —</text>
      <text x="336" y="234" class="lbl x-fill" font-size="12">at the root, through h. Everywhere else is latent.</text>

      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" class="arrow-fill"/>
        </marker>
      </defs>
    </svg>
  </div>
  <figcaption>The three functions and how search uses them. <code>h</code>
  converts the real board into a latent exactly once, at the root.
  <code>g</code> walks the tree downward, latent to latent, with no board
  in sight. <code>f</code> reads off a policy and a value at every node,
  root included.</figcaption>
</figure>
""".strip()


def diagram_training_graph() -> str:
    """Diagram 2: the unrolled training graph.

    Observation into h into s0, then K dynamics steps across, with a
    prediction head dropping down from every latent to its three
    losses. Marks where the 1/K loss scaling and the half gradient
    apply.
    """
    steps = 5
    x0 = 70
    dx = 118
    y_lat = 90
    y_pred = 190
    y_loss = 300

    svg_parts = []
    svg_parts.append(f'<text x="{x0 - 40}" y="20" class="lbl faint" font-size="12">observation</text>')
    svg_parts.append(
        f'<line x1="{x0 - 40}" y1="28" x2="{x0}" y2="{y_lat - 22}" class="rule-line" marker-end="url(#arrow2)"/>'
    )
    svg_parts.append(f'<text x="{x0 - 30}" y="55" class="lbl x-fill" font-size="12">h</text>')

    for k in range(steps + 1):
        cx = x0 + k * dx
        svg_parts.append(f'<circle cx="{cx}" cy="{y_lat}" r="22" class="latent-node"/>')
        svg_parts.append(
            f'<text x="{cx}" y="{y_lat + 5}" text-anchor="middle" class="lbl on-node">s{"⁰¹²³⁴⁵"[k]}</text>'
        )
        # f drops down to a prediction head
        svg_parts.append(
            f'<line x1="{cx}" y1="{y_lat + 22}" x2="{cx}" y2="{y_pred - 20}" class="rule-line" marker-end="url(#arrow2)"/>'
        )
        svg_parts.append(f'<text x="{cx + 6}" y="{y_lat + 45}" class="lbl o-fill" font-size="12">f</text>')
        svg_parts.append(f'<rect x="{cx - 44}" y="{y_pred - 20}" width="88" height="34" rx="5" class="predict-box"/>')
        svg_parts.append(
            f'<text x="{cx}" y="{y_pred + 2}" text-anchor="middle" class="lbl" font-size="11">π{"⁰¹²³⁴⁵"[k]} v{"⁰¹²³⁴⁵"[k]}</text>'
        )
        # loss box
        svg_parts.append(
            f'<line x1="{cx}" y1="{y_pred + 14}" x2="{cx}" y2="{y_loss - 18}" class="rule-line" marker-end="url(#arrow2)"/>'
        )
        scale_label = "loss ×1" if k == 0 else "loss ×1/K"
        svg_parts.append(f'<rect x="{cx - 52}" y="{y_loss - 18}" width="104" height="36" rx="5" class="loss-box"/>')
        svg_parts.append(
            f'<text x="{cx}" y="{y_loss - 3}" text-anchor="middle" class="lbl" font-size="10">&#960;, v, r</text>'
        )
        svg_parts.append(
            f'<text x="{cx}" y="{y_loss + 11}" text-anchor="middle" class="lbl x-fill" font-size="10">{scale_label}</text>'
        )

        if k < steps:
            svg_parts.append(
                f'<line x1="{cx + 22}" y1="{y_lat}" x2="{cx + dx - 22}" y2="{y_lat}" class="rule-line" marker-end="url(#arrow2)"/>'
            )
            svg_parts.append(f'<text x="{cx + dx / 2 - 6}" y="{y_lat - 10}" class="lbl x-fill" font-size="12">g</text>')
            svg_parts.append(
                f'<text x="{cx + dx / 2 - 24}" y="{y_lat + 22}" class="lbl faint" font-size="9">×½ grad</text>'
            )

    total_width = x0 + steps * dx + 90
    body = "\n      ".join(svg_parts)

    return f"""
<figure class="diagram">
  <div class="fig-frame">
    <svg viewBox="0 0 {total_width} 350" role="img" aria-labelledby="diag2-title diag2-desc"
         style="max-width:100%; height:auto; font-family:'IBM Plex Mono',monospace;">
      <title id="diag2-title">The unrolled training graph</title>
      <desc id="diag2-desc">Observation into representation into latent s0,
        then K dynamics steps across to s1 through s5, with a prediction
        head dropping down from every latent to its policy, value and
        reward loss. Step 0's loss counts fully; every later step's loss
        is scaled by 1/K. Gradient flowing back across each dynamics step
        is halved.</desc>
      {body}
      <defs>
        <marker id="arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" class="arrow-fill"/>
        </marker>
      </defs>
    </svg>
  </div>
  <figcaption>The unrolled training graph. Every arrow crossing a
  <code>g</code> step forward also carries a gradient backward scaled by
  the half gradient (<code>×½</code>); the loss at step 0 counts fully and
  every later step's loss is scaled by <code>1/K</code>.</figcaption>
</figure>
""".strip()


DIAGRAM_STYLE = """
/* ---- additions for docs/algorithm-explained.html, not part of the ---- */
/* ---- stylesheet copied verbatim from the sibling project above.      - */
/* ---- these style the two SVG diagrams. The page shell, the sticky   - */
/* ---- table of contents and the masthead all come from the copied    - */
/* ---- stylesheet above, which already defines them.                  - */
/* A long, hyphen-free identifier (a test name, a namespaced symbol) has
   no natural break point; without this it can force horizontal page
   scroll on a narrow window even though normal prose wraps fine. */
p, li, blockquote p, figcaption, td, th{ overflow-wrap:anywhere; }
blockquote{
  border-left:3px solid var(--rule-strong);
  margin:18px 0;
  padding:2px 18px;
  color:var(--ink-dim);
  max-width:64ch;
}
blockquote p{ margin:0 0 12px; }
blockquote p:last-child{ margin-bottom:0; }
figure.diagram{ margin:28px 0; }
figure.diagram .fig-frame{
  background:var(--surface);
  border:1px solid var(--rule);
  border-radius:8px;
  padding:18px;
  box-shadow:var(--shadow);
  overflow-x:auto;
}
figure.diagram svg{ display:block; max-width:100%; height:auto; }
figure.diagram figcaption{
  margin-top:10px;
  font-size:13px;
  color:var(--ink-dim);
  max-width:66ch;
}
svg .lbl{ fill:var(--ink); }
svg .lbl.faint{ fill:var(--ink-faint); }
svg .lbl.x-fill{ fill:var(--x); }
svg .lbl.o-fill{ fill:var(--o); }
svg .lbl.on-node{ fill:var(--surface); font-weight:600; }
svg .rule-line{ stroke:var(--ink-dim); stroke-width:1.5; fill:none; }
svg .rule-line-static{ stroke:var(--rule); stroke-width:1; fill:none; }
svg .arrow-fill{ fill:var(--ink-dim); }
svg .board-box{ fill:var(--x-soft); stroke:var(--x); stroke-width:1.5; }
svg .latent-node{ fill:var(--x); stroke:var(--rule-strong); stroke-width:1; }
svg .latent-node.root-node{ fill:var(--x); stroke:var(--ink); stroke-width:2; }
svg .predict-box{ fill:var(--o-soft); stroke:var(--o); stroke-width:1.2; }
svg .loss-box{ fill:var(--surface-2); stroke:var(--rule-strong); stroke-width:1.2; }
""".strip()


TOC_ITEM = re.compile(r"^-\s+\[(?P<num>\d+)\s*—\s*(?P<title>.+?)\]\((?P<href>#[^)]+)\)\s*$")


def render_toc(list_markdown: str):
    """Turn the markdown "## Contents" list into the sticky sidebar.

    The list stays in the markdown because that is the source of truth and
    reads correctly on its own; here it becomes a nav that stays put while
    the article scrolls, which is what the sibling project's page does.
    Returns None if the list is not in the expected shape, so a malformed
    Contents falls back to rendering inline rather than vanishing.
    """
    items = []
    for line in list_markdown.split("\n"):
        if not line.strip():
            continue
        m = TOC_ITEM.match(line)
        if not m:
            return None
        items.append(
            f'      <li><a href="{m["href"]}"><span class="n">{m["num"]}</span>'
            f'{render_inline(m["title"])}</a></li>'
        )
    if not items:
        return None
    return "\n".join(
        ['  <nav class="toc">', '    <div class="toc-label">Contents</div>', "    <ol>"]
        + items
        + ["    </ol>", "  </nav>"]
    )


# The sibling project's page is the design this one matches: an eyebrow
# with the two player chips, a big title over a dek, a rule-topped meta
# row, numbered section heads with the number set small and grey, a
# concept chip under each heading, and asides as left-ruled tinted
# callouts. Every rule for all of that already lives in the stylesheet
# copied from that project; these builders emit the markup that uses it,
# so the two documents read as a matched pair rather than as two pages
# that happen to share a colour scheme.
EYEBROW = "rlexp / muzero-tictactoe — code walkthrough"

SECTION_HEADING = re.compile(r"^##\s+(?P<num>\d+)\s*—\s*(?P<title>.+)$")
CONCEPT_LINE = re.compile(r"^\*\*Concept:\s*(?P<what>.+?)\*\*$", re.S)
META_SPLIT = re.compile(r"\*\*([^*]+):\*\*")
CALLOUT_LABEL = re.compile(r"^\s*<p><strong>(?P<label>[^<]+?)[.:]</strong>\s*(?P<rest>.*)$", re.S)


def build_masthead(raw_blocks):
    """The title block, from everything above the Contents list."""
    title, dek, meta = None, None, None
    for raw in raw_blocks:
        text = raw.strip()
        if text.startswith("# "):
            title = text[2:].strip()
        elif text.startswith("**Language:**"):
            meta = text
        elif text.startswith("*("):
            # The "a richer, illustrated version exists" note. This IS that
            # version, so it says nothing here.
            continue
        elif dek is None and not text.startswith(("#", "-", ">", "```", "|")):
            dek = " ".join(text.split())

    if title is None or dek is None or meta is None:
        raise RuntimeError("masthead needs a '# title', a lead paragraph and a "
                           "'**Language:** ...' line above '## Contents'")

    parts = META_SPLIT.split(meta)
    spans = []
    for i in range(1, len(parts) - 1, 2):
        label = parts[i].strip()
        value = " ".join(parts[i + 1].split()).strip().rstrip("·").strip()
        spans.append(f"      <span><b>{label}</b> {render_inline(value)}</span>")
    if not spans:
        raise RuntimeError("could not split the '**Language:** ...' line into meta fields")

    return "\n".join([
        f'    <div class="eyebrow"><span class="chip-x"></span>'
        f'<span class="chip-o"></span> {EYEBROW}</div>',
        f'    <h1 class="title">{render_inline(title)}</h1>',
        f'    <p class="dek">{render_inline(dek)}</p>',
        '    <div class="meta-row">',
        *spans,
        "    </div>",
    ])


def as_callout(html_block: str) -> str:
    """Render a blockquote as the sibling's left-ruled tinted callout.

    A leading bolded phrase becomes the small monospace label above the
    body; the stylesheet uppercases it. A blockquote without one still
    becomes a callout, just without a label.
    """
    inner = html_block.strip()
    if not inner.startswith("<blockquote>"):
        return html_block
    inner = inner[len("<blockquote>"):]
    if inner.endswith("</blockquote>"):
        inner = inner[: -len("</blockquote>")]
    inner = inner.strip()

    m = CALLOUT_LABEL.match(inner)
    if m:
        label = m["label"].strip().lower()
        body = "<p>" + m["rest"].lstrip()
        return ('      <div class="callout">\n'
                f'        <span class="cal-label">{label}</span>\n'
                f"        {body}\n"
                "      </div>")
    return f'      <div class="callout">\n        {inner}\n      </div>'


def build_html(markdown_text: str, stylesheet_html: str) -> str:
    blocks = parse_markdown(markdown_text)

    diag1 = diagram_search_tree()
    diag2 = diagram_training_graph()

    # Split the document into the masthead (everything above "## Contents"),
    # the sidebar nav, and the article. The sibling project's page is laid
    # out the same way: a two-column grid with a sticky table of contents.
    toc_html = None
    masthead_raw = []
    rendered = []
    seen_contents = False
    expecting_list = False
    expecting_rule = False
    open_section = False

    diag1_inserted = False
    diag2_inserted = False
    for raw, html_block in blocks:
        stripped = raw.strip()
        if not seen_contents:
            if stripped == "## Contents":
                seen_contents = True
                expecting_list = True
                continue
            masthead_raw.append(raw)
            continue
        if expecting_list:
            expecting_list = False
            candidate = render_toc(stripped)
            if candidate is not None:
                toc_html = candidate
                expecting_rule = True
                continue
            # Unrecognised shape: keep it inline rather than losing it.
            rendered.append(html_block)
        if expecting_rule:
            expecting_rule = False
            if stripped == "---":
                continue   # the rule that separated Contents from the body

        heading = SECTION_HEADING.match(stripped)
        if heading:
            if open_section:
                rendered.append("    </section>")
            # Reuse the id the parser already assigned. Calling slugify
            # again would look like a second heading with the same text and
            # earn a "-1" suffix, silently breaking every sidebar link.
            existing = re.search(r'id="([^"]+)"', html_block)
            if existing is None:
                raise RuntimeError(f"heading block has no id: {stripped[:60]}")
            slug = existing.group(1)
            rendered.append(
                f'    <section>\n'
                f'      <div class="sec-head"><span class="sec-num">{heading["num"]}</span>'
                f'<h2 id="{slug}">{render_inline(heading["title"])}</h2></div>'
            )
            open_section = True
            continue

        concept = CONCEPT_LINE.match(stripped)
        if concept:
            what = " ".join(concept["what"].split())
            rendered.append(f'      <span class="concept">concept — {render_inline(what)}</span>')
            continue

        if stripped.startswith(">"):
            rendered.append(as_callout(html_block))
            continue

        rendered.append(html_block)
        if not diag1_inserted and "Everything below is latents." in raw:
            rendered.append(diag1)
            diag1_inserted = True
        if not diag2_inserted and "the other keeps gradients from compounding backward." in raw:
            rendered.append(diag2)
            diag2_inserted = True

    if not diag1_inserted or not diag2_inserted:
        raise RuntimeError("diagram insertion anchor not found -- markdown source changed?")
    if open_section:
        rendered.append("    </section>")
    if toc_html is None:
        raise RuntimeError('could not build the sidebar from the "## Contents" list -- '
                           "has its format changed?")

    article = "\n\n".join(rendered)
    masthead_html = build_masthead(masthead_raw)

    # Insert our additional CSS right before the closing </style> so the
    # copied block above it remains byte-for-byte verbatim.
    stylesheet_html = stylesheet_html.replace(
        "</style>", f"{DIAGRAM_STYLE}\n</style>", 1
    )

    # A real document, not a bare fragment. Without a declared charset a
    # browser has to guess, and Safari guesses a legacy encoding for local
    # files -- which turns every em-dash, arrow and subscript in this
    # document into mojibake. Chrome happens to sniff UTF-8 and looks fine,
    # which is what makes the bug easy to miss. The meta must also sit in
    # the first 1024 bytes to be honoured, so it goes first.
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MuZero from Scratch</title>
{stylesheet_html}
</head>
<body>
<div class="shell">
  <div class="masthead">
{masthead_html}
  </div>
{toc_html}
  <main>
{article}
  </main>
</div>
</body>
</html>
"""
    return page


def main():
    if not MD_PATH.exists():
        print(f"error: {MD_PATH} not found", file=sys.stderr)
        return 1
    if not SIBLING_HTML_PATH.exists():
        print(f"error: sibling stylesheet {SIBLING_HTML_PATH} not found", file=sys.stderr)
        return 1

    sibling_lines = SIBLING_HTML_PATH.read_text(encoding="utf-8").split("\n")
    try:
        link_index = next(i for i, l in enumerate(sibling_lines)
                          if l.lstrip().startswith("<link rel=\"stylesheet\""))
        style_start = next(i for i, l in enumerate(sibling_lines)
                           if l.lstrip().startswith("<style"))
        style_end = next(i for i, l in enumerate(sibling_lines)
                         if l.lstrip().startswith("</style>"))
    except StopIteration:
        print("error: could not find the font <link> and <style> block in "
              f"{SIBLING_HTML_PATH}", file=sys.stderr)
        return 1
    if not style_start < style_end:
        print("error: sibling <style> block looks malformed", file=sys.stderr)
        return 1
    stylesheet_html = "\n".join(
        [sibling_lines[link_index]] + sibling_lines[style_start:style_end + 1]
    )

    markdown_text = MD_PATH.read_text(encoding="utf-8")
    page = build_html(markdown_text, stylesheet_html)
    HTML_PATH.write_text(page, encoding="utf-8")
    print(f"wrote {HTML_PATH} ({page.count(chr(10)) + 1} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
