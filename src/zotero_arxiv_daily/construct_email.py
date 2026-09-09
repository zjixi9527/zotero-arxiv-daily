"""HTML email rendering for Zotero-arXiv-Daily.

This module turns a list of :class:`Paper` objects into the final HTML body that
is emailed to the user.

Safety note (Q3): every field that originates from external data (paper title,
authors, affiliations, TLDR text, URLs) is HTML-escaped before being embedded in
the template. Previously the template used ``str.format`` with raw values, which
(a) crashed with a ``KeyError`` whenever a title happened to contain ``{`` or
``}`` (common in LaTeX/math-heavy titles), and (b) left fields unescaped, which
could both inject markup and corrupt the HTML. Escaping fixes both.
"""

import html
import math

from .protocol import Paper


def _e(text: object) -> str:
    """HTML-escape an arbitrary external value for safe embedding in markup.

    ``None`` renders as an empty string. Every paper-provided field should pass
    through this helper before it lands in the template.
    """
    if text is None:
        return ""
    return html.escape(str(text), quote=False)


framework = """<!DOCTYPE HTML>
<html>
<head>
  <style>
    .star-wrapper {
      font-size: 1.3em; /* 调整星星大小 */
      line-height: 1; /* 确保垂直对齐 */
      display: inline-flex;
      align-items: center; /* 保持对齐 */
    }
    .half-star {
      display: inline-block;
      width: 0.5em; /* 半颗星的宽度 */
      overflow: hidden;
      white-space: nowrap;
      vertical-align: middle;
    }
    .full-star {
      vertical-align: middle;
    }
  </style>
</head>
<body>

<div>
    __CONTENT__
</div>

<br><br>
<div>
To unsubscribe, remove your email in your Github Action setting.
</div>

</body>
</html>
"""


# Block template with ``{name}`` placeholders. Placeholders are substituted with
# *already-escaped* values via str.replace (not str.format), so literal ``{`` or
# ``}`` inside content can never collide with the template or raise an error.
_BLOCK_TEMPLATE = """\
<table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
<tr>
    <td style="font-size: 20px; font-weight: bold; color: #333;">
        {title}
    </td>
</tr>
<tr>
    <td style="font-size: 14px; color: #666; padding: 8px 0;">
        {authors}
        <br>
        <i>{affiliations}</i>
    </td>
</tr>
<tr>
    <td style="font-size: 14px; color: #333; padding: 8px 0;">
        <strong>Relevance:</strong> {rate}
    </td>
</tr>
<tr>
    <td style="font-size: 14px; color: #333; padding: 8px 0;">
        <strong>TLDR:</strong> {tldr}
    </td>
</tr>
<tr>
    <td style="padding: 8px 0;">
        <a href="{pdf_url}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #d9534f; padding: 8px 16px; border-radius: 4px;">PDF</a>
    </td>
</tr>
</table>
"""


def get_empty_html():
    block_template = """
    <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
  <tr>
    <td style="font-size: 20px; font-weight: bold; color: #333;">
        No Papers Today. Take a Rest!
    </td>
  </tr>
  </table>
  """
    return block_template


def _render_single_block(
    title,
    authors,
    rate,
    tldr,
    pdf_url,
    affiliations=None,
):
    """Render one paper block, escaping every external field.

    Falls back to safe ``str.replace``-based substitution instead of
    ``str.format``, so literal ``{``/``}`` in content cannot crash the render.
    ``affiliations=None`` renders as an empty string.
    """
    out = _BLOCK_TEMPLATE
    substitutions = (
        ("title", _e(title)),
        ("authors", _e(authors)),
        ("affiliations", _e(affiliations)),
        ("rate", _e(rate)),
        ("tldr", _e(tldr)),
        # PDF href is an attribute context: escape double quotes too.
        ("pdf_url", html.escape(str(pdf_url), quote=True)),
    )
    for key, value in substitutions:
        out = out.replace("{" + key + "}", value)
    return out


def get_block_html(title, authors, rate, tldr, pdf_url, affiliations=None):
    """Compatibility wrapper kept for existing callers and tests.

    Delegates to :func:`_render_single_block`, which applies HTML escaping to all
    fields. ``affiliations=None`` renders as empty markup.
    """
    return _render_single_block(title, authors, rate, tldr, pdf_url, affiliations)


def get_stars(score: float):
    full_star = '<span class="full-star">⭐</span>'
    half_star = '<span class="half-star">⭐</span>'
    low = 6
    high = 8
    if score <= low:
        return ""
    elif score >= high:
        return full_star * 5
    else:
        interval = (high - low) / 10
        star_num = math.ceil((score - low) / interval)
        full_star_num = int(star_num / 2)
        half_star_num = star_num - full_star_num * 2
        return '<div class="star-wrapper">' + full_star * full_star_num + half_star * half_star_num + "</div>"


def _affiliation_text(affiliations):
    if not affiliations:
        return "Unknown Affiliation"
    shown = affiliations[:5]
    text = ", ".join(str(a) for a in shown)
    if len(affiliations) > 5:
        text += ", ..."
    return text


def render_email(papers: list[Paper]) -> str:
    parts = []
    if len(papers) == 0:
        return framework.replace("__CONTENT__", get_empty_html())

    for p in papers:
        # rate = get_stars(p.score)
        rate = round(p.score, 1) if p.score is not None else "Unknown"
        author_list = [a for a in p.authors]
        num_authors = len(author_list)
        if num_authors <= 5:
            authors = ", ".join(author_list)
        else:
            authors = ", ".join(author_list[:3] + ["..."] + author_list[-2:])
        parts.append(
            _render_single_block(
                title=p.title,
                authors=authors,
                rate=str(rate),
                tldr=p.tldr or "",
                pdf_url=p.pdf_url,
                affiliations=_affiliation_text(p.affiliations),
            )
        )

    content = "<br>" + "</br><br>".join(parts) + "</br>"
    return framework.replace("__CONTENT__", content)
