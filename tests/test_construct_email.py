"""Tests for zotero_arxiv_daily.construct_email: render_email, get_stars, get_block_html."""

from tests.canned_responses import make_sample_paper
from zotero_arxiv_daily.construct_email import get_block_html, get_empty_html, get_stars, render_email


def test_render_email_with_papers():
    papers = [make_sample_paper(score=7.5, tldr="A great paper.", affiliations=["MIT"])]
    html = render_email(papers)
    assert "Sample Paper Title" in html
    assert "A great paper." in html
    assert "MIT" in html


def test_render_email_empty_list():
    html = render_email([])
    assert "No Papers Today" in html


def test_render_email_author_truncation():
    authors = [f"Author {i}" for i in range(10)]
    paper = make_sample_paper(authors=authors, score=7.0, tldr="ok")
    html = render_email([paper])
    assert "Author 0" in html
    assert "Author 1" in html
    assert "Author 2" in html
    assert "..." in html
    assert "Author 8" in html
    assert "Author 9" in html
    # Middle authors should be truncated
    assert "Author 5" not in html


def test_render_email_affiliation_truncation():
    affiliations = [f"Uni {i}" for i in range(8)]
    paper = make_sample_paper(affiliations=affiliations, score=7.0, tldr="ok")
    html = render_email([paper])
    assert "Uni 0" in html
    assert "Uni 4" in html
    assert "..." in html
    assert "Uni 7" not in html


def test_render_email_no_affiliations():
    paper = make_sample_paper(affiliations=None, score=7.0, tldr="ok")
    html = render_email([paper])
    assert "Unknown Affiliation" in html


def test_get_stars_low_score():
    assert get_stars(5.0) == ""
    assert get_stars(6.0) == ""


def test_get_stars_high_score():
    stars = get_stars(8.0)
    assert stars.count("full-star") == 5


def test_get_stars_mid_score():
    stars = get_stars(7.0)
    assert "star" in stars
    assert stars.count("full-star") + stars.count("half-star") > 0


def test_get_block_html_contains_all_fields():
    html = get_block_html("Title", "Auth", "3.5", "Summary", "http://pdf.url", "MIT")
    assert "Title" in html
    assert "Auth" in html
    assert "3.5" in html
    assert "Summary" in html
    assert "http://pdf.url" in html
    assert "MIT" in html


def test_get_empty_html():
    html = get_empty_html()
    assert "No Papers Today" in html


# ---------------------------------------------------------------------------
# Q3: HTML-escaping security tests
# ---------------------------------------------------------------------------


def test_render_email_title_with_braces_does_not_crash():
    """A LaTeX/math-heavy title containing { } must not raise KeyError."""
    paper = make_sample_paper(title="Attention {Is All} You Need $x^{2}$", score=7.0, tldr="ok")
    html = render_email([paper])
    assert "Attention" in html
    assert "All" in html


def test_render_email_escapes_html_in_title():
    paper = make_sample_paper(title="<script>alert('xss')</script>", score=7.0, tldr="ok")
    html = render_email([paper])
    # Raw tag must not appear verbatim; escaped form must.
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_render_email_escapes_html_in_affiliations():
    paper = make_sample_paper(
        affiliations=["<b>MIT</b>", '"><img src=x onerror=alert(1)>'],
        score=7.0,
        tldr="ok",
    )
    html = render_email([paper])
    assert "<b>MIT</b>" not in html
    assert "&lt;b&gt;MIT&lt;/b&gt;" in html


def test_render_email_escapes_quotes_in_pdf_url_href():
    """pdf_url is embedded in an attribute (href=...), so quotes must be escaped."""
    paper = make_sample_paper(
        pdf_url='https://arxiv.org/pdf/2026.00001" onmouseover="alert(1)',
        score=7.0,
        tldr="ok",
    )
    html = render_email([paper])
    assert 'href="https://arxiv.org/pdf/2026.00001"' in html or 'href="https:&#x2F;&#x2F;arxiv.org' in html
    # The injected attribute must not survive.
    assert 'onmouseover="alert(1)' not in html


def test_get_block_html_escapes_all_external_fields():
    html = get_block_html(
        "<Title>",
        "Auth<br>or",
        "3.5",
        "<Summary>",
        'http://pdf.url?a=1&b=2"',
        "<MIT>",
    )
    assert "<Title>" not in html
    assert "&lt;Title&gt;" in html
    assert "Auth&lt;br&gt;or" in html
    assert "&lt;Summary&gt;" in html
    assert "&lt;MIT&gt;" in html
