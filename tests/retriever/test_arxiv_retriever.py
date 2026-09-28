"""Tests for ArxivRetriever."""

import time
from types import SimpleNamespace

import zotero_arxiv_daily.retriever.arxiv_retriever as arxiv_retriever
from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever, _run_with_hard_timeout


def _sleep_and_return(value: str, delay_seconds: float) -> str:
    time.sleep(delay_seconds)
    return value


def _raise_runtime_error() -> None:
    raise RuntimeError("boom")


# ``_run_with_hard_timeout`` runs its callable in a fresh ``multiprocessing``
# process.  Where the ``fork`` start method is unavailable (Windows, and macOS
# with Python >= 3.8) the child must re-execute the interpreter and re-import
# the project's native dependencies before it can run anything, which costs
# several seconds even for a trivial callable.  Tests that exercise the *happy*
# path therefore use a budget well above that startup cost, so they assert
# behaviour rather than process-startup latency.  The timeout branch keeps a
# deliberately tiny budget, which is precise on every platform.
_STARTUP_SAFE_TIMEOUT = 30


def test_arxiv_retriever(config, mock_feedparser, monkeypatch):
    # The RSS fixture gives us paper IDs.  After feedparser, the code calls
    # arxiv.Client().results(search) which makes real HTTP requests.  We mock
    # the arxiv Client so the test stays offline.
    new_entries = [e for e in mock_feedparser.entries if e.get("arxiv_announce_type", "new") == "new"]

    # Build fake ArxivResult-like objects matching each RSS entry
    fake_results = []
    for entry in new_entries:
        pid = entry.id.removeprefix("oai:arXiv.org:")
        fake_results.append(
            SimpleNamespace(
                title=entry.title,
                authors=[SimpleNamespace(name="Test Author")],
                summary="Test abstract",
                pdf_url=f"https://arxiv.org/pdf/{pid}",
                entry_id=f"https://arxiv.org/abs/{pid}",
                source_url=lambda pid=pid: f"https://arxiv.org/e-print/{pid}",
            )
        )

    class FakeClient:
        def __init__(self, **kw):
            pass

        def results(self, search):
            return iter(fake_results)

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)

    # Skip file downloads in convert_to_paper
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", lambda paper: None)

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == len(new_entries)
    assert set(p.title for p in papers) == set(e.title for e in new_entries)


def test_run_with_hard_timeout_returns_value():
    result = _run_with_hard_timeout(
        _sleep_and_return,
        ("done", 0.01),
        timeout=_STARTUP_SAFE_TIMEOUT,
        operation="test op",
        paper_title="paper",
    )
    assert result == "done"


def test_run_with_hard_timeout_returns_none_on_timeout(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 1.0), timeout=0.01, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "timed out" in warnings[0]


def test_run_with_hard_timeout_returns_none_on_failure(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _raise_runtime_error,
        (),
        timeout=_STARTUP_SAFE_TIMEOUT,
        operation="test op",
        paper_title="paper",
    )
    assert result is None
    assert "boom" in warnings[0]
