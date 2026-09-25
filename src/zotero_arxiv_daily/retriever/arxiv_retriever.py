import multiprocessing
import os
from collections.abc import Callable
from queue import Empty
from tempfile import TemporaryDirectory
from time import sleep
from typing import Any

import arxiv
import feedparser
import requests
from arxiv import Result as ArxivResult
from loguru import logger
from tqdm import tqdm

from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from .base import BaseRetriever, register_retriever

DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180


def _download_file(url: str, path: str) -> None:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _run_in_subprocess[T](
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout[T](
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context(
        "fork" if "fork" in start_methods else start_methods[0]
    )

    result_queue = context.Queue()
    process = context.Process(
        target=_run_in_subprocess,
        args=(result_queue, func, args),
    )
    process.start()

    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()

        process.join(5)
        result_queue.close()
        result_queue.join_thread()

        logger.warning(
            f"{operation} timed out for {paper_title} "
            f"after {timeout} seconds"
        )
        return None

    process.join(5)
    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(
        f"{operation} failed for {paper_title}: {payload}"
    )
    return None


def _extract_text_from_pdf_worker(pdf_url: str) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        _download_file(pdf_url, path)
        return extract_markdown_from_pdf(path)


def _extract_text_from_html_worker(html_url: str) -> str | None:
    import trafilatura

    downloaded = trafilatura.fetch_url(html_url)

    if downloaded is None:
        raise ValueError(
            f"Failed to download HTML from {html_url}"
        )

    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
    )

    if not text:
        raise ValueError(
            f"No text extracted from {html_url}"
        )

    return text


def _extract_text_from_tar_worker(
    source_url: str,
    paper_id: str,
    paper_title: str | None = None,
) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")

        _download_file(source_url, path)

        file_contents = extract_tex_code_from_tar(
            path,
            paper_id,
            paper_title=paper_title,
        )

        if not file_contents or "all" not in file_contents:
            raise ValueError("Main tex file not found.")

        return file_contents["all"]


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    # convert_to_paper downloads full text via subprocess + network,
    # so conversion must stay serial to avoid overloading arXiv.
    concurrency_safe = False

    def __init__(self, config):
        super().__init__(config)

        if self.config.source.arxiv.category is None:
            raise ValueError(
                "category must be specified for arxiv."
            )

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        """
        Retrieve latest papers from arXiv.

        Strategy:
        1. Read paper IDs from arXiv RSS.
        2. Query arXiv API in batches of 20.
        3. Retry HTTP 429 with backoff.
        4. If a batch fails with 406 or another HTTP error,
           automatically fall back to per-paper requests.
        5. If a single paper still fails, skip it instead of
           crashing the whole workflow.
        """

        client = arxiv.Client(
            num_retries=10,
            delay_seconds=10,
        )

        query = "+".join(
            self.config.source.arxiv.category
        )

        include_cross_list = (
            self.config.source.arxiv.get(
                "include_cross_list",
                False,
            )
        )

        # --------------------------------------------------
        # 1. Get latest paper IDs from arXiv RSS feed
        # --------------------------------------------------
        feed_url = f"https://rss.arxiv.org/atom/{query}"

        logger.info(
            f"Fetching arXiv RSS feed: {feed_url}"
        )

        feed = feedparser.parse(feed_url)

        feed_title = getattr(
            feed.feed,
            "title",
            "",
        )

        if "Feed error for query" in feed_title:
            raise Exception(
                f"Invalid ARXIV_QUERY: {query}."
            )

        allowed_announce_types = (
            {"new", "cross"}
            if include_cross_list
            else {"new"}
        )

        all_paper_ids = [
            entry.id.removeprefix(
                "oai:arXiv.org:"
            )
            for entry in feed.entries
            if entry.get(
                "arxiv_announce_type",
                "new",
            )
            in allowed_announce_types
        ]

        if self.config.executor.debug:
            all_paper_ids = all_paper_ids[:10]

        logger.info(
            f"Found {len(all_paper_ids)} "
            "arXiv paper IDs from RSS"
        )

        if not all_paper_ids:
            logger.warning(
                "No arXiv papers found from RSS feed."
            )
            return []

        # --------------------------------------------------
        # 2. Query metadata from arXiv API
        # --------------------------------------------------
        raw_papers: list[ArxivResult] = []

        bar = tqdm(
            total=len(all_paper_ids),
            desc="Retrieving arXiv papers",
        )

        batch_size = 20

        max_batch_retries = 5
        batch_retry_delay = 30

        batch_interval = 3
        single_paper_interval = 1

        for i in range(
            0,
            len(all_paper_ids),
            batch_size,
        ):
            batch_ids = all_paper_ids[
                i : i + batch_size
            ]

            batch_number = (
                i // batch_size
            ) + 1

            total_batches = (
                len(all_paper_ids)
                + batch_size
                - 1
            ) // batch_size

            logger.info(
                f"Retrieving arXiv batch "
                f"{batch_number}/{total_batches} "
                f"({len(batch_ids)} papers)"
            )

            search = arxiv.Search(
                id_list=batch_ids
            )

            batch_completed = False

            # ----------------------------------------------
            # Try normal batch request
            # ----------------------------------------------
            for attempt in range(
                max_batch_retries
            ):
                try:
                    batch = list(
                        client.results(search)
                    )

                    raw_papers.extend(batch)

                    # Update by requested IDs,
                    # not returned results.
                    bar.update(len(batch_ids))

                    logger.info(
                        f"arXiv batch "
                        f"{batch_number} succeeded: "
                        f"{len(batch)}/"
                        f"{len(batch_ids)} papers returned"
                    )

                    batch_completed = True
                    break

                except arxiv.HTTPError as exc:
                    status = getattr(
                        exc,
                        "status",
                        None,
                    )

                    # --------------------------------------
                    # HTTP 429: rate limited
                    # --------------------------------------
                    if status == 429:
                        if (
                            attempt
                            < max_batch_retries - 1
                        ):
                            wait = (
                                batch_retry_delay
                                * (attempt + 1)
                            )

                            logger.warning(
                                f"arXiv API HTTP 429 "
                                f"on batch "
                                f"{batch_number}. "
                                f"Retry "
                                f"{attempt + 1}/"
                                f"{max_batch_retries} "
                                f"in {wait}s"
                            )

                            sleep(wait)
                            continue

                        logger.warning(
                            f"arXiv API HTTP 429 "
                            f"on batch "
                            f"{batch_number} "
                            f"after "
                            f"{max_batch_retries} "
                            "attempts. "
                            "Falling back to "
                            "per-paper requests."
                        )

                    # --------------------------------------
                    # HTTP 406 / 403 / 5xx / etc.
                    # --------------------------------------
                    else:
                        logger.warning(
                            f"arXiv API batch request "
                            f"failed on batch "
                            f"{batch_number} "
                            f"with HTTP {status}. "
                            "Falling back to "
                            "per-paper requests."
                        )

                    # --------------------------------------
                    # Fallback:
                    # retrieve papers one by one
                    # --------------------------------------
                    fallback_batch: list[
                        ArxivResult
                    ] = []

                    for index, paper_id in enumerate(
                        batch_ids
                    ):
                        logger.info(
                            f"Fallback arXiv request "
                            f"{index + 1}/"
                            f"{len(batch_ids)}: "
                            f"{paper_id}"
                        )

                        try:
                            single_search = (
                                arxiv.Search(
                                    id_list=[
                                        paper_id
                                    ]
                                )
                            )

                            single_results = list(
                                client.results(
                                    single_search
                                )
                            )

                            if single_results:
                                fallback_batch.extend(
                                    single_results
                                )

                            else:
                                logger.warning(
                                    "No arXiv result "
                                    f"returned for "
                                    f"{paper_id}"
                                )

                        except arxiv.HTTPError as (
                            paper_exc
                        ):
                            paper_status = getattr(
                                paper_exc,
                                "status",
                                None,
                            )

                            logger.warning(
                                "Skipping arXiv "
                                f"paper {paper_id} "
                                "because the API "
                                "returned HTTP "
                                f"{paper_status}"
                            )

                        except Exception as (
                            paper_exc
                        ):
                            logger.warning(
                                "Skipping arXiv "
                                f"paper {paper_id} "
                                "because of "
                                f"{type(paper_exc).__name__}: "
                                f"{paper_exc}"
                            )

                        if (
                            index + 1
                            < len(batch_ids)
                        ):
                            sleep(
                                single_paper_interval
                            )

                    raw_papers.extend(
                        fallback_batch
                    )

                    # Mark entire batch attempted,
                    # even if a few papers were skipped.
                    bar.update(len(batch_ids))

                    logger.info(
                        f"arXiv fallback "
                        f"completed for batch "
                        f"{batch_number}: "
                        f"{len(fallback_batch)}/"
                        f"{len(batch_ids)} "
                        "papers retrieved"
                    )

                    batch_completed = True
                    break

                except Exception as exc:
                    # Unexpected errors should not
                    # silently kill the whole run.
                    logger.exception(
                        "Unexpected error while "
                        f"retrieving arXiv batch "
                        f"{batch_number}: "
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )

                    # Count this batch as attempted.
                    bar.update(len(batch_ids))

                    batch_completed = True
                    break

            if not batch_completed:
                logger.warning(
                    f"arXiv batch "
                    f"{batch_number} "
                    "could not be completed."
                )

                bar.update(len(batch_ids))

            # Avoid querying arXiv too aggressively
            if (
                i + batch_size
                < len(all_paper_ids)
            ):
                sleep(batch_interval)

        bar.close()

        logger.info(
            f"Successfully retrieved "
            f"{len(raw_papers)} "
            "arXiv papers in total"
        )

        return raw_papers

    def convert_to_paper(
        self,
        raw_paper: ArxivResult,
    ) -> Paper:
        title = raw_paper.title

        authors = [
            author.name
            for author in raw_paper.authors
        ]

        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url

        # ----------------------------------------------
        # Try source tar first
        # ----------------------------------------------
        full_text = extract_text_from_tar(
            raw_paper
        )

        # ----------------------------------------------
        # Then HTML
        # ----------------------------------------------
        if full_text is None:
            full_text = extract_text_from_html(
                raw_paper
            )

        # ----------------------------------------------
        # Finally PDF
        # ----------------------------------------------
        if full_text is None:
            full_text = extract_text_from_pdf(
                raw_paper
            )

        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text,
        )


def extract_text_from_html(
    paper: ArxivResult,
) -> str | None:
    html_url = paper.entry_id.replace(
        "/abs/",
        "/html/",
    )

    try:
        return _extract_text_from_html_worker(
            html_url
        )

    except Exception as exc:
        logger.warning(
            f"HTML extraction failed "
            f"for {paper.title}: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )
        return None


def extract_text_from_pdf(
    paper: ArxivResult,
) -> str | None:
    if paper.pdf_url is None:
        logger.warning(
            f"No PDF URL available "
            f"for {paper.title}"
        )
        return None

    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (paper.pdf_url,),
        timeout=PDF_EXTRACT_TIMEOUT,
        operation="PDF extraction",
        paper_title=paper.title,
    )


def extract_text_from_tar(
    paper: ArxivResult,
) -> str | None:
    source_url = paper.source_url()

    if source_url is None:
        logger.warning(
            f"No source URL available "
            f"for {paper.title}"
        )
        return None

    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (
            source_url,
            paper.entry_id,
            paper.title,
        ),
        timeout=TAR_EXTRACT_TIMEOUT,
        operation="Tar extraction",
        paper_title=paper.title,
    )
