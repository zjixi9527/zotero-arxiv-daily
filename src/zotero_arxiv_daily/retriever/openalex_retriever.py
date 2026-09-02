from datetime import datetime, timedelta, timezone
import os

import requests
from loguru import logger

from .base import BaseRetriever, register_retriever
from ..protocol import Paper


OPENALEX_WORKS_URL = "https://api.openalex.org/works"
REQUEST_TIMEOUT = 30


def reconstruct_abstract(inverted_index: dict | None) -> str:
    """Reconstruct OpenAlex abstract from abstract_inverted_index."""
    if not inverted_index:
        return ""

    positions = []

    for word, indices in inverted_index.items():
        for index in indices:
            positions.append((index, word))

    positions.sort(key=lambda x: x[0])

    return " ".join(word for _, word in positions)


@register_retriever("openalex")
class OpenAlexRetriever(BaseRetriever):

    def __init__(self, config):
        super().__init__(config)

        self.api_key = os.getenv("OPENALEX_API_KEY")

        self.lookback_days = int(
            self.retriever_config.get("lookback_days", 7)
        )

        self.per_page = min(
            int(self.retriever_config.get("per_page", 50)),
            100,
        )

        self.tracked_authors = (
            self.retriever_config.get("tracked_authors") or []
        )

        # Zotero corpus used for semantic discovery
        self.corpus = []

        self.semantic_search = bool(
            self.retriever_config.get("semantic_search", True)
        )

        self.semantic_seed_count = int(
            self.retriever_config.get("semantic_seed_count", 5)
        )

        # OpenAlex semantic search returns at most 50 works
        self.semantic_per_page = min(
            int(
                self.retriever_config.get(
                    "semantic_per_page",
                    50,
                )
            ),
            50,
        )

        self.semantic_types = list(
            self.retriever_config.get("semantic_types")
            or ["article", "conference-paper"]
        )

    def set_corpus(self, corpus):
        """Receive the filtered Zotero corpus from Executor."""
        self.corpus = list(corpus or [])

    def _get_headers(self):
        headers = {}

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        return headers

    def _retrieve_raw_papers(self) -> list[dict]:
        from_date = (
            datetime.now(timezone.utc)
            - timedelta(days=self.lookback_days)
        ).date().isoformat()

        headers = self._get_headers()

        raw_papers = []
        seen_work_ids = set()

        # ---------------------------------------------------------
        # 1. Retrieve recent papers from explicitly tracked authors
        # ---------------------------------------------------------
        for author in self.tracked_authors:
            author_id = author.get("openalex_id")

            if not author_id:
                logger.warning(
                    f"Skipping OpenAlex author without openalex_id: "
                    f"{author}"
                )
                continue

            author_name = author.get("name", author_id)

            logger.info(
                f"Retrieving recent OpenAlex works for "
                f"{author_name} ({author_id})"
            )

            params = {
                "filter": (
                    f"authorships.author.id:{author_id},"
                    f"from_publication_date:{from_date}"
                ),
                "sort": "publication_date:desc",
                "per_page": self.per_page,
            }

            response = requests.get(
                OPENALEX_WORKS_URL,
                params=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )

            response.raise_for_status()

            results = response.json().get("results", [])

            if self.config.executor.debug:
                results = results[:5]

            for work in results:
                work_id = work.get("id")

                if not work_id:
                    continue

                if work_id in seen_work_ids:
                    continue

                seen_work_ids.add(work_id)
                raw_papers.append(work)

            logger.info(
                f"Found {len(results)} recent works for "
                f"{author_name}"
            )

        # ---------------------------------------------------------
        # 2. Semantic discovery based on recent Zotero papers
        # ---------------------------------------------------------
        if self.semantic_search and self.corpus:
            seed_papers = sorted(
                self.corpus,
                key=lambda paper: paper.added_date,
                reverse=True,
            )[:self.semantic_seed_count]

            query_parts = []

            for paper in seed_papers:
                title = (paper.title or "").strip()
                abstract = (paper.abstract or "").strip()

                # Use title plus a compact abstract fragment.
                query_parts.append(
                    f"{title}. {abstract[:350]}"
                )

            # OpenAlex semantic search uses up to 2,000 characters.
            semantic_query = "\n".join(query_parts)[:2000]

            if semantic_query:
                logger.info(
                    f"Searching OpenAlex semantically using "
                    f"{len(seed_papers)} recent Zotero papers"
                )

                filters = [
                    f"from_publication_date:{from_date}",
                    "has_abstract:true",
                ]

                if self.semantic_types:
                    filters.append(
                        "type:" + "|".join(self.semantic_types)
                    )

                params = {
                    "search.semantic": semantic_query,
                    "filter": ",".join(filters),
                    "per_page": self.semantic_per_page,
                }

                response = requests.get(
                    OPENALEX_WORKS_URL,
                    params=params,
                    headers=headers,
                    timeout=REQUEST_TIMEOUT,
                )

                response.raise_for_status()

                semantic_results = response.json().get(
                    "results",
                    [],
                )

                if self.config.executor.debug:
                    semantic_results = semantic_results[:10]

                added_count = 0

                for work in semantic_results:
                    work_id = work.get("id")

                    if not work_id:
                        continue

                    if work_id in seen_work_ids:
                        continue

                    seen_work_ids.add(work_id)
                    raw_papers.append(work)
                    added_count += 1

                logger.info(
                    f"Added {added_count} semantic OpenAlex works"
                )

        logger.info(
            f"Retrieved {len(raw_papers)} unique OpenAlex works"
        )

        return raw_papers

    def convert_to_paper(self, raw_paper: dict) -> Paper:
        title = (
            raw_paper.get("title")
            or raw_paper.get("display_name")
            or "Untitled"
        )

        authors = []

        for authorship in raw_paper.get("authorships", []):
            author = authorship.get("author") or {}
            name = author.get("display_name")

            if name:
                authors.append(name)

        abstract = reconstruct_abstract(
            raw_paper.get("abstract_inverted_index")
        )

        # Keep Jina reranking usable even if abstract is unavailable.
        if not abstract:
            abstract = title

        best_oa_location = (
            raw_paper.get("best_oa_location") or {}
        )

        primary_location = (
            raw_paper.get("primary_location") or {}
        )

        pdf_url = best_oa_location.get("pdf_url")

        url = (
            raw_paper.get("doi")
            or primary_location.get("landing_page_url")
            or raw_paper.get("id")
        )

        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=url,
            pdf_url=pdf_url,
            full_text=None,
        )
