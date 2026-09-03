from datetime import datetime, timedelta, timezone
import os

import requests
from loguru import logger

from .base import BaseRetriever, register_retriever
from ..protocol import Paper


OPENALEX_WORKS_URL = "https://api.openalex.org/works"
REQUEST_TIMEOUT = 30


def reconstruct_abstract(
    inverted_index: dict | None,
) -> str:
    """
    Reconstruct plain-text abstract from
    OpenAlex abstract_inverted_index.
    """

    if not inverted_index:
        return ""

    positions = []

    for word, indices in inverted_index.items():
        for index in indices:
            positions.append(
                (index, word)
            )

    positions.sort(
        key=lambda item: item[0]
    )

    return " ".join(
        word
        for _, word in positions
    )


@register_retriever("openalex")
class OpenAlexRetriever(BaseRetriever):

    def __init__(self, config):
        super().__init__(config)

        self.api_key = os.getenv(
            "OPENALEX_API_KEY"
        )

        self.lookback_days = int(
            self.retriever_config.get(
                "lookback_days",
                7,
            )
        )

        self.per_page = min(
            int(
                self.retriever_config.get(
                    "per_page",
                    50,
                )
            ),
            100,
        )

        self.tracked_authors = (
            self.retriever_config.get(
                "tracked_authors"
            )
            or []
        )

        # Zotero papers used to define
        # the user's current research interests.
        self.corpus = []

        self.semantic_search = bool(
            self.retriever_config.get(
                "semantic_search",
                True,
            )
        )

        self.semantic_seed_count = int(
            self.retriever_config.get(
                "semantic_seed_count",
                5,
            )
        )

        self.semantic_per_page = min(
            int(
                self.retriever_config.get(
                    "semantic_per_page",
                    30,
                )
            ),
            50,
        )

        self.semantic_types = list(
            self.retriever_config.get(
                "semantic_types"
            )
            or ["article"]
        )

    def set_corpus(self, corpus):
        """
        Receive the filtered Zotero corpus
        from Executor.
        """

        self.corpus = list(
            corpus or []
        )

    def _get_headers(self) -> dict:
        headers = {}

        if self.api_key:
            headers["Authorization"] = (
                f"Bearer {self.api_key}"
            )

        return headers

    def _add_unique_works(
        self,
        works: list[dict],
        raw_papers: list[dict],
        seen_work_ids: set,
    ) -> int:
        """
        Add works while removing duplicates
        based on OpenAlex Work ID.
        """

        added = 0

        for work in works:
            work_id = work.get("id")

            if not work_id:
                continue

            if work_id in seen_work_ids:
                continue

            seen_work_ids.add(work_id)
            raw_papers.append(work)

            added += 1

        return added

    def _retrieve_raw_papers(
        self,
    ) -> list[dict]:

        from_date = (
            datetime.now(timezone.utc)
            - timedelta(
                days=self.lookback_days
            )
        ).date().isoformat()

        headers = self._get_headers()

        raw_papers = []
        seen_work_ids = set()

        # =========================================================
        # 1. Track explicitly selected researchers
        # =========================================================

        for author in self.tracked_authors:
            author_id = author.get(
                "openalex_id"
            )

            if not author_id:
                logger.warning(
                    "Skipping OpenAlex author "
                    "without openalex_id: "
                    f"{author}"
                )
                continue

            author_name = author.get(
                "name",
                author_id,
            )

            logger.info(
                "Retrieving recent OpenAlex "
                f"works for {author_name} "
                f"({author_id})"
            )

            params = {
                "filter": (
                    f"authorships.author.id:"
                    f"{author_id},"
                    f"from_publication_date:"
                    f"{from_date}"
                ),
                "sort": (
                    "publication_date:desc"
                ),
                "per_page": self.per_page,
            }

            try:
                response = requests.get(
                    OPENALEX_WORKS_URL,
                    params=params,
                    headers=headers,
                    timeout=REQUEST_TIMEOUT,
                )

                if not response.ok:
                    logger.warning(
                        "OpenAlex author request "
                        f"failed for {author_name}: "
                        f"HTTP {response.status_code} - "
                        f"{response.text[:500]}"
                    )
                    continue

                results = (
                    response.json()
                    .get("results", [])
                )

            except requests.RequestException as exc:
                logger.warning(
                    "OpenAlex author request "
                    f"failed for {author_name}: "
                    f"{exc}"
                )
                continue

            if self.config.executor.debug:
                results = results[:5]

            added = self._add_unique_works(
                results,
                raw_papers,
                seen_work_ids,
            )

            logger.info(
                f"Found {len(results)} recent "
                f"works for {author_name}; "
                f"added {added} unique works"
            )

        # =========================================================
        # 2. Discover papers based on Zotero research interests
        # =========================================================

        if (
            self.semantic_search
            and self.corpus
        ):
            seed_papers = sorted(
                self.corpus,
                key=lambda paper: (
                    paper.added_date
                ),
                reverse=True,
            )[
                : self.semantic_seed_count
            ]

            # Use titles only.
            #
            # This is deliberately simple:
            # OpenAlex does the first semantic search,
            # then Jina uses the whole Zotero corpus
            # for the final ranking.
            seed_titles = [
                (paper.title or "").strip()
                for paper in seed_papers
                if (paper.title or "").strip()
            ]

            semantic_query = " ; ".join(
                seed_titles
            )[:700]

            if semantic_query:
                logger.info(
                    "Searching OpenAlex "
                    "semantically using "
                    f"{len(seed_titles)} "
                    "recent Zotero paper titles"
                )

                filters = [
                    (
                        "from_publication_date:"
                        f"{from_date}"
                    ),
                    "has_abstract:true",
                ]

                if self.semantic_types:
                    filters.append(
                        "type:"
                        + "|".join(
                            self.semantic_types
                        )
                    )

                params = {
                    "search.semantic": (
                        semantic_query
                    ),
                    "filter": ",".join(
                        filters
                    ),
                    "per_page": (
                        self.semantic_per_page
                    ),
                }

                try:
                    response = requests.get(
                        OPENALEX_WORKS_URL,
                        params=params,
                        headers=headers,
                        timeout=REQUEST_TIMEOUT,
                    )

                    if not response.ok:
                        logger.warning(
                            "OpenAlex semantic "
                            "search failed: "
                            f"HTTP "
                            f"{response.status_code} - "
                            f"{response.text[:1000]}"
                        )

                    else:
                        semantic_results = (
                            response.json()
                            .get(
                                "results",
                                [],
                            )
                        )

                        if (
                            self.config
                            .executor
                            .debug
                        ):
                            semantic_results = (
                                semantic_results[
                                    :10
                                ]
                            )

                        added_count = (
                            self._add_unique_works(
                                semantic_results,
                                raw_papers,
                                seen_work_ids,
                            )
                        )

                        logger.info(
                            "OpenAlex semantic "
                            "search returned "
                            f"{len(semantic_results)} "
                            "works; added "
                            f"{added_count} "
                            "unique works"
                        )

                except requests.RequestException as exc:
                    # Semantic discovery is optional.
                    # Never let it break the entire
                    # daily recommendation workflow.
                    logger.warning(
                        "OpenAlex semantic "
                        f"search failed: {exc}"
                    )

        logger.info(
            f"Retrieved "
            f"{len(raw_papers)} "
            "unique OpenAlex works"
        )

        return raw_papers

    def convert_to_paper(
        self,
        raw_paper: dict,
    ) -> Paper:

        title = (
            raw_paper.get("title")
            or raw_paper.get(
                "display_name"
            )
            or "Untitled"
        )

        authors = []

        for authorship in raw_paper.get(
            "authorships",
            [],
        ):
            author = (
                authorship.get("author")
                or {}
            )

            name = author.get(
                "display_name"
            )

            if name:
                authors.append(name)

        abstract = reconstruct_abstract(
            raw_paper.get(
                "abstract_inverted_index"
            )
        )

        # Jina needs text to embed.
        # If OpenAlex has no abstract,
        # fall back to the title.
        if not abstract:
            abstract = title

        best_oa_location = (
            raw_paper.get(
                "best_oa_location"
            )
            or {}
        )

        primary_location = (
            raw_paper.get(
                "primary_location"
            )
            or {}
        )

        pdf_url = (
            best_oa_location.get(
                "pdf_url"
            )
        )

        url = (
            raw_paper.get("doi")
            or primary_location.get(
                "landing_page_url"
            )
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
