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
                3,
            )
        )

        self.per_page = min(
            int(
                self.retriever_config.get(
                    "per_page",
                    20,
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

        # Zotero corpus used to describe
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

        # OpenAlex semantic search supports
        # at most 50 results per query.
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
            self.retriever_config.get(
                "semantic_types"
            )
            or ["article"]
        )

    def set_corpus(self, corpus):
        """
        Receive filtered Zotero papers from Executor.

        These papers are used to automatically
        represent the user's current research interests.
        """

        self.corpus = list(
            corpus or []
        )

    def _get_headers(self) -> dict:
        """
        Construct OpenAlex request headers.
        """

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
        Add OpenAlex works while removing duplicates
        according to OpenAlex Work ID.
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

        now = datetime.now(timezone.utc)

        from_date = (
            now
            - timedelta(
                days=self.lookback_days
            )
        ).date().isoformat()

        current_year = now.year

        headers = self._get_headers()

        raw_papers = []
        seen_work_ids = set()

        # =====================================================
        # 1. Retrieve papers from explicitly tracked authors
        # =====================================================

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
                        f"HTTP "
                        f"{response.status_code} - "
                        f"{response.text[:500]}"
                    )
                    continue

                results = (
                    response.json()
                    .get(
                        "results",
                        [],
                    )
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

        # =====================================================
        # 2. Semantic discovery based on Zotero interests
        # =====================================================

        if (
            self.semantic_search
            and self.corpus
        ):

            # Use recently added Zotero papers
            # as the current research-interest profile.
            seed_papers = sorted(
                self.corpus,
                key=lambda paper: (
                    paper.added_date
                ),
                reverse=True,
            )[
                : self.semantic_seed_count
            ]

            # Titles are intentionally used instead
            # of long abstracts.
            #
            # OpenAlex performs the broad semantic retrieval,
            # then Jina compares the returned candidates
            # against the entire Zotero corpus.
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

                # IMPORTANT:
                #
                # OpenAlex Semantic Search currently rejects
                # from_publication_date.
                #
                # Therefore:
                # 1. restrict the API query to current year;
                # 2. retrieve semantic candidates;
                # 3. locally keep only papers within
                #    lookback_days.
                filters = [
                    f"publication_year:{current_year}",
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

                        all_semantic_results = (
                            response.json()
                            .get(
                                "results",
                                [],
                            )
                        )

                        logger.info(
                            "OpenAlex semantic "
                            "search returned "
                            f"{len(all_semantic_results)} "
                            f"candidate works "
                            f"from {current_year}"
                        )

                        # -------------------------------------
                        # Apply exact recent-date filtering
                        # locally.
                        #
                        # OpenAlex publication_date uses
                        # YYYY-MM-DD format, so lexical
                        # comparison is valid here.
                        # -------------------------------------

                        semantic_results = []

                        for work in all_semantic_results:

                            publication_date = (
                                work.get(
                                    "publication_date"
                                )
                                or ""
                            )

                            if not publication_date:
                                continue

                            if publication_date < from_date:
                                continue

                            semantic_results.append(
                                work
                            )

                        logger.info(
                            f"{len(semantic_results)} "
                            "semantic OpenAlex works "
                            f"were published since "
                            f"{from_date}"
                        )

                        if (
                            self.config
                            .executor
                            .debug
                        ):
                            semantic_results = (
                                semantic_results[:10]
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
                            "search added "
                            f"{added_count} "
                            "unique recent works"
                        )

                except requests.RequestException as exc:

                    # Semantic discovery is supplementary.
                    # It should never break the entire
                    # daily paper recommendation workflow.
                    logger.warning(
                        "OpenAlex semantic "
                        f"search failed: {exc}"
                    )

        # =====================================================
        # Final retrieval summary
        # =====================================================

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
        """
        Convert an OpenAlex work object
        into the project's Paper format.
        """

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

        # Jina needs some text to embed.
        # If OpenAlex has no abstract,
        # use the paper title instead.
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
