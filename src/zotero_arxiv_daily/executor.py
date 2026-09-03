import random
from datetime import datetime

from loguru import logger
from omegaconf import DictConfig, ListConfig
from openai import OpenAI
from pyzotero import zotero
from tqdm import tqdm

from .construct_email import render_email
from .protocol import CorpusPaper
from .reranker import get_reranker_cls
from .retriever import get_retriever_cls
from .utils import glob_match, send_email


def normalize_path_patterns(
    patterns: list[str] | ListConfig | None,
    config_key: str,
) -> list[str] | None:

    if patterns is None:
        return None

    if not isinstance(patterns, (list, ListConfig)):
        raise TypeError(
            f"config.zotero.{config_key} must be a list of "
            f"glob patterns or null, for example "
            f'["2026/survey/**"]. '
            f"Single strings are not supported."
        )

    if any(not isinstance(pattern, str) for pattern in patterns):
        raise TypeError(
            f"config.zotero.{config_key} must contain only "
            f"glob pattern strings."
        )

    return list(patterns)


class Executor:

    def __init__(self, config: DictConfig):
        self.config = config

        self.include_path_patterns = normalize_path_patterns(
            config.zotero.include_path,
            "include_path",
        )

        self.ignore_path_patterns = normalize_path_patterns(
            config.zotero.ignore_path,
            "ignore_path",
        )

        self.retrievers = {
            source: get_retriever_cls(source)(config)
            for source in config.executor.source
        }

        self.reranker = get_reranker_cls(
            config.executor.reranker
        )(config)

        self.openai_client = OpenAI(
            api_key=config.llm.api.key,
            base_url=config.llm.api.base_url,
        )

    def fetch_zotero_corpus(self) -> list[CorpusPaper]:
        logger.info("Fetching zotero corpus")

        zot = zotero.Zotero(
            self.config.zotero.user_id,
            "user",
            self.config.zotero.api_key,
        )

        collections = zot.everything(
            zot.collections()
        )

        collections = {
            c["key"]: c
            for c in collections
        }

        corpus = zot.everything(
            zot.items(
                itemType=(
                    "conferencePaper || "
                    "journalArticle || "
                    "preprint"
                )
            )
        )

        corpus = [
            c
            for c in corpus
            if c["data"]["abstractNote"] != ""
        ]

        def get_collection_path(col_key: str) -> str:
            parent = collections[col_key]["data"][
                "parentCollection"
            ]

            if parent:
                return (
                    get_collection_path(parent)
                    + "/"
                    + collections[col_key]["data"]["name"]
                )

            return collections[col_key]["data"]["name"]

        for c in corpus:
            paths = [
                get_collection_path(col)
                for col in c["data"]["collections"]
            ]

            c["paths"] = paths

        logger.info(
            f"Fetched {len(corpus)} zotero papers"
        )

        return [
            CorpusPaper(
                title=c["data"]["title"],
                abstract=c["data"]["abstractNote"],
                added_date=datetime.strptime(
                    c["data"]["dateAdded"],
                    "%Y-%m-%dT%H:%M:%SZ",
                ),
                paths=c["paths"],
            )
            for c in corpus
        ]

    def filter_corpus(
        self,
        corpus: list[CorpusPaper],
    ) -> list[CorpusPaper]:

        if self.include_path_patterns:
            logger.info(
                "Selecting zotero papers matching "
                f"include_path: "
                f"{self.include_path_patterns}"
            )

            corpus = [
                c
                for c in corpus
                if any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.include_path_patterns
                )
            ]

        if self.ignore_path_patterns:
            logger.info(
                "Excluding zotero papers matching "
                f"ignore_path: "
                f"{self.ignore_path_patterns}"
            )

            corpus = [
                c
                for c in corpus
                if not any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.ignore_path_patterns
                )
            ]

        if (
            self.include_path_patterns
            or self.ignore_path_patterns
        ):
            samples = random.sample(
                corpus,
                min(5, len(corpus)),
            )

            samples = "\n".join(
                [
                    c.title
                    + " - "
                    + "\n".join(c.paths)
                    for c in samples
                ]
            )

            logger.info(
                f"Selected {len(corpus)} "
                f"zotero papers:\n"
                f"{samples}\n..."
            )

        return corpus

    def run(self):
        corpus = self.fetch_zotero_corpus()
        corpus = self.filter_corpus(corpus)

        if len(corpus) == 0:
            logger.error(
                "No zotero papers found. "
                "Please check your zotero settings:\n"
                f"{self.config.zotero}"
            )
            return

        all_papers = []

        for source, retriever in self.retrievers.items():
            logger.info(
                f"Retrieving {source} papers..."
            )

            # OpenAlex uses the Zotero corpus
            # to construct the semantic search profile.
            if hasattr(retriever, "set_corpus"):
                retriever.set_corpus(corpus)

            papers = retriever.retrieve_papers()

            if len(papers) == 0:
                logger.info(
                    f"No {source} papers found"
                )
                continue

            logger.info(
                f"Retrieved {len(papers)} "
                f"{source} papers"
            )

            all_papers.extend(papers)

        logger.info(
            f"Total {len(all_papers)} papers "
            f"retrieved from all sources"
        )

        reranked_papers = []

        if len(all_papers) > 0:
            logger.info("Reranking papers...")

            reranked_papers = self.reranker.rerank(
                all_papers,
                corpus,
            )

            reranked_papers = reranked_papers[
                : self.config.executor.max_paper_num
            ]

            logger.info(
                "Generating TLDR and affiliations..."
            )

            for paper in tqdm(reranked_papers):
                paper.generate_tldr(
                    self.openai_client,
                    self.config.llm,
                )

                paper.generate_affiliations(
                    self.openai_client,
                    self.config.llm,
                )

        elif not self.config.executor.send_empty:
            logger.info(
                "No new papers found. "
                "No email will be sent."
            )
            return

        logger.info("Sending email...")

        email_content = render_email(
            reranked_papers
        )

        send_email(
            self.config,
            email_content,
        )

        logger.info(
            "Email sent successfully"
        )
