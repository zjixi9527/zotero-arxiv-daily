import random
import socket
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
        raise TypeError(f"config.zotero.{config_key} must contain only glob pattern strings.")

    return list(patterns)


class Executor:
    # Required, non-empty configuration fields (dotted path -> human-readable label).
    # These fail fast at construction time instead of deep inside the pipeline.
    _REQUIRED_FIELDS = {
        "zotero.user_id": "Zotero user ID",
        "zotero.api_key": "Zotero API key",
        "email.sender": "email sender",
        "email.receiver": "email receiver",
        "email.smtp_server": "SMTP server",
        "email.smtp_port": "SMTP port",
        "email.sender_password": "SMTP password",
        "llm.api.key": "LLM API key",
        "llm.api.base_url": "LLM API base URL",
        "llm.generation_kwargs.model": "LLM model",
    }

    def __init__(self, config: DictConfig):
        self.config = config

        self._validate_config()

        self.include_path_patterns = normalize_path_patterns(
            config.zotero.include_path,
            "include_path",
        )

        self.ignore_path_patterns = normalize_path_patterns(
            config.zotero.ignore_path,
            "ignore_path",
        )

        self.retrievers = {source: get_retriever_cls(source)(config) for source in config.executor.source}

        self.reranker = get_reranker_cls(config.executor.reranker)(config)

        self.openai_client = OpenAI(
            api_key=config.llm.api.key,
            base_url=config.llm.api.base_url,
        )

    @staticmethod
    def _get_config_field(config: DictConfig, dotted_path: str):
        """Return the value at ``dotted_path`` (e.g. "email.smtp_port") or ``None`` if absent."""
        node = config
        for part in dotted_path.split("."):
            try:
                node = getattr(node, part)
            except AttributeError:
                return None
        return node

    def _validate_config(self) -> None:
        """Fail fast with a friendly message when the config is clearly unusable.

        Catches empty/missing required fields, an out-of-range SMTP port, and
        references to retriever sources that have no config section. Live
        connectivity (Zotero/SMTP handshake) is intentionally left to a
        ``doctor``-style check so construction stays deterministic and quick.
        """
        errors: list[str] = []
        cfg = self.config

        for dotted_path, label in self._REQUIRED_FIELDS.items():
            value = self._get_config_field(cfg, dotted_path)
            if value is None or (isinstance(value, str) and not value.strip()):
                errors.append(f"config.{dotted_path} ({label}) is empty or missing")

        port = self._get_config_field(cfg, "email.smtp_port")
        if port is not None:
            try:
                port_value = int(port)
            except (TypeError, ValueError):
                errors.append(f"config.email.smtp_port must be an integer, got {port!r}")
            else:
                if not 1 <= port_value <= 65535:
                    errors.append(f"config.email.smtp_port must be in 1..65535, got {port_value}")

        # Every enabled source must have a config section under config.source.
        enabled_sources = self._get_config_field(cfg, "executor.source")
        if enabled_sources:
            for source in enabled_sources:
                if not hasattr(cfg.source, source):
                    errors.append(
                        f"config.executor.source references {source!r}, but no config.source.{source} section exists"
                    )

        # Best-effort SMTP hostname resolution: warn only, never fail the run,
        # since the host may become resolvable later (e.g. in the runner VPC).
        smtp_server = self._get_config_field(cfg, "email.smtp_server")
        if isinstance(smtp_server, str) and smtp_server:
            try:
                socket.gethostbyname(smtp_server)
            except OSError:
                logger.warning(f"Could not resolve SMTP server hostname {smtp_server!r}; will retry at send time.")

        if errors:
            raise ValueError("Configuration validation failed:\n  - " + "\n  - ".join(errors))

    def fetch_zotero_corpus(self) -> list[CorpusPaper]:
        logger.info("Fetching zotero corpus")

        zot = zotero.Zotero(
            self.config.zotero.user_id,
            "user",
            self.config.zotero.api_key,
        )

        collections = zot.everything(zot.collections())

        collections = {c["key"]: c for c in collections}

        corpus = zot.everything(zot.items(itemType=("conferencePaper || journalArticle || preprint")))

        corpus = [c for c in corpus if c["data"]["abstractNote"] != ""]

        def get_collection_path(col_key: str) -> str:
            parent = collections[col_key]["data"]["parentCollection"]

            if parent:
                return get_collection_path(parent) + "/" + collections[col_key]["data"]["name"]

            return collections[col_key]["data"]["name"]

        for c in corpus:
            paths = [get_collection_path(col) for col in c["data"]["collections"]]

            c["paths"] = paths

        logger.info(f"Fetched {len(corpus)} zotero papers")

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
            logger.info(f"Selecting zotero papers matching include_path: {self.include_path_patterns}")

            corpus = [
                c
                for c in corpus
                if any(glob_match(path, pattern) for path in c.paths for pattern in self.include_path_patterns)
            ]

        if self.ignore_path_patterns:
            logger.info(f"Excluding zotero papers matching ignore_path: {self.ignore_path_patterns}")

            corpus = [
                c
                for c in corpus
                if not any(glob_match(path, pattern) for path in c.paths for pattern in self.ignore_path_patterns)
            ]

        if self.include_path_patterns or self.ignore_path_patterns:
            samples = random.sample(
                corpus,
                min(5, len(corpus)),
            )

            samples = "\n".join([c.title + " - " + "\n".join(c.paths) for c in samples])

            logger.info(f"Selected {len(corpus)} zotero papers:\n{samples}\n...")

        return corpus

    def run(self):
        corpus = self.fetch_zotero_corpus()
        corpus = self.filter_corpus(corpus)

        if len(corpus) == 0:
            logger.error(f"No zotero papers found. Please check your zotero settings:\n{self.config.zotero}")
            return

        all_papers = []

        for source, retriever in self.retrievers.items():
            logger.info(f"Retrieving {source} papers...")

            # OpenAlex uses the Zotero corpus
            # to construct the semantic search profile.
            if hasattr(retriever, "set_corpus"):
                retriever.set_corpus(corpus)

            papers = retriever.retrieve_papers()

            if len(papers) == 0:
                logger.info(f"No {source} papers found")
                continue

            logger.info(f"Retrieved {len(papers)} {source} papers")

            all_papers.extend(papers)

        logger.info(f"Total {len(all_papers)} papers retrieved from all sources")

        reranked_papers = []

        if len(all_papers) > 0:
            logger.info("Reranking papers...")

            reranked_papers = self.reranker.rerank(
                all_papers,
                corpus,
            )

            reranked_papers = reranked_papers[: self.config.executor.max_paper_num]

            logger.info("Generating TLDR and affiliations...")

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
            logger.info("No new papers found. No email will be sent.")
            return

        logger.info("Sending email...")

        email_content = render_email(reranked_papers)

        send_email(
            self.config,
            email_content,
        )

        logger.info("Email sent successfully")
