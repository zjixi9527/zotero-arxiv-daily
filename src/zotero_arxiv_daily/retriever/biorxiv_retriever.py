from datetime import datetime
from typing import Any

import requests
from loguru import logger

from ..protocol import Paper
from ..utils import call_with_retry
from .base import BaseRetriever, register_retriever


@register_retriever("biorxiv")
class BiorxivRetriever(BaseRetriever):
    server = "biorxiv"

    def __init__(self, config):
        super().__init__(config)
        if self.retriever_config.category is None:
            raise ValueError(f"category must be specified for {self.name}")

    def _retrieve_raw_papers(self) -> list[dict[str, Any]]:
        api_url = f"https://api.biorxiv.org/details/{self.server}/2d"

        def _fetch() -> requests.Response:
            response = requests.get(api_url)
            response.raise_for_status()
            return response

        response = call_with_retry(_fetch, retries=10, base_delay=10, what="fetch biorxiv papers")
        result = response.json()
        collection = result["collection"]
        if len(collection) == 0:
            logger.warning(f"No paper found. API Message: {result['messages']}")
            return []
        dated_collection = [(datetime.strptime(c["date"], "%Y-%m-%d").date(), c) for c in collection]
        latest_date = max(date for date, _ in dated_collection)
        collection = [c for date, c in dated_collection if date == latest_date]
        categories = [c.lower() for c in self.retriever_config.category]
        collection = [c for c in collection if c["category"] in categories]
        if self.config.executor.debug:
            collection = collection[:10]
        return collection

    def convert_to_paper(self, raw_paper: dict[str, Any]) -> Paper | None:
        title = raw_paper["title"]
        authors = [a.strip() for a in raw_paper["authors"].split(";")]
        abstract = raw_paper["abstract"]
        pdf_url = f"https://www.{self.server}.org/content/{raw_paper['doi']}v{raw_paper['version']}.full.pdf"
        full_text = None  # biorxiv forbids scraping its pdf
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=pdf_url,
            pdf_url=pdf_url,
            full_text=full_text,
        )
