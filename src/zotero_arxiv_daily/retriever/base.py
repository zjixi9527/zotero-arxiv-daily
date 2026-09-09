from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from ..protocol import Paper, RawPaperItem


class BaseRetriever(ABC):
    name: str

    # Subclasses whose ``convert_to_paper`` performs blocking network / subprocess
    # work (e.g. fetching paper full text) must set this to False so conversion
    # stays serial and does not hammer the upstream service with concurrent jobs.
    concurrency_safe: bool = True

    def __init__(self, config: DictConfig):
        self.config = config
        self.retriever_config = getattr(config.source, self.name)

    @abstractmethod
    def _retrieve_raw_papers(self) -> list[RawPaperItem]:
        pass

    @abstractmethod
    def convert_to_paper(self, raw_paper: RawPaperItem) -> Paper | None:
        pass

    def _convert_one(self, raw_paper: RawPaperItem) -> Paper | None:
        try:
            return self.convert_to_paper(raw_paper)
        except Exception as exc:
            logger.warning(f"Skipping paper {getattr(raw_paper, 'title', raw_paper)}: {exc}")
            return None

    def retrieve_papers(self) -> list[Paper]:
        raw_papers = self._retrieve_raw_papers()
        logger.info("Processing papers...")

        if not self.concurrency_safe:
            papers = []
            for raw_paper in tqdm(raw_papers, total=len(raw_papers), desc="Converting papers"):
                paper = self._convert_one(raw_paper)
                if paper is not None:
                    papers.append(paper)
            return papers

        # Network-free sources can convert papers in parallel instead of
        # sleeping serially between each one.
        with ThreadPoolExecutor() as pool:
            futures = [pool.submit(self._convert_one, raw_paper) for raw_paper in raw_papers]
            papers = []
            for future in tqdm(futures, total=len(futures), desc="Converting papers"):
                paper = future.result()
                if paper is not None:
                    papers.append(paper)
        return papers


registered_retrievers = {}


def register_retriever(name: str):
    def decorator(cls):
        registered_retrievers[name] = cls
        cls.name = name
        return cls

    return decorator


def get_retriever_cls(name: str) -> type[BaseRetriever]:
    if name not in registered_retrievers:
        raise ValueError(f"Retriever {name} not found")
    return registered_retrievers[name]
