from .base import register_retriever
from .biorxiv_retriever import BiorxivRetriever


@register_retriever("medrxiv")
class MedrxivRetriever(BiorxivRetriever):
    server = "medrxiv"
