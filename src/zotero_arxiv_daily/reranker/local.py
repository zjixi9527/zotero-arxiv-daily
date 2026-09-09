import logging
import warnings

import numpy as np

from .base import BaseReranker, register_reranker


@register_reranker("local")
class LocalReranker(BaseReranker):
    def get_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional extra
            raise ModuleNotFoundError(
                "The local reranker requires heavy ML dependencies "
                "(sentence-transformers, torch, scikit-learn, peft) that are not installed.\n"
                "Install them with:  uv sync --extra local\n"
                "or use a different reranker (e.g. `reranker: openai`) in your config."
            ) from exc

        if not self.config.executor.debug:
            from huggingface_hub.utils import logging as hf_logging
            from transformers.utils import logging as transformers_logging

            transformers_logging.set_verbosity_error()
            hf_logging.set_verbosity_error()
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.ERROR)
            logging.getLogger("transformers").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
            warnings.filterwarnings("ignore", category=FutureWarning)

        encoder = SentenceTransformer(self.config.reranker.local.model, trust_remote_code=True)
        encode_kwargs = self.config.reranker.local.encode_kwargs or {}
        s1_feature = encoder.encode(s1, **encode_kwargs, show_progress_bar=True)
        s2_feature = encoder.encode(s2, **encode_kwargs, show_progress_bar=True)
        sim = encoder.similarity(s1_feature, s2_feature)
        return sim.numpy()
