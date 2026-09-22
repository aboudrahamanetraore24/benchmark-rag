"""
Système RAG classique (baseline). Expose ask().

Recherche vectorielle plate top-k. Référence de comparaison du benchmark.
"""

import time

from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding

from src.config import EMBED_MODEL, get_top_k, BASELINE_INDEX_DIR
from src.systems.base import Reponse
from src.systems.generation import repondre

load_dotenv()

_embedder = OpenAIEmbedding(model=EMBED_MODEL)
_retriever = None


def _get_retriever():
    """Charge l'index baseline une seule fois, puis le réutilise."""
    global _retriever
    if _retriever is None:
        if not BASELINE_INDEX_DIR.exists():
            raise FileNotFoundError(
                f"Index baseline introuvable dans {BASELINE_INDEX_DIR}. "
                f"Lance d'abord : python -m src.build.baseline_build"
            )
        storage = StorageContext.from_defaults(persist_dir=str(BASELINE_INDEX_DIR))
        index = load_index_from_storage(storage, embed_model=_embedder)
        _retriever = index.as_retriever(similarity_top_k=get_top_k("baseline"))
    return _retriever


def ask(question: str) -> Reponse:
    start = time.perf_counter()
    nodes = _get_retriever().retrieve(question)
    return repondre(question, nodes, start)
