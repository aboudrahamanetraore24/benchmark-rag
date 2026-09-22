"""
Système RAPTOR. Expose ask().

Retrieval hiérarchique sur l'arbre de résumés (Sarthi et al., 2024).

L'arbre a trois catégories de nœuds :
  level=0    → chunks originaux         (détails précis)
  level=1..N → résumés de clusters      (contexte intermédiaire)
  level=max  → racine unique            (vue globale du corpus)

À chaque requête, on récupère des représentants des trois catégories
pour que le LLM ait à la fois la vue d'ensemble et les détails.
"""

import time

from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding

from src.config import EMBED_MODEL, get_top_k, RAPTOR_INDEX_DIR
from src.systems.base import Reponse
from src.systems.generation import repondre

load_dotenv()

_embedder = OpenAIEmbedding(model=EMBED_MODEL)
_index = None


def _get_index():
    global _index
    if _index is None:
        if not RAPTOR_INDEX_DIR.exists():
            raise FileNotFoundError(
                f"Index RAPTOR introuvable dans {RAPTOR_INDEX_DIR}. "
                f"Exécuter d'abord : python -m src.build.raptor_build"
            )
        storage = StorageContext.from_defaults(persist_dir=str(RAPTOR_INDEX_DIR))
        _index = load_index_from_storage(storage, embed_model=_embedder)
    return _index


def ask(question: str) -> Reponse:
    start = time.perf_counter()
    top_k = get_top_k("raptor")

    # Récupère un large pool pour avoir tous les niveaux représentés
    retriever = _get_index().as_retriever(similarity_top_k=top_k * 3)
    candidats = retriever.retrieve(question)

    if not candidats:
        return repondre(question, [], start)

    # Sépare les nœuds par catégorie
    max_lvl = max(n.metadata.get("level", 0) for n in candidats)

    racine  = [n for n in candidats if n.metadata.get("level", 0) == max_lvl]
    mid     = [n for n in candidats if 0 < n.metadata.get("level", 0) < max_lvl]
    chunks  = [n for n in candidats if n.metadata.get("level", 0) == 0]

    # Distribution des slots : 1 racine + ~1/5 résumés mid + reste en chunks
    n_root  = min(1, len(racine))
    n_mid   = min(len(mid),   max(1, top_k // 5))
    n_chunk = min(len(chunks), top_k - n_root - n_mid)

    nodes = racine[:n_root] + mid[:n_mid] + chunks[:n_chunk]

    # Repli si un niveau est absent (corpus très petit ou arbre à 1 niveau)
    if not nodes:
        nodes = candidats[:top_k]

    return repondre(question, nodes, start)
