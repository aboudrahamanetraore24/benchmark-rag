"""
Système RAG hybride. Expose ask().

Combine recherche vectorielle dense (LlamaIndex) et recherche lexicale sparse (BM25).
Les résultats des deux retrievers sont fusionnés via Reciprocal Rank Fusion (RRF).
"""

import json
import time

from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from rank_bm25 import BM25Okapi

from src.config import EMBED_MODEL, get_top_k, HYBRID_INDEX_DIR
from src.systems.base import Reponse
from src.systems.generation import repondre

load_dotenv()

_embedder = OpenAIEmbedding(model=EMBED_MODEL)
_vec_retriever = None
_bm25 = None
_bm25_corpus: list[dict] | None = None

# Initialisé dans _charger() depuis get_top_k("hybrid") pour respecter le mode actuel.
_CANDIDATS: int = 0


def _charger():
    global _vec_retriever, _bm25, _bm25_corpus, _CANDIDATS

    if _vec_retriever is not None:
        return

    # Nombre de candidats récupérés par chaque retriever avant fusion RRF.
    # Multiplier par 2 laisse suffisamment de matière à RRF pour reclasser.
    _CANDIDATS = get_top_k("hybrid") * 2

    vector_dir = HYBRID_INDEX_DIR / "vector"
    if not vector_dir.exists():
        raise FileNotFoundError(
            f"Index hybride introuvable dans {HYBRID_INDEX_DIR}. "
            f"Lance d'abord : python -m src.build.hybrid_build"
        )

    storage = StorageContext.from_defaults(persist_dir=str(vector_dir))
    index = load_index_from_storage(storage, embed_model=_embedder)
    _vec_retriever = index.as_retriever(similarity_top_k=_CANDIDATS)

    bm25_path = HYBRID_INDEX_DIR / "bm25_corpus.json"
    _bm25_corpus = json.loads(bm25_path.read_text(encoding="utf-8"))
    tokenized = [chunk["text"].lower().split() for chunk in _bm25_corpus]
    _bm25 = BM25Okapi(tokenized)


def _rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion sur plusieurs classements de node_ids."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rang, nid in enumerate(ranking):
            scores[nid] = scores.get(nid, 0.0) + 1.0 / (k + rang + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def ask(question: str) -> Reponse:
    start = time.perf_counter()
    _charger()

    # Recherche dense
    vec_nodes = _vec_retriever.retrieve(question)
    vec_ids = [n.node_id for n in vec_nodes]
    vec_map = {n.node_id: n for n in vec_nodes}

    # Recherche sparse BM25
    bm25_scores = _bm25.get_scores(question.lower().split())
    top_indices = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:_CANDIDATS]
    bm25_ids = [_bm25_corpus[i]["id"] for i in top_indices]
    corpus_map = {chunk["id"]: chunk for chunk in _bm25_corpus}

    # Fusion RRF → top-K
    fused = _rrf([vec_ids, bm25_ids])[:get_top_k("hybrid")]

    # Reconstruction des NodeWithScore avec le score RRF
    nodes: list[NodeWithScore] = []
    for nid, rrf_score in fused:
        if nid in vec_map:
            nodes.append(NodeWithScore(node=vec_map[nid].node, score=rrf_score))
        elif nid in corpus_map:
            chunk = corpus_map[nid]
            text_node = TextNode(
                text=chunk["text"],
                id_=nid,
                metadata={"doc_id": chunk.get("doc_id", "")},
            )
            nodes.append(NodeWithScore(node=text_node, score=rrf_score))

    return repondre(question, nodes, start)
