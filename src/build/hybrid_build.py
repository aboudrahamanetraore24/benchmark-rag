"""
Construit l'index du RAG hybride : vectoriel + BM25, fusion par RRF à la requête.

Produit deux artefacts dans HYBRID_INDEX_DIR :
  - vector/          : index LlamaIndex (FAISS) pour la recherche dense
  - bm25_corpus.json : corpus sérialisé pour la recherche lexicale BM25
"""

import json
import logging
import time

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding

from src.config import EMBED_MODEL, HYBRID_INDEX_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from src.corpus import documents_du_benchmark
from src.metrics import log_metric, cout_embedding_usd, compter_tokens

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("hybrid-build")

embedder = OpenAIEmbedding(model=EMBED_MODEL)
splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def main():
    t0 = time.perf_counter()

    textes, ids = documents_du_benchmark()
    docs = []
    for doc_id, text in zip(ids, textes):
        for i, piece in enumerate(splitter.split_text(text)):
            docs.append(Document(text=piece, id_=f"{doc_id}_{i:04d}"))
    log.info(f"Chunks créés : {len(docs)}")

    HYBRID_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Index vectoriel (recherche dense)
    index = VectorStoreIndex.from_documents(docs, embed_model=embedder)
    vector_dir = HYBRID_INDEX_DIR / "vector"
    vector_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(vector_dir))

    # Corpus BM25 : texte + identifiants de chaque chunk
    corpus = [
        {
            "id": doc.doc_id,
            "text": doc.text,
            "doc_id": doc.doc_id.rsplit("_", 1)[0],
        }
        for doc in docs
    ]
    bm25_path = HYBRID_INDEX_DIR / "bm25_corpus.json"
    bm25_path.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")
    log.info(f"Corpus BM25 sauvegardé : {bm25_path}")

    temps = time.perf_counter() - t0
    log.info(f"Indexation terminée. Temps : {temps:.2f}s")

    tokens_corpus = sum(compter_tokens(doc.text, EMBED_MODEL) for doc in docs)
    cout_embedding = cout_embedding_usd(EMBED_MODEL, tokens_corpus)
    log_metric({
        "systeme": "RAG hybride",
        "phase": "indexation",
        "n_chunks": len(docs),
        "temps_s": round(temps, 2),
        "cout_usd": round(cout_embedding, 6),
        "note": "BM25 + vectoriel, fusion RRF (k=60) à la requête",
    })


if __name__ == "__main__":
    main()
