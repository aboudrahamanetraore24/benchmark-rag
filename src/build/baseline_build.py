"""
Construit l'index du RAG baseline : recherche vectorielle plate sur les chunks.
"""

import logging
import time

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding

from src.config import EMBED_MODEL, BASELINE_INDEX_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from src.corpus import documents_du_benchmark
from src.metrics import log_metric, cout_embedding_usd, compter_tokens

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("baseline-build")

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

    index = VectorStoreIndex.from_documents(docs, embed_model=embedder)
    BASELINE_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(BASELINE_INDEX_DIR))

    temps = time.perf_counter() - t0
    log.info(f"Indexation terminée. Temps : {temps:.2f}s")

    tokens_corpus = sum(compter_tokens(doc.text, EMBED_MODEL) for doc in docs)
    cout_embedding = cout_embedding_usd(EMBED_MODEL, tokens_corpus)
    log_metric({
        "systeme": "RAG classique",
        "phase": "indexation",
        "n_chunks": len(docs),
        "temps_s": round(temps, 2),
        "cout_usd": round(cout_embedding, 6),
        "note": "coût d'embedding du corpus estimé via tiktoken",
    })


if __name__ == "__main__":
    main()
