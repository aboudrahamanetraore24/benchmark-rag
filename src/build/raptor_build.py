"""
Construit l'arbre RAPTOR (clustering + résumés) et sauvegarde l'index.

Fidèle à RAPTOR (Sarthi et al., 2024) :
clustering à deux niveaux (global + local), UMAP + GMM + BIC, soft clustering.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import umap
from dotenv import load_dotenv
from sklearn.mixture import GaussianMixture
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBED_MODEL, MODEL_NAME, RAPTOR_INDEX_DIR, llm_kwargs
from src.corpus import documents_du_benchmark
from src.metrics import cout_embedding_usd, cout_usd, compter_tokens, log_metric

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("raptor-build")

UMAP_DIM           = 10
SOFT_THRESHOLD     = 0.1
MAX_CLUSTERS       = 50
MAX_CLUSTER_TOKENS = 3500
MAX_SUMMARY_TOKENS = 1_200
RANDOM_STATE       = 42
SUMMARY_PROMPT     = "Résume en français les informations clés de ces extraits :\n\n{context}\n\nRésumé :"

embedder = OpenAIEmbedding(model=EMBED_MODEL)
llm      = OpenAI(**llm_kwargs())
splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def embed(texts):
    return np.array(embedder.get_text_embedding_batch(texts, show_progress=True))


def _get_optimal_k(embeddings):
    max_k = min(MAX_CLUSTERS, len(embeddings))
    bics = [GaussianMixture(n_components=k, random_state=RANDOM_STATE).fit(embeddings).bic(embeddings)
            for k in range(1, max_k)]
    return int(np.argmin(bics)) + 1


def _gmm_labels(embeddings):
    gm = GaussianMixture(n_components=_get_optimal_k(embeddings), random_state=RANDOM_STATE).fit(embeddings)
    return [np.where(p > SOFT_THRESHOLD)[0] for p in gm.predict_proba(embeddings)], gm.n_components


def _perform_clustering(embeddings):
    """Clustering à deux niveaux (global puis local) — fidèle au papier RAPTOR."""
    n   = len(embeddings)
    dim = min(UMAP_DIM, n - 2)

    if dim < 1:
        return [np.array([0]) for _ in range(n)]

    global_reduced = umap.UMAP(
        n_neighbors=max(2, int((n - 1) ** 0.5)),
        n_components=dim,
        metric="cosine",
        random_state=RANDOM_STATE,
    ).fit_transform(embeddings)
    global_labels, n_global = _gmm_labels(global_reduced)

    all_labels = [np.array([]) for _ in range(n)]
    total = 0

    for i in range(n_global):
        mask    = np.array([i in lbl for lbl in global_labels])
        sub_emb = embeddings[mask]
        sub_idx = np.where(mask)[0]

        if len(sub_emb) <= dim + 1:
            for idx in sub_idx:
                all_labels[idx] = np.append(all_labels[idx], total)
            total += 1
        else:
            local_reduced = umap.UMAP(
                n_neighbors=min(10, len(sub_emb) - 1),
                n_components=min(dim, len(sub_emb) - 2),
                metric="cosine",
                random_state=RANDOM_STATE,
            ).fit_transform(sub_emb)
            local_labels, n_local = _gmm_labels(local_reduced)

            for j in range(n_local):
                for idx, lbl in zip(sub_idx, local_labels):
                    if j in lbl:
                        all_labels[idx] = np.append(all_labels[idx], j + total)
            total += n_local

    return all_labels


def get_clusters(texts, embeddings):
    """Regroupe les textes par cluster. Recluster récursivement si un cluster dépasse MAX_CLUSTER_TOKENS."""
    labels = _perform_clustering(embeddings)

    groups = {}
    for i, cluster_labels in enumerate(labels):
        for label in cluster_labels:
            groups.setdefault(int(label), []).append(i)

    result = []
    for indices in groups.values():
        cluster = [texts[i] for i in indices]
        if len(indices) == 1:
            result.append(cluster)
            continue
        if sum(compter_tokens(t, MODEL_NAME) for t in cluster) > MAX_CLUSTER_TOKENS:
            result.extend(get_clusters(cluster, embeddings[np.array(indices)]))
        else:
            result.append(cluster)
    return result


def summarize(cluster):
    prompt = SUMMARY_PROMPT.format(context="\n\n".join(cluster))
    resp   = llm.complete(prompt, max_tokens=MAX_SUMMARY_TOKENS)
    text   = resp.text.strip() or " ".join(cluster)[:2000]
    return text, compter_tokens(prompt, MODEL_NAME), compter_tokens(text, MODEL_NAME)


def build_tree(texts):
    all_nodes = [(t, 0) for t in texts]
    current   = list(texts)
    total_in  = total_out = 0
    level     = 0

    while len(current) > 1:
        level += 1
        log.info(f"Niveau {level} : {len(current)} nœuds")
        vectors  = embed(current)
        clusters = get_clusters(current, vectors)

        if len(clusters) >= len(current):
            log.info("Clustering infaisable, arrêt.")
            break

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(summarize, clusters))

        summaries = []
        for text, t_in, t_out in results:
            summaries.append(text)
            total_in  += t_in
            total_out += t_out

        all_nodes.extend([(s, level) for s in summaries])
        current = summaries

    return all_nodes, total_in, total_out


def main():
    t0 = time.perf_counter()

    textes, _ = documents_du_benchmark()
    texts     = [piece for text in textes for piece in splitter.split_text(text)]
    log.info(f"Chunks créés : {len(texts)}")

    nodes, tokens_in, tokens_out = build_tree(texts)
    log.info(f"Total nœuds (chunks + résumés) : {len(nodes)}")

    index = VectorStoreIndex.from_documents(
        [Document(text=t, metadata={"level": lvl}) for t, lvl in nodes],
        embed_model=embedder,
    )
    RAPTOR_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(RAPTOR_INDEX_DIR))

    temps      = time.perf_counter() - t0
    tokens_idx = sum(compter_tokens(t, EMBED_MODEL) for t, _ in nodes)
    cout_total = cout_usd(MODEL_NAME, tokens_in, tokens_out) + cout_embedding_usd(EMBED_MODEL, tokens_idx)

    log.info(f"Indexation terminée. Temps : {temps:.2f}s | Coût : ${cout_total:.4f}")
    log_metric({
        "systeme":  "RAPTOR",
        "phase":    "indexation",
        "n_chunks": len(texts),
        "n_noeuds": len(nodes),
        "temps_s":  round(temps, 2),
        "cout_usd": round(cout_total, 6),
    })


if __name__ == "__main__":
    main()
