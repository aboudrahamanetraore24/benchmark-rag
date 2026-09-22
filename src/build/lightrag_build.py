"""
Construit l'index LightRAG (graphe d'entités + relations) et le sauvegarde.

Le coût est mesuré via l'usage réel retourné par l'API OpenAI.
"""

import asyncio
import logging
import shutil
import time

import openai
from dotenv import load_dotenv

from lightrag import LightRAG
from lightrag.llm.openai import openai_embed
from lightrag.utils import EmbeddingFunc, setup_logger
from lightrag.kg.shared_storage import initialize_pipeline_status

from src.config import MODEL_NAME, EMBED_MODEL, EMBED_DIM, LIGHTRAG_INDEX_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from src.corpus import documents_du_benchmark
from src.metrics import log_metric, CompteurLLM, usage_reel

load_dotenv()

setup_logger("lightrag", level="INFO")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("lightrag-build")

compteur = CompteurLLM(MODEL_NAME)
_openai_client = openai.AsyncOpenAI()


_LIGHTRAG_KWARGS = frozenset({
    "hashing_kv", "openai_client_configs", "enable_cot",
    "token_tracker", "base_url", "api_key",
    "use_azure", "azure_deployment", "api_version", "image_inputs",
})


async def _llm_avec_usage(prompt, system_prompt=None, history_messages=None, **kwargs):
    """Appel LLM direct qui capture l'usage réel (tokens in/out) via l'objet réponse OpenAI."""
    for key in _LIGHTRAG_KWARGS:
        kwargs.pop(key, None)

    wants_json = kwargs.pop("entity_extraction", False) or kwargs.pop("keyword_extraction", False)
    if wants_json and kwargs.get("response_format") is None:
        kwargs["response_format"] = {"type": "json_object"}

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for m in (history_messages or []):
        messages.append(m if isinstance(m, dict) else {"role": "user", "content": str(m)})
    messages.append({"role": "user", "content": prompt})

    response = await _openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        **kwargs,
    )

    t_in, t_out = usage_reel(response)
    compteur.tokens_in += t_in
    compteur.tokens_out += t_out

    return response.choices[0].message.content or ""



async def _embed(texts):
    return await openai_embed(texts, model=EMBED_MODEL)


async def build():
    # LightRAG mémorise les IDs traités dans kv_store_doc_status.json.
    # Le répertoire est supprimé avant chaque build pour forcer le re-traitement complet.
    if LIGHTRAG_INDEX_DIR.exists():
        shutil.rmtree(LIGHTRAG_INDEX_DIR)
    LIGHTRAG_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    textes, ids = documents_du_benchmark()  # mêmes documents que baseline/RAPTOR
    log.info(f"Documents chargés : {len(textes)}")

    rag = LightRAG(
        working_dir=str(LIGHTRAG_INDEX_DIR),
        llm_model_func=_llm_avec_usage,
        llm_model_name=MODEL_NAME,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBED_DIM,
            max_token_size=8192,
            func=_embed,
        ),
        chunk_token_size=CHUNK_SIZE,
        chunk_overlap_token_size=CHUNK_OVERLAP,
        enable_llm_cache=False,
        enable_llm_cache_for_entity_extract=False,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    try:
        t0 = time.perf_counter()
        await rag.ainsert(textes, ids=ids)
        temps_indexation = time.perf_counter() - t0

        cout_indexation = compteur.cout()

        log.info(
            f"Indexation terminée. Temps : {temps_indexation:.2f}s | "
            f"Coût réel : ${cout_indexation:.4f} "
            f"(tokens in={compteur.tokens_in}, out={compteur.tokens_out})"
        )

        log_metric({
            "systeme": "LightRAG",
            "phase": "indexation",
            "n_chunks": len(textes),
            "temps_s": round(temps_indexation, 2),
            "cout_usd": round(cout_indexation, 6),
            "note": f"{len(textes)} documents ingérés (pas des chunks). LightRAG gère le découpage en interne selon la taille configurée ({CHUNK_SIZE} tokens, overlap {CHUNK_OVERLAP})",
        })
    finally:
        await rag.finalize_storages()


def main():
    asyncio.run(build())


if __name__ == "__main__":
    main()
