"""
Système LightRAG. Expose ask().

Interroge le graphe d'entités/relations en mode mix (local + global).
"""

import asyncio
import logging
import time

import openai
from dotenv import load_dotenv

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

from src.config import (
    MODEL_NAME, EMBED_MODEL, EMBED_DIM, get_top_k, LIGHTRAG_INDEX_DIR, TEMPERATURE,
    PROMPT_GENERATION,
)
from src.metrics import CompteurLLM, usage_reel, cout_embedding_usd, compter_tokens
from src.systems.base import Reponse, Source

load_dotenv()

for _name in ("nano-vectordb", "lightrag", "lightrag.operate",
              "httpx", "httpcore", "openai",
              "llama_index", "llama_index.core"):
    logging.getLogger(_name).setLevel(logging.ERROR)

MODE = "mix"  # local + global combinés

# Même prompt de génération que tous les autres systèmes : LightRAG fait sa propre
# recherche (graphe + chunks) mais reçoit les mêmes instructions de réponse.
# {context} -> {context_data}, seul nom de placeholder accepté par aquery(system_prompt=...).
_SYSTEM_PROMPT = (
    PROMPT_GENERATION.split("\n\nQuestion :")[0].replace("{context}", "{context_data}")
)

# Event loop persistant : les asyncio.Lock internes de LightRAG sont liés au loop
# de leur création ; un nouveau loop à chaque appel lèverait "bound to a different event loop".
_loop = asyncio.new_event_loop()

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

    # Flags legacy LightRAG signalant une extraction JSON (entités ou mots-clés).
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


async def _make_rag():
    """Charge l'index LightRAG persisté et initialise les stockages."""
    if not LIGHTRAG_INDEX_DIR.exists():
        raise FileNotFoundError(
            f"Index LightRAG introuvable dans {LIGHTRAG_INDEX_DIR}. "
            f"Exécuter d'abord : python -m src.build.lightrag_build"
        )
    rag = LightRAG(
        working_dir=str(LIGHTRAG_INDEX_DIR),
        llm_model_func=_llm_avec_usage,
        llm_model_name=MODEL_NAME,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBED_DIM, max_token_size=8192, func=_embed,
        ),
        # Cache désactivé pour mesurer la latence et le coût réels à chaque requête.
        enable_llm_cache=False,
        enable_llm_cache_for_entity_extract=False,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


async def _recuperer_contexte(rag, question: str) -> list[Source]:
    """Refait une requête en only_need_context pour exposer ce sur quoi LightRAG s'est appuyé."""
    try:
        contexte = await rag.aquery(
            question,
            param=QueryParam(mode=MODE, top_k=get_top_k("lightrag"), only_need_context=True, enable_rerank=False),
        )
        if isinstance(contexte, str) and contexte.strip():
            return [Source(texte=contexte.strip(), identifiant="lightrag_context")]
    except Exception:
        pass
    return []


async def _ask_async(question: str) -> Reponse:
    compteur.reset()

    rag = await _make_rag()
    try:
        start = time.perf_counter()
        reponse = await rag.aquery(
            question,
            param=QueryParam(mode=MODE, top_k=get_top_k("lightrag"), enable_rerank=False),
            system_prompt=_SYSTEM_PROMPT,
        )
        latence = time.perf_counter() - start
        if not reponse or not isinstance(reponse, str):
            raise RuntimeError(f"LightRAG a retourné une réponse vide ou invalide : {reponse!r}")
        sources = await _recuperer_contexte(rag, question)
    finally:
        await rag.finalize_storages()

    cout_total = compteur.cout() + cout_embedding_usd(EMBED_MODEL, compter_tokens(question, EMBED_MODEL))
    return Reponse(
        texte=reponse.strip(),
        latence_s=latence,
        cout_usd=cout_total,
        sources=sources,
    )


def ask(question: str) -> Reponse:
    return _loop.run_until_complete(_ask_async(question))
