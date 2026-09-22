"""
Génération partagée par baseline, RAPTOR et hybrid.

Même prompt, même modèle, même comptage de coût : toute différence de réponse
entre ces systèmes vient uniquement du retrieval.
"""

import time

from llama_index.llms.openai import OpenAI

from src.config import MODEL_NAME, EMBED_MODEL, PROMPT_GENERATION, llm_kwargs
from src.metrics import cout_usd, usage_reel, compter_tokens, cout_embedding_usd
from src.systems.base import Reponse, Source

_llm = OpenAI(**llm_kwargs())


def _nodes_vers_sources(nodes) -> list[Source]:
    """Convertit les nœuds LlamaIndex récupérés en objets Source traçables."""
    sources = []
    for n in nodes:
        sources.append(Source(
            texte=n.text,
            score=getattr(n, "score", None),
            identifiant=getattr(n, "node_id", None),
        ))
    return sources


def repondre(question: str, nodes, start: float) -> Reponse:
    """Génère la réponse à partir des nœuds récupérés.

    start doit être pris avant le retrieval pour que la latence couvre retrieval + génération.
    """
    context = "\n\n".join(n.text for n in nodes)
    prompt = PROMPT_GENERATION.format(context=context, question=question)

    resp = _llm.complete(prompt)
    latence = time.perf_counter() - start

    texte = resp.text.strip()
    t_in, t_out = usage_reel(resp.raw)

    # embedding de la question ajouté pour cohérence inter-systèmes
    cout = cout_usd(MODEL_NAME, t_in, t_out)
    cout += cout_embedding_usd(EMBED_MODEL, compter_tokens(question, EMBED_MODEL))

    return Reponse(
        texte=texte,
        latence_s=latence,
        cout_usd=cout,
        sources=_nodes_vers_sources(nodes),
    )
