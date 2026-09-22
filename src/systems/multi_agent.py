"""
Système Multi-Agent RAG (Manager/Workers). Expose ask().

Pipeline en trois phases :
  1. Décomposition  : un appel LLM décompose la question en N_SOUS_AGENTS
                      sous-questions complémentaires et non redondantes.
  2. Recherche      : N_SOUS_AGENTS sous-agents FunctionAgent tournent en parallèle
                      via asyncio.gather(), chacun traitant une sous-question.
  3. Synthèse       : un appel LLM agrège les N_SOUS_AGENTS résultats en une
                      réponse finale cohérente.

Le parallélisme réel (asyncio.gather) distingue ce système du Mono-Agentic :
les appels API des sous-agents partent simultanément, la latence totale ≈
latence d'un seul sous-agent (et non N × latence).

Réutilise l'index du RAG classique — pas d'indexation propre.

Limites documentées :
  - Coût des embeddings des sous-questions internes non comptabilisé (reformulations).
  - Sources intermédiaires (chunks bruts) non exposées.
"""

import asyncio
import logging
import re
import time

from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage, PromptTemplate
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.tools import QueryEngineTool
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events.llm import LLMChatEndEvent
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from src.config import (
    MODEL_NAME, EMBED_MODEL, get_top_k,
    BASELINE_INDEX_DIR, llm_kwargs, N_SOUS_AGENTS,
    PROMPT_GENERATION,
)
from src.metrics import cout_usd, usage_reel, compter_tokens, cout_embedding_usd
from src.systems.base import Reponse

load_dotenv()

for _name in ("llama_index", "llama_index.core", "httpx", "httpcore", "openai"):
    logging.getLogger(_name).setLevel(logging.ERROR)

_embedder = OpenAIEmbedding(model=EMBED_MODEL)
_llm = None
_sous_agents: list[FunctionAgent] | None = None


class _CompteurTokensMA(BaseEventHandler):
    """Additionne les tokens de tous les appels LLM du pipeline (décomposition + sous-agents + synthèse)."""

    tokens_in: int = 0
    tokens_out: int = 0

    @classmethod
    def class_name(cls) -> str:
        return "_CompteurTokensMultiAgent"

    def reset(self):
        self.tokens_in = 0
        self.tokens_out = 0

    def handle(self, event, **kwargs):
        if not isinstance(event, LLMChatEndEvent):
            return
        raw = getattr(event.response, "raw", None)
        t_in, t_out = usage_reel(raw)
        if t_in == 0 and event.messages:
            for msg in event.messages:
                content = msg.content if isinstance(msg.content, str) else ""
                t_in += compter_tokens(content, MODEL_NAME)
            if event.response and event.response.message:
                t_out = compter_tokens(
                    event.response.message.content or "", MODEL_NAME
                )
        self.tokens_in += t_in
        self.tokens_out += t_out


_compteur = _CompteurTokensMA()
get_dispatcher().add_event_handler(_compteur)


def _build():
    """Construit le LLM partagé et les N_SOUS_AGENTS sous-agents une seule fois."""
    global _llm, _sous_agents
    if _llm is not None:
        return _llm, _sous_agents

    if not BASELINE_INDEX_DIR.exists():
        raise FileNotFoundError(
            f"Index introuvable dans {BASELINE_INDEX_DIR}. "
            f"Exécuter d'abord : python -m src.build.baseline_build"
        )

    _llm = OpenAI(**llm_kwargs())

    storage = StorageContext.from_defaults(persist_dir=str(BASELINE_INDEX_DIR))
    index = load_index_from_storage(storage, embed_model=_embedder)
    qa_template = PromptTemplate(
        PROMPT_GENERATION.replace("{context}", "{context_str}").replace("{question}", "{query_str}")
    )
    query_engine = index.as_query_engine(
        llm=_llm,
        similarity_top_k=get_top_k("multi_agent"),
        text_qa_template=qa_template,
    )

    outil_recherche = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="recherche_documents",
        description=(
            "Recherche dans la base documentaire et retourne les passages les plus "
            "pertinents pour la requête donnée. Utiliser une requête courte et précise."
        ),
    )

    _sous_agents = [
        FunctionAgent(
            tools=[outil_recherche],
            llm=_llm,
            name=f"SousAgent_{i}",
            system_prompt=(
                "Tu es un agent de recherche documentaire. "
                "Pour chaque sous-question, effectue plusieurs requêtes variées avant de conclure. "
                "Retourne toutes les informations pertinentes trouvées."
            ),
            initial_tool_choice="required",
        )
        for i in range(N_SOUS_AGENTS)
    ]

    return _llm, _sous_agents


def _parser_sous_questions(texte: str, question: str, k: int) -> list[str]:
    """Extrait jusqu'à k sous-questions d'une réponse numérotée. Fallback sur la question originale."""
    sous_questions = []
    for ligne in texte.splitlines():
        match = re.match(r'^\d+[.)]\s*(.+)', ligne.strip())
        if match:
            sq = match.group(1).strip()
            if sq:
                sous_questions.append(sq)
    if not sous_questions:
        return [question] * k
    return sous_questions[:k]


async def _decomposer(llm, question: str, k: int) -> list[str]:
    """Phase 1 : décompose la question en au plus k sous-questions complémentaires (1 appel LLM)."""
    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                f"Tu décomposes une question en au plus {k} sous-questions "
                "complémentaires, non redondantes, couvrant des aspects différents du sujet. "
                "Propose-en moins si la question ne s'y prête pas. "
                f"Réponds uniquement avec les sous-questions numérotées (1. ... 2. ...). "
                "Aucun autre texte."
            ),
        ),
        ChatMessage(
            role=MessageRole.USER,
            content=f"Question : {question}",
        ),
    ]
    response = await llm.achat(messages)
    return _parser_sous_questions(response.message.content or "", question, k)


async def _synthétiser(llm, question: str, resultats: list) -> str:
    """Phase 3 : synthétise les résultats des K sous-agents en une réponse finale (1 appel LLM)."""
    contexte = "\n\n---\n\n".join(str(r) for r in resultats)
    prompt = PROMPT_GENERATION.format(context=contexte, question=question)
    messages = [
        ChatMessage(role=MessageRole.USER, content=prompt),
    ]
    response = await llm.achat(messages)
    return (response.message.content or "").strip()


async def _ask_async(question: str) -> Reponse:
    llm, sous_agents = _build()
    _compteur.reset()

    start = time.perf_counter()

    # Phase 1 : décomposition de la question en N_SOUS_AGENTS sous-questions
    sous_questions = await _decomposer(llm, question, N_SOUS_AGENTS)

    # Phase 2 : N_SOUS_AGENTS sous-agents tournent en parallèle
    resultats = await asyncio.gather(*[
        agent.run(sq)
        for agent, sq in zip(sous_agents, sous_questions)
    ])

    # Phase 3 : synthèse des résultats
    texte_final = await _synthétiser(llm, question, list(resultats))

    latence = time.perf_counter() - start

    # LLM : décomposition + tous les sous-agents + synthèse (capturés par le compteur)
    cout = cout_usd(MODEL_NAME, _compteur.tokens_in, _compteur.tokens_out)
    # Embeddings : une par sous-question envoyée au retriever
    for sq in sous_questions:
        cout += cout_embedding_usd(EMBED_MODEL, compter_tokens(sq, EMBED_MODEL))

    return Reponse(
        texte=texte_final,
        latence_s=latence,
        cout_usd=cout,
        sources=[],
    )


def ask(question: str) -> Reponse:
    return asyncio.run(_ask_async(question))
