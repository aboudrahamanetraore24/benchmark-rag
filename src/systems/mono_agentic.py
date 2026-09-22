"""
Système Mono-Agentic RAG. Expose ask().

Un seul FunctionAgent reçoit la question et dispose d'un outil de recherche sur l'index
du RAG classique. Il décide lui-même comment formuler ses requêtes et en émet
plusieurs avant de synthétiser la réponse finale.

Limites documentées :
  - Coût des embeddings de reformulations internes non comptabilisé.
  - Sources intermédiaires (chunks bruts) non exposées.
"""

import asyncio
import logging
import time

from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events.llm import LLMChatEndEvent
from llama_index.llms.openai import OpenAI 
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.core import PromptTemplate
from src.config import MODEL_NAME, EMBED_MODEL, get_top_k, BASELINE_INDEX_DIR, llm_kwargs, PROMPT_GENERATION
from src.metrics import cout_usd, usage_reel, compter_tokens, cout_embedding_usd
from src.systems.base import Reponse, Source

load_dotenv()

for _name in ("llama_index", "llama_index.core", "httpx", "httpcore", "openai"):
    logging.getLogger(_name).setLevel(logging.ERROR)

_embedder = OpenAIEmbedding(model=EMBED_MODEL)
_agent = None

# Mêmes instructions de réponse que tous les autres systèmes (PROMPT_GENERATION),
# sans {context}/{question} : ici le contexte arrive via l'outil de recherche, pas
# par injection statique. Seul ajout : l'orchestration propre à l'agent.
_AGENT_SYSTEM_PROMPT = (
    PROMPT_GENERATION.split("\n\nContexte :")[0]
    + "\n\nUtilise l'outil de recherche avec plusieurs requêtes variées pour rassembler "
      "le contexte nécessaire avant de répondre."
)


class _CompteurTokens(BaseEventHandler):
    """Additionne les tokens de tous les appels LLM de l'agent (planification + synthèse)."""

    tokens_in: int = 0
    tokens_out: int = 0

    @classmethod
    def class_name(cls) -> str:
        return "_CompteurTokensMonoAgentic"

    def reset(self):
        self.tokens_in = 0
        self.tokens_out = 0

    def handle(self, event, **kwargs):
        if not isinstance(event, LLMChatEndEvent):
            return
        raw = getattr(event.response, "raw", None)
        t_in, t_out = usage_reel(raw)
        if t_in == 0 and event.messages:
            # fallback tiktoken si usage absent (streaming résiduel ou chunk sans usage)
            for msg in event.messages:
                content = msg.content if isinstance(msg.content, str) else ""
                t_in += compter_tokens(content, MODEL_NAME)
            if event.response and event.response.message:
                t_out = compter_tokens(
                    event.response.message.content or "", MODEL_NAME
                )
        self.tokens_in += t_in
        self.tokens_out += t_out


_compteur = _CompteurTokens()
get_dispatcher().add_event_handler(_compteur)


def _build_agent():
    """Construit l'agent une seule fois : index → outil de recherche → agent."""
    global _agent
    if _agent is not None:
        return _agent

    if not BASELINE_INDEX_DIR.exists():
        raise FileNotFoundError(
            f"Index introuvable dans {BASELINE_INDEX_DIR}. "
            f"Exécuter d'abord : python -m src.build.baseline_build"
        )

    llm = OpenAI(**llm_kwargs())

    storage = StorageContext.from_defaults(persist_dir=str(BASELINE_INDEX_DIR))
    index = load_index_from_storage(storage, embed_model=_embedder)
    qa_template = PromptTemplate(
        PROMPT_GENERATION.replace("{context}", "{context_str}").replace("{question}", "{query_str}")
    )
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=get_top_k("mono_agentic"),
        text_qa_template=qa_template,
    )

    outil_recherche = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="recherche_documents",
        description=(
            "Recherche dans la base documentaire interne et retourne les passages "
            "les plus pertinents pour une requête donnée. Utiliser plusieurs requêtes "
            "ciblées pour couvrir différents aspects d'un sujet large."
        ),
    )
    _agent = FunctionAgent(
        tools=[outil_recherche],
        llm=llm,
        streaming=False,
        # "required" : le modèle DOIT appeler au moins un outil au premier tour,
        # ce qui empêche l'agent de demander des clarifications sans chercher.
        initial_tool_choice="required",
        system_prompt=_AGENT_SYSTEM_PROMPT,
        )
    return _agent


async def _ask_async(question: str) -> Reponse:
    agent = _build_agent()
    _compteur.reset()

    start = time.perf_counter()
    reponse = await agent.run(question)
    latence = time.perf_counter() - start

    # LLM : somme de tous les tours (sélection d'outil + synthèse finale)
    cout = cout_usd(MODEL_NAME, _compteur.tokens_in, _compteur.tokens_out)
    # Embedding de la question initiale (les reformulations internes ne sont pas comptées)
    cout += cout_embedding_usd(EMBED_MODEL, compter_tokens(question, EMBED_MODEL))

    return Reponse(
        texte=str(reponse).strip(),
        latence_s=latence,
        cout_usd=cout,
        sources=[],
    )


def ask(question: str) -> Reponse:
    return asyncio.run(_ask_async(question))
