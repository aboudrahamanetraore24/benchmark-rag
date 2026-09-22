"""
Configuration partagée entre tous les systèmes du benchmark.

Invariant : tous les systèmes partent du même ensemble de documents (voir corpus.py).
La seule variable autorisée entre deux systèmes est leur architecture de retrieval.
Modèle, embeddings, top_k et prompt de génération sont fixés ici.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

# Modèle de génération commun à tous les systèmes.
# Mettre à jour MODEL_NAME, SUPPORTE_TEMPERATURE et les tarifs dans metrics.py
# lors d'un changement de modèle.
MODEL_NAME = "gpt-5.4"

# Modèle utilisé comme juge indépendant (évaluation LLM-as-judge).
# Volontairement différent de MODEL_NAME pour éviter le biais d'auto-évaluation.
JUDGE_MODEL = "gpt-5.5"

# Les modèles de raisonnement (o3, o4…) rejettent le paramètre temperature via l'API.
# Mettre à False pour ces modèles afin d'éviter une erreur 400.
SUPPORTE_TEMPERATURE = True

# Reproductibilité des réponses. Ignoré si SUPPORTE_TEMPERATURE est False.
TEMPERATURE = 0.1


def llm_kwargs() -> dict:
    """Retourne les kwargs du client LLM selon le modèle configuré.

    Point d'entrée unique pour tous les systèmes : centralise la logique
    de compatibilité temperature afin d'éviter toute duplication.
    """
    kwargs = {"model": MODEL_NAME}
    if SUPPORTE_TEMPERATURE:
        kwargs["temperature"] = TEMPERATURE
    return kwargs


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

# Nombre de sous-agents lancés en parallèle par le système Multi-Agent.
N_SOUS_AGENTS = 3

# Nombre de résultats retournés par le retriever.
#
# Deux modes disponibles via TOP_K_MODE :
#   "global"   → tous les systèmes utilisent TOP_K
#   "variable" → chaque système utilise sa propre valeur dans TOP_K_PAR_SYSTEME
TOP_K = 10
TOP_K_MODE = "global"

TOP_K_PAR_SYSTEME = {
    "baseline":     8,   # chunks plats 512 tok → ~4k contexte
    "hybrid":       8,   # idem ; la fusion RRF reclasse, ne réduit pas le besoin
    "mono_agentic": 6,   # l'agent re-filtre en aval, k brut un peu plus serré
    "multi_agent":  5,   # PAR sous-agent → cumulé ≈ 5×nb_agents, donc petit
    "raptor":       5,   # nœuds = résumés de clusters, denses → moins d'unités
    "lightrag":     10,  # unités graphe fragmentées → plus d'unités pr/ couvrir
}

def get_top_k(systeme: str) -> int:
    """Retourne le top-k du système selon le mode actuel (global ou variable)."""
    if TOP_K_MODE == "variable":
        return TOP_K_PAR_SYSTEME.get(systeme, TOP_K)
    return TOP_K

# ---------------------------------------------------------------------------
# Corpus & Chunking
# ---------------------------------------------------------------------------

# Nombre de documents retenus pour le benchmark (None = corpus entier).
# RAPTOR et LightRAG effectuant des appels LLM à l'indexation, cette valeur
# agit comme garde-fou de coût sur les gros corpus.
# Sélection déterministe par ordre d'apparition dans documents_extracted.jsonl — voir corpus.py.
SAMPLE_SIZE = None

# Paramètres de découpage utilisés par baseline_build et raptor_build.
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# ---------------------------------------------------------------------------
# Prompt de génération
# ---------------------------------------------------------------------------

# Prompt de génération commun à tous les systèmes RAG du benchmark.
# Modifier ici pour affecter baseline, RAPTOR, hybrid, mono-agentic et multi-agent.
PROMPT_GENERATION = (
    "Tu es un assistant qui répond uniquement à partir du contexte fourni.\n"
    "Utilise exclusivement les informations du contexte pour construire ta réponse.\n"
    "Si la réponse ne figure pas dans le contexte, dis-le clairement.\n"
    "Réponds de manière concise et précise.\n\n"
    "Contexte :\n"
    "{context}\n\n"
    "Question :\n"
    "{question}\n\n"
    "Réponse :"
)

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")

BASELINE_INDEX_DIR = DATA_DIR / "baseline_index"
RAPTOR_INDEX_DIR   = DATA_DIR / "raptor_index"
LIGHTRAG_INDEX_DIR = DATA_DIR / "lightrag_index"
HYBRID_INDEX_DIR   = DATA_DIR / "hybrid_index"
