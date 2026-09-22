"""
Interface commune à tous les systèmes RAG.

Chaque système expose ask(question: str) -> Reponse.
Reponse porte le texte généré, les sources récupérées, la latence et le coût.
Sources incluses pour distinguer un mauvais retrieval d'une mauvaise génération.
"""

from dataclasses import dataclass, field


@dataclass
class Source:
    """Un chunk récupéré par un système, avec son score de similarité si dispo."""
    texte: str
    score: float | None = None        # score de similarité (None si non exposé)
    identifiant: str | None = None     # chunk_id / doc_id si disponible


@dataclass
class Reponse:
    texte: str
    latence_s: float
    cout_usd: float
    sources: list[Source] = field(default_factory=list)
