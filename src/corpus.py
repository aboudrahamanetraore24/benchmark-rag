"""
Source unique de vérité sur le corpus du benchmark.

Tous les systèmes (baseline, raptor, lightrag) passent par ce module,
garantissant qu'ils opèrent sur exactement le même ensemble de documents.

Invariant de comparabilité :
    Chaque système reçoit les mêmes documents sources. Le découpage (chunking)
    est laissé à chaque système selon son architecture.

Échantillonnage :
    - Unité = le document (une ligne dans documents_extracted.jsonl = un fichier source).
    - Déterministe : ordre d'apparition dans le fichier, puis troncature à SAMPLE_SIZE.
    - SAMPLE_SIZE = None → corpus entier.
"""

import json
from pathlib import Path

from src.config import SAMPLE_SIZE

EXTRACTED_PATH = Path("data/extracted/documents_extracted.jsonl")


def _lire_documents_extraits() -> list[dict]:
    with EXTRACTED_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def doc_ids_du_benchmark(sample_size: int | None = SAMPLE_SIZE) -> list[str]:
    """Retourne la liste déterministe des doc_id composant le benchmark."""
    rows = _lire_documents_extraits()
    if sample_size is not None:
        rows = rows[:sample_size]
    return [row["doc_id"] for row in rows]


def documents_du_benchmark(sample_size: int | None = SAMPLE_SIZE) -> tuple[list[str], list[str]]:
    """Retourne (textes, ids) des documents retenus.

    Chaque système est responsable de son propre découpage.
    """
    rows = _lire_documents_extraits()
    if sample_size is not None:
        rows = rows[:sample_size]
    return [row["text"] for row in rows], [row["doc_id"] for row in rows]
