"""
Outils de mesure (temps, coût, comptage de tokens) partagés par tous les systèmes.

Contient CompteurLLM : un wrapper unique qui instrumente les appels LLM pour
compter les tokens consommés, utilisé par LightRAG (indexation et requête).
"""

import fcntl
import json
import time
from pathlib import Path

import tiktoken

RESULTS_PATH = Path("data/results.jsonl")

# Tarifs OpenAI en dollars par million de tokens (source : openai.com/pricing).
# À mettre à jour lors d'un changement de modèle dans config.py.
PRIX = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-5-mini":  {"input": 0.25, "output": 2.00},
    "o4-mini":     {"input": 1.10, "output": 4.40},
    "gpt-5.4":     {"input": 2.50, "output": 15.00},
    "gpt-5.5":     {"input": 5.00, "output": 30.00},
}

PRIX_EMBEDDING = {
    "text-embedding-3-small": 0.02,
}

_ENCODERS = {}


def _encoder(modele: str):
    if modele not in _ENCODERS:
        try:
            _ENCODERS[modele] = tiktoken.encoding_for_model(modele)
        except KeyError:
            _ENCODERS[modele] = tiktoken.get_encoding("o200k_base")
    return _ENCODERS[modele]


def compter_tokens(texte: str, modele: str = "gpt-4o-mini") -> int:
    return len(_encoder(modele).encode(texte or ""))


def cout_usd(modele: str, tokens_in: int, tokens_out: int) -> float:
    p = PRIX.get(modele, {"input": 0, "output": 0})
    return (tokens_in * p["input"] + tokens_out * p["output"]) / 1_000_000


def cout_embedding_usd(modele: str, tokens: int) -> float:
    prix = PRIX_EMBEDDING.get(modele, 0)
    return tokens * prix / 1_000_000


def usage_reel(raw) -> tuple[int, int]:
    """Lit (tokens_in, tokens_out) depuis la réponse LLM quand disponible."""
    if raw is None:
        return 0, 0

    # Extraire l'objet usage
    if isinstance(raw, dict):
        usage = raw.get("usage", raw)
    else:
        usage = getattr(raw, "usage", raw)

    if usage is None:
        return 0, 0

    def _get(obj, *noms):
        for n in noms:
            v = obj.get(n) if isinstance(obj, dict) else getattr(obj, n, None)
            if v:
                return v
        return 0

    t_in = _get(usage, "prompt_tokens", "input_tokens")
    t_out = _get(usage, "completion_tokens", "output_tokens")
    return t_in, t_out


def chrono(fn, *args, **kwargs):
    """Exécute fn en la chronométrant. Renvoie (résultat, durée_en_secondes)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


def log_metric(record: dict, path: Path = RESULTS_PATH):
    """Append une ligne JSON. Verrou exclusif requis : build et évaluation
    tournent parfois en parallèle dans des process séparés et des écritures
    concurrentes non verrouillées peuvent fusionner deux lignes sans saut de
    ligne, corrompant le fichier au parsing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


class CompteurLLM:
    """Compteur de tokens in/out. Fournit wrap() pour instrumenter un appel LLM async."""

    def __init__(self, modele: str):
        self.modele = modele
        self.tokens_in = 0
        self.tokens_out = 0

    def reset(self):
        self.tokens_in = 0
        self.tokens_out = 0

    def ajouter_entree(self, texte: str):
        self.tokens_in += compter_tokens(texte, self.modele)

    def ajouter_sortie(self, texte: str):
        self.tokens_out += compter_tokens(texte, self.modele)

    def cout(self) -> float:
        return cout_usd(self.modele, self.tokens_in, self.tokens_out)

    def wrap(self, complete_fn):
        """Enrobe complete_fn pour compter ses tokens. Le nom de modèle est injecté depuis self.modele."""

        async def _wrapped(prompt, system_prompt=None, history_messages=None, **kwargs):
            history_messages = history_messages or []

            entree = system_prompt or ""
            for m in history_messages:
                entree += m.get("content", "") if isinstance(m, dict) else str(m)
            entree += prompt
            self.ajouter_entree(entree)

            resp = await complete_fn(
                self.modele,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **kwargs,
            )
            self.ajouter_sortie(resp)
            return resp

        return _wrapped
