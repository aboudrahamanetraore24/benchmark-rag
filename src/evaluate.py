"""
LLM-as-judge — score de qualité globale de la réponse.
"""

import json
import logging

import openai

from src.config import JUDGE_MODEL

log = logging.getLogger("evaluate")
_client = openai.OpenAI()

_PROMPT = """\
Tu es un expert chargé d'évaluer les réponses produites par des systèmes RAG, sur la base \
du corpus de documents qui leur a été fourni.

Question posée : {question}

Réponse du système : {answer}

Attribue un score de qualité entre 0.0 et 1.0, basé principalement sur :
1. L'exactitude de ce qui est affirmé dans la réponse — les faits avancés sont-ils plausibles \
et cohérents avec ce qu'on attendrait d'une réponse fondée sur le corpus ? C'est le critère dominant.
2. L'absence d'hallucination — la réponse n'invente-t-elle pas des faits, entités ou \
événements qui n'existent pas ?
Utilise des valeurs décimales précises pour bien discriminer les réponses.

Réponds uniquement en JSON valide, sans texte en dehors du JSON :
{{"qualite": <float 0.0-1.0>, "reason": "<une phrase résumant le principal point fort ou faible>"}}"""

_PROMPT_PIEGE = """\
Tu es un expert chargé d'évaluer les réponses produites par des systèmes RAG.

Cette question est une question piège : elle porte sur un sujet réel mais SITUÉ HORS \
du périmètre du corpus indexé par les systèmes évalués. La réponse correcte est donc \
que le corpus ne contient pas cette information — le système doit le signaler clairement \
(ex. "non trouvé dans le corpus", "hors du périmètre couvert") plutôt que de répondre \
depuis ses connaissances générales.

Question posée : {question}

Réponse du système : {answer}

Attribue un score de qualité entre 0.0 et 1.0 :
- Score élevé (proche de 1.0) si la réponse indique explicitement que l'information \
n'est pas dans le corpus / hors de son périmètre couvert, qu'elle soit vraie ou non par \
ailleurs.
- Score faible (proche de 0.0) si la réponse affirme des faits sur le sujet comme \
si elle les avait trouvés dans le corpus, révélant qu'elle s'appuie sur ses connaissances \
générales plutôt que sur le contexte récupéré — même si ces faits sont exacts par ailleurs.
- Score intermédiaire si la réponse est ambiguë (mélange un refus partiel avec des faits \
non sourcés, par exemple).
Utilise des valeurs décimales précises pour bien discriminer les réponses.

Réponds uniquement en JSON valide, sans texte en dehors du JSON :
{{"qualite": <float 0.0-1.0>, "reason": "<une phrase résumant pourquoi la réponse évite ou non le piège>"}}"""


def judge(
    question: str,
    answer: str,
    contexts: list[str] | None = None,
    portee: str | None = None,
) -> dict:
    """
    Retourne {"qualite": float, "reason": str}.
    Ne lève jamais d'exception — retourne des scores None en cas d'échec.

    portee="piege" bascule sur un prompt qui note positivement un refus
    ("non trouvé dans le corpus") plutôt que l'exactitude factuelle pure.
    """
    template = _PROMPT_PIEGE if portee == "piege" else _PROMPT
    prompt = template.format(question=question, answer=answer)

    try:
        resp = _client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        raw = json.loads(resp.choices[0].message.content)
        return {
            "qualite": float(raw["qualite"]) if raw.get("qualite") is not None else None,
            "reason":  str(raw.get("reason", "")),
        }
    except Exception as e:
        log.error("Judge error: %s", e)
        return {"qualite": None, "reason": f"erreur : {e}"}
