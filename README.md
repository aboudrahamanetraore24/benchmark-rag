# Benchmark RAG

Compare six architectures de retrieval-augmented generation sur les mêmes questions, le même corpus et le même modèle de génération.

| Système | Retrieval | Clé `--systemes` |
|---|---|---|
| RAG classique | Vectoriel plat (cosine) | `baseline` |
| RAG hybride | BM25 + vectoriel, fusion RRF | `hybrid` |
| Mono-Agentic | Agent unique multi-requêtes (index baseline) | `mono_agentic` |
| Multi-Agent | Manager → N sous-agents parallèles → synthèse (index baseline) | `multi_agent` |
| RAPTOR | Arbre de résumés hiérarchique | `raptor` |
| LightRAG | Graphe d'entités/relations | `lightrag` |

---

## Installation

```bash
# Prérequis : Python 3.12, uv (https://docs.astral.sh/uv/)
uv sync          # crée le venv et installe toutes les dépendances
cp .env.example .env   # puis renseigner OPENAI_API_KEY dans .env
```

> Les versions de `numba`, `numpy` et `scikit-learn` sont contraintes dans `pyproject.toml` pour assurer la compatibilité avec `umap-learn`. Ne pas les upgrader manuellement.

---

## Pipeline d'exécution

### 0. Préparer le corpus

Placer les fichiers sources dans `data/raw/` (sous-dossiers autorisés).  
Formats supportés : `.pdf`, `.docx`, `.pptx`, `.txt`.

```
data/
└── raw/
    ├── document1.pdf
    ├── document2.docx
    └── sous-dossier/
        └── document3.pptx
```

> Il est aussi possible de fournir directement un `documents_extracted.jsonl` sans passer par `data/raw/`, tant que chaque ligne respecte le format `{"doc_id": "...", "source_file": "...", "file_type": "...", "text": "..."}`.

### 1. Extraction (une seule fois par corpus)

```bash
python -m src.prepare
```

Produit `data/extracted/documents_extracted.jsonl`.

### 2. Indexation (une seule fois, ou après changement de corpus)

L'ordre ci-dessous doit être respecté : **hybrid avant baseline**. Dans `main.py`, le RAG hybride s'exécute en premier à froid ; lancer son indexation en premier garantit que les mesures de temps sont cohérentes (pas de biais lié au warm-up de la connexion HTTP vers l'API d'embedding).

```bash
python -m src.build.hybrid_build     # index vectoriel + corpus BM25, fusion RRF à la requête  ← en premier
python -m src.build.baseline_build   # index vectoriel plat  (réutilisé par Mono-Agentic et Multi-Agent)
python -m src.build.raptor_build     # arbre RAPTOR : UMAP + GMM + BIC + résumés LLM
python -m src.build.lightrag_build   # graphe d'entités/relations (long, appels LLM)
```

### 3. Benchmark

```bash
python -m main                                              # tous les systèmes, toutes les questions
python -m main --systemes baseline hybrid                   # sous-ensemble de systèmes
python -m main --systemes baseline mono_agentic multi_agent # agents uniquement
python -m main --systemes baseline raptor lightrag          # systèmes avec indexation propre
python -m main --judge                                      # + évaluation qualité par LLM
```

Sorties générées dans `data/` :
- `results.jsonl` — métriques brutes (latence, coût, tokens) par (système, question)
- `results.md` — rapport texte
- `evaluation.html` — grille de notation côte à côte (+ scores du judge si `--judge`)

---

## Évaluation automatique (`--judge`)

Le flag `--judge` active un LLM-as-judge indépendant (`JUDGE_MODEL`, ici `gpt-5.5`) qui attribue à chaque réponse un score de **qualité globale** (0.0 → 1.0) accompagné d'une phrase d'explication. Le judge est volontairement différent du modèle de génération (`MODEL_NAME`) pour éviter le biais d'auto-évaluation.

Le rapport HTML affiche pour chaque réponse la note et l'explication du judge, ainsi qu'un bar chart des scores moyens par système.

```bash
python -m main --judge
python -m main --systemes baseline raptor --judge
```

---

## Questions

Les questions du benchmark sont définies dans [`questions.py`](questions.py) sous la forme d'une liste de dictionnaires `{"texte": "..."}`. Adapter ce fichier au corpus cible.

Chaque question accepte optionnellement une clé `"portee"` (`"locale"`, `"globale"` ou `"piege"`) :

```python
QUESTIONS = [
    {"texte": "Question sans portée définie"},                              
    {"texte": "Fait précis, passage ciblé du corpus", "portee": "locale"},
    {"texte": "Synthèse transversale sur tout le corpus", "portee": "globale"},
    {"texte": "Événement réel mais hors du périmètre du corpus", "portee": "piege"},
]
```

La clé `"portee"` est entièrement facultative : le benchmark fonctionne à l'identique sans elle (voir `questions.py` pour un exemple sans portée). Quand elle est renseignée :
- elle s'affiche en badge sur chaque question du rapport HTML ;
- en mode `--judge`, `"piege"` bascule le judge sur un prompt qui note positivement un refus ("non trouvé dans le corpus") plutôt que l'exactitude factuelle ;
- le bar chart de scores du judge ventile alors les moyennes par portée au lieu d'afficher une seule série "Qualité".

Idéalement, un corpus est couvert par un mélange de questions locales, globales et pièges.

---

## Configuration

Tous les paramètres partagés (modèle, TOP_K, SAMPLE_SIZE, CHUNK_SIZE…) sont centralisés dans [`src/config.py`](src/config.py).

- `MODEL_NAME` — modèle utilisé par tous les systèmes RAG pour la génération
- `JUDGE_MODEL` — modèle utilisé par le LLM-as-judge (indépendant de `MODEL_NAME`)

### TOP_K — mode global ou variable

Par défaut (`TOP_K_MODE = "global"`), tous les systèmes utilisent la même valeur `TOP_K`.

```python
TOP_K_MODE = "global"
TOP_K = 10  # valeur commune à tous les systèmes
```

Pour que chaque système utilise sa propre valeur, passer en mode variable :

```python
TOP_K_MODE = "variable"

TOP_K_PAR_SYSTEME = {
    "baseline":     8,
    "hybrid":       8,
    "mono_agentic": 6,
    "multi_agent":  5,
    "raptor":       5,
    "lightrag":     10,
}
```

### N_SOUS_AGENTS — nombre de sous-agents parallèles (Multi-Agent)

Le système Multi-Agent décompose chaque question en `N_SOUS_AGENTS` sous-questions traitées simultanément via `asyncio.gather()`. Configurable dans `src/config.py` :

```python
N_SOUS_AGENTS = 3
```

---

## Structure

```
├── main.py              # point d'entrée du benchmark et génération des rapports
├── questions.py         # questions soumises à tous les systèmes
├── pyproject.toml       # dépendances (uv)
└── src/
    ├── config.py        # paramètres partagés (modèle, TOP_K, chemins…)
    ├── corpus.py        # sélection déterministe du corpus (invariant du benchmark)
    ├── evaluate.py      # LLM-as-judge (qualité sans référence)
    ├── metrics.py       # coûts, comptage tokens, log JSONL
    ├── prepare.py       # extraction texte → documents_extracted.jsonl
    ├── build/
    │   ├── baseline_build.py   # chunking 512 tokens + index vectoriel
    │   ├── hybrid_build.py     # chunking 512 tokens + index vectoriel + corpus BM25
    │   ├── raptor_build.py     # chunking 512 tokens + arbre RAPTOR
    │   └── lightrag_build.py   # graphe LightRAG (chunking interne)
    └── systems/
        ├── base.py             # dataclasses Reponse et Source
        ├── generation.py       # génération partagée (baseline, hybrid, RAPTOR)
        ├── baseline.py         # recherche vectorielle plate
        ├── hybrid.py           # BM25 + vectoriel, Reciprocal Rank Fusion
        ├── mono_agentic.py     # agent unique avec outil de recherche
        ├── multi_agent.py      # Manager/Workers : décomposition → N sous-agents parallèles → synthèse
        ├── raptor.py           # retrieval sur arbre de résumés
        └── lightrag.py         # requêtes sur graphe d'entités
```

---

## Rapports

Le dossier [`docs/`](docs/) contient un rapport d'évaluation généré par ce benchmark, ainsi que le mémoire qui a inspiré ce projet :

- [`benchmark-histoire-de-france-michelet-gpt-5.4.html`](docs/benchmark-histoire-de-france-michelet-gpt-5.4.html) : grille de notation (modèle `gpt-5.4`) du benchmark exécuté sur les tomes V à VII de l'*Histoire de France* de Jules Michelet (1364-1465 : fin du règne de Charles V, folie de Charles VI, traité de Troyes, Jeanne d'Arc, reconquête du royaume par Charles VII). Les questions pièges portent sur des événements postérieurs à 1465 (bataille de Marignan, mort de Charles le Téméraire, édit de Nantes…) afin de vérifier que chaque système signale l'absence de l'information dans le corpus plutôt que de répondre depuis ses connaissances générales.
- [`memoire-comparaison-architectures-rag.pdf`](docs/memoire-comparaison-architectures-rag.pdf) : mémoire de master d'Erik Lundberg (Lund University), *"When Does Graph RAG Pay Off? A Systematic Comparison of LightRAG and Chunk-Based RAG Pipelines"*, dont la comparaison systématique d'architectures RAG a inspiré la conception de ce benchmark.

> Le benchmark a aussi été exécuté sur un corpus de documents internes d'entreprise (projets clients) durant le stage, non inclus ici pour des raisons de confidentialité.

---

## Références

- [Searching for Best Practices in Retrieval-Augmented Generation](https://arxiv.org/pdf/2502.11371) : étude comparative exhaustive de différentes architectures et configurations RAG.
- [RAG : arrêtez de chercher, commencez à classer](https://www.askaibrain.com/posts/rag-arretez-de-chercher-commencez-a-classer) : synthèse de l'état de l'art actuel sur les approches RAG.
