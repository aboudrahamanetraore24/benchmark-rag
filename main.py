"""
Lance le benchmark : chaque système répond à chaque question.

Produit :
  - data/results.jsonl   : métriques (latence, coût) par (système, question)
  - data/results.md      : rapport texte lisible
  - data/evaluation.html : grille de notation manuelle côte à côte

Usage :
    python -m main
    python -m main --systemes baseline raptor
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import html
import json
import logging
from datetime import datetime
from pathlib import Path

from src.config import get_top_k, TOP_K_MODE, MODEL_NAME, SAMPLE_SIZE, N_SOUS_AGENTS
from src.metrics import log_metric
from src.evaluate import judge
from src.systems import baseline, mono_agentic, multi_agent, raptor, lightrag, hybrid


# --- Silence des logs verbeux ---------------------------------------------
class _NoLowLevelKeywords(logging.Filter):
    def filter(self, record):
        return "low_level_keywords" not in record.getMessage()


logging.getLogger("lightrag").addFilter(_NoLowLevelKeywords())
logging.getLogger().addFilter(_NoLowLevelKeywords())
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("benchmark")

for _name in ("llama_index", "llama_index.core", "nano-vectordb", "lightrag",
              "httpx", "httpcore", "openai"):
    logging.getLogger(_name).setLevel(logging.WARNING)

from questions import QUESTIONS

RESULTS_MD = Path("data/results.md")
RESULTS_HTML = Path("data/evaluation.html")

SYSTEMES = [
    ("RAG hybride", hybrid),
    ("RAG classique", baseline),
    ("Mono-Agentic", mono_agentic),
    ("Multi-Agent", multi_agent),
    ("RAPTOR", raptor),
    ("LightRAG", lightrag),
]

CONDITIONS = {
    "RAG classique": f"recherche vectorielle plate, top_k={get_top_k('baseline')}",
    "RAG hybride": f"BM25 + vectoriel, fusion RRF (k=60), top_k={get_top_k('hybrid')}",
    "Mono-Agentic": f"agent unique + outil de recherche (index RAG classique), top_k={get_top_k('mono_agentic')}",
    "Multi-Agent": f"Manager → {N_SOUS_AGENTS} sous-agents parallèles (asyncio.gather) → synthèse, top_k={get_top_k('multi_agent')}",
    "RAPTOR": f"arbre de résumés, top_k={get_top_k('raptor')}",
    "LightRAG": f"graphe d'entités, mode={lightrag.MODE}, top_k={get_top_k('lightrag')}",
}


# ======================================================================
# Génération de la grille HTML de notation
# ======================================================================
HTML_STYLE = """
<style>
  body { font-family: -apple-system, system-ui, sans-serif; margin: 0; padding: 24px;
         background: #f5f5f4; color: #1c1917; line-height: 1.5; }
  h1 { font-size: 22px; }
  .meta { color: #57534e; font-size: 13px; margin-bottom: 24px; }
.question-block { margin-bottom: 48px; }
  .question-title { font-size: 17px; background: #1c1917; color: #fafaf9;
                    padding: 12px 16px; border-radius: 8px; display: flex;
                    align-items: center; gap: 10px; flex-wrap: wrap; }
  .question-scope { font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em;
                    border-radius: 999px; padding: 2px 8px; white-space: nowrap; }
  .scope-locale  { background: #e0f2fe; color: #075985; }
  .scope-globale { background: #fee2e2; color: #991b1b; }
  .scope-piege   { background: #fef3c7; color: #92400e; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
          gap: 16px; margin-top: 16px; }
  .card { background: #fff; border: 1px solid #e7e5e4; border-radius: 10px;
          padding: 16px; display: flex; flex-direction: column; }
  .card-header { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap;
                 margin-bottom: 2px; }
  .sys-name { font-weight: 700; font-size: 15px; }
  .sys-tag { font-size: 10px; font-weight: 600; text-transform: uppercase;
             letter-spacing: 0.06em; border-radius: 4px; padding: 2px 6px;
             white-space: nowrap; }
  .tag-vectoriel  { background: #dbeafe; color: #1e40af; }
  .tag-hybride    { background: #dcfce7; color: #166534; }
  .tag-mono-agentic { background: #fef9c3; color: #854d0e; }
  .tag-multi-agent  { background: #ffedd5; color: #9a3412; }
  .tag-hierarchique { background: #f3e8ff; color: #6b21a8; }
  .tag-graphe     { background: #ffe4e6; color: #9f1239; }
  .metrics { font-family: monospace; font-size: 12px; color: #57534e; margin: 4px 0 12px; }
  .answer { font-size: 14px; flex-grow: 1; }
  .answer.markdown-rendered > *:first-child { margin-top: 0; }
  .answer.markdown-rendered > *:last-child { margin-bottom: 0; }
  .answer.markdown-rendered p { margin: 0 0 10px; }
  .answer.markdown-rendered ul, .answer.markdown-rendered ol { margin: 0 0 10px; padding-left: 22px; }
  .answer.markdown-rendered li { margin: 2px 0; }
  .answer.markdown-rendered li > p { margin: 0; }
  .answer.markdown-rendered h1, .answer.markdown-rendered h2, .answer.markdown-rendered h3,
  .answer.markdown-rendered h4 { font-size: 14px; margin: 14px 0 6px; }
  .answer.markdown-rendered strong { font-weight: 700; }
  .answer.markdown-rendered code { background: #f5f5f4; border-radius: 4px; padding: 1px 4px;
                                    font-size: 12.5px; }
  .answer.markdown-rendered pre { background: #f5f5f4; border-radius: 6px; padding: 8px;
                                   overflow-x: auto; }
  .answer.markdown-rendered pre code { background: none; padding: 0; }
  .answer.markdown-rendered blockquote { border-left: 3px solid #d6d3d1; margin: 0 0 10px;
                                          padding-left: 10px; color: #57534e; }
  .answer.markdown-rendered table { border-collapse: collapse; margin: 0 0 10px; }
  .answer.markdown-rendered th, .answer.markdown-rendered td { border: 1px solid #e7e5e4;
                                                                 padding: 4px 8px; }
  details { margin-top: 12px; font-size: 13px; }
  details summary { cursor: pointer; color: #0369a1; }
  .source { background: #f5f5f4; border-left: 3px solid #d6d3d1; padding: 8px;
            margin: 6px 0; font-size: 12px; white-space: pre-wrap; }
  .score-pill { font-size:12px; font-weight:700; border-radius:999px;
                padding:2px 10px; white-space:nowrap; }
  .score-high   { background:#dcfce7; color:#166534; }
  .score-mid    { background:#fef9c3; color:#854d0e; }
  .score-low    { background:#fee2e2; color:#991b1b; }
  .score-none   { background:#f5f5f4; color:#78716c; }
  .score-grid   { display:flex; gap:16px; margin: 10px 0; flex-wrap:wrap; }
  .score-item   { display:flex; flex-direction:column; align-items:center; gap:4px; }
  .score-label  { font-size:10px; font-weight:600; text-transform:uppercase;
                  letter-spacing:0.05em; color:#78716c; }
  .score-reason { font-size:11px; color:#78716c; font-style:italic; margin-top:6px; }
</style>
"""

# Badge affiché sur chaque carte pour identifier la méthode de retrieval.
_TAGS: dict[str, tuple[str, str]] = {
    "RAG classique": ("vectoriel",     "tag-vectoriel"),
    "RAG hybride":   ("vectoriel + BM25", "tag-hybride"),
    "Mono-Agentic":  ("mono-agent",    "tag-mono-agentic"),
    "Multi-Agent":   ("multi-agent",   "tag-multi-agent"),
    "RAPTOR":        ("hierarchique",  "tag-hierarchique"),
    "LightRAG":      ("graphe",        "tag-graphe"),
}

_COULEURS = {
    "RAG classique": "#3b82f6",
    "RAG hybride":   "#22c55e",
    "Mono-Agentic":  "#eab308",
    "Multi-Agent":   "#f97316",
    "RAPTOR":        "#a855f7",
    "LightRAG":      "#f43f5e",
}

# Portées de question optionnelles (voir questions.py) : si une question n'a
# pas de "portee", elle n'est pas notée séparément et n'affiche aucun badge.
_PORTEE_LABELS = {
    "locale": "Locale",
    "globale": "Globale",
    "piege": "Piège",
}

_PORTEE_COULEURS = {
    "locale": "#0ea5e9",
    "globale": "#dc2626",
    "piege": "#f59e0b",
}


def _esc(t: str) -> str:
    return html.escape(t or "")


def lire_requetes() -> dict[str, dict]:
    """Retourne latence et coût moyens par système depuis results.jsonl (phase=requete)."""
    path = Path("data/results.jsonl")
    if not path.exists():
        return {}
    totaux: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
            if rec.get("phase") != "requete":
                continue
            nom = rec["systeme"]
            if nom not in totaux:
                totaux[nom] = {"latence_s": 0.0, "cout_usd": 0.0, "n": 0}
            totaux[nom]["latence_s"] += rec.get("latence_s", 0)
            totaux[nom]["cout_usd"] += rec.get("cout_usd", 0)
            totaux[nom]["n"] += 1
        except Exception:
            continue
    return {
        nom: {
            "latence_s": round(v["latence_s"] / v["n"], 2),
            "cout_usd": round(v["cout_usd"] / v["n"], 6),
            "n": v["n"],
        }
        for nom, v in totaux.items() if v["n"] > 0
    }


def lire_indexation() -> dict:
    """Retourne la dernière entrée d'indexation par système depuis results.jsonl."""
    path = Path("data/results.jsonl")
    if not path.exists():
        return {}
    derniers = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
            if rec.get("phase") == "indexation":
                derniers[rec["systeme"]] = rec
        except Exception:
            continue
    return derniers


def _section_graphiques_html(systemes_affiches: list[str]) -> str:
    """Génère 4 bar charts Chart.js : temps/coût indexation + temps/coût réponse."""
    idx = lire_indexation()
    req = lire_requetes()
    if not idx and not req:
        return ""

    sys_idx = [n for n in systemes_affiches if n in idx]
    sys_req = [n for n in systemes_affiches if n in req]
    if not sys_idx and not sys_req:
        return ""

    n_questions = max((req[n]["n"] for n in sys_req), default=0)
    sous_titre = f"moyenne sur {n_questions} question{'s' if n_questions > 1 else ''}"

    idx_labels  = json.dumps(sys_idx)
    idx_temps   = json.dumps([idx[n].get("temps_s", 0) for n in sys_idx])
    idx_cout    = json.dumps([round(idx[n].get("cout_usd", 0), 6) for n in sys_idx])
    idx_colors  = json.dumps([_COULEURS.get(n, "#888") for n in sys_idx])

    req_labels  = json.dumps(sys_req)
    req_latence = json.dumps([req[n]["latence_s"] for n in sys_req])
    req_cout    = json.dumps([req[n]["cout_usd"] for n in sys_req])
    req_colors  = json.dumps([_COULEURS.get(n, "#888") for n in sys_req])

    js_fn = """
const mkChart = (id, labels, data, colors, fmtY) => new Chart(document.getElementById(id), {
  type: 'bar',
  data: {
    labels,
    datasets: [{ data, backgroundColor: colors.map(c => c + 'bb'), borderColor: colors, borderWidth: 1 }]
  },
  options: {
    responsive: true,
    plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => fmtY(ctx.parsed.y) } } },
    scales: { y: { beginAtZero: true, ticks: { callback: fmtY } } }
  }
});"""

    js_calls = (
        f"mkChart('chartIdxTemps', {idx_labels}, {idx_temps},   {idx_colors}, v => v + ' s');\n"
        f"mkChart('chartIdxCout',  {idx_labels}, {idx_cout},    {idx_colors}, v => '$' + v.toFixed(6));\n"
        f"mkChart('chartReqTemps', {req_labels}, {req_latence}, {req_colors}, v => v + ' s');\n"
        f"mkChart('chartReqCout',  {req_labels}, {req_cout},    {req_colors}, v => '$' + v.toFixed(6));\n"
    )

    return (
        "<h2>Comparaison des systèmes</h2>"
        "<div style='display:grid;grid-template-columns:1fr 1fr;gap:32px;margin:16px 0 48px;'>"
        "<div>"
        "<h3 style='font-size:14px;font-weight:600;margin:0 0 8px;'>Temps d'indexation"
        " <span style='font-weight:400;color:#78716c'>(s)</span></h3>"
        "<canvas id='chartIdxTemps'></canvas></div>"
        "<div>"
        "<h3 style='font-size:14px;font-weight:600;margin:0 0 8px;'>Coût d'indexation"
        " <span style='font-weight:400;color:#78716c'>($)</span></h3>"
        "<canvas id='chartIdxCout'></canvas></div>"
        "<div>"
        f"<h3 style='font-size:14px;font-weight:600;margin:0 0 8px;'>Temps de réponse"
        f" <span style='font-weight:400;color:#78716c'>{sous_titre} · s</span></h3>"
        "<canvas id='chartReqTemps'></canvas></div>"
        "<div>"
        f"<h3 style='font-size:14px;font-weight:600;margin:0 0 8px;'>Coût de réponse"
        f" <span style='font-weight:400;color:#78716c'>{sous_titre} · $</span></h3>"
        "<canvas id='chartReqCout'></canvas></div>"
        "</div>"
        "<script src='https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js'></script>"
        f"<script>{js_fn}\n{js_calls}</script>"
    )


def _pill(v: float | None) -> str:
    if v is None:
        return '<span class="score-pill score-none">N/A</span>'
    pct = int(v * 100)
    css = "score-high" if v >= 0.75 else ("score-mid" if v >= 0.4 else "score-low")
    return f'<span class="score-pill {css}">{pct}%</span>'



def _section_scores_juge_html(resultats: list) -> str:
    """Bar chart groupé : score de qualité moyen par système, un score par portée."""
    from collections import defaultdict

    totaux: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    portees_vues: list = []
    for _, portee, reps in resultats:
        if portee not in portees_vues:
            portees_vues.append(portee)
        for nom, rep, err, verdict in reps:
            if err is None and verdict:
                v = verdict.get("qualite")
                if v is not None:
                    totaux[nom][portee].append(float(v))
    if not totaux:
        return ""

    noms = list(totaux.keys())
    datasets = [
        {
            "label": _PORTEE_LABELS.get(portee, "Qualité") if portee else "Qualité",
            "data": [
                round(sum(totaux[n][portee]) / len(totaux[n][portee]), 3)
                if totaux[n][portee] else 0
                for n in noms
            ],
            "backgroundColor": _PORTEE_COULEURS.get(portee, "#3b82f6") + "bb",
            "borderColor": _PORTEE_COULEURS.get(portee, "#3b82f6"),
            "borderWidth": 1, "borderRadius": 4,
        }
        for portee in portees_vues
    ]

    return (
        "<h2>Qualité de la réponse par portée (LLM as judge)</h2>"
        "<div style='max-width:900px;margin:0 0 32px'>"
        "<canvas id='chartScoresJuge'></canvas></div>"
        "<script src='https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js'></script>"
        f"<script>new Chart(document.getElementById('chartScoresJuge'),{{"
        f"type:'bar',data:{{labels:{json.dumps(noms)},datasets:{json.dumps(datasets)}}},"
        f"options:{{responsive:true,"
        f"plugins:{{legend:{{position:'top'}},"
        f"tooltip:{{callbacks:{{label:ctx=>ctx.dataset.label+' : '+ctx.parsed.y.toFixed(2)}}}}}},"
        f"scales:{{y:{{min:0,max:1,ticks:{{callback:v=>v.toFixed(1)}}}}}}}}}})</script>"
    )


def _carte_html(nom, rep, verdict=None) -> str:
    tag_label, tag_cls = _TAGS.get(nom, ("", ""))
    badge = (
        f'<span class="sys-tag {tag_cls}">{_esc(tag_label)}</span>'
        if tag_label else ""
    )

    judge_html = ""
    if verdict:
        q = verdict.get("qualite")
        reason = verdict.get("reason", "")
        pill = _pill(q)
        judge_html = (
            f'<div class="score-grid"><div class="score-item">'
            f'<span class="score-label">Qualité</span>{pill}</div></div>'
            f'<div class="score-reason">{_esc(reason)}</div>'
        )

    sources_html = ""
    if rep.sources:
        blocs = ""
        for k, s in enumerate(rep.sources, 1):
            score = f" (score {s.score:.3f})" if s.score is not None else ""
            blocs += f'<div class="source">[{k}]{score}\n{_esc(s.texte[:1200])}</div>'
        sources_html = (
            f'<details><summary>Sources récupérées ({len(rep.sources)})</summary>'
            f'{blocs}</details>'
        )
    else:
        sources_html = (
            '<details><summary>Sources récupérées (non disponibles)</summary>'
            '<div class="source">Ce système n\'expose pas ses chunks bruts '
            'individuellement (voir limites dans le rapport).</div></details>'
        )

    return f"""
    <div class="card">
      <div class="card-header">
        <span class="sys-name">{_esc(nom)}</span>{badge}
      </div>
      <div class="metrics">{rep.latence_s:.2f}s · ${rep.cout_usd:.6f}</div>
      {judge_html}
      <div class="answer markdown-source" hidden>{_esc(rep.texte)}</div>
      <div class="answer markdown-rendered"></div>
      {sources_html}
    </div>
    """


def ecrire_html(resultats, lignes_conditions, judge_mode: bool = False):
    """resultats : liste de (question, portee, [(nom, Reponse|None, erreur, verdict|None)])."""
    systemes_affiches = [nom for nom, _, _, _ in resultats[0][2]] if resultats else []
    parts = [
        "<!doctype html><html lang='fr'><head><meta charset='utf-8'>",
        "<title>Grille de notation — Benchmark RAG</title>",
        HTML_STYLE, "</head><body>",
        "<h1>Grille de notation — Benchmark RAG</h1>",
        f"<div class='meta'>Généré le {datetime.now():%Y-%m-%d %H:%M} · "
        f"modèle {MODEL_NAME} · top_k={'variable' if TOP_K_MODE == 'variable' else get_top_k('baseline')} · "
        f"échantillon={SAMPLE_SIZE if SAMPLE_SIZE else 'complet'}</div>",
        "<h2>Conditions des systèmes</h2><ul>",
        *[f"<li>{l}</li>" for l in lignes_conditions],
        "</ul>",
        _section_graphiques_html(systemes_affiches),
        _section_scores_juge_html(resultats) if judge_mode else "",
    ]

    for qi, (question, portee, reps) in enumerate(resultats):
        scope_badge = (
            f'<span class="question-scope scope-{portee}">{_PORTEE_LABELS[portee]}</span>'
            if portee else ""
        )
        parts.append('<div class="question-block">')
        parts.append(
            f'<div class="question-title">Q{qi + 1} — {_esc(question)}{scope_badge}</div>'
        )
        parts.append('<div class="grid">')
        for nom, rep, err, verdict in reps:
            if err is not None:
                parts.append(
                    f'<div class="card"><div class="sys-name">{_esc(nom)}</div>'
                    f'<div class="answer" style="color:#b91c1c">Échec : {_esc(err)}</div></div>'
                )
            else:
                parts.append(_carte_html(nom, rep, verdict if judge_mode else None))
        parts.append('</div></div>')
    parts.append(
        "<script src='https://cdn.jsdelivr.net/npm/marked/marked.min.js'></script>"
        "<script src='https://cdn.jsdelivr.net/npm/dompurify@3/dist/purify.min.js'></script>"
        "<script>"
        "document.querySelectorAll('.markdown-source').forEach(function (el) {"
        "  var html = DOMPurify.sanitize(marked.parse(el.textContent));"
        "  el.nextElementSibling.innerHTML = html;"
        "});"
        "</script>"
    )
    parts.append("</body></html>")
    RESULTS_HTML.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_HTML.write_text("\n".join(parts), encoding="utf-8")


# ======================================================================
# Boucle principale
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="Benchmark RAG")
    parser.add_argument(
        "--systemes", nargs="*", default=None,
        help="Sous-ensemble de systèmes (ex: baseline raptor). Défaut : tous.",
    )
    parser.add_argument(
        "--judge", action="store_true",
        help="Active l'évaluation automatique par LLM judge.",
    )
    args = parser.parse_args()
    run_judge = args.judge

    cle = {"baseline": "RAG classique", "hybrid": "RAG hybride",
           "mono_agentic": "Mono-Agentic", "multi_agent": "Multi-Agent",
           "raptor": "RAPTOR", "lightrag": "LightRAG"}
    if args.systemes:
        voulus = {cle.get(s, s) for s in args.systemes}
        systemes = [(n, m) for n, m in SYSTEMES if n in voulus]
    else:
        systemes = SYSTEMES

    lignes_conditions = [f"<strong>{n}</strong> : {CONDITIONS[n]}" for n, _ in systemes]

    lignes_md = ["# Résultats du benchmark RAG\n",
                 f"_Généré le {datetime.now():%Y-%m-%d %H:%M}_\n",
                 "## Conditions des systèmes\n"]
    for nom, _ in systemes:
        lignes_md.append(f"- **{nom}** : {CONDITIONS[nom]}")
    lignes_md.append("")

    resultats_html = []

    for q in QUESTIONS:
        question = q["texte"]
        portee = q.get("portee")
        entete = f"## [{_PORTEE_LABELS[portee]}] {question}" if portee else f"## {question}"
        print("\n" + "=" * 80)
        print(entete)
        print("=" * 80)
        lignes_md.append(entete + "\n")

        reps_question = []
        for nom, systeme in systemes:
            try:
                rep = systeme.ask(question)
            except Exception as e:
                log.error(f"{nom} a échoué sur '{question}' : {e}")
                lignes_md.append(f"### {nom}\n\n_Échec : {e}_\n")
                reps_question.append((nom, None, str(e), None))
                continue

            contexts = [s.texte for s in rep.sources] if rep.sources else []
            verdict = judge(question, rep.texte, contexts, portee=portee) if run_judge else None

            if run_judge and verdict:
                _s = lambda v: f"{v:.2f}" if isinstance(v, float) else "N/A"
                score_str = f" | qualite={_s(verdict.get('qualite'))}"
            else:
                score_str = ""

            print(f"\n--- {nom} — {rep.latence_s:.2f}s | ${rep.cout_usd:.6f}{score_str} ---")
            if not run_judge:
                print(rep.texte)

            lignes_md.append(
                f"### {nom}  \n`{rep.latence_s:.2f}s · ${rep.cout_usd:.6f}{score_str}`\n\n{rep.texte}\n"
            )
            reps_question.append((nom, rep, None, verdict))

            log_metric({
                "systeme": nom,
                "phase": "requete",
                "question": question,
                "portee": portee,
                "latence_s": round(rep.latence_s, 2),
                "cout_usd": round(rep.cout_usd, 6),
                "n_sources": len(rep.sources),
                **({"judge_qualite": verdict.get("qualite"),
                    "judge_reason": verdict.get("reason")} if verdict else {}),
            })

        resultats_html.append((question, portee, reps_question))

    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_MD.write_text("\n".join(lignes_md), encoding="utf-8")

    ecrire_html(resultats_html, lignes_conditions, judge_mode=run_judge)

    log.info(f"Page du rapport : {RESULTS_HTML}")


if __name__ == "__main__":
    main()
