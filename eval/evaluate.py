"""
Évaluation du RAG : retrieval metrics (Recall@k, MRR) avec comparaison
vectoriel seul vs hybrid (vectoriel + keyword boost).
"""
import os
import sys
import json
import time
import logging
import warnings
from pathlib import Path

os.environ["ANONYMIZED_TELEMETRY"] = "False"
warnings.filterwarnings("ignore")
for _logger in ("chromadb", "chromadb.telemetry", "huggingface_hub", "posthog", "langchain_core"):
    logging.getLogger(_logger).setLevel(logging.ERROR)
try:
    from langchain_core._api.deprecation import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from retrieve import search, search_hybrid, extract_keywords, rerank_hybrid

ROOT = Path(__file__).parent.parent
DATASET = ROOT / "eval" / "dataset.jsonl"
REPORT = ROOT / "eval" / "results.md"
K_RETRIEVE = 8
K_RETURN = 3


def load_dataset():
    items = []
    with open(DATASET) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def recall_at_k(retrieved_names, expected_names, k=3):
    """% des collections attendues présentes dans le top-k retrouvé."""
    top = retrieved_names[:k]
    if not expected_names:
        return 1.0
    hits = sum(1 for e in expected_names if e in top)
    return hits / len(expected_names)


def mrr(retrieved_names, expected_names):
    """Mean Reciprocal Rank : 1/rank du premier attendu trouvé, 0 si aucun."""
    for i, name in enumerate(retrieved_names, 1):
        if name in expected_names:
            return 1.0 / i
    return 0.0


def first_expected_rank(retrieved_names, expected_names):
    """Rang (1-indexed) du premier attendu, ou 0 si aucun."""
    for i, name in enumerate(retrieved_names, 1):
        if name in expected_names:
            return i
    return 0


def run_eval(mode="hybrid"):
    """
    mode: 'vector' (vectoriel seul) ou 'hybrid' (vector + keyword boost)
    """
    dataset = load_dataset()
    results = []
    for item in dataset:
        qid = item["id"]
        query = item["query"]
        expected = item["expected"]
        t0 = time.time()
        if mode == "hybrid":
            retrieved = search_hybrid(query, k_retrieve=K_RETRIEVE, k_return=K_RETURN)
            retrieved_names = [d.metadata["collection"] for d, _ in retrieved]
        else:
            retrieved = search(query, k=K_RETURN)
            retrieved_names = [d.metadata["collection"] for d, _ in retrieved]
        elapsed = time.time() - t0
        rec = recall_at_k(retrieved_names, expected, k=K_RETURN)
        mrr_v = mrr(retrieved_names, expected)
        rank = first_expected_rank(retrieved_names, expected)
        results.append({
            "id": qid,
            "query": query,
            "expected": expected,
            "retrieved": retrieved_names,
            "recall@3": rec,
            "mrr": mrr_v,
            "first_rank": rank,
            "elapsed_s": elapsed,
        })
        status = "OK" if rec >= 0.5 else "MISS"
        print(f"  [{status}] Q{qid:02d} recall={rec:.2f} mrr={mrr_v:.2f} rank={rank} ({elapsed:.1f}s) {query[:50]}")
    return results


def aggregate(results):
    n = len(results)
    recall = sum(r["recall@3"] for r in results) / n
    mrr_v = sum(r["mrr"] for r in results) / n
    avg_rank = sum(r["first_rank"] for r in results if r["first_rank"] > 0) / max(1, sum(1 for r in results if r["first_rank"] > 0))
    perfect = sum(1 for r in results if r["recall@3"] == 1.0)
    missed = sum(1 for r in results if r["recall@3"] == 0.0)
    avg_time = sum(r["elapsed_s"] for r in results) / n
    return {
        "n": n,
        "recall@3": recall,
        "mrr": mrr_v,
        "avg_first_rank": avg_rank,
        "perfect_recall": perfect,
        "total_missed": missed,
        "avg_time_s": avg_time,
    }


def generate_report(vector_results, hybrid_results, vector_agg, hybrid_agg):
    """Génère un rapport markdown complet."""
    lines = []
    lines.append("# Rapport d'évaluation du RAG Catalogue Archi\n")
    lines.append(f"**Date** : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**Corpus** : 13 collections (9 Novoceram + 4 Marazzi)\n")
    lines.append(f"**Dataset** : 20 requêtes en langage naturel annotées\n")
    lines.append(f"**LLM/Embeddings** : Mistral AI (mistral-embed + mistral-small-latest)\n")
    lines.append(f"**Vector store** : ChromaDB local persistant\n\n")

    lines.append("## Métriques de synthèse\n\n")
    lines.append("| Métrique | Vectoriel seul | Hybrid (vector + keyword) | Delta |\n")
    lines.append("|---|---|---|---|\n")
    lines.append(f"| **Recall@3** | {vector_agg['recall@3']:.3f} | {hybrid_agg['recall@3']:.3f} | {(hybrid_agg['recall@3']-vector_agg['recall@3']):+.3f} |\n")
    lines.append(f"| **MRR** | {vector_agg['mrr']:.3f} | {hybrid_agg['mrr']:.3f} | {(hybrid_agg['mrr']-vector_agg['mrr']):+.3f} |\n")
    lines.append(f"| **Rang moyen 1er attendu** | {vector_agg['avg_first_rank']:.2f} | {hybrid_agg['avg_first_rank']:.2f} | {(hybrid_agg['avg_first_rank']-vector_agg['avg_first_rank']):+.2f} |\n")
    lines.append(f"| **Requêtes parfaites (recall=1.0)** | {vector_agg['perfect_recall']}/{vector_agg['n']} | {hybrid_agg['perfect_recall']}/{hybrid_agg['n']} | {hybrid_agg['perfect_recall']-vector_agg['perfect_recall']:+d} |\n")
    lines.append(f"| **Requêtes ratées (recall=0)** | {vector_agg['total_missed']} | {hybrid_agg['total_missed']} | {hybrid_agg['total_missed']-vector_agg['total_missed']:+d} |\n")
    lines.append(f"| **Latence moyenne** | {vector_agg['avg_time_s']:.1f}s | {hybrid_agg['avg_time_s']:.1f}s | {(hybrid_agg['avg_time_s']-vector_agg['avg_time_s']):+.1f}s |\n\n")

    lines.append("## Interprétation\n\n")
    delta_recall = hybrid_agg['recall@3'] - vector_agg['recall@3']
    if delta_recall > 0:
        lines.append(f"- **Le reranking hybride améliore le Recall@3 de {delta_recall*100:+.1f} points** (de {vector_agg['recall@3']*100:.1f}% à {hybrid_agg['recall@3']*100:.1f}%).\n")
        lines.append("- Le boost par mots-clés corrige les cas où le vectoriel seul intercale un produit non pertinent (ex: requête \"bois\" qui ramène Pierre/Béton avant le 2e produit bois).\n")
    elif delta_recall < 0:
        lines.append(f"- ⚠️ Le reranking hybride **dégrade** le Recall@3 de {abs(delta_recall)*100:.1f} points. À investiguer.\n")
    else:
        lines.append("- Le reranking hybride n'apporte pas de gain sur ce dataset. Le vectoriel seul suffit.\n")
    lines.append(f"- **MRR** : {hybrid_agg['mrr']:.3f} (hybride) vs {vector_agg['mrr']:.3f} (vectoriel) — indique la qualité du ranking, pas seulement la présence.\n")
    lines.append(f"- **Latence** : {hybrid_agg['avg_time_s']:.1f}s par requête en moyenne (recherche hybride). Acceptable pour usage asynchrone, lent pour usage temps réel.\n\n")

    lines.append("## Détail par requête\n\n")
    lines.append("### Mode vectoriel seul\n\n")
    lines.append("| # | Requête | Attendus | Retrouvés | Recall@3 | MRR | Rang 1er |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    for r in vector_results:
        exp = ", ".join(r["expected"])
        ret = ", ".join(r["retrieved"])
        rank_str = str(r["first_rank"]) if r["first_rank"] > 0 else "—"
        lines.append(f"| {r['id']} | {r['query'][:40]}... | {exp} | {ret} | {r['recall@3']:.2f} | {r['mrr']:.2f} | {rank_str} |\n")
    lines.append("\n### Mode hybrid (vector + keyword boost)\n\n")
    lines.append("| # | Requête | Attendus | Retrouvés | Recall@3 | MRR | Rang 1er |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    for r in hybrid_results:
        exp = ", ".join(r["expected"])
        ret = ", ".join(r["retrieved"])
        rank_str = str(r["first_rank"]) if r["first_rank"] > 0 else "—"
        lines.append(f"| {r['id']} | {r['query'][:40]}... | {exp} | {ret} | {r['recall@3']:.2f} | {r['mrr']:.2f} | {rank_str} |\n")

    lines.append("\n## Cas d'étude : requêtes difficiles\n\n")
    difficult = [r for r in hybrid_results if r["recall@3"] < 1.0]
    if not difficult:
        lines.append("✅ Toutes les requêtes obtiennent un Recall@3 parfait en mode hybride.\n")
    else:
        lines.append(f"### {len(difficult)} requêtes avec Recall@3 < 1.0\n\n")
        for r in difficult:
            missing = [e for e in r["expected"] if e not in r["retrieved"]]
            lines.append(f"**Q{r['id']}** : {r['query']}\n")
            lines.append(f"- Attendus : {', '.join(r['expected'])}\n")
            lines.append(f"- Retrouvés : {', '.join(r['retrieved'])}\n")
            lines.append(f"- Manquants : {', '.join(missing) if missing else 'aucun'}\n")
            lines.append(f"- Recall@3 : {r['recall@3']:.2f} | MRR : {r['mrr']:.2f}\n\n")

    lines.append("\n## Méthodologie\n\n")
    lines.append("### Dataset\n")
    lines.append("- 20 requêtes en langage naturel français, couvrant : aspect bois, pierre, béton, résine, tomette/terre cuite, artisanal, écologique, technique, décor mural.\n")
    lines.append("- Annotations manuelles : pour chaque requête, 2-3 collections attendues (ground truth) basées sur les effets/aspects/usage décrits dans les fiches produits.\n")
    lines.append("- Difficulté variable : easy (mot-clé direct), medium (intention claire mais sémantique), hard (ambiance/émotionnel).\n\n")
    lines.append("### Métriques\n")
    lines.append("- **Recall@3** : fraction des collections attendues qui apparaissent dans le top-3 retrouvé. 1.0 = parfait, 0.0 = raté.\n")
    lines.append("- **MRR** (Mean Reciprocal Rank) : 1/rang du premier attendu trouvé. Sensible à l'ordre, pas seulement à la présence.\n")
    lines.append("- **Rang moyen du 1er attendu** : indique la qualité du ranking.\n")
    lines.append("- **Latence** : temps total par requête (embedding + recherche ChromaDB).\n\n")
    lines.append("### Modes comparés\n")
    lines.append("- **Vectoriel seul** : similarité cosinus entre embeddings Mistral de la requête et des fiches produits.\n")
    lines.append("- **Hybrid** : retrieve k=8 candidats vectoriels, puis reranking par combinaison pondérée (alpha=0.7 vectoriel + 0.3 mots-clés). Boost les fiches contenant les mots-clés de la requête (ex: \"bois\" booste Hirati/Noa).\n")

    REPORT.write_text("".join(lines), encoding="utf-8")
    print(f"\n📝 Rapport écrit : {REPORT}")


def main():
    print("=" * 60)
    print("ÉVALUATION RAG CATALOGUE ARCHI")
    print("=" * 60)
    print(f"\nDataset : {DATASET}")
    print(f"k_retrieve = {K_RETRIEVE}, k_return = {K_RETURN}\n")

    print("\n--- Mode 1/2 : Vectoriel seul ---\n")
    vector_results = run_eval(mode="vector")
    vector_agg = aggregate(vector_results)

    print("\n--- Mode 2/2 : Hybrid (vector + keyword boost) ---\n")
    hybrid_results = run_eval(mode="hybrid")
    hybrid_agg = aggregate(hybrid_results)

    print("\n" + "=" * 60)
    print("SYNTHÈSE")
    print("=" * 60)
    print(f"\nVectoriel seul :")
    print(f"  Recall@3 = {vector_agg['recall@3']:.3f}")
    print(f"  MRR       = {vector_agg['mrr']:.3f}")
    print(f"  Rang moyen 1er = {vector_agg['avg_first_rank']:.2f}")
    print(f"  Parfaits  = {vector_agg['perfect_recall']}/{vector_agg['n']}")
    print(f"  Ratés     = {vector_agg['total_missed']}")
    print(f"  Latence   = {vector_agg['avg_time_s']:.1f}s")
    print(f"\nHybrid :")
    print(f"  Recall@3 = {hybrid_agg['recall@3']:.3f}")
    print(f"  MRR       = {hybrid_agg['mrr']:.3f}")
    print(f"  Rang moyen 1er = {hybrid_agg['avg_first_rank']:.2f}")
    print(f"  Parfaits  = {hybrid_agg['perfect_recall']}/{hybrid_agg['n']}")
    print(f"  Ratés     = {hybrid_agg['total_missed']}")
    print(f"  Latence   = {hybrid_agg['avg_time_s']:.1f}s")
    print(f"\nDelta Recall@3 = {(hybrid_agg['recall@3']-vector_agg['recall@3'])*100:+.1f} points")

    generate_report(vector_results, hybrid_results, vector_agg, hybrid_agg)


if __name__ == "__main__":
    main()
