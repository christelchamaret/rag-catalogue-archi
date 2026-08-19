# RAG Catalogue Archi

Assistant de recommandation de carrelages et revêtements architecturaux basé sur un **RAG hybride** (vectoriel + keyword boost) — Mistral AI + ChromaDB + Streamlit.

## Résultats d'évaluation

| Métrique | Vectoriel seul | **Hybrid** | Delta |
|---|---|---|---|
| **Recall@3** | 0.650 | **0.758** | **+10.8 pts** |
| **MRR** | 0.875 | **1.000** | +0.125 |
| **Rang moyen 1er attendu** | 1.25 | **1.00** | parfait |
| **Requêtes parfaites** | 4/20 | **9/20** | +5 |
| **Requêtes ratées** | 0/20 | 0/20 | — |

→ Le reranking hybride fait passer **chaque requête** avec le bon produit en **position 1** (MRR=1.000) et améliore le Recall@3 de **+10,8 points**. Voir [`eval/results.md`](eval/results.md) pour le rapport complet.

## Stack

- **LLM & Embeddings** : Mistral AI (`mistral-embed` + `mistral-small-latest`) — souveraineté européenne
- **Vector store** : ChromaDB local persistant
- **Framework** : LangChain
- **UI** : Streamlit (charte graphique TRAIT)
- **Python** : 3.12

## Architecture du RAG

```
Requête utilisateur (langage naturel ambigu)
       ↓
[1] Embedding Mistral mistral-embed (1024-dim)
       ↓
[2] Retrieval ChromaDB k=8 candidats (similarité cosinus)
       ↓
[3] Reranking hybride :
       score_final = 0.7 × score_vectoriel + 0.3 × score_mots-clés
       (boost les fiches contenant les mots-clés de la requête)
       ↓
[4] Top-3 candidats → Mistral-small-latest génère une recommandation
       argumentée (aspects techniques, ambiance, formats, couleurs, lien source)
       ↓
[5] UI Streamlit : 3 cartes produit (photo + specs) + bloc recommandation LLM
```

## Corpus

**13 collections** indexées, couvrant tous les aspects archi :

| Marque | Collections | Aspects |
|---|---|---|
| Novoceram | Hirati, Noa | Bois |
| Novoceram | Sérac, Tiber, Performance | Pierre / Travertin |
| Novoceram | Zen, Vertige, Performance, Teranga | Béton / Ciment / Résine |
| Novoceram | Osmose | Tomette / Terre cuite |
| Marazzi | Resin, Cementum Wall | Béton / Résine (Thin Wall Coverings) |
| Marazzi | ArtCraft, Terramater | Artisanal / Terre cuite (Crogiolo) |

Sources : fiches produits publiques Novoceram (novoceram.fr) et Marazzi (marazzi.fr).

## Démarrage

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env  # renseigner MISTRAL_API_KEY

python src/ingest.py          # indexer les 13 collections dans ChromaDB
streamlit run src/app.py      # lancer l'UI sur http://localhost:8501
```

## Évaluation

```bash
python eval/evaluate.py        # compare vectoriel seul vs hybrid sur 20 requêtes
```

- **Dataset** : 20 requêtes en langage naturel français annotées (effets/aspects/usage), difficulté variable (easy/medium/hard)
- **Métriques** : Recall@3, MRR, rang moyen du 1er attendu, % de requêtes parfaites, latence
- **Comparaison** : vectoriel seul vs hybrid (vector + keyword boost, alpha=0.7/0.3)

Voir [`eval/results.md`](eval/results.md) pour le rapport détaillé par requête.

## Structure

```
rag-catalogue-archi/
├── data/processed/     # 13 JSON de collections (Hirati, Noa, Sérac, Tiber, Performance,
│                       # Zen, Vertige, Teranga, Osmose, Marazzi Resin, ArtCraft,
│                       # Cementum Wall, Terramater)
├── src/
│   ├── ingest.py       # indexation ChromaDB (embeddings Mistral)
│   ├── retrieve.py     # search + search_hybrid (reranking) + answer (LLM)
│   └── app.py          # UI Streamlit charte TRAIT
├── eval/
│   ├── dataset.jsonl   # 20 requêtes annotées (ground truth)
│   ├── evaluate.py     # script d'éval (Recall@3, MRR, comparaison)
│   └── results.md      # rapport généré
├── chroma_db/          # vector store persistant (non commité)
├── requirements.txt
└── README.md
```

## Innovations clés

1. **Hybrid retrieval** : combinaison pondérée (alpha=0.7) de similarité vectorielle dense (Mistral) et de matching mots-clés sparse. Corrige les cas où le vectoriel seul intercale un produit non pertinent (ex : requête "bois" qui remonte Pierre/Béton avant le 2e produit Bois).

2. **Charte graphique TRAIT** : UI Streamlit applique la charte officielle (Noir Profond `#1A1A1A` + Or Champagne `#C9A961` + Garamond), avec cartes produits + photos + recommandation LLM argumentée.

3. **Souveraineté data** : 100% Mistral AI (pas OpenAI) 

## Auteur

**Christel Chamaret** — Docteure en informatique (thèse : *Harmonie des couleurs : modélisation expérimentale et algorithmique*, Université Rennes 1, 2016).

- 16 ans R&D en IA/traitement d'image/vidéo chez Technicolor & Interdigital (40+ brevets, supervision multi-site 25 personnes, productisation POC IA pour effets spéciaux)
- Architecte d'intérieur depuis 2021 (Trait de Couleur)
- Fondatrice TRAIT (2026) — offre de formation et infrastructure IA pour les métiers de l'architecture

Contact : christel@traitdecouleur.fr · https://traitdecouleur.fr
