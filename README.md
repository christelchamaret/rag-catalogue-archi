# RAG Catalogue Produits Archi

Assistant de recommandation de produits d'architecture intérieure (carrelages, faïences, peintures) basé sur un RAG sémantique Mistral + ChromaDB.

## Stack

- **LLM & Embeddings** : Mistral AI (`mistral-small-latest` + `mistral-embed`)
- **Vector store** : ChromaDB (local, persistant)
- **Framework** : LangChain
- **UI** : Streamlit
- **Langue** : Python 3.12

## Cas d'usage

L'utilisateur décrit une ambiance en langage naturel ambigu (ex : "terracotta mat pour cuisine nord sombre"). Le système récupère les fiches produits les plus pertinentes du corpus et fournit une recommandation argumentée.

## Structure

```
rag-catalogue-archi/
├── data/raw/           # PDF/HTML bruts
├── data/processed/     # JSON structurés
├── src/
│   ├── ingest.py       # extraction + embeddings + indexation ChromaDB
│   ├── retrieve.py     # fonction de recherche
│   └── app.py          # UI Streamlit
├── eval/               # datasets d'évaluation
├── requirements.txt
└── README.md
```

## Démarrage

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env  # renseigner MISTRAL_API_KEY

python src/ingest.py          # indexer le corpus
streamlit run src/app.py      # lancer l'UI
```

## Données

Fiches produits publiques issues de fabricants (Cuisines Morel, Cementin, Novoceram, Tollet, Panonia, etc.).

## Auteur

Christel Chamaret — Docteure en informatique (harmonie des couleurs), ex-Ingénieure chercheuse IA/vidéo, architecte d'intérieur.
