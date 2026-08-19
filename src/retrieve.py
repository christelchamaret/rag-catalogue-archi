"""Retrieval : cherche dans ChromaDB et génère une recommandation LLM."""
import os
import sys
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

from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

try:
    import chromadb
    _CHROMA_SETTINGS = chromadb.config.Settings(allow_anonymous_telemetry=False, anonymized_telemetry=False)
except Exception:
    _CHROMA_SETTINGS = None

load_dotenv()

ROOT = Path(__file__).parent.parent
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "carrelages"

SYSTEM_PROMPT = """Tu es un assistant expert en produits d'architecture intérieure (carrelages, faïences).
À partir des fiches produits fournies ci-dessous, recommande les 3 produits les plus pertinents
pour la demande utilisateur. Pour chaque produit, explique pourquoi il correspond (aspects, ambiance,
usage technique, couleurs, formats), donne les informations techniques clés (formats, couleurs, finition)
et propose le lien source.

Sois concret, factuel et structuré. Cite les produits par leur nom de collection."""

HUMAN_PROMPT = """Demande utilisateur :
{query}

Fiches produits candidates :
{context}

Réponds en français, en structurant ta réponse en 3 produits recommandés."""


def format_docs(docs):
    return "\n\n---\n\n".join(
        f"Collection: {d.metadata['collection']}\n"
        f"Marque: {d.metadata['marque']}\n"
        f"URL: {d.metadata['url_source']}\n"
        f"{d.page_content}"
        for d in docs
    )


def search(query: str, k: int = 5):
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    kwargs = dict(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    if _CHROMA_SETTINGS is not None:
        kwargs["client_settings"] = _CHROMA_SETTINGS
    vectordb = Chroma(**kwargs)
    return vectordb.similarity_search_with_relevance_scores(query, k=k)


# Stop words français basiques pour l'extraction de mots-clés
STOP_WORDS = {
    "le", "la", "les", "un", "une", "des", "du", "de", "de", "à", "au", "aux",
    "et", "ou", "pour", "dans", "sur", "avec", "sans", "par", "est", "sont",
    "ce", "cette", "ces", "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses",
    "je", "tu", "il", "elle", "nous", "vous", "ils", "elles",
    "qui", "que", "quoi", "dont", "où", "comme", "mais", "donc", "or", "ni", "car",
    "pas", "ne", "n", "m", "t", "s", "l", "d", "qu", "j",
    "plus", "moins", "très", "trop", "bien", "mal",
    "pièce", "de", "vie", "salle", "bain", "cuisine", "salon", "chambre",
    "finition", "aspect", "effet", "carrelage", "modèle", "collection",
}


def extract_keywords(query: str):
    """Extrait les mots-clés significatifs de la requête."""
    import re
    words = re.findall(r"[a-zàâäéèêëïîôöùûüç]+", query.lower())
    return [w for w in words if len(w) >= 3 and w not in STOP_WORDS]


def keyword_score(doc, keywords):
    """Score = fraction de mots-clés présents dans la fiche (case-insensitive)."""
    if not keywords:
        return 0.0
    content = doc.page_content.lower()
    matched = sum(1 for kw in keywords if kw in content)
    return matched / len(keywords)


def rerank_hybrid(results, keywords, alpha=0.7):
    """
    Hybrid retrieval : combine score vectoriel (dense) et score mots-clés (sparse).
    alpha = poids du score vectoriel, (1-alpha) = poids mots-clés.
    """
    scored = []
    for doc, vec_score in results:
        kw_score = keyword_score(doc, keywords)
        final = alpha * vec_score + (1 - alpha) * kw_score
        scored.append((doc, vec_score, kw_score, final))
    scored.sort(key=lambda x: x[3], reverse=True)
    return scored


def search_hybrid(query: str, k_retrieve: int = 8, k_return: int = 3):
    """Recherche hybride : retrieve large + reranking par mots-clés + vector."""
    results = search(query, k=k_retrieve)
    keywords = extract_keywords(query)
    ranked = rerank_hybrid(results, keywords)
    return [(d, s) for d, s, _, _ in ranked[:k_return]]


def answer(query: str, k: int = 3):
    results = search_hybrid(query, k_retrieve=8, k_return=k)
    docs = [d for d, _ in results]
    llm = ChatMistralAI(model="mistral-small-latest", temperature=0.3)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])
    chain = (
        {"context": lambda _: format_docs(docs), "query": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return results, chain.invoke(query)


def main():
    if len(sys.argv) < 2:
        print('Usage: python retrieve.py "votre requête"')
        return
    query = sys.argv[1]
    results, response = answer(query, k=5)

    print("\n=== TOP 5 PRODUITS RETROUVÉS ===\n")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. [score={score:.3f}] {doc.metadata['collection']} ({doc.metadata['marque']})")
        print(f"   Aspects: {doc.metadata.get('effets', '')}")
        print(f"   Couleurs: {doc.metadata.get('couleurs', '')}")
        print(f"   URL: {doc.metadata.get('url_source', '')}")
        print()

    print("\n=== RÉPONSE LLM ===\n")
    print(response)


if __name__ == "__main__":
    main()
