"""Retrieval : cherche dans ChromaDB et génère une recommandation LLM."""
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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
    vectordb = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    return vectordb.similarity_search_with_relevance_scores(query, k=k)


def answer(query: str, k: int = 5):
    results = search(query, k=k)
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
