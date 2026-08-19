"""Ingestion : lit les JSON de data/processed, embeddings Mistral, stockage ChromaDB."""
import os
import json
import glob
import warnings
from pathlib import Path

os.environ["ANONYMIZED_TELEMETRY"] = "False"
warnings.filterwarnings("ignore")
try:
    from langchain_core._api.deprecation import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except Exception:
    pass

from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

try:
    import chromadb
    _CHROMA_SETTINGS = chromadb.config.Settings(allow_anonymous_telemetry=False, anonymized_telemetry=False)
except Exception:
    _CHROMA_SETTINGS = None

load_dotenv()

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "processed"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "carrelages"


def build_document(c: dict) -> Document:
    text = "\n".join([
        f"Collection: {c['collection']}",
        f"Marque: {c['marque']}",
        f"Aspects/Effets: {', '.join(c.get('effets', []))}",
        f"Utilisations: {', '.join(c.get('utilisations', []))}",
        f"Technologie: {c.get('technologie', '')}",
        f"Formats: {', '.join(c.get('formats', []))}",
        f"Couleurs: {', '.join(c.get('couleurs', []))}",
        f"Finition: {c.get('finition', '')}",
        f"Antidérapant: {'oui' if c.get('antiderapant') else 'non'}",
        f"Accessoires: {', '.join(c.get('accessoires', []))}",
        f"Certifications: {', '.join(c.get('certifications', []))}",
        f"Description: {c.get('description_marketing', '')}",
        f"Argument design: {c.get('argument_design', '')}",
        f"Destinations suggérées: {', '.join(c.get('destinations_suggerees', []))}",
    ])
    return Document(
        page_content=text,
        metadata={
            "collection": c["collection"],
            "marque": c["marque"],
            "url_source": c.get("url_source", ""),
            "image_url": c.get("image_url", ""),
            "effets": ", ".join(c.get("effets", [])),
            "couleurs": ", ".join(c.get("couleurs", [])),
            "formats": ", ".join(c.get("formats", [])),
        },
    )


def main():
    files = sorted(glob.glob(str(DATA_DIR / "*.json")))
    if not files:
        print(f"Aucun JSON trouvé dans {DATA_DIR}")
        return

    print(f"{len(files)} fichiers JSON trouvés:")
    docs = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        doc = build_document(data)
        docs.append(doc)
        print(f"  - {doc.metadata['collection']} ({doc.metadata['marque']})")

    print(f"\nInitialisation embeddings Mistral (model=mistral-embed)...")
    embeddings = MistralAIEmbeddings(model="mistral-embed")

    print(f"Indexation dans ChromaDB ({CHROMA_DIR})...")
    if CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
    kwargs = dict(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )
    if _CHROMA_SETTINGS is not None:
        kwargs["client_settings"] = _CHROMA_SETTINGS
    Chroma.from_documents(**kwargs)
    print(f"\n✅ {len(docs)} collections indexées dans {CHROMA_DIR}")


if __name__ == "__main__":
    main()
