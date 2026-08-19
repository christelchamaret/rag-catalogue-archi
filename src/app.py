"""UI Streamlit : interface de recommandation de carrelages."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import streamlit as st
from retrieve import search, answer

st.set_page_config(
    page_title="RAG Catalogue Archi",
    page_icon="🎨",
    layout="wide",
)

st.title("🎨 Assistant carrelage — RAG Mistral + ChromaDB")
st.caption(
    "Décris une ambiance ou un besoin en langage naturel. "
    "Le système récupère les fiches produits les plus pertinentes "
    "et te propose 3 recommandations argumentées."
)

col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_area(
        "Ta demande :",
        value="Ambiance terracotta cozy pour cuisine nord sombre, quelque chose de chaleureux et naturel",
        height=100,
    )
with col2:
    k = st.slider("Nombre de produits à récupérer (k)", 3, 9, 5)
    st.caption("Corpus : 9 collections Novoceram (Bois, Pierre, Travertin, Béton, Ciment, Résine, Tomette)")

if st.button("🔍 Recommander", type="primary"):
    if not query.strip():
        st.warning("Décris ta demande ci-dessus.")
        st.stop()

    with st.spinner("Recherche dans le catalogue..."):
        results, response = answer(query, k=k)

    st.subheader("📚 Top produits retrouvés")
    for i, (doc, score) in enumerate(results, 1):
        with st.expander(
            f"{i}. {doc.metadata['collection']} — score: {score:.3f}",
            expanded=(i <= 3),
        ):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown(f"**Marque :** {doc.metadata['marque']}")
                st.markdown(f"**Aspects :** {doc.metadata.get('effets', '')}")
                st.markdown(f"**Couleurs :** {doc.metadata.get('couleurs', '')}")
                st.markdown(f"**Formats :** {doc.metadata.get('formats', '')}")
            with c2:
                st.markdown(f"🔗 [Voir la fiche]({doc.metadata.get('url_source', '')})")
            st.markdown("**Fiche complète :**")
            st.text(doc.page_content)

    st.subheader("🤖 Recommandation LLM")
    st.markdown(response)

st.divider()
st.caption(
    "MVP RAG — Christel Chamaret · Mistral (mistral-embed + mistral-small-latest) · "
    "LangChain · ChromaDB · Streamlit"
)
