"""UI Streamlit : interface de recommandation de carrelages — Charte TRAIT."""
import sys
import logging
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
for _logger in ("chromadb", "chromadb.telemetry", "huggingface_hub", "posthog", "langchain_core"):
    logging.getLogger(_logger).setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent))
import streamlit as st
from retrieve import search, answer

st.set_page_config(
    page_title="Assistant Carrelage · TRAIT",
    page_icon="▪",
    layout="wide",
)

st.markdown("""
<style>
/* Charte TRAIT */
:root {
  --trait-noir: #1A1A1A;
  --trait-or: #C9A961;
  --trait-cuivre: #B87333;
  --trait-anthracite: #3D3D3D;
  --trait-perle: #E8E8E8;
  --trait-casse: #F8F8F8;
  --trait-blanc: #FFFFFF;
}

html, body, [class*="css"] {
  font-family: 'Georgia', 'Garamond', serif;
  color: var(--trait-anthracite);
}

/* Header noir */
.trait-header {
  background: var(--trait-noir);
  color: var(--trait-casse);
  padding: 2rem;
  margin: -1rem -1rem 2rem -1rem;
  border-bottom: 4px solid var(--trait-or);
}
.trait-header h1 {
  font-family: 'Georgia', 'Garamond', serif;
  font-size: 2.6rem;
  font-weight: normal;
  color: var(--trait-casse);
  margin-bottom: 0.3rem;
}
.trait-header .subtitle {
  font-family: 'Arial', sans-serif;
  font-size: 0.9rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--trait-or);
}

/* Sous-titre section */
.trait-section-title {
  font-family: 'Georgia', 'Garamond', serif;
  font-size: 1.8rem;
  font-weight: normal;
  color: var(--trait-noir);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-top: 2rem;
  margin-bottom: 0.5rem;
}
.trait-section-title::after {
  content: '';
  display: block;
  width: 60px;
  height: 3px;
  background: var(--trait-or);
  margin-top: 0.5rem;
}

/* Carte produit */
.trait-card {
  background: var(--trait-blanc);
  border: 1px solid var(--trait-perle);
  border-top: 3px solid var(--trait-or);
  border-radius: 4px;
  padding: 1.5rem;
  height: 100%;
}
.trait-card h3 {
  font-family: 'Georgia', 'Garamond', serif;
  font-size: 1.4rem;
  font-weight: normal;
  color: var(--trait-noir);
  margin-top: 0;
  margin-bottom: 0.3rem;
}
.trait-card .marque {
  font-family: 'Arial', sans-serif;
  font-size: 0.85rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--trait-or);
  margin-bottom: 1rem;
}
.trait-card .meta {
  font-family: 'Arial', sans-serif;
  font-size: 0.9rem;
  color: var(--trait-anthracite);
  line-height: 1.5;
}
.trait-card .meta strong {
  color: var(--trait-noir);
  font-weight: bold;
}
.trait-card .puce {
  display: inline-block;
  width: 8px;
  height: 8px;
  background: var(--trait-or);
  margin-right: 8px;
  vertical-align: middle;
}
.trait-card .source-link {
  display: inline-block;
  margin-top: 1rem;
  font-family: 'Arial', sans-serif;
  font-size: 0.85rem;
  color: var(--trait-noir);
  text-decoration: none;
  border-bottom: 1px solid var(--trait-or);
  padding-bottom: 2px;
}
.trait-card .source-link:hover {
  color: var(--trait-or);
}
.trait-card .score {
  font-family: 'Arial', sans-serif;
  font-size: 0.75rem;
  color: var(--trait-cuivre);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-top: 0.8rem;
}

/* Bouton CTA */
.stButton > button {
  background: var(--trait-noir) !important;
  color: var(--trait-blanc) !important;
  border: 2px solid var(--trait-noir) !important;
  font-family: 'Arial', sans-serif !important;
  font-size: 0.9rem !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  padding: 0.75rem 2rem !important;
  border-radius: 0 !important;
  transition: all 0.2s ease;
}
.stButton > button:hover {
  background: transparent !important;
  color: var(--trait-noir) !important;
  border-color: var(--trait-or) !important;
}

/* Zone de saisie */
.stTextArea textarea, .stTextArea label {
  font-family: 'Arial', sans-serif !important;
}

/* Footer */
.trait-footer {
  background: var(--trait-noir);
  color: var(--trait-casse);
  padding: 2rem;
  margin: 3rem -1rem -2rem -1rem;
  text-align: center;
  font-family: 'Arial', sans-serif;
  font-size: 0.85rem;
}
.trait-footer a {
  color: var(--trait-or);
  text-decoration: none;
}

/* Recommandation LLM */
.reco-llm {
  background: var(--trait-casse);
  border-left: 4px solid var(--trait-or);
  padding: 2rem;
  margin: 1rem 0;
}
.reco-llm h2 {
  font-family: 'Georgia', 'Garamond', serif;
  font-size: 1.8rem;
  font-weight: normal;
  color: var(--trait-noir);
  margin-top: 0;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.reco-llm h3 {
  font-family: 'Georgia', 'Garamond', serif;
  font-weight: normal;
  color: var(--trait-noir);
  border-bottom: 1px solid var(--trait-perle);
  padding-bottom: 0.5rem;
}
.reco-llm p, .reco-llm li {
  font-family: 'Arial', sans-serif;
  color: var(--trait-anthracite);
  line-height: 1.6;
}
.reco-llm strong {
  color: var(--trait-noir);
}
.reco-llm table {
  width: 100%;
  border-collapse: collapse;
  margin: 1rem 0;
}
.reco-llm table th {
  background: var(--trait-noir);
  color: var(--trait-or);
  padding: 0.75rem;
  text-align: left;
  font-family: 'Arial', sans-serif;
  font-size: 0.85rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  border-bottom: 2px solid var(--trait-or);
}
.reco-llm table td {
  padding: 0.75rem;
  border-bottom: 1px solid var(--trait-perle);
  font-family: 'Arial', sans-serif;
  font-size: 0.9rem;
}
.reco-llm table tr:nth-child(even) td {
  background: var(--trait-casse);
}

/* Spinner */
.stSpinner > div {
  border-top-color: var(--trait-or) !important;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="trait-header">
  <h1>Assistant Carrelage</h1>
  <div class="subtitle">Recommandation par RAG · Mistral + ChromaDB</div>
</div>
""", unsafe_allow_html=True)

# Zone de saisie (k=3 fixe)
query = st.text_area(
    "Ta demande :",
    value="Ambiance terracotta cozy pour cuisine nord sombre, quelque chose de chaleureux et naturel",
    height=100,
)
st.caption("Corpus : 9 collections Novoceram · Bois, Pierre, Travertin, Béton, Ciment, Résine, Tomette")

if st.button("▪ Recommander"):
    if not query.strip():
        st.warning("Décris ta demande ci-dessus.")
        st.stop()

    with st.spinner("Recherche sémantique dans le catalogue..."):
        results, response = answer(query, k=3)

    # Top 3 produits en cartes avec photo
    st.markdown('<div class="trait-section-title">Trois collections retrouvées</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, ((doc, score), col) in enumerate(zip(results[:3], cols), 1):
        with col:
            image_url = doc.metadata.get("image_url", "")
            if image_url:
                st.image(image_url, use_container_width=True)
            st.markdown(f"""
<div class="trait-card">
  <h3>{doc.metadata['collection']}</h3>
  <div class="marque">{doc.metadata['marque']}</div>
  <div class="meta">
    <div><span class="puce"></span><strong>Aspects</strong> : {doc.metadata.get('effets', '')}</div>
    <div><span class="puce"></span><strong>Couleurs</strong> : {doc.metadata.get('couleurs', '')}</div>
    <div><span class="puce"></span><strong>Formats</strong> : {doc.metadata.get('formats', '')}</div>
  </div>
  <a class="source-link" href="{doc.metadata.get('url_source', '')}" target="_blank">Voir la fiche complète ↗</a>
  <div class="score">Score de pertinence · {score:.3f}</div>
</div>
""", unsafe_allow_html=True)

    # Recommandation LLM
    st.markdown('<div class="trait-section-title">Analyse & recommandation</div>', unsafe_allow_html=True)
    st.markdown(f"""
<div class="reco-llm">
{response}
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="trait-footer">
  <strong>TRAIT</strong> · IA pour Architecture & Design<br>
  Mistral AI · LangChain · ChromaDB · Streamlit · MVP RAG
</div>
""", unsafe_allow_html=True)
