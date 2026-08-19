# Rapport d'évaluation du RAG Catalogue Archi
**Date** : 2026-08-19 17:36:46
**Corpus** : 13 collections (9 Novoceram + 4 Marazzi)
**Dataset** : 20 requêtes en langage naturel annotées
**LLM/Embeddings** : Mistral AI (mistral-embed + mistral-small-latest)
**Vector store** : ChromaDB local persistant

## Métriques de synthèse

| Métrique | Vectoriel seul | Hybrid (vector + keyword) | Delta |
|---|---|---|---|
| **Recall@3** | 0.650 | 0.758 | +0.108 |
| **MRR** | 0.875 | 1.000 | +0.125 |
| **Rang moyen 1er attendu** | 1.25 | 1.00 | -0.25 |
| **Requêtes parfaites (recall=1.0)** | 4/20 | 9/20 | +5 |
| **Requêtes ratées (recall=0)** | 0 | 0 | +0 |
| **Latence moyenne** | 1.7s | 0.6s | -1.1s |

## Interprétation

- **Le reranking hybride améliore le Recall@3 de +10.8 points** (de 65.0% à 75.8%).
- Le boost par mots-clés corrige les cas où le vectoriel seul intercale un produit non pertinent (ex: requête "bois" qui ramène Pierre/Béton avant le 2e produit bois).
- **MRR** : 1.000 (hybride) vs 0.875 (vectoriel) — indique la qualité du ranking, pas seulement la présence.
- **Latence** : 0.6s par requête en moyenne (recherche hybride). Acceptable pour usage asynchrone, lent pour usage temps réel.

## Détail par requête

### Mode vectoriel seul

| # | Requête | Attendus | Retrouvés | Recall@3 | MRR | Rang 1er |
|---|---|---|---|---|---|---|
| 1 | finition bois pour une pièce de vie... | Hirati, Noa | Hirati, Performance, Noa | 1.00 | 1.00 | 1 |
| 2 | ambiance terracotta cozy pour cuisine no... | Sérac, Osmose, Terramater | Teranga, Terramater, Osmose | 0.67 | 0.50 | 2 |
| 3 | sol extérieur terrasse moderne minéral g... | Teranga, Tiber, Performance | Terramater, Teranga, Sérac | 0.33 | 0.50 | 2 |
| 4 | carrelage effet résine sobre et contempo... | Vertige, Zen, Resin | Resin, Vertige, Hirati | 0.67 | 1.00 | 1 |
| 5 | pierre calcaire ancienne pour salon lumi... | Sérac, Tiber, Terramater | Performance, Sérac, Vertige | 0.33 | 0.50 | 2 |
| 6 | carrelage artisanal terre cuite pour cui... | Terramater, ArtCraft, Osmose | Terramater, ArtCraft, Teranga | 0.67 | 1.00 | 1 |
| 7 | revêtement mural effet béton moderne ave... | Cementum Wall, Vertige, Zen | Cementum Wall, Performance, Zen | 0.67 | 1.00 | 1 |
| 8 | tomettes hexagonales décoratives pour cu... | Osmose, Terramater, ArtCraft | Osmose, Terramater, Performance | 0.67 | 1.00 | 1 |
| 9 | pierre travertin beige pour terrasse ext... | Tiber, Sérac, Teranga | Teranga, Vertige, Tiber | 0.67 | 1.00 | 1 |
| 10 | sol technique haute fréquentation pour b... | Performance, Teranga, Cementum Wall | Performance, Resin, Osmose | 0.33 | 1.00 | 1 |
| 11 | ciment vieilli avec craquelures pour amb... | Teranga, Zen, Cementum Wall | Cementum Wall, Sérac, Teranga | 0.67 | 1.00 | 1 |
| 12 | bois naturel hyper réaliste avec veines ... | Hirati, Noa | Hirati, Noa, Sérac | 1.00 | 1.00 | 1 |
| 13 | effet résine colorée pour architecture c... | Vertige, Resin, Zen | Resin, Vertige, Performance | 0.67 | 1.00 | 1 |
| 14 | carrelage écologique avec matière recycl... | Terramater, ArtCraft, Osmose | Terramater, Hirati, ArtCraft | 0.67 | 1.00 | 1 |
| 15 | pierre de Bordeaux Renaissance beige ave... | Sérac, Tiber | Sérac, Teranga, Cementum Wall | 0.50 | 1.00 | 1 |
| 16 | ciment avec craquelures et grand format ... | Teranga, Resin | Cementum Wall, Teranga, Osmose | 0.50 | 0.50 | 2 |
| 17 | décoration murale florale ou géométrique... | Cementum Wall, ArtCraft, Terramater | Osmose, ArtCraft, Cementum Wall | 0.67 | 0.50 | 2 |
| 18 | bois clair type lin ou neutre pour cuisi... | Hirati, Noa | Hirati, Noa, Osmose | 1.00 | 1.00 | 1 |
| 19 | travertin romain avec aspérités et tons ... | Tiber, Sérac, Terramater | Tiber, Teranga, Vertige | 0.33 | 1.00 | 1 |
| 20 | sol garage ou boutique technique épaisse... | Performance, Teranga | Performance, Teranga, Sérac | 1.00 | 1.00 | 1 |

### Mode hybrid (vector + keyword boost)

| # | Requête | Attendus | Retrouvés | Recall@3 | MRR | Rang 1er |
|---|---|---|---|---|---|---|
| 1 | finition bois pour une pièce de vie... | Hirati, Noa | Hirati, Noa, Performance | 1.00 | 1.00 | 1 |
| 2 | ambiance terracotta cozy pour cuisine no... | Sérac, Osmose, Terramater | Terramater, Sérac, Noa | 0.67 | 1.00 | 1 |
| 3 | sol extérieur terrasse moderne minéral g... | Teranga, Tiber, Performance | Teranga, Sérac, Performance | 0.67 | 1.00 | 1 |
| 4 | carrelage effet résine sobre et contempo... | Vertige, Zen, Resin | Resin, Zen, Vertige | 1.00 | 1.00 | 1 |
| 5 | pierre calcaire ancienne pour salon lumi... | Sérac, Tiber, Terramater | Sérac, Performance, Teranga | 0.33 | 1.00 | 1 |
| 6 | carrelage artisanal terre cuite pour cui... | Terramater, ArtCraft, Osmose | Terramater, ArtCraft, Osmose | 1.00 | 1.00 | 1 |
| 7 | revêtement mural effet béton moderne ave... | Cementum Wall, Vertige, Zen | Cementum Wall, Performance, Zen | 0.67 | 1.00 | 1 |
| 8 | tomettes hexagonales décoratives pour cu... | Osmose, Terramater, ArtCraft | Osmose, Terramater, Performance | 0.67 | 1.00 | 1 |
| 9 | pierre travertin beige pour terrasse ext... | Tiber, Sérac, Teranga | Tiber, Sérac, Teranga | 1.00 | 1.00 | 1 |
| 10 | sol technique haute fréquentation pour b... | Performance, Teranga, Cementum Wall | Performance, Resin, Osmose | 0.33 | 1.00 | 1 |
| 11 | ciment vieilli avec craquelures pour amb... | Teranga, Zen, Cementum Wall | Teranga, Sérac, Osmose | 0.33 | 1.00 | 1 |
| 12 | bois naturel hyper réaliste avec veines ... | Hirati, Noa | Hirati, Noa, Sérac | 1.00 | 1.00 | 1 |
| 13 | effet résine colorée pour architecture c... | Vertige, Resin, Zen | Vertige, Resin, Zen | 1.00 | 1.00 | 1 |
| 14 | carrelage écologique avec matière recycl... | Terramater, ArtCraft, Osmose | Terramater, ArtCraft, Vertige | 0.67 | 1.00 | 1 |
| 15 | pierre de Bordeaux Renaissance beige ave... | Sérac, Tiber | Sérac, Tiber, Teranga | 1.00 | 1.00 | 1 |
| 16 | ciment avec craquelures et grand format ... | Teranga, Resin | Teranga, Osmose, Sérac | 0.50 | 1.00 | 1 |
| 17 | décoration murale florale ou géométrique... | Cementum Wall, ArtCraft, Terramater | ArtCraft, Osmose, Terramater | 0.67 | 1.00 | 1 |
| 18 | bois clair type lin ou neutre pour cuisi... | Hirati, Noa | Hirati, Noa, Teranga | 1.00 | 1.00 | 1 |
| 19 | travertin romain avec aspérités et tons ... | Tiber, Sérac, Terramater | Tiber, Sérac, Resin | 0.67 | 1.00 | 1 |
| 20 | sol garage ou boutique technique épaisse... | Performance, Teranga | Performance, Teranga, Vertige | 1.00 | 1.00 | 1 |

## Cas d'étude : requêtes difficiles

### 11 requêtes avec Recall@3 < 1.0

**Q2** : ambiance terracotta cozy pour cuisine nord sombre, quelque chose de chaleureux et naturel
- Attendus : Sérac, Osmose, Terramater
- Retrouvés : Terramater, Sérac, Noa
- Manquants : Osmose
- Recall@3 : 0.67 | MRR : 1.00

**Q3** : sol extérieur terrasse moderne minéral gris
- Attendus : Teranga, Tiber, Performance
- Retrouvés : Teranga, Sérac, Performance
- Manquants : Tiber
- Recall@3 : 0.67 | MRR : 1.00

**Q5** : pierre calcaire ancienne pour salon lumineux et discret
- Attendus : Sérac, Tiber, Terramater
- Retrouvés : Sérac, Performance, Teranga
- Manquants : Tiber, Terramater
- Recall@3 : 0.33 | MRR : 1.00

**Q7** : revêtement mural effet béton moderne avec décors design
- Attendus : Cementum Wall, Vertige, Zen
- Retrouvés : Cementum Wall, Performance, Zen
- Manquants : Vertige
- Recall@3 : 0.67 | MRR : 1.00

**Q8** : tomettes hexagonales décoratives pour cuisine
- Attendus : Osmose, Terramater, ArtCraft
- Retrouvés : Osmose, Terramater, Performance
- Manquants : ArtCraft
- Recall@3 : 0.67 | MRR : 1.00

**Q10** : sol technique haute fréquentation pour boutique ou restaurant
- Attendus : Performance, Teranga, Cementum Wall
- Retrouvés : Performance, Resin, Osmose
- Manquants : Teranga, Cementum Wall
- Recall@3 : 0.33 | MRR : 1.00

**Q11** : ciment vieilli avec craquelures pour ambiance industrielle
- Attendus : Teranga, Zen, Cementum Wall
- Retrouvés : Teranga, Sérac, Osmose
- Manquants : Zen, Cementum Wall
- Recall@3 : 0.33 | MRR : 1.00

**Q14** : carrelage écologique avec matière recyclée et savoir-faire artisanal italien
- Attendus : Terramater, ArtCraft, Osmose
- Retrouvés : Terramater, ArtCraft, Vertige
- Manquants : Osmose
- Recall@3 : 0.67 | MRR : 1.00

**Q16** : ciment avec craquelures et grand format 120x120
- Attendus : Teranga, Resin
- Retrouvés : Teranga, Osmose, Sérac
- Manquants : Resin
- Recall@3 : 0.50 | MRR : 1.00

**Q17** : décoration murale florale ou géométrique artisanale
- Attendus : Cementum Wall, ArtCraft, Terramater
- Retrouvés : ArtCraft, Osmose, Terramater
- Manquants : Cementum Wall
- Recall@3 : 0.67 | MRR : 1.00

**Q19** : travertin romain avec aspérités et tons beige chaleureux
- Attendus : Tiber, Sérac, Terramater
- Retrouvés : Tiber, Sérac, Resin
- Manquants : Terramater
- Recall@3 : 0.67 | MRR : 1.00


## Méthodologie

### Dataset
- 20 requêtes en langage naturel français, couvrant : aspect bois, pierre, béton, résine, tomette/terre cuite, artisanal, écologique, technique, décor mural.
- Annotations manuelles : pour chaque requête, 2-3 collections attendues (ground truth) basées sur les effets/aspects/usage décrits dans les fiches produits.
- Difficulté variable : easy (mot-clé direct), medium (intention claire mais sémantique), hard (ambiance/émotionnel).

### Métriques
- **Recall@3** : fraction des collections attendues qui apparaissent dans le top-3 retrouvé. 1.0 = parfait, 0.0 = raté.
- **MRR** (Mean Reciprocal Rank) : 1/rang du premier attendu trouvé. Sensible à l'ordre, pas seulement à la présence.
- **Rang moyen du 1er attendu** : indique la qualité du ranking.
- **Latence** : temps total par requête (embedding + recherche ChromaDB).

### Modes comparés
- **Vectoriel seul** : similarité cosinus entre embeddings Mistral de la requête et des fiches produits.
- **Hybrid** : retrieve k=8 candidats vectoriels, puis reranking par combinaison pondérée (alpha=0.7 vectoriel + 0.3 mots-clés). Boost les fiches contenant les mots-clés de la requête (ex: "bois" booste Hirati/Noa).
