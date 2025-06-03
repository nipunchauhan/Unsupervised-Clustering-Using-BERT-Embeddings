# Unsupervised BERT-Based Clustering of News Articles

This project applies BERT embeddings with dimensionality reduction and clustering techniques to perform unsupervised topic discovery on the AG News dataset. It demonstrates how semantic understanding from transformer models enables effective document grouping without labels.

## Problem Statement

In today's fast-paced news cycle, it's infeasible to manually classify or label large volumes of content. We leverage unsupervised learning to semantically cluster headlines and articles, allowing editorial teams to identify trending topics and reduce redundancy.

## Tools and Technologies

- **Python**, **Jupyter Notebook**
- **HuggingFace Datasets (AG News)**
- **Sentence-BERT (all-MiniLM-L6-v2)**
- **UMAP** for dimensionality reduction
- **KMeans** for clustering
- **NLTK** for preprocessing
- **Silhouette Score** for evaluation
- **Matplotlib / Seaborn** for visualization

## Methodology

- **Preprocessing:** Lowercasing, punctuation/URL removal, lemmatization, stopword filtering
- **Embedding:** SentenceTransformer BERT model (384-dim vectors)
- **Reduction:** UMAP to 10D (for clustering), 2D (for visualization)
- **Clustering:** KMeans with optimal `k` selected via Silhouette Score
- **Evaluation:** Quantitative (score ≈ 0.5582), qualitative (coherent topics)

## Results

- Distinct and meaningful clusters of news articles
- Visual confirmation via 2D UMAP plots
- Clusters mapped to real-world semantic groupings
- No labeled data was required

## How to Run

1. Clone this repo
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
