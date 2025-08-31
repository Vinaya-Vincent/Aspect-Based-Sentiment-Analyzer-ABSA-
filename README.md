# Aspect-Based Sentiment Analyzer for Product Reviews  

A **Streamlit-based web application** that performs **Aspect-Based Sentiment Analysis (ABSA)** on product reviews.  
Unlike traditional sentiment analysis, which gives only an overall rating, this system identifies **specific product aspects** (e.g., *battery, screen, camera*) and predicts the sentiment toward each aspect.  

---

##  Features
- **Aspect Extraction** → POS tagging to extract nouns and adjective–noun pairs as aspects.  
- **TF-IDF Analysis** → Highlights the most important words influencing sentiment.  
- **Word2Vec + Cosine Similarity** → Groups semantically similar aspects (e.g., *“screen” ≈ “display”*).  
- **Multiple Models** →  
  - Naive Bayes  
  - Support Vector Machine (SVM)  
  - Random Forest  
  - DistilBERT (Transformer-based model)  
- **Interactive Visualizations** → Aspect comparisons, semantic groups, confusion matrices, word clouds, and more.  
- **Helpful Reviews Identification** → Finds the most helpful review for a chosen aspect.  

---

##  Project Structure
