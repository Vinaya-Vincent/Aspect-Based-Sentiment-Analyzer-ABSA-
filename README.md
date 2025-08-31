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


---

## Dataset
- Source: **Amazon Electronics Reviews (UCSD, 2014)**  
- Full dataset: ~1.7M reviews  
- This project uses a **sample of 10,000 reviews** for efficient training and prototyping.  

---

##  Installation & Setup
Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/ABSA-Analyzer.git
cd ABSA-Analyzer
pip install -r requirements.txt

