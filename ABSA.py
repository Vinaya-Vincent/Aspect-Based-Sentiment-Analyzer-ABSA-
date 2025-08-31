# ==============================================================================
# Enhanced Streamlit Aspect-Based Sentiment Analysis Web Application
#
# FIX: Corrected the @st.cache_data UnhashableParamError for the TF-IDF vectorizer.
#
# New Features:
# - Aspect Comparison with side-by-side charts
# - Semantic Aspect Grouping using Word2Vec and Cosine Similarity
# - Spelling Correction using TextBlob
# - Detailed TF-IDF implementation and visualization
# - Comprehensive evaluation metrics for both models
# - Advanced POS tagging for better aspect extraction
# - Word2Vec similarity analysis
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import warnings
from collections import Counter, defaultdict
import os
import sys
import subprocess
import pickle
from pathlib import Path

# --- Best Practice Note ---
# For reliable deployment, it's better to use a `requirements.txt` file
# and install packages before running the app. This code installs packages
# at runtime for convenience.
# A sample `requirements.txt` would contain:
# streamlit
# pandas
# numpy
# nltk
# scikit-learn
# transformers
# torch
# matplotlib
# seaborn
# textblob
# gensim
# plotly
# wordcloud

# Auto-install function with error correction
def install_package(package_name, import_name=None):
    """Install package if not found and handle specific numpy errors."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
    except (ImportError, ValueError) as e:
        # Specifically catch the numpy binary incompatibility error
        if 'numpy.dtype size changed' in str(e):
            st.error(f" NumPy incompatibility detected with `{package_name}`.")
            with st.spinner(f"Attempting to fix by reinstalling `{package_name}` and `numpy`..."):
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--upgrade",
                    "--force-reinstall", "numpy", package_name
                ])
            st.success("Libraries have been reinstalled successfully!")
            st.info("Please refresh the page or restart the app for the changes to take effect.")
            st.stop()
        
        # Handle standard ImportError
        else:
            st.warning(f"Installing {package_name}... Please wait.")
            with st.spinner(f"Downloading and installing {package_name}..."):
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            st.success(f"{package_name} installed successfully!")
            # Re-attempt import after installation
            try:
                __import__(import_name)
            except ImportError:
                st.error(f"Failed to import {import_name} after installation. Please restart the app.")
                st.stop()

# Install required packages
required_packages = [
    ('nltk', 'nltk'),
    ('scikit-learn', 'sklearn'),
    ('gensim', 'gensim'), # Moved gensim up to catch error sooner if present
    ('transformers', 'transformers'),
    ('torch', 'torch'),
    ('matplotlib', 'matplotlib'),
    ('seaborn', 'seaborn'),
    ('textblob', 'textblob'),
    ('plotly', 'plotly'),
    ('wordcloud', 'wordcloud')
]

for package, import_name in required_packages:
    install_package(package, import_name)

# Now import everything
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud

# Suppress warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Enhanced Aspect-Based Sentiment Analysis",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .comparison-card {
        border: 2px solid #e1e5e9;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f8f9fa;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
session_vars = ['data_loaded', 'models_loaded', 'aspects_extracted', 'word2vec_trained',
                'semantic_groups_created', 'tfidf_analyzed']
for var in session_vars:
    if var not in st.session_state:
        st.session_state[var] = False

@st.cache_data
def install_nltk_data():
    """Download required NLTK data"""
    required_packages = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
                         'punkt_tab', 'maxent_ne_chunker', 'words']
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            nltk.download(package, quiet=True)

@st.cache_data
def load_and_validate_data(uploaded_file):
    """Load and validate the dataset with comprehensive checks"""
    try:
        df = pd.read_csv(uploaded_file)

        # Validation checks
        required_columns = ['reviewText']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            st.info("Required columns: reviewText (required), overall (optional)")
            return None

        # Data cleaning and validation
        initial_count = len(df)
        df = df.dropna(subset=['reviewText'])
        df = df[df['reviewText'].str.strip() != '']

        if len(df) == 0:
            st.error("No valid reviews found after cleaning!")
            return None

        # Handle overall rating column
        if 'overall' not in df.columns:
            st.warning("No 'overall' rating column found. Creating dummy ratings for demo.")
            df['overall'] = np.random.choice([1, 2, 3, 4, 5], size=len(df))
        else:
            df['overall'] = pd.to_numeric(df['overall'], errors='coerce')
            df = df.dropna(subset=['overall'])

        # Handle helpful column
        if 'helpful' in df.columns:
            df['helpful'] = df['helpful'].apply(lambda x: [int(i) for i in re.findall(r'\d+', str(x))] if pd.notna(x) else [0, 0])
        else:
            df['helpful'] = [[0, 0] for _ in range(len(df))]

        st.success(f"Data loaded successfully! {len(df)}/{initial_count} reviews retained after cleaning.")
        return df

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def apply_spelling_correction(text):
    """Apply spelling correction using TextBlob"""
    try:
        blob = TextBlob(text)
        return str(blob.correct())
    except:
        return text

@st.cache_data
def advanced_preprocess_text(text, apply_spell_check=False):
    """Enhanced text preprocessing with spelling correction option"""
    if not isinstance(text, str):
        return ""

    # Apply spelling correction if requested
    if apply_spell_check:
        text = apply_spelling_correction(text)

    # Initialize tools
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Basic cleaning
    text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenization and POS tagging
    words = word_tokenize(text)
    pos_tags = pos_tag(words)

    # Advanced filtering based on POS tags
    cleaned_words = []
    for word, tag in pos_tags:
        if (word not in stop_words and
            len(word) > 2 and
            tag in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']):
            cleaned_words.append(lemmatizer.lemmatize(word, pos='v' if tag.startswith('V') else 'n'))

    return " ".join(cleaned_words)

@st.cache_data
def extract_aspects_advanced(df, top_n=20, min_freq=5):
    """Advanced aspect extraction using POS tagging and frequency analysis"""
    aspect_candidates = []

    for text in df['cleaned_review']:
        words = word_tokenize(text)
        pos_tags = pos_tag(words)

        # Extract nouns and noun phrases
        for i, (word, tag) in enumerate(pos_tags):
            if tag in ['NN', 'NNS']:
                # Check for adjective-noun pairs
                if i > 0 and pos_tags[i-1][1] in ['JJ', 'JJR', 'JJS']:
                    aspect_candidates.append(f"{pos_tags[i-1][0]} {word}")
                aspect_candidates.append(word)

    # Count frequencies and filter
    aspect_counts = Counter(aspect_candidates)
    filtered_aspects = {aspect: count for aspect, count in aspect_counts.items()
                        if count >= min_freq and len(aspect.split()) <= 2}

    top_aspects = [aspect for aspect, count in
                   Counter(filtered_aspects).most_common(top_n)]

    return top_aspects, aspect_counts

@st.cache_resource
def train_word2vec_model(df, vector_size=100, window=5, min_count=2):
    """Train Word2Vec model on the corpus"""
    sentences = []
    for text in df['cleaned_review']:
        sentences.append(simple_preprocess(text))

    model = Word2Vec(sentences, vector_size=vector_size, window=window,
                     min_count=min_count, workers=4, sg=1)
    return model

@st.cache_data
def create_semantic_groups(_word2vec_model, aspects, similarity_threshold=0.6):
    """Group semantically similar aspects using Word2Vec and cosine similarity"""
    groups = []
    used_aspects = set()

    for aspect in aspects:
        if aspect in used_aspects:
            continue

        # Try to get vector for aspect (handle multi-word aspects)
        aspect_words = aspect.split()
        try:
            if len(aspect_words) == 1:
                if aspect in _word2vec_model.wv:
                    aspect_vector = _word2vec_model.wv[aspect]
                else:
                    continue
            else:
                # Average vectors for multi-word aspects
                vectors = []
                for word in aspect_words:
                    if word in _word2vec_model.wv:
                        vectors.append(_word2vec_model.wv[word])
                if not vectors:
                    continue
                aspect_vector = np.mean(vectors, axis=0)
        except:
            continue

        # Find similar aspects
        group = [aspect]
        used_aspects.add(aspect)

        for other_aspect in aspects:
            if other_aspect in used_aspects:
                continue

            try:
                other_words = other_aspect.split()
                if len(other_words) == 1:
                    if other_aspect in _word2vec_model.wv:
                        other_vector = _word2vec_model.wv[other_aspect]
                    else:
                        continue
                else:
                    vectors = []
                    for word in other_words:
                        if word in _word2vec_model.wv:
                            vectors.append(_word2vec_model.wv[word])
                    if not vectors:
                        continue
                    other_vector = np.mean(vectors, axis=0)

                # Calculate cosine similarity
                similarity = cosine_similarity([aspect_vector], [other_vector])[0][0]

                if similarity >= similarity_threshold:
                    group.append(other_aspect)
                    used_aspects.add(other_aspect)

            except:
                continue

        if len(group) > 1:  # Only keep groups with multiple aspects
            groups.append({
                'group_name': f"Group_{len(groups)+1}_{group[0]}",
                'aspects': group,
                'representative': group[0]
            })

    return groups

@st.cache_resource
def train_comprehensive_models(df):
    """Train multiple models with comprehensive evaluation"""
    # Prepare data
    df_model = df[df['overall'] != 3].copy()  # Remove neutral reviews
    df_model['sentiment'] = df_model['overall'].apply(lambda r: 1 if r > 3 else 0)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df_model['cleaned_review'], df_model['sentiment'],
        test_size=0.2, random_state=42, stratify=df_model['sentiment']
    )

    # TF-IDF Vectorization with detailed parameters
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        max_df=0.8,
        min_df=2,
        stop_words='english'
    )

    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Train multiple models
    models = {
        'Naive Bayes': MultinomialNB(alpha=1.0),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}

    for name, model in models.items():
        # Train model
        model.fit(X_train_tfidf, y_train)

        # Predictions
        y_pred = model.predict(X_test_tfidf)
        y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1] if hasattr(model, 'predict_proba') else None

        # Cross-validation
        cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring='f1')

        # Comprehensive metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None

        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'classification_report': classification_report(y_test, y_pred, target_names=['Negative', 'Positive'], output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

    return results, tfidf_vectorizer, X_test, y_test

# THIS IS THE CORRECTED FUNCTION
@st.cache_data
def analyze_tfidf_features(_tfidf_vectorizer, top_n=20):
    """Analyze TF-IDF features and their importance.
    _tfidf_vectorizer is ignored by the cache.
    """
    feature_names = _tfidf_vectorizer.get_feature_names_out()

    # Get feature importance by summing tfidf scores for each term
    tfidf_scores = _tfidf_vectorizer.transform(feature_names).toarray().diagonal()

    feature_importance = list(zip(feature_names, tfidf_scores))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    return feature_importance[:top_n]


@st.cache_resource
def load_transformer_model():
    """Load pre-trained transformer model with error handling"""
    try:
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            return_all_scores=True
        )
        return sentiment_pipeline
    except Exception as e:
        st.error(f"Error loading transformer model: {str(e)}")
        return None

def analyze_aspect_sentiment_comparison(df, aspect1, aspect2, transformer_model, sample_size=200):
    """Compare sentiment between two aspects"""
    # Sample data for faster processing
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)

    aspect1_sentiments = []
    aspect2_sentiments = []

    progress_bar = st.progress(0, text="Comparing aspects...")

    for idx, (_, row) in enumerate(sample_df.iterrows()):
        progress_bar.progress((idx + 1) / len(sample_df), text=f"Processing review {idx+1}/{len(sample_df)}")

        sentences = sent_tokenize(row['reviewText'])

        for sentence in sentences:
            if aspect1.lower() in sentence.lower():
                truncated = sentence[:512]
                result = transformer_model(truncated)[0]
                sentiment = max(result, key=lambda x: x['score'])
                aspect1_sentiments.append(sentiment['label'])

            if aspect2.lower() in sentence.lower():
                truncated = sentence[:512]
                result = transformer_model(truncated)[0]
                sentiment = max(result, key=lambda x: x['score'])
                aspect2_sentiments.append(sentiment['label'])

    progress_bar.empty()

    # Create comparison data
    comparison_data = {
        'Aspect': [aspect1.capitalize()] * len(aspect1_sentiments) + [aspect2.capitalize()] * len(aspect2_sentiments),
        'Sentiment': aspect1_sentiments + aspect2_sentiments
    }

    return pd.DataFrame(comparison_data)

def create_comparison_charts(comparison_df):
    """Create side-by-side comparison charts"""
    if comparison_df.empty:
        st.warning("No data found for comparison. Please try different aspects.")
        return None, ""

    # Count sentiments for each aspect
    sentiment_counts = comparison_df.groupby(['Aspect', 'Sentiment']).size().unstack(fill_value=0)
    sentiment_counts['Total'] = sentiment_counts.sum(axis=1)
    sentiment_counts['Positive_Rate'] = (sentiment_counts.get('POSITIVE', 0) / sentiment_counts['Total']).fillna(0)

    fig = px.bar(sentiment_counts.reset_index(), x="Aspect", y=["POSITIVE", "NEGATIVE"],
                 title="Side-by-Side Aspect Sentiment Comparison",
                 color_discrete_map={'POSITIVE': '#2ecc71', 'NEGATIVE': '#e74c3c'},
                 labels={'value': 'Number of Mentions', 'variable': 'Sentiment'},
                 barmode='group')

    fig.update_layout(height=500)

    # Generate summary sentence
    if len(sentiment_counts) == 2:
        aspect1_name, aspect2_name = sentiment_counts.index
        aspect1_pos_rate = sentiment_counts.loc[aspect1_name, 'Positive_Rate']
        aspect2_pos_rate = sentiment_counts.loc[aspect2_name, 'Positive_Rate']

        if aspect1_pos_rate > aspect2_pos_rate:
            summary = f" **'{aspect1_name}'** receives more positive mentions ({aspect1_pos_rate:.1%}) compared to **'{aspect2_name}'** ({aspect2_pos_rate:.1%})."
        elif aspect2_pos_rate > aspect1_pos_rate:
            summary = f"**'{aspect2_name}'** receives more positive mentions ({aspect2_pos_rate:.1%}) compared to **'{aspect1_name}'** ({aspect1_pos_rate:.1%})."
        else:
            summary = f" Both aspects have a similar rate of positive mentions ({aspect1_pos_rate:.1%})."
    else:
        summary = "Comparison summary requires two distinct aspects."

    return fig, summary

def create_model_comparison_chart(results):
    """Create comprehensive model comparison visualization"""
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    models = list(results.keys())

    data = []
    for metric in metrics:
        for model in models:
            data.append({'Model': model, 'Metric': metric.capitalize(), 'Score': results[model][metric]})

    df = pd.DataFrame(data)

    fig = px.bar(df, x="Metric", y="Score", color="Model", barmode="group",
                 title="Overall Model Performance Comparison",
                 labels={'Score': 'Score (0.0 to 1.0)'},
                 height=500)

    return fig

def create_tfidf_visualization(feature_importance):
    """Create TF-IDF feature importance visualization"""
    features, scores = zip(*feature_importance)

    df = pd.DataFrame({'Feature': features, 'Score': scores})

    fig = px.bar(df, x='Score', y='Feature', orientation='h',
                 title="Top 20 Most Important Features (TF-IDF)",
                 labels={'Score': 'TF-IDF Importance Score'},
                 color='Score', color_continuous_scale='Viridis',
                 height=600)

    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return fig

def create_confusion_matrix_viz(cm, model_name):
    """Create confusion matrix visualization"""
    fig = px.imshow(cm, text_auto=True,
                    labels=dict(x="Predicted Label", y="True Label", color="Count"),
                    x=['Negative', 'Positive'],
                    y=['Negative', 'Positive'],
                    color_continuous_scale='Blues',
                    title=f"Confusion Matrix: {model_name}")
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header"> Enhanced Aspect-Based Sentiment Analysis</h1>', unsafe_allow_html=True)
    st.markdown("Complete toolkit with semantic grouping, model comparison, and advanced analytics")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header(" Configuration")

        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="Upload your review dataset (must contain 'reviewText' column)"
        )

        st.markdown("###  Analysis Settings")

        # Analysis parameters
        num_aspects = st.slider("Number of aspects to extract", 10, 50, 20)
        min_aspect_freq = st.slider("Minimum aspect frequency", 3, 15, 5)
        sample_size = st.slider("Sample size for analysis", 200, 1000, 500)

        # Advanced options
        st.markdown("###  Advanced Options")
        spell_check = st.checkbox("Apply spelling correction", help="Slower but potentially more accurate")
        similarity_threshold = st.slider("Semantic similarity threshold", 0.3, 0.9, 0.6)

       
    # Main content
    if uploaded_file is not None:
        # Initialize NLTK
        with st.spinner("Setting up NLTK and TextBlob..."):
            install_nltk_data()

        # Load and validate data
        if not st.session_state.data_loaded:
            with st.spinner("Loading and validating dataset..."):
                df = load_and_validate_data(uploaded_file)
                if df is not None:
                    with st.spinner(f"Preprocessing {len(df)} reviews... (this may take a while with spell check)"):
                        df['cleaned_review'] = df['reviewText'].apply(
                            lambda x: advanced_preprocess_text(x, spell_check)
                        )
                    st.session_state.df = df
                    st.session_state.data_loaded = True

                    # Display enhanced data overview
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric(" Total Reviews", f"{len(df):,}")
                    with col2:
                        st.metric(" Avg Rating", f"{df['overall'].mean():.1f}")
                    with col3:
                        st.metric(" Rating Std", f"{df['overall'].std():.1f}")
                    with col4:
                        st.metric(" Avg Length", f"{df['reviewText'].str.len().mean():.0f}")
                    with col5:
                        st.metric(" Features", df.shape[1])

        if st.session_state.data_loaded:
            df = st.session_state.df

            # Extract aspects with advanced method
            if not st.session_state.aspects_extracted:
                with st.spinner("Extracting aspects using advanced POS tagging..."):
                    aspects, aspect_counts = extract_aspects_advanced(df, num_aspects, min_aspect_freq)
                    st.session_state.aspects = aspects
                    st.session_state.aspect_counts = aspect_counts
                    st.session_state.aspects_extracted = True

            aspects = st.session_state.aspects

            # Train Word2Vec model
            if not st.session_state.word2vec_trained:
                with st.spinner("Training Word2Vec model for semantic analysis..."):
                    w2v_model = train_word2vec_model(df)
                    st.session_state.w2v_model = w2v_model
                    st.session_state.word2vec_trained = True

            w2v_model = st.session_state.w2v_model

            # Create semantic groups
            if not st.session_state.semantic_groups_created:
                with st.spinner("Creating semantic aspect groups using cosine similarity..."):
                    semantic_groups = create_semantic_groups(w2v_model, aspects, similarity_threshold)
                    st.session_state.semantic_groups = semantic_groups
                    st.session_state.semantic_groups_created = True

            semantic_groups = st.session_state.semantic_groups

            # Train comprehensive models
            if not st.session_state.models_loaded:
                with st.spinner("Training Naive Bayes, SVM, and Random Forest models..."):
                    model_results, tfidf_vectorizer, X_test, y_test = train_comprehensive_models(df)
                    st.session_state.model_results = model_results
                    st.session_state.tfidf_vectorizer = tfidf_vectorizer
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test

                with st.spinner("Loading pre-trained Transformer model (DistilBERT)..."):
                    transformer_model = load_transformer_model()
                    if transformer_model:
                        st.session_state.transformer_model = transformer_model
                        st.session_state.models_loaded = True

            # Analyze TF-IDF features
            if not st.session_state.tfidf_analyzed:
                with st.spinner("Analyzing TF-IDF features..."):
                    tfidf_features = analyze_tfidf_features(st.session_state.tfidf_vectorizer)
                    st.session_state.tfidf_features = tfidf_features
                    st.session_state.tfidf_analyzed = True

            if st.session_state.models_loaded:
                # Create enhanced tabs
                tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
                    " Single Review", " Aspect Comparison", " Semantic Groups",
                    " Model Performance", "TF-IDF Analysis", " Word2Vec Explorer",
                    "Top Aspects", " Helpful Reviews"
                ])

                with tab1:
                    st.markdown('<div class="section-header">Enhanced Single Review Analysis</div>', unsafe_allow_html=True)

                    user_review = st.text_area(
                        "Enter a review to analyze:",
                        value="The screen quality is amazing with vibrant colors, but the battery drains too quickly. The camera performance is outstanding in daylight.",
                        height=120
                    )

                    if st.button("Analyze Review", type="primary"):
                        if user_review and aspects:
                            cleaned_review = advanced_preprocess_text(user_review, spell_check)
                            sentences = sent_tokenize(user_review)
                            results_found = False

                            for aspect in aspects:
                                for sentence in sentences:
                                    if f" {aspect.lower().split()[-1]} " in f" {sentence.lower()} ":
                                        results_found = True
                                        with st.expander(f" Aspect Detected: **{aspect.capitalize()}**"):
                                            st.markdown(f"> *{sentence.strip()}*")

                                            col1, col2 = st.columns(2)
                                            # Classical Model Analysis
                                            with col1:
                                                st.subheader("Classical Model (Naive Bayes)")
                                                vectorized_sent = st.session_state.tfidf_vectorizer.transform([cleaned_review])
                                                nb_model = st.session_state.model_results['Naive Bayes']['model']
                                                pred_prob = nb_model.predict_proba(vectorized_sent)[0]
                                                sentiment = "Positive" if pred_prob[1] > pred_prob[0] else "Negative"
                                                color = "green" if sentiment == "Positive" else "red"
                                                st.markdown(f"Sentiment: <strong style='color:{color};'>{sentiment}</strong>", unsafe_allow_html=True)
                                                st.progress(max(pred_prob))
                                                st.write(f"Confidence: {max(pred_prob):.2%}")

                                            # Transformer Model Analysis
                                            with col2:
                                                st.subheader("Transformer Model (DistilBERT)")
                                                transformer_result = st.session_state.transformer_model(sentence[:512])[0]
                                                positive_score = next(item['score'] for item in transformer_result if item['label'] == 'POSITIVE')
                                                negative_score = next(item['score'] for item in transformer_result if item['label'] == 'NEGATIVE')
                                                sentiment = "Positive" if positive_score > negative_score else "Negative"
                                                color = "green" if sentiment == "Positive" else "red"
                                                st.markdown(f"Sentiment: <strong style='color:{color};'>{sentiment}</strong>", unsafe_allow_html=True)
                                                st.progress(max(positive_score, negative_score))
                                                st.write(f"Confidence: {max(positive_score, negative_score):.2%}")
                            if not results_found:
                                st.warning("No predefined aspects found in the review. Try a different review or adjust aspect extraction settings.")

                with tab2:
                    st.markdown('<div class="section-header">Aspect Comparison </div>', unsafe_allow_html=True)
                    if len(aspects) >= 2:
                        col1, col2 = st.columns(2)
                        with col1:
                            aspect1 = st.selectbox("Select first aspect:", aspects, index=0)
                        with col2:
                            aspect2 = st.selectbox("Select second aspect:", aspects, index=1)

                        if st.button("Compare Aspects", type="primary"):
                            if aspect1 == aspect2:
                                st.warning("Please select two different aspects for comparison.")
                            else:
                                comparison_df = analyze_aspect_sentiment_comparison(
                                    df, aspect1, aspect2, st.session_state.transformer_model, sample_size
                                )
                                fig, summary = create_comparison_charts(comparison_df)
                                if fig:
                                    st.info(summary)
                                    st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Not enough aspects extracted to perform a comparison. Please adjust settings in the sidebar.")

                with tab3:
                    st.markdown('<div class="section-header">Semantic Aspect Groups</div>', unsafe_allow_html=True)
                    st.info("Aspects are grouped based on semantic similarity using Word2Vec vectors and cosine similarity. This helps consolidate related topics.")
                    if semantic_groups:
                        for group in semantic_groups:
                            with st.expander(f"**{group['representative'].capitalize()}** Group"):
                                st.write(", ".join([a.capitalize() for a in group['aspects']]))
                    else:
                        st.warning("No semantic groups could be formed with the current settings. Try lowering the similarity threshold.")

                with tab4:
                    st.markdown('<div class="section-header">Comprehensive Model Performance</div>', unsafe_allow_html=True)
                    st.plotly_chart(create_model_comparison_chart(st.session_state.model_results), use_container_width=True)

                    for model_name, results in st.session_state.model_results.items():
                        with st.expander(f"Detailed Metrics for **{model_name}**"):
                            st.plotly_chart(create_confusion_matrix_viz(results['confusion_matrix'], model_name), use_container_width=True)
                            report_df = pd.DataFrame(results['classification_report']).transpose()
                            st.dataframe(report_df.round(3))
                            st.write(f"**Cross-Validation F1-Score:** {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")

                with tab5:
                    st.markdown('<div class="section-header">TF-IDF Feature Importance</div>', unsafe_allow_html=True)
                    st.info("This chart shows the terms that the classical models consider most important for classifying sentiment, based on their TF-IDF scores.")
                    st.plotly_chart(create_tfidf_visualization(st.session_state.tfidf_features), use_container_width=True)

                with tab6:
                    st.markdown('<div class="section-header">Word2Vec Explorer</div>', unsafe_allow_html=True)
                    st.info("Explore semantic relationships between words based on the trained Word2Vec model.")
                    search_word = st.text_input("Enter a word to find similar terms:", "battery")
                    if st.button("Find Similar Words"):
                        try:
                            similar_words = w2v_model.wv.most_similar(search_word, topn=10)
                            df_similar = pd.DataFrame(similar_words, columns=['Word', 'Cosine Similarity'])
                            st.dataframe(df_similar)
                        except KeyError:
                            st.error(f"'{search_word}' not found in the model's vocabulary.")

                with tab7:
                    st.markdown('<div class="section-header">Top Product Aspects</div>', unsafe_allow_html=True)
                    aspect_df = pd.DataFrame(
                        Counter(st.session_state.aspect_counts).most_common(num_aspects),
                        columns=['Aspect', 'Mentions']
                    )
                    fig = px.bar(aspect_df, y='Aspect', x='Mentions', orientation='h', title='Top Product Aspects by Frequency',
                                 color='Mentions', color_continuous_scale='Plasma')
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)

                with tab8:
                    st.markdown('<div class="section-header">Helpful Review Analysis</div>', unsafe_allow_html=True)
                    selected_aspect = st.selectbox("Select aspect to find most helpful review for:", aspects)

                    if st.button("Find Most Helpful Review", type="primary"):
                        aspect_df = df[df['reviewText'].str.contains(selected_aspect, case=False)].copy()

                        if not aspect_df.empty:
                            aspect_df['helpfulness_ratio'] = aspect_df['helpful'].apply(
                                lambda x: x[0] / x[1] if len(x) == 2 and x[1] > 5 else 0 # min 5 votes
                            )
                            most_helpful = aspect_df.sort_values(by='helpfulness_ratio', ascending=False).iloc[0]

                            st.subheader(f"Most Helpful Review for '{selected_aspect.capitalize()}'")
                            with st.container(border=True):
                                # Linguistic Analysis
                                review_text = most_helpful['reviewText']
                                word_count = len(word_tokenize(review_text))
                                sentence_count = len(sent_tokenize(review_text))
                                aspects_in_review = [asp for asp in aspects if asp in review_text.lower()]

                                c1, c2, c3, c4 = st.columns(4)
                                c1.metric("Word Count", word_count)
                                c2.metric("Sentence Count", sentence_count)
                                c3.metric("Aspects Mentioned", len(aspects_in_review))
                                c4.metric("Rating", f"{most_helpful['overall']}/5 ")

                                st.markdown(f"**Helpfulness:** {most_helpful['helpful'][0]} out of {most_helpful['helpful'][1]} found helpful ({most_helpful['helpfulness_ratio']:.1%})")
                                st.markdown(f"**Review:**")
                                st.info(f"{review_text}")
                        else:
                            st.warning(f"No reviews found containing the aspect '{selected_aspect}'")

    else:
        # Welcome screen
        st.info("Welcome! Please upload a CSV file using the sidebar to begin the analysis.")
        st.markdown("### How to Get Started")
        st.markdown("""
        1.  **Upload Your Data:** Use the file uploader in the sidebar. Your CSV must have a `reviewText` column. Columns like `overall` and `helpful` are optional but recommended.
        2.  **Configure Settings:** Adjust the analysis parameters in the sidebar to control aspect extraction and processing.
        3.  **Explore the Tabs:** Once the data is processed, navigate through the tabs to explore different facets of the analysis, from high-level model performance to individual review deep-dives.
        """)

if __name__ == "__main__":
    main()