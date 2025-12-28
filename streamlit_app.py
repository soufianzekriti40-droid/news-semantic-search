import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from sentence_transformers import SentenceTransformer
import time

# Page configuration
st.set_page_config(
    page_title="News Semantic Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        margin-bottom: 1rem;
        background-color: #f9f9f9;
    }
    .similarity-score {
        font-weight: bold;
        font-size: 1.1rem;
    }
    .category-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: bold;
        display: inline-block;
        margin-right: 0.5rem;
    }
    .category-world { background-color: #e74c3c; color: white; }
    .category-sports { background-color: #3498db; color: white; }
    .category-business { background-color: #2ecc71; color: white; }
    .category-scitech { background-color: #f39c12; color: white; }
</style>
""", unsafe_allow_html=True)

# Cache loading functions

@st.cache_resource
def load_models():
    """Load spaCy and transformer models"""
    import subprocess
    import sys
    
    # Try to load spaCy model, download if not available
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        st.info("Downloading spaCy model... This may take a few minutes on first run.")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load('en_core_web_sm')
    
    transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
    return nlp, transformer_model

@st.cache_data
def load_data():
    """Load processed data and embeddings"""
    df = pd.read_csv('processed_data.csv')
    embeddings_spacy = np.load('embeddings_spacy.npy')
    embeddings_transformer = np.load('embeddings_transformer.npy')
    return df, embeddings_spacy, embeddings_transformer

# Load everything
with st.spinner('Loading models and data...'):
    nlp, transformer_model = load_models()
    df, embeddings_spacy, embeddings_transformer = load_data()

# Helper functions
def get_spacy_embedding(text):
    """Get spaCy embedding for text"""
    doc = nlp(text)
    vectors = [token.vector for token in doc if token.has_vector]
    if len(vectors) == 0:
        return np.zeros(300)
    return np.mean(vectors, axis=0)

def get_transformer_embedding(text):
    """Get transformer embedding for text"""
    return transformer_model.encode([text])[0]

def search_documents(query, method='spacy', top_k=10, category_filter=None, min_similarity=0.0):
    """Search documents using specified method"""
    start_time = time.time()
    
    # Get query embedding
    if method == 'spacy':
        query_emb = get_spacy_embedding(query).reshape(1, -1)
        doc_embeddings = embeddings_spacy
    else:
        query_emb = get_transformer_embedding(query).reshape(1, -1)
        doc_embeddings = embeddings_transformer
    
    # Calculate similarities
    similarities = cosine_similarity(query_emb, doc_embeddings)[0]
    
    # Apply filters
    df['similarity'] = similarities
    filtered_df = df.copy()
    
    if category_filter and category_filter != 'All':
        filtered_df = filtered_df[filtered_df['Category'] == category_filter]
    
    filtered_df = filtered_df[filtered_df['similarity'] >= min_similarity]
    
    # Sort and get top results
    results = filtered_df.nlargest(top_k, 'similarity')
    
    search_time = time.time() - start_time
    
    return results, search_time

def get_category_color(category):
    """Get color for category badge"""
    colors = {
        'World': '#e74c3c',
        'Sports': '#3498db',
        'Business': '#2ecc71',
        'Sci/Tech': '#f39c12'
    }
    return colors.get(category, '#95a5a6')

def display_result_card(rank, row, method_name):
    """Display a single result card"""
    category_class = row['Category'].lower().replace('/', '').replace(' ', '')
    
    st.markdown(f"""
    <div class="result-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <div>
                <span class="category-badge category-{category_class}">{row['Category']}</span>
                <span style="color: #999;">Rank #{rank}</span>
            </div>
            <span class="similarity-score" style="color: {get_category_color(row['Category'])};">
                {row['similarity']:.4f}
            </span>
        </div>
        <h4 style="margin: 0.5rem 0;">{row['Title']}</h4>
        <p style="color: #666; margin: 0;">{row['Description'][:200]}...</p>
    </div>
    """, unsafe_allow_html=True)

# Main UI
st.markdown('<div class="main-header">🔍 News Semantic Search Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Comparing Word Vectors vs Transformer Embeddings</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Search Settings")
    
    # Search method
    comparison_mode = st.checkbox("Compare Both Methods", value=True)
    
    if not comparison_mode:
        method = st.radio(
            "Select Method",
            ["spaCy (Word2Vec)", "Transformer (BERT)"],
            help="Choose which embedding method to use"
        )
        method = 'spacy' if 'spaCy' in method else 'transformer'
    
    # Number of results
    top_k = st.slider("Number of Results", min_value=3, max_value=20, value=5)
    
    # Category filter
    category_filter = st.selectbox(
        "Filter by Category",
        ['All', 'World', 'Sports', 'Business', 'Sci/Tech']
    )
    
    # Similarity threshold
    min_similarity = st.slider(
        "Minimum Similarity",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="Filter results below this similarity score"
    )
    
    st.divider()
    
    # Sample queries
    st.subheader("📝 Sample Queries")
    sample_queries = [
        "football match championship",
        "stock market trading",
        "space exploration NASA",
        "political elections government",
        "artificial intelligence technology",
        "international diplomacy peace"
    ]
    
    selected_sample = st.selectbox(
        "Try a sample query",
        [""] + sample_queries,
        help="Click to use a pre-made query"
    )
    
    st.divider()
    
    # Statistics
    st.subheader("📊 Dataset Stats")
    st.metric("Total Articles", f"{len(df):,}")
    st.metric("Categories", "4")
    st.metric("Embeddings", "2 types")

# Main content
query = st.text_input(
    "Enter your search query:",
    value=selected_sample if selected_sample else "",
    placeholder="e.g., football championship, stock market crash, NASA mission...",
    help="Enter keywords or a natural language query"
)

search_button = st.button("🔍 Search", type="primary", use_container_width=True)

if search_button and query:
    
    if comparison_mode:
        # Side-by-side comparison
        st.subheader("📊 Comparison Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔵 spaCy (Word2Vec)")
            with st.spinner('Searching with spaCy...'):
                results_spacy, time_spacy = search_documents(
                    query, method='spacy', top_k=top_k,
                    category_filter=category_filter, min_similarity=min_similarity
                )
            
            st.caption(f"⏱️ Search time: {time_spacy:.3f}s | Found: {len(results_spacy)} articles")
            
            if len(results_spacy) > 0:
                for idx, row in results_spacy.iterrows():
                    display_result_card(results_spacy.index.get_loc(idx) + 1, row, "spaCy")
            else:
                st.warning("No results found. Try adjusting filters.")
        
        with col2:
            st.markdown("### 🟠 Transformer (BERT)")
            with st.spinner('Searching with Transformer...'):
                results_transformer, time_transformer = search_documents(
                    query, method='transformer', top_k=top_k,
                    category_filter=category_filter, min_similarity=min_similarity
                )
            
            st.caption(f"⏱️ Search time: {time_transformer:.3f}s | Found: {len(results_transformer)} articles")
            
            if len(results_transformer) > 0:
                for idx, row in results_transformer.iterrows():
                    display_result_card(results_transformer.index.get_loc(idx) + 1, row, "Transformer")
            else:
                st.warning("No results found. Try adjusting filters.")
        
        # Comparison metrics
        if len(results_spacy) > 0 and len(results_transformer) > 0:
            st.divider()
            st.subheader("📈 Performance Comparison")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("spaCy Avg Similarity", f"{results_spacy['similarity'].mean():.4f}")
            with col2:
                st.metric("Transformer Avg Similarity", f"{results_transformer['similarity'].mean():.4f}")
            with col3:
                speed_diff = ((time_spacy / time_transformer) - 1) * 100
                st.metric("Speed Difference", f"{abs(speed_diff):.1f}%", 
                         delta="spaCy faster" if time_spacy < time_transformer else "Transformer faster",
                         delta_color="normal" if time_spacy < time_transformer else "inverse")
            with col4:
                # Calculate agreement (how many same articles in top results)
                common = len(set(results_spacy['Title'].head(5)) & set(results_transformer['Title'].head(5)))
                st.metric("Top-5 Agreement", f"{common}/5")
    
    else:
        # Single method search
        st.subheader(f"📊 Results using {method.upper()}")
        
        with st.spinner(f'Searching with {method}...'):
            results, search_time = search_documents(
                query, method=method, top_k=top_k,
                category_filter=category_filter, min_similarity=min_similarity
            )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Results Found", len(results))
        with col2:
            st.metric("Search Time", f"{search_time:.3f}s")
        with col3:
            if len(results) > 0:
                st.metric("Avg Similarity", f"{results['similarity'].mean():.4f}")
        
        st.divider()
        
        if len(results) > 0:
            for idx, row in results.iterrows():
                display_result_card(results.index.get_loc(idx) + 1, row, method)
        else:
            st.warning("No results found. Try adjusting your query or filters.")

# About section
with st.expander("ℹ️ About This Project"):
    st.markdown("""
    ### Research Question
    **How well can traditional word vectors (spaCy embeddings) capture semantic similarity 
    in news articles compared to modern transformer-based embeddings (BERT)?**
    
    ### Methods
    - **spaCy (Word2Vec)**: Traditional word embeddings using mean pooling of word vectors
    - **Transformer (BERT)**: Contextual sentence embeddings using sentence-transformers
    
    ### Dataset
    - **Source**: AG News Topic Classification Dataset
    - **Size**: 10,000 articles (balanced across 4 categories)
    - **Categories**: World, Sports, Business, Sci/Tech
    
    ### Key Findings
    1. **Transformer achieves 6.7% higher precision** (91.7% vs 85.0%)
    2. **Better semantic separation** in embedding space
    3. **Word vectors remain valuable** for speed and efficiency
    4. **Context matters**: Transformers excel at abstract and nuanced queries
    
    ### Technologies Used
    - Python, spaCy, Sentence-Transformers, Streamlit
    - Scikit-learn, NumPy, Pandas, Plotly
    
    ---
    *Computational Linguistics for Discourse Analysis - Group Project*  
    *Master's in Data and Discourse Studies*
    """)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Built with ❤️ using Streamlit | Master's in Data and Discourse Studies<br>
    Group Project for Computational Linguistics for Discourse Analysis</p>
</div>

""", unsafe_allow_html=True)




