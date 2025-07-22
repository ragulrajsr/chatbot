import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["PYTORCH_NO_MPS"] = "1"


import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

# Cache loading CSV data (news highlights)
@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv("data/news_highlights.csv")

highlights = load_data()
documents = highlights["representative_title"].tolist()

# Cache setup of embedding model and FAISS index
@st.cache_resource(show_spinner=False)
def setup_faiss_index(docs):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(docs, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return embedder, index

embedder, index = setup_faiss_index(documents)

# Cache loading local text generation pipeline (GPT-2) - CPU only
@st.cache_resource(show_spinner=False)
def load_generator():
    generator = pipeline(
        'text-generation',
        model='gpt2',
        device=-1,            # Use CPU explicitly
        max_length=150
    )
    return generator

generator = load_generator()

# Retrieve most relevant documents from FAISS index based on query
def retrieve_docs(query, k=3):
    q_vec = embedder.encode([query], convert_to_numpy=True)
    _, idxs = index.search(q_vec, k)
    return [documents[i] for i in idxs[0]]

# Generate answer given retrieved contexts and user question
def generate_answer(contexts, question):
    context_text = "\n".join(contexts)
    prompt = f"Context:\n{context_text}\nQuestion: {question}\nAnswer:"
    outputs = generator(prompt, max_length=150, num_return_sequences=1)
    # Remove prompt prefix from generated text for cleaner output
    generated = outputs[0]['generated_text']
    answer = generated[len(prompt):].strip()
    return answer

# Streamlit UI
st.title("News Highlights Chatbot")

# Category filter in sidebar
categories = highlights["category"].unique()
selected_cat = st.sidebar.selectbox("Select Category", categories)

# Filter news by category
filtered = highlights[highlights["category"] == selected_cat]

st.header(f"Top Highlights - {selected_cat.capitalize()}")

for _, row in filtered.iterrows():
    st.subheader(row["representative_title"])
    st.write(f"Sources reporting: {row['source_count']}")
    st.markdown("---")

st.header("Ask questions about these news highlights")

user_question = st.text_input("Your question:")

if user_question:
    with st.spinner("Generating answer..."):
        retrieved = retrieve_docs(user_question)
        answer = generate_answer(retrieved, user_question)
    st.markdown("**Answer:**")
    st.write(answer)
