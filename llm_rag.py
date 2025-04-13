import streamlit as st
import requests
from pypdf import PdfReader
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# ---------- Set Page Config ---------- 
st.set_page_config(page_title="Q&A - LLM RAG", layout="wide")

# ---------- PDF Utilities ----------
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        document_text = ""
        for page in pdf_reader.pages:
            document_text += page.extract_text()

    document_text = re.sub(r'\d+', '', document_text)
    document_text = re.sub(r'\s+', ' ', document_text)
    document_text = document_text.strip()

    return document_text

# ---------- FAISS Utilities ----------
def create_faiss_index(text):
    sentences = text.split(".")
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(sentences).toarray()

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors, dtype=np.float32))

    return index, sentences, vectorizer

def query_document(query, index, vectorizer, sentences):
    query_vector = vectorizer.transform([query]).toarray()
    distances, indices = index.search(np.array(query_vector, dtype=np.float32), k=3)

    relevant_sentences = [sentences[i] for i in indices[0]]
    scores = [1 / (1 + d) for d in distances[0]]

    results = list(zip(relevant_sentences, scores))
    return results

# ---------- Mistral API ----------
def query_mistral(prompt, context):
    api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
    headers = {"Authorization": f"Bearer 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"}  # Replace with your Hugging Face token

    payload = {
        "inputs": prompt + " Context: " + context,
        "parameters": {
            "temperature": 0.5,
            "max_new_tokens": 500
        }
    }

    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]['generated_text']
    else:
        return f"Error: {response.status_code}, {response.text}"

# ---------- Streamlit App ----------
def run():
    st.markdown("""
        <style>
            .stButton > button {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 0.5em 2em;
                font-size: 18px;
                transition: background-color 0.3s ease;
            }
            .stButton > button:hover {
                background-color: #c0392b;
            }
            .block {
                background-color: #fdfdfd;
                padding: 2rem;
                border-radius: 20px;
                box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
                margin-bottom: 2rem;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("💬 Smart Q&A Assistant")
    st.markdown("Ask any question about Academic City student policies. The assistant will use both retrieval and a large language model to provide accurate answers.")

    if st.button("⬅️ Back to Services"):
        st.session_state.page = "services"
        st.rerun()

    st.markdown("---")

    # Load & preprocess document
    st.markdown("### 📚 Policy Document")
    with st.spinner("Loading and processing document..."):
        pdf_path = "C:/Users/Michael/Documents/AI EXAM PROJECT/Student_Policies.pdf"

        document_text = extract_text_from_pdf(pdf_path)
        index, sentences, vectorizer = create_faiss_index(document_text)

    if 'responses' not in st.session_state:
        st.session_state.responses = []

    # Previous interactions
    if st.session_state.responses:
        st.markdown("### 📜 Previous Q&A")
        for q_a in st.session_state.responses:
            with st.container():
                st.markdown(f"**❓ Question:** {q_a['question']}")
                st.markdown(f"**💡 Answer:** {q_a['answer']}")
                st.markdown(f"**✅ Confidence Score:** `{q_a['top_score']:.2f}`")
                st.markdown("---")

    # Ask new question
    st.markdown("### 🧠 Ask a New Question")
    unique_key = f"user_query_{len(st.session_state.responses)}"
    user_query = st.text_input("🔎 Enter your question here:", key=unique_key)

    ask_button = st.button("Ask", key=f"ask_button_{unique_key}")

    if ask_button and user_query.strip():
        with st.spinner("🔄 Generating answer..."):
            results = query_document(user_query, index, vectorizer, sentences)
            context = " ".join([res[0] for res in results])
            response = query_mistral(user_query, context)

            st.session_state.responses.append({
                "question": user_query,
                "answer": response,
                "top_score": results[0][1]
            })

            st.rerun()

    st.caption("🚀 Powered by Mistral-7B via HuggingFace Inference API")

if __name__ == "__main__":
    run()
