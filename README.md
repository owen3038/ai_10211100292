# ai_10211100292

```markdown
# 🤖 Smart Academic Assistant - AI Exam Project

This project is a comprehensive AI-powered academic assistant designed to showcase practical applications of machine learning, deep learning, and natural language processing using **Streamlit**. The system supports:

- 📈 **Regression Modeling**
- 📊 **Clustering Analysis**
- 🧠 **Neural Network Classification** (with PyTorch)
- 💬 **LLM-powered Q&A** on Academic City Student Policies (RAG system)

---

## 🚀 Features

### 1. Regression
- Upload CSV files for regression tasks.
- Choose a target column and input features.
- Preprocess data (handle missing values, normalization).
- Train and visualize regression models.
- Predict new values based on trained models.

### 2. Clustering
- Upload dataset.
- Select the number of clusters.
- Visualize results using PCA or t-SNE.
- See cluster assignments and structure.

### 3. Neural Network Classification (PyTorch)
- Upload dataset and define the target column.
- Train a feedforward neural network with adjustable epochs and learning rate.
- Evaluate model accuracy and loss.
- Predict test values or upload your own model and data.
- Download predictions as CSV.

### 4. LLM RAG (Retrieval-Augmented Generation)
- Ask questions based on **Academic City Student Policy** PDF.
- FAISS vector search finds relevant context.
- LLM (Mistral-7B via HuggingFace API) generates accurate, document-based answers.
- Confidence score based on sentence similarity is displayed.

---

## 🧠 Architecture

### LLM Q&A (RAG) System
- **Input**: User query
- **FAISS Index**: Built from cleaned policy PDF using TF-IDF vectors.
- **Retrieval**: Top 3 semantically similar sentences.
- **Prompt Creation**: Combines context + question.
- **LLM**: Mistral-7B Instruct model via HuggingFace API.
- **Output**: Generated answer with a confidence score.

![Architecture](architecture.png)

---

## 📄 Dataset & Models

### Datasets
- Academic City Student Policy (PDF) for the Q&A system.
- User-uploaded CSVs for Regression, Clustering, and Neural Networks.

### Models Used
- Linear Regression (`scikit-learn`)
- KMeans Clustering (`scikit-learn`)
- Feedforward Neural Network (`PyTorch`)
- Mistral-7B LLM (`HuggingFace API`)
- Semantic Search with `FAISS` and `TF-IDF`

---

## ⚙️ How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/academic-assistant-ai.git
   cd academic-assistant-ai
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run main.py
   ```

4. Use the sidebar to explore features:
   - Services → Regression, Clustering, Neural Networks, LLM Q&A

---

## 📊 Example Q&A Comparison

| Query                     | ChatGPT Response                          | LLM RAG Response                                                                 |
|--------------------------|-------------------------------------------|----------------------------------------------------------------------------------|
| What is the dress code?  | "It typically includes modest attire."    | "The policy specifies students must wear formal attire on designated days."     |
| Who to contact for absence? | "Generally your lecturer or head."      | "The policy states to contact the Registrar’s office via email at acity.edu.gh."|

---

## 📁 Project Structure

```
📂 academic-assistant-ai
│
├── main.py                   # Main Streamlit app
├── regression.py             # Regression logic
├── clustering.py             # Clustering logic
├── neural_networks.py        # Neural network (PyTorch)
├── llm_rag.py                # LLM Q&A (RAG system)
├── requirements.txt          # Dependencies
├── architecture.png          # System architecture image
├── Student_Policies.pdf      # Academic City Policy Document
└── README.md                 # This file
```

---

## 📬 Contact

For any questions or issues, feel free to reach out:
- **Name:** Michael Onwuachi
- **Institution:** Academic City University 
- **Email:** michael.onwuachi@acity.edu.gh
