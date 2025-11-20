# AmbedkarGPT-Intern-Task


# Kalpit Pvt Ltd – AI Intern Hiring  
## Assignment 01 + Assignment 02 

---

## 📁 Folder Structure

The updated folder now looks like this:

```
001Assignment/
│
├── Assignment-01.py        # CLI-based RAG system (Assignment 1)
├── Assignment-02.py        # Final evaluation script (renamed from Assignment-03.py)
│
├── speech.txt              # Input speech for Assignment 1
│
├── Document/               # Corpus for Assignment 2 (6 documents)
│     ├── speech1.txt
│     ├── speech2.txt
│     ├── speech3.txt
│     ├── speech4.txt
│     ├── speech5.txt
│     └── speech6.txt
│
├── test_dataset.json       # Provided evaluation dataset
└── test_results.json       # Generated results
```

---

# 🧩 Assignment 01 – RAG QA System

### ✔ Purpose  
Build a simple command-line Retrieval-Augmented Generation (RAG) system using:

- Text loading  
- Chunking  
- Embeddings  
- Chroma vector store  
- Ollama LLM  
- Retrieval + answer generation  

### ▶ Run Assignment 01

```bash
python Assignment-01.py
```

Ask any question:

```
Question: What is the central theme of the speech?
```

Exit:

```
exit
```

---

# 🧪 Assignment 02 – Evaluation Framework  
(Using the final script previously named `Assignment-03.py`)

### ✔ Goal  
Evaluate the RAG system across a multi-document corpus using 25 predefined questions.

### ✔ Metrics Implemented

#### **Retrieval**
- Hit Rate  
- Precision@5  
- MRR@5  

#### **Answer Quality**
- ROUGE-L  
- Semantic Similarity  

#### **Faithfulness**
- Cosine similarity between answer and retrieved context  

### ▶ Run Assignment 02

```bash
python Assignment-02.py
```

This produces:

- `test_results.json`  
- Console output summarizing metrics  

---

# 📄 Input Files

### `Document/` folder  
Must contain the 6 corpus files:

```
speech1.txt
speech2.txt
speech3.txt
speech4.txt
speech5.txt
speech6.txt
```

### `test_dataset.json`  
Contains 25 evaluation questions with:

- id  
- question  
- ground_truth  
- source_documents  
- question_type  
- answerable  

---

# 📊 Output Format

`test_results.json` example:

```json
{
  "id": 1,
  "question": "...",
  "gold_answer": "...",
  "predicted_answer": "...",
  "hit_rate": 0.0,
  "precision@5": 0.0,
  "mrr@5": 0.0,
  "rouge_l": 0.42,
  "semantic_similarity": 0.61,
  "faithfulness": 0.78
}
```

---

# 📦 Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```
langchain
langchain-core
langchain-community
langchain-huggingface
langchain-ollama

chromadb

sentence-transformers
scikit-learn
pandas
numpy

rouge-score
```

---
