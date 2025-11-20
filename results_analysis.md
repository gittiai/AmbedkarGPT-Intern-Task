# Results Analysis – Assignment 02  
This document summarizes the performance of the RAG evaluation pipeline applied to a 6-document corpus related to Dr. B.R. Ambedkar. The test set included 25 factual, conceptual, comparative, and unanswerable questions.

---

## 1. Retrieval Performance

The retrieval module struggled to identify the correct document for every question:

- **Hit Rate:** 0/25  
- **Precision@5:** 0  
- **MRR@5:** 0  

This indicates that the retriever did not surface any relevant document among the top-k results.  
The LLM therefore generated answers without the necessary context, leading to partially correct or incorrect outputs.

### Likely Reasons
- Chunk size (300 chars) too small → context fragmented across chunks  
- Embeddings (MiniLM-L6) not well suited for philosophical or political text  
- No reranker used  
- Comparative questions require multi-document retrieval, which was not enabled  

---

## 2. Answer Quality

Despite poor retrieval, the LLM produced a wide range of answer quality.

### ROUGE-L
- High scores (0.5–0.7) appeared when the model relied on generic descriptions similar to the ground truth  
- Low scores (0.0–0.2) for document-specific factual or comparative questions  

### Semantic Similarity
- Scores ranged from ~0.07 to ~0.80  
- Higher values for conceptual definitions or generic responses  
- Lower values when the question required precise document grounding  

### Faithfulness
- Faithfulness was often high even when the answer was wrong  
- This means the LLM stuck to *retrieved* context, but the retrieved context was irrelevant  

---

## 3. Unanswerable Questions

The system performed well on unanswerable questions.  
It consistently responded with:  
**“The answer is not present in the document.”**

This matched the ground truth and yielded high ROUGE-L and semantic similarity scores.

---

## 4. Common Failures

### a. Document-specific factual questions  
The system frequently missed clear answers that existed in the corpus.

### b. Comparative questions  
Both documents required for comparison were rarely retrieved together, leading to incomplete or incorrect conclusions.

### c. Multi-sentence answers  
Some answers contained partial correctness but lacked detail due to missing context.

---

## 5. Recommendations

### **1. Improve Retrieval Quality**
- Increase chunk size (recommended: **800–1200 characters**)  
- Use stronger embeddings (e.g., `nomic-embed-text`, BGE models)  
- Add a cross-encoder reranker (e.g., `bge-reranker-base`)  

### **2. Support Multi-Document Queries**
- Use multi-query retrieval  
- Use hybrid retrievers (BM25 + embeddings)  

### **3. Reduce Irrelevant Context**
- Apply reranking before passing context to the LLM  

### **4. Improve Handling of Comparative Questions**
- Retrieve from multiple documents explicitly  
- Merge or concatenate top chunks from each relevant document  

---

