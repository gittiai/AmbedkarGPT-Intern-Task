import json
import pandas as pd
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

corpus_path = "/Users/adityashivhare/Downloads/python_project/001Assignment/Document"
loader = DirectoryLoader(
    corpus_path,
    glob="**/*.txt",   
)
docs = loader.load()

splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(docs)

#embedding_model = OllamaEmbeddings(model="mistral")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(chunks, embedding_model, persist_directory="db")
retriever = vectordb.as_retriever()

llm = OllamaLLM(model="mistral")

prompt = ChatPromptTemplate.from_template("""
Answer the question ONLY using the context provided.
If the answer is not present, reply:
"The answer is not present in the document."

<context>
{context}
</context>

Question: {input}
""")

doc_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, doc_chain)


def hit_rate(relevant_docs, retrieved_docs):
    return 1.0 if any(doc in retrieved_docs for doc in relevant_docs) else 0.0

def precision_at_k(relevant_docs, retrieved_docs):
    if len(retrieved_docs) == 0:
        return 0
    hits = sum(1 for doc in retrieved_docs if doc in relevant_docs)
    return hits / len(retrieved_docs)

def mrr(relevant_docs, retrieved_docs):
    for idx, doc in enumerate(retrieved_docs):
        if doc in relevant_docs:
            return 1 / (idx + 1)
    return 0.0

def rouge_l_score(pred, gold):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(gold, pred)["rougeL"].fmeasure

def cosine_sim(a, b, model):
    emb1 = model.encode([a])
    emb2 = model.encode([b])
    return cosine_similarity(emb1, emb2)[0][0]

json_path = "/Users/adityashivhare/Downloads/python_project/001Assignment/test_dataset.json"

with open(json_path, "r") as f:
     test_data = json.load(f)["test_questions"] 

semantic_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")


results = []

for item in test_data:
    q = item["question"]
    gold = item["ground_truth"]

    output = rag_chain.invoke({"input": q})
    pred = output["answer"]

    retrieved_docs = retriever.invoke(q)
    retrieved_texts = [d.page_content for d in retrieved_docs]

    hr = hit_rate([gold], retrieved_texts)
    pk = precision_at_k([gold], retrieved_texts)
    mrr_score = mrr([gold], retrieved_texts)
    rouge_l = rouge_l_score(pred, gold)
    sem_sim = cosine_sim(pred, gold, semantic_model)
    faith = cosine_sim(pred, " ".join(retrieved_texts), semantic_model)

    results.append({
    "id": int(item["id"]),
    "question": q,
    "gold_answer": gold,
    "predicted_answer": pred,
    "hit_rate": float(hr),
    "precision@5": float(pk),
    "mrr@5": float(mrr_score),
    "rouge_l": float(rouge_l),
    "semantic_similarity": float(sem_sim),
    "faithfulness": float(faith)
})


with open("test_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Evaluation complete. Results saved to test_results.json")
