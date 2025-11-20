from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, load_summarize_chain

loader = TextLoader("/Users/adityashivhare/Downloads/python_project/001Assignment/speech.txt")
docs = loader.load()

splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectordb = Chroma.from_documents(chunks, embedding, persist_directory="db")
retriever = vectordb.as_retriever()

llm = OllamaLLM(model="mistral")

prompt = ChatPromptTemplate.from_template("""
Answer the question ONLY using the context provided below.
If the answer is not found in the context, say:
"The answer is not present in the document."

<context>
{context}
</context>

Question: {input}
""")
doc_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, doc_chain)

while True:
    q = input("Question: ")

    if q.lower().strip() == "exit":
        break

    result = rag_chain.invoke({"input": q})
    print("Answer:", result["answer"], "\n")

