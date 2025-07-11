
query = input("Enter your query: ")

import os
from langchain_community.document_loaders import PyPDFLoader

data_dir = "data"
pages = []

for filename in os.listdir(data_dir):
    if filename.endswith(".pdf"):
        file_path = os.path.join(data_dir, filename)
        loader = PyPDFLoader(file_path, extract_images=False)
        page = loader.load()
        pages.extend(page)

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=20)
docs = text_splitter.split_documents(pages)

from langchain_huggingface import HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever()
results = retriever.invoke(query)
top_chunks = results[:3]

context = "\n\n".join(doc.page_content for doc in top_chunks)
prompt = f"""You are an Elektrobit assistant. Use the following context to answer the user's question.

Context:
{context}

Question:
{query}

Answer:
Start your response with 'Hello, I am ElektrobitAI.' Then provide a clear, concise, and point-wise answer based only on the provided context.
"""

from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="deepseek-coder-v2:latest")
response = llm.invoke(prompt)
print(response)

