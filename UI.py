import os
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM

# Source path for dataset and Vector database which is being used in the RAG layer
data_dir = "data"
persist_dir = "./chroma_db"

# Embedding model is HuggingFace MiniLM
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Reading all the pdf files using PyPDFLoader from langchain
pages = []
for filename in os.listdir(data_dir):
    if filename.endswith(".pdf"):
        file_path = os.path.join(data_dir, filename)
        loader = PyPDFLoader(file_path, extract_images=False)
        page = loader.load()
        pages.extend(page)
 
# Splitting test into chunks with overlaps
# You can change this parameters to generate more accurate responses once you get cloud access
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
docs = text_splitter.split_documents(pages)

# Storing all the data obtained in ChromaDB
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory=persist_dir
)

# Creating an object of the VectorDB to retrieve relevant document
retriever = vectorstore.as_retriever()

# You can change the LLM model being used here
# Once you get cloud storage please download 650B parameters model and try to use it here 
# If i have already left by then please tell me how good it is, I never used that in my life (sed life)
llm = OllamaLLM(model="deepseek-coder-v2:latest")

# You could pass history of chat as a list of Strings
# I have not done that so the variable is not being used
# I am retrieving top 5 elements from VectorDB here, if you get access to good models will large input token limit, increase it
def answer_query(query, history): 
    results = retriever.invoke(query)
    top_chunks = results[:5]
    context = "\n\n".join(doc.page_content for doc in top_chunks)
    print(context)

    # This is the prompt that is being sent to LLM 
    # Change it according to your usage
    prompt = f"""You are a very smart and super fast assistant. Use the following context to answer the user's question.

    Context:
    {context}

    Question:
    {query}

    Answer:
    Provide a clear and point-wise answer based only on the provided context.
    """
    
    # I dont have to explain this I guess
    # If you are too dumb then its getting the response from the LLM and printing it
    response = llm.invoke(prompt)
    return response

# Gradio UI is being used here
# Its very basic to use just to make UI for chat bots
with gr.Blocks() as app:
    gr.ChatInterface(
        fn=answer_query,
        title="ElektrobitAI",
        chatbot=gr.Chatbot(type='messages'),
        textbox=gr.Textbox(placeholder="Enter your query here...", lines=2),
        type="messages"
    )

app.launch()

#Thanks I guess idk 