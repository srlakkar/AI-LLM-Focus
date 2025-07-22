import streamlit as st
import os
import tempfile
import openai
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid

#Configuration
load_dotenv()

#Extract and split text
def load_and_split_pdfs(pdf_folder):
    all_docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    import os
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder, file))
            documents = loader.load()
            all_docs.extend(text_splitter.split_documents(documents))
    return all_docs

docs = load_and_split_pdfs(r"C:\Users\srira\AI\AI LLM Focus\RAG Basics\data")
print(f"Loaded and split {len(docs)} document chunks.")


#Create embeddings and store in Qdrant

# Connect to local Qdrant
#qdrant = QdrantClient(":memory:")  # in-memory for testing
qdrant = QdrantClient(host="localhost", port=6333)  #local qdrant with docker
collection_name = "pdf_rag"

# Create collection
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
vector_size = embedding_model.get_sentence_embedding_dimension()

# Check if collection exists, delete if so, then create
if qdrant.collection_exists(collection_name):
    qdrant.delete_collection(collection_name=collection_name)
qdrant.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
)

# Insert vectors
payloads = [{"text": doc.page_content} for doc in docs]
vectors = [embedding_model.encode(doc.page_content) for doc in docs]

qdrant.upsert(
    collection_name=collection_name,
    points=[
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vectors[i],
            payload=payloads[i]
        ) for i in range(len(vectors))
    ]
)

print("Documents embedded and stored in Qdrant.")


#Build the RAG function

def retrieve_and_answer(query, top_k=3):
    query_vector = embedding_model.encode(query)
    search_result = qdrant.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k
    )

    context = "\n".join([hit.payload["text"] for hit in search_result])
    prompt = f"Answer the question based on the following context:\n{context}\n\nQuestion: {query}\nAnswer:"

    client = openai.OpenAI()  # Make sure your API key is set in the environment or pass it here
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

