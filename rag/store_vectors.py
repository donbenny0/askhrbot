import re
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from typing import List
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv


load_dotenv()
azure_openapi_azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openapi_api_key=os.getenv("AZURE_OPENAI_API_KEY")
azure_openapi_deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
azure_openapi_api_version="2024-12-01-preview"


# Initialize embeddings
embedding_model = AzureOpenAIEmbeddings(
    model="text-embedding-3-large",
    azure_endpoint=azure_openapi_azure_endpoint,
    api_key=azure_openapi_api_key,
    openai_api_version=azure_openapi_api_version
)

# initialize initial text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    separators=[r"\n\n", r"\n", r"\. ", " ", ""],
    keep_separator=True
)

# read docs
def read_docs(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return ""
    
    
# create initial chunks
def create_initial_chunks(file_path: str) -> List[str]:
    text = read_docs(file_path)
    if not text:
        return []
    text = re.sub(r'\s+', ' ', text).strip()
    documents = text_splitter.create_documents([text])
    return [doc.page_content for doc in documents]


# create semantic chunks
def create_semantic_chunks(paragraphs: List[str], 
                         similarity_threshold: float = 0.82) -> List[List[str]]:
    if not paragraphs:
        return []
    
    # Batch process all embeddings at once
    para_embeddings = embedding_model.embed_documents(paragraphs)
    para_embeddings = [np.array(e).reshape(1, -1) for e in para_embeddings]
    
    semantic_chunks = []
    current_chunk = []
    
    for i in range(len(paragraphs)):
        if not current_chunk:
            current_chunk.append(paragraphs[i])
            continue
            
        # Compare with all paragraphs in current chunk
        similarities = [cosine_similarity(para_embeddings[i], e)[0][0] 
                       for e in para_embeddings[:i]]
        max_similarity = max(similarities) if similarities else 0
        
        if max_similarity > similarity_threshold:
            current_chunk.append(paragraphs[i])
        else:
            semantic_chunks.append(current_chunk)
            current_chunk = [paragraphs[i]]
    
    if current_chunk:
        semantic_chunks.append(current_chunk)
        
    return semantic_chunks



# Create vectorstore
persist_directory = "askhr_bot_vectorstore"
collection_name = "askhr_bot_vectorstore_collection"

vectorstore = Chroma(
    collection_name=collection_name,
    embedding_function=embedding_model,
    persist_directory=persist_directory,
    collection_metadata={"hnsw:space": "cosine"}
)


# Store into vectorstore
def store_chunks_in_chroma(semantic_chunks: List[List[str]]) -> str:

    if not semantic_chunks:
        return "No chunks to store."
    
    docs = []
    for idx, chunk_group in enumerate(semantic_chunks):
        combined_text = ' '.join(chunk_group).strip()
        if not combined_text:
            continue
            
    
        title = ' '.join(combined_text.split()[:5]) + "..."
        
        doc = Document(
            page_content=combined_text,
            metadata={
                "chunk_id": idx,
                "source": "employee_handbook_india",
                "length": len(combined_text),
                "num_paragraphs": len(chunk_group),
                "title": title,
                "type": "policy"
            }
        )
        docs.append(doc)
    
    if docs:
        # Batch add documents
        vectorstore.add_documents(docs)
        return f"Stored {len(docs)} semantic chunks in Chroma."
    return "No valid documents to store."



# Process the document using the pipeline
file_path = "docs/policies.txt"

print("Creating initial chunks...")
paragraphs = create_initial_chunks(file_path)
print(f"Created {len(paragraphs)} initial chunks.")

print("Creating semantic chunks...")
semantic_chunks = create_semantic_chunks(paragraphs)
print(f"Created {len(semantic_chunks)} semantic chunks.")

print("Storing in Chroma...")
result = store_chunks_in_chroma(semantic_chunks)
print(result)


