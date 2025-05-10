import os
from time import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema.runnable import RunnablePassthrough,RunnableLambda
from langchain_groq import ChatGroq
from dotenv import load_dotenv


# load_dotenv()
load_dotenv()


# read docs
def read_docs(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return ""

# Initialize embeddings
print("Initializing embeddings...")
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5", model_kwargs={"device": "cpu"},encode_kwargs={"normalize_embeddings": True})

# initialize LLM
print("Initializing LLM...")
llm = ChatGroq(temperature=0,model_name="meta-llama/llama-4-scout-17b-16e-instruct",api_key=os.getenv("GROQ_API_KEY"))


# load vectorstore
persist_directory = "rag/askhr_bot_vectorstore"
collection_name = "askhr_bot_vectorstore_collection"

print("Loading vectorstore...")
vectorstore = Chroma(
    collection_name=collection_name,
    embedding_function=embedding_model,
    persist_directory=persist_directory,
    collection_metadata={"hnsw:space": "cosine"}
)


# initalize base retriver
print("Retrieving base retriever...")
base_retriever = vectorstore.as_retriever(search_type="mmr",search_kwargs={"k": 5,"fetch_k": 10,   "lambda_mult": 0.5})

# initialize query rephraser
print("Retrieving query rephraser...")
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=read_docs("rag/docs/query_rephraser_template.txt"),
)

# initialize multi-query retriever
print("Initializing multi-query retriever...")
retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
    prompt=QUERY_PROMPT,
    parser_key="lines",  
    include_original=True  
)


# initalize template
print("Retrieving template...")
template = read_docs("rag/docs/chat_template.txt")
prompt = ChatPromptTemplate.from_template(template)

# debug retriever
print("Debugging retriever...")
def print_context(inputs):
    docs = inputs["context"]
    print("🔍 Retrieved context:\n")
    for doc in docs:
        print(doc.page_content)
        print("-" * 80)
    return inputs


# Rag pipeline
print("Initializing RAG pipeline...")
query_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | RunnableLambda(print_context)
    | prompt
    | llm
    | StrOutputParser()
)



def timed_query(question: str) -> str:
    start_time = time()
    try:
        print("Processing query...")
        print(f"Question: {question}")
        response = query_chain.invoke(question)
        elapsed = time() - start_time
        print(f"Response time: {elapsed:.2f} seconds")
        return response
    except Exception as e:
        return f"Error processing query: {str(e)}"
    
# print(timed_query("What is the remote work policy?"))