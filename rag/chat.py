import os
from time import time
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema.runnable import RunnablePassthrough
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings


# load_dotenv()
load_dotenv()
azure_openapi_azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openapi_api_key=os.getenv("AZURE_OPENAI_API_KEY")
azure_openapi_deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
azure_openapi_api_version="2024-12-01-preview"


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
embedding_model = AzureOpenAIEmbeddings(
    model="text-embedding-3-large",
    azure_endpoint=azure_openapi_azure_endpoint,
    api_key=azure_openapi_api_key,
    openai_api_version=azure_openapi_api_version
)

# initialize LLM
print("Initializing LLM...")
llm = AzureChatOpenAI(
    api_key=azure_openapi_api_key,
    azure_endpoint=azure_openapi_azure_endpoint,
    api_version=azure_openapi_api_version,
    deployment_name=azure_openapi_deployment_name,
    temperature=0,
)


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
# print("Debugging retriever...")
# def print_context(inputs):
#     docs = inputs["context"]
#     print("🔍 Retrieved context:\n")
#     for doc in docs:
#         print(doc.page_content)
#         print("-" * 80)
#     return inputs


# Rag pipeline
print("Initializing RAG pipeline...")
query_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    # | RunnableLambda(print_context)
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