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
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever

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
prompt_template = read_docs("rag/docs/chat_template.txt")


retriever_prompt = (
    "Given a chat history and the latest user question which might reference context in the chat history,"
    "formulate a standalone question which can be understood without the chat history."
    "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt  = ChatPromptTemplate.from_messages(
    [
        ("system", retriever_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),


     ]
)

history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_template),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Rag pipeline
print("Initializing RAG pipeline...")


def timed_query(user_input: str) -> str:
    chat_history = []

    # Step 1: Read chat history from file if it exists
    try:
        with open('chat_history.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 2):
                if i + 1 < len(lines):
                    chat_history.extend([
                        HumanMessage(content=lines[i].strip()),
                        AIMessage(content=lines[i + 1].strip())
                    ])
    except FileNotFoundError:
        # No existing chat history
        pass

    # Step 2: Invoke RAG chain
    start_time = time()
    try:
        response = rag_chain.invoke({
            "input": user_input,
            "chat_history": chat_history
        })

        # Add the new interaction to history
        user_message = HumanMessage(content=user_input)
        ai_message = AIMessage(content=response["answer"])
        chat_history.extend([user_message, ai_message])

        # Step 3: Append new interaction to chat history file
        with open('chat_history.txt', 'a', encoding='utf-8') as f:
            f.write(f"{user_input}\n{response['answer']}\n")

        elapsed = time() - start_time
        print(f"Response time: {elapsed:.2f} seconds")
        return response["answer"]

    except Exception as e:
        return f"Error processing query: {str(e)}"
    