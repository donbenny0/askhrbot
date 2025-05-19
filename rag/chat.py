import os
import json
import random
from time import time
from datetime import datetime
from typing import List, Dict, Any
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage


load_dotenv()

class Config:
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    API_VERSION = "2024-02-01"
    PERSIST_DIRECTORY = "rag/askhr_bot_vectorstore"
    COLLECTION_NAME = "askhr_bot_vectorstore_collection"
    
embedding_model = AzureOpenAIEmbeddings(
model="text-embedding-3-large",
    azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
    api_key=Config.AZURE_OPENAI_API_KEY,
    openai_api_version=Config.API_VERSION
)

llm = AzureChatOpenAI(
    api_key=Config.AZURE_OPENAI_API_KEY,
    azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
    api_version=Config.API_VERSION,
    deployment_name=Config.AZURE_OPENAI_DEPLOYMENT_NAME,
    temperature=0,
)

vectorstore = Chroma(
    collection_name=Config.COLLECTION_NAME,
    embedding_function=embedding_model,
    persist_directory=Config.PERSIST_DIRECTORY,
    collection_metadata={"hnsw:space": "cosine"}
)

def initialize_retrievers():
    try:
        raw = vectorstore.get(include=["documents", "metadatas"])
        docs = [
            Document(page_content=content, metadata=metadata)
            for content, metadata in zip(raw["documents"], raw["metadatas"])
        ]

        bm25_retriever = BM25Retriever.from_documents(docs, k=5)

        vector_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 7, "fetch_k": 20, "lambda_mult": 0.6, "score_threshold": 0.7}
        )

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.4, 0.6]
        )

        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate five 
            different versions of the given user question to retrieve relevant documents from a 
            vector database. Provide these alternative questions separated by newlines.
            Original question: {question}"""
        )

        return MultiQueryRetriever.from_llm(
            retriever=ensemble_retriever,
            llm=llm,
            prompt=QUERY_PROMPT,
            include_original=True
        )

    except Exception as e:
        print(f"Error initializing retrievers: {e}. Falling back to simple retriever.")
        return vectorstore.as_retriever(search_kwargs={"k": 5})


retriever = initialize_retrievers()


class ChatHistoryManager:
    def __init__(self, user_id: str = "default", session_id: str = "default_session"):
        self.user_id = user_id
        self.session_id = session_id
        self.history_file = f"chat_history_{user_id}.json"

    def load_history(self) -> List[Dict[str, Any]]:
        try:
            with open(self.history_file, 'r') as f:
                all_history = json.load(f)
            return [entry for entry in all_history if entry.get("session_id") == self.session_id]
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def save_history(self, history: List[Dict[str, Any]]):
        try:
            with open(self.history_file, 'r') as f:
                all_history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            all_history = []

        # remove old session data
        all_history = [entry for entry in all_history if entry.get("session_id") != self.session_id]
        all_history.extend(history)
        with open(self.history_file, 'w') as f:
            json.dump(all_history[-100:], f, indent=2)

    def summarize_if_needed(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        unsummarized_blocks = [
            entry for entry in history
            if not entry.get("summarized", False)
            and entry.get("user_message") and entry.get("assistant_response")
        ]

        if len(unsummarized_blocks) < 5:
            return history

        history_text = "\n".join(
            f"User: {entry['user_message']}\nAssistant: {entry['assistant_response']}"
            for entry in unsummarized_blocks[:10]
        )

        summary_prompt = f"""
        Summarize the following 10 interactions into one concise but informative summary:
        {history_text}
        """

        summary = llm.invoke(summary_prompt.strip())

        # ✅ Fix: Ensure AIMessage is converted to string
        if isinstance(summary, AIMessage):
            summary = summary.content

        summary_entry = {
            "role": "system",
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "summarized": True,
            "summary_of": [entry["timestamp"] for entry in unsummarized_blocks[:10]]
        }

        # Replace the first 10 unsummarized entries with the summary
        history = [entry for entry in history if entry not in unsummarized_blocks[:10]]
        history.insert(0, summary_entry)
        return history

    def append_chat_pair(self, history: List[Dict[str, Any]], user_msg: str, assistant_msg: str) -> List[Dict[str, Any]]:
        history.append({
            "user_message": user_msg,
            "assistant_response": assistant_msg,
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "summarized": False
        })
        return self.summarize_if_needed(history)


def get_dynamic_prompt(user_input: str, history: List) -> PromptTemplate:
    sensitive_keywords = ["complaint", "harassment", "grievance", "termination"]
    policy_keywords = ["policy", "rule", "guideline"]
    benefit_keywords = ["benefit", "pto", "leave", "insurance"]
    
    if any(kw in user_input.lower() for kw in sensitive_keywords):
        instructions = "This is a sensitive topic. Be professional and direct the user to official HR channels if appropriate."
    elif any(kw in user_input.lower() for kw in policy_keywords):
        instructions = "Provide exact policy details with reference to the policy document when possible."
    elif any(kw in user_input.lower() for kw in benefit_keywords):
        instructions = "Include eligibility requirements and any limitations for benefits mentioned."
    else:
        instructions = "Respond helpfully and professionally."
    
    template = f"""You are an HR assistant for a company. Use the following context to answer the question at the end.
If you don't know the answer, say you don't know. Be concise but helpful.

Context:
{{context}}

Conversation history:
{{chat_history}}

Question: {{input}}

Considerations:
1. {instructions}
2. Format lists and important details clearly
3. Provide sources when available

Answer:"""
    
    return PromptTemplate.from_template(template)


def docs_to_serializable(docs: List[Document]) -> List[Dict[str, Any]]:
    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in docs
    ]
    
def chat(user_input: str, user_id: str = "default", session_id: str = "default_session") -> str:
    history_manager = ChatHistoryManager(user_id, session_id)
    chat_history = history_manager.load_history()

    formatted_history = []
    for entry in chat_history:
        if entry.get("summarized", False):
            formatted_history.append(("system", entry["summary"]))
        else:
            formatted_history.append(("user", entry["user_message"]))
            formatted_history.append(("assistant", entry["assistant_response"]))



    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, formulate a standalone question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)
    qa_prompt = get_dynamic_prompt(user_input, chat_history)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    start_time = time()
    try:
        response = rag_chain.invoke({
            "input": user_input,
            "chat_history": formatted_history
        })
        elapsed = time() - start_time

        # Make sure response["answer"] is a plain string
        answer = response["answer"]
        if isinstance(answer, AIMessage):
            answer = answer.content


        chat_history = history_manager.append_chat_pair(
            chat_history, user_msg=user_input, assistant_msg=answer
        )

        history_manager.save_history(chat_history)

        return answer

    except Exception as e:
        print(f"Error in RAG chain: {e}")
        fallback_responses = [
            f"I'm having trouble accessing that information. Could you rephrase your question? (Error: {str(e)[:50]})",
            "My knowledge base seems to be unavailable at the moment. Please try again later.",
            "I encountered an unexpected error while processing your request."
        ]
        return random.choice(fallback_responses)
