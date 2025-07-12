from fastapi import APIRouter, HTTPException, status, Request
from pydantic import BaseModel
from threading import Lock
import os
from typing import Dict

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

# Import your custom modules
from app.core.config import settings
# Changed import from get_pinecone_vectorstore to get_faiss_vectorstore
from app.db.vector_db import get_faiss_vectorstore
from app.utils.prompts import prompt_template

# Initialize the FastAPI router
router = APIRouter()

# --- Global Components (Initialized Once) ---
# Initialize the vector store (now FAISS in-memory)
docsearch = None # Initialize as None
try:
    docsearch = get_faiss_vectorstore() # Call the new FAISS function
except Exception as e:
    print(f"Failed to initialize FAISS vector store at startup: {e}")
    # docsearch remains None, and the chat endpoint will handle this.

# Initialize the LLM (Using OpenAI)
try:
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set in environment variables.")

    llm = ChatOpenAI(
        openai_api_key=settings.OPENAI_API_KEY,
        model_name="gpt-3.5-turbo", # Use a suitable OpenAI model
        temperature=0.8,
        max_tokens=512
    )
    print(f"Successfully initialized OpenAI LLM: {llm.model_name}")
except ValueError as e:
    print(f"Failed to initialize OpenAI LLM at startup: {e}")
    llm = None
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"OpenAI API Key not configured. Error: {e}"
    )
except Exception as e:
    print(f"Failed to initialize OpenAI LLM at startup: {e}")
    llm = None
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"An error occurred while initializing the OpenAI LLM. Error: {e}"
    )


# Define the prompt template for the RAG chain
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# Global lock for model inference to handle concurrent requests gracefully
inference_lock = Lock()

# In-memory storage for conversation history (for demonstration purposes)
conversation_memories: Dict[str, ConversationBufferMemory] = {}

# --- Helper to get or create conversation memory for a session ---
def get_conversation_memory(session_id: str) -> ConversationBufferMemory:
    """
    Retrieves or creates a ConversationBufferMemory instance for a given session ID.
    """
    if session_id not in conversation_memories:
        print(f"Creating new conversation memory for session: {session_id}")
        conversation_memories[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    return conversation_memories[session_id]

# --- Pydantic model for request body validation ---
class ChatRequest(BaseModel):
    message: str
    session_id: str

# --- Chat Endpoint ---
@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Handles chat requests, processes messages using the RAG pipeline,
    and maintains conversation history.
    """
    user_message = request.message.strip()
    session_id = request.session_id.strip()

    if not user_message:
        print("Received empty message. Returning error.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty."
        )

    # Check if docsearch was successfully initialized
    if docsearch is None:
        print("FAISS vector store not initialized. Returning error.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chatbot's knowledge base is not ready. Please check backend logs for FAISS initialization errors."
        )
    if llm is None:
        print("LLM model not initialized. Returning error.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chatbot's brain is not ready. Please check backend logs for LLM loading errors."
        )

    memory = get_conversation_memory(session_id)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=False,
        chain_type_kwargs=chain_type_kwargs,
        memory=memory
    )

    bot_response = "Oops! The Totot Assistant is having trouble right now. Please try again later."

    if not inference_lock.acquire(blocking=False):
        print("Rapid request detected. Model is busy.")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="The chatbot is currently busy. Please try again in a moment."
        )

    try:
        print(f"Processing message for session {session_id}: '{user_message}'")
        result = qa_chain.invoke({"query": user_message})
        bot_response = result.get("result", bot_response)
        print(f"Bot response for session {session_id}: '{bot_response}'")

    except Exception as e:
        print(f"Error during model inference for session {session_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An error occurred while processing your request: {str(e)}"
        )
    finally:
        inference_lock.release()

    return {"answer": bot_response}
