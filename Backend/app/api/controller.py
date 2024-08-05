# from fastapi import FastAPI, APIRouter, HTTPException, Depends, status, Form
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Annotated
# from sqlalchemy import create_engine, select
# from sqlalchemy.orm import sessionmaker, Session
# from .model import Base, ChatHistory
# import requests

# from langchain_chroma import Chroma
# from langchain_community.document_loaders import TextLoader, DocumentLoader
# from langchain_community.embeddings.sentence_transformer import (
#     SentenceTransformerEmbeddings,
# )
# from langchain_text_splitters import CharacterTextSplitter


# from llm.ollama import get_llama3, get_dolphin_phi, get_llama3_1
# from llm.gemini import get_gemini


# # app = FastAPI()
# router = APIRouter(tags=["chat with llm"], prefix="/chat-with-llm")

# # Database logic

# DATABASE_URL = "mysql+pymysql://root:Mysql%40001@127.0.0.1:3306/mydemo_bot"
# engine = create_engine(DATABASE_URL)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base.metadata.create_all(bind=engine)


# class ChatRequest(BaseModel):
#     prompt: str


# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()


# db_dependency = Annotated[Session, Depends(get_db)]

# # Vector store configuration

# def fetch_documents(db:Session):
#      # session = SessionLocal(d)
#      documents = db.query(ChatHistory).all()
#      return documents

# class DBLoader(DocumentLoader):
#      def __init__(self, db: Session):
#           self.db = db
          
#      def load(self):
#           documents = fetch_documents(self.db)
#           return [{"page_content": doc.user_input + "\n" + doc.model_response} for doc in documents]



# # api endpoints
# @router.get("/history", status_code=status.HTTP_200_OK)
# async def get_history(db: db_dependency):
#     try:
#         chat_history = db.query(ChatHistory).all()
#         return chat_history
#     except Exception as e:
#         raise HTTPException(
#             status_code=404, detail="Error retrieving chat history: " + str(e)
#         )


# @router.post("/chat", status_code=status.HTTP_200_OK)
# async def start_convo(db: db_dependency, input: ChatRequest, model: str):
#      try:
#           loader = DBLoader()
#           documents = loader.load()

#           text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#           docs = text_splitter.split_documents(documents)

#           # create the open-source embedding function
#           embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

#           # load it into Chroma
#           vector_db = Chroma.from_documents(docs, embedding_function)

#           # Retrieve top 5 relevant chat history entries
#           query_embedding = embedding_function.embed_query(input.prompt)
#           similar_chats = vector_db.similarity_search_with_embeddings(query_embedding, k=5)          
#           history_prompt = "\n".join([f"User: {chat.user_input}\nModel: {chat.model_response}" for chat in similar_chats])
#           # history_prompt = "\n".join([f"User: {chat.user_input}\nModel: {chat.model_response}" for chat in chat_history])
#           history_prompt += f"\nUser: {input.prompt}"
#           try:
#                if model == "llama3":
#                     response = get_llama3(history_prompt)
#                elif model == "llama3.1":
#                     response = get_llama3_1(history_prompt)
#                elif model == "dolphin-phi":
#                     response = get_dolphin_phi(history_prompt)
#                elif model == "gemini":
#                     response = get_gemini(user_question = input, chat_history = history_prompt)
#                chat_entry = ChatHistory(user_input=input.prompt, model_response=response, model_name = model)
#                db.add(chat_entry)
#                db.commit()
#                return {"response": response}
#           except requests.exceptions.JSONDecodeError:
#                raise HTTPException(status_code=500, detail="Invalid JSON response from model API")
#      except requests.ConnectionError:
#           raise HTTPException(status_code=500, detail="Unable to connect to the server")








from fastapi import FastAPI, APIRouter, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Annotated
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from .model import Base, ChatHistory
import requests

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader  # Remove DocumentLoader if it does not exist
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from llm.ollama import get_llama3, get_dolphin_phi, get_llama3_1
from llm.gemini import get_gemini

# Initialize FastAPI and router
app = FastAPI()
router = APIRouter(tags=["chat with llm"], prefix="/chat-with-llm")

# Database setup
DATABASE_URL = "mysql+pymysql://root:Mysql%40001@127.0.0.1:3306/mydemo_bot"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)

class ChatRequest(BaseModel):
    prompt: str

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]

def fetch_documents(db: Session):
    try:
        # Query the ChatHistory table to retrieve all records
        documents = db.query(ChatHistory).all()
        return documents
    except Exception as e:
        # Handle exceptions such as database connection issues
        raise HTTPException(status_code=500, detail="Error fetching documents from the database: " + str(e))


# Vector store configuration
# from typing import Optional

# class Document:
#     def __init__(self, page_content: str):
#         self.page_content = page_content
#      #    self.metadata = metadata or {}


class DBLoader():
    def __init__(self, db: Session):
        self.db = db

    def load(self):
        documents = fetch_documents(self.db)
        # Ensure Document has both page_content and metadata
        return [Document(page_content=doc.user_input + "\n" + doc.model_response,metadata={"source": "database"}) for doc in documents]

# API endpoints
@router.get("/history", status_code=status.HTTP_200_OK)
async def get_history(db: db_dependency):
    try:
        chat_history = db.query(ChatHistory).all()
        return chat_history
    except Exception as e:
        raise HTTPException(status_code=404, detail="Error retrieving chat history: " + str(e))

@router.post("/chat", status_code=status.HTTP_200_OK)
async def start_convo(db: db_dependency, input: ChatRequest, model: str):
    try:
        loader = DBLoader(db)
        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)  # This should now work with Document instances

        # Create the open-source embedding function
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        # Load it into Chroma
        vector_db = Chroma.from_documents(docs, embedding_function)

        # Retrieve top 5 relevant chat history entries
        query_embedding = embedding_function.embed_query(input.prompt)
        similar_chats = vector_db.similarity_search_with_score(query_embedding, k=5)
        history_prompt = "\n".join([f"User: {chat.user_input}\nModel: {chat.model_response}" for chat in similar_chats])
        history_prompt += f"\nUser: {input.prompt}"

        try:
            if model == "llama3":
                response = get_llama3(history_prompt)
            elif model == "llama3.1":
                response = get_llama3_1(history_prompt)
            elif model == "dolphin-phi":
                response = get_dolphin_phi(history_prompt)
            elif model == "gemini":
                response = get_gemini(user_question=input.prompt, chat_history=history_prompt)
            else:
                raise HTTPException(status_code=400, detail="Invalid model specified")

            chat_entry = ChatHistory(user_input=input.prompt, model_response=response, model_name=model)
            db.add(chat_entry)
            db.commit()
            return {"response": response}
        except requests.exceptions.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Invalid JSON response from model API")
    except requests.ConnectionError:
        raise HTTPException(status_code=500, detail="Unable to connect to the server")

# Include the router in the FastAPI app
app.include_router(router)

def embed_documents(self, texts):
    # Ensure texts is a list of strings
    if isinstance(texts, list) and all(isinstance(text, str) for text in texts):
        texts = list(map(lambda x: x.replace("\n", " "), texts))
    else:
        raise ValueError("Input texts must be a list of strings.")
    
    # Continue with embedding logic
    embeddings = self.model.encode(texts)
    return embeddings
