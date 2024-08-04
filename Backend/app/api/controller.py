from fastapi import FastAPI, APIRouter, HTTPException, Depends, status, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Annotated
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from .model import Base, ChatHistory
import requests

from llm.ollama import get_llama3, get_dolphin_phi, get_llama3_1
from llm.gemini import get_gemini


# app = FastAPI()
router = APIRouter(tags=["chat with llm"], prefix="/chat-with-llm")


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


@router.get("/history", status_code=status.HTTP_200_OK)
async def get_history(db: db_dependency):
    try:
        chat_history = db.query(ChatHistory).all()
        return chat_history
    except Exception as e:
        raise HTTPException(
            status_code=404, detail="Error retrieving chat history: " + str(e)
        )


@router.post("/chat", status_code=status.HTTP_200_OK)
async def start_convo(db: db_dependency, input: ChatRequest, model: str):
     try:
          chat_history = db.query(ChatHistory).all()
          history_prompt = "\n".join([f"User: {chat.user_input}\nModel: {chat.model_response}" for chat in chat_history])
          history_prompt += f"\nUser: {input.prompt}"
          try:
               if model == "llama3":
                    response = get_llama3(history_prompt)
               elif model == "llama3.1":
                    response = get_llama3_1(history_prompt)
               elif model == "dolphin-phi":
                    response = get_dolphin_phi(history_prompt)
               elif model == "gemini":
                    response = get_gemini(user_question = input, chat_history = history_prompt)
               chat_entry = ChatHistory(user_input=input.prompt, model_response=response, model_name = model)
               db.add(chat_entry)
               db.commit()
               return {"response": response}
          except requests.exceptions.JSONDecodeError:
               raise HTTPException(status_code=500, detail="Invalid JSON response from model API")
     except requests.ConnectionError:
          raise HTTPException(status_code=500, detail="Unable to connect to the server")
