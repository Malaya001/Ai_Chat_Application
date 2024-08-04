from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key = True, index = True)
    user_input = Column(Text, nullable = False)
    model_name = Column(Text, nullable = False)
    model_response = Column(Text, nullable = False)