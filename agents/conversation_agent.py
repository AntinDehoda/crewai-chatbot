import os
from dotenv import load_dotenv
from crewai import Agent
from langchain_openai import ChatOpenAI

# Завантаження змінних оточення
load_dotenv()


def create_conversation_agent():
    """
    Створює базовий Conversation Agent з використанням OpenAI GPT-4o-mini
    """
    
    # Ініціалізація LLM
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Створення агента
    conversation_agent = Agent(
        role="Асистент для розмови",
        goal="Підтримувати природну та корисну розмову з користувачем",
        backstory="""Ти - дружній та компетентний асистент, який допомагає 
        користувачам у різних питаннях. Ти завжди намагаєшся зрозуміти контекст 
        розмови та надати релевантні та корисні відповіді.""",
        verbose=True,
        memory=True,
        llm=llm,
        allow_delegation=False
    )
   
    return conversation_agent