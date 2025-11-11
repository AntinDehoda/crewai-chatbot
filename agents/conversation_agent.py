import os
from dotenv import load_dotenv
from crewai import Agent
from langchain_openai import ChatOpenAI
from typing import Optional, List

# Завантаження змінних оточення
load_dotenv()


def create_conversation_agent(tools: Optional[List] = None):
    """
    Створює базовий Conversation Agent з використанням OpenAI GPT-4o-mini

    Args:
        tools: Список інструментів для агента (опціонально)
    """

    # Ініціалізація LLM
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Оновлений backstory з інформацією про RAG
    backstory = """Ти - дружній та компетентний асистент, який допомагає
    користувачам у різних питаннях. Ти завжди намагаєшся зрозуміти контекст
    розмови та надати релевантні та корисні відповіді.

    У тебе є доступ до інструменту пошуку в завантажених PDF документах.
    Коли користувач запитує про вміст документів або конкретну інформацію,
    яка може бути в PDF файлах, використовуй інструмент 'Search PDF Documents'
    для пошуку релевантної інформації.

    Якщо інформації немає в документах або документи не завантажені,
    відповідай на основі своїх загальних знань."""

    # Створення агента
    agent_params = {
        "role": "Асистент для розмови",
        "goal": "Підтримувати природну та корисну розмову з користувачем та допомагати знаходити інформацію в документах",
        "backstory": backstory,
        "verbose": True,
        "memory": True,
        "llm": llm,
        "allow_delegation": False
    }

    # Додаємо інструменти якщо надані
    if tools:
        agent_params["tools"] = tools

    conversation_agent = Agent(**agent_params)

    return conversation_agent