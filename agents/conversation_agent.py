"""
Conversation Agent - базовий агент для текстових відповідей
"""
import os
from dotenv import load_dotenv
from crewai import Agent
from langchain_anthropic import ChatAnthropic

# Завантаження змінних оточення
load_dotenv()


def create_conversation_agent():
    """
    Створює Conversation Agent з розширеними можливостями
    
    Returns:
        Agent: Налаштований conversation agent
    """
    
    # Ініціалізація Claude LLM
    llm = ChatAnthropic(
        model=os.getenv("MODEL_NAME", "claude-sonnet-4-5-20250929"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=float(os.getenv("TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("MAX_TOKENS", "4096"))
    )
    
    # Створення агента
    agent = Agent(
        role="Розумний асистент для спілкування",
        goal="Надавати корисні, точні та дружні відповіді на запитання користувачів, "
             "підтримувати природну розмову та запам'ятовувати контекст",
        backstory="""Ти досвідчений AI асистент, який спеціалізується на природному 
        спілкуванні з людьми. Ти маєш глибокі знання в різних областях і вмієш 
        адаптувати свій стиль спілкування до потреб користувача. Ти завжди ввічливий, 
        уважний і прагнеш дати максимально корисну відповідь. Ти розмовляєш українською 
        мовою та добре розумієш культурний контекст.""",
        verbose=True,
        memory=True,
        allow_delegation=False,
        llm=llm,
        max_iter=15,  # Максимальна кількість ітерацій для виконання завдання
    )
    
    return agent


def create_enhanced_conversation_agent(custom_role=None, custom_goal=None, custom_backstory=None):
    """
    Створює кастомізований Conversation Agent
    
    Args:
        custom_role: Спеціальна роль для агента
        custom_goal: Спеціальна мета для агента
        custom_backstory: Спеціальна історія для агента
    
    Returns:
        Agent: Налаштований conversation agent
    """
    
    llm = ChatAnthropic(
        model=os.getenv("MODEL_NAME", "claude-sonnet-4-5-20250929"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=float(os.getenv("TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("MAX_TOKENS", "4096"))
    )
    
    agent = Agent(
        role=custom_role or "Розумний асистент для спілкування",
        goal=custom_goal or "Надавати корисні відповіді користувачам",
        backstory=custom_backstory or "Ти досвідчений AI асистент",
        verbose=True,
        memory=True,
        allow_delegation=False,
        llm=llm,
        max_iter=15,
    )
    
    return agent


if __name__ == "__main__":
    # Тест агента
    print("Створення Conversation Agent...")
    agent = create_conversation_agent()
    print(f"✓ Агент створено: {agent.role}")
    print(f"✓ Мета: {agent.goal}")
