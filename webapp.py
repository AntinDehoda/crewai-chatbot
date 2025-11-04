import os
import streamlit as st
from dotenv import load_dotenv
from crewai import Task, Crew, Process
from crewai.memory import ShortTermMemory
from agents.conversation_agent import create_conversation_agent

# Завантаження змінних оточення
load_dotenv()


def initialize_crew():#
    """Ініціалізація агента та crew один раз для сесії"""
    agent = create_conversation_agent()
    
    short_term_memory = ShortTermMemory(
        embedder_config={
            "provider": "openai",
            "model": "text-embedding-3-small",
        }
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[],
        verbose=False,  # Вимкнули для чистішого UI
        process=Process.sequential,
        memory=False,
        short_term_memory=short_term_memory
    )
    
    return crew, agent


def format_conversation_context(messages, last_n=5):
    """
    Форматує останні N повідомлень як контекст для агента
    """
    if not messages:
        return ""
    
    # Беремо останні N повідомлень
    recent_messages = messages[-last_n:] if len(messages) > last_n else messages
    
    context = "Історія розмови:\n"
    for msg in recent_messages:
        role = "Користувач" if msg["role"] == "user" else "Асистент"
        context += f"{role}: {msg['content']}\n"
    
    return context


# Ініціалізація session state
if "crew" not in st.session_state:
    st.session_state.crew, st.session_state.agent = initialize_crew()
    st.session_state.messages = []

# UI
st.title("🤖 AI Чат-бот")
st.caption("Чат-бот з короткостроковою пам'яттю")

# Кнопка очищення історії
if st.button("🗑️ Очистити історію"):
    st.session_state.messages = []
    st.rerun()

# Відображення історії чату
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Поле для вводу
if prompt := st.chat_input("Напиши повідомлення..."):
    # Додати повідомлення користувача в історію
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
    
    # Форматування контексту з історії
    context = format_conversation_context(st.session_state.messages)
    
    # Створити task з контекстом
    task = Task(
        description=f"""{context}

Нове повідомлення користувача: {prompt}

Відповісти на повідомлення користувача, враховуючи всю історію розмови вище. 
Використовуй інформацію з попередніх повідомлень для персоналізованої відповіді.""",
        agent=st.session_state.agent,
        expected_output="Природна та корисна відповідь на повідомлення користувача з урахуванням контексту розмови"
    )
    
    st.session_state.crew.tasks = [task]
    
    with st.chat_message("assistant"):
        with st.spinner("Думаю..."):
            response = st.session_state.crew.kickoff()
            st.write(response)
    
    # Додати відповідь бота в історію
    st.session_state.messages.append({"role": "assistant", "content": str(response)})
