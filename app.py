import os
import streamlit as st
from dotenv import load_dotenv
from crewai import Task, Crew, Process
from crewai.memory import ShortTermMemory
from agents.conversation_agent import create_conversation_agent

# Завантаження змінних оточення
load_dotenv()


def initialize_crew():
    """Ініціалізація агента та crew один раз для сесії"""
    agent = create_conversation_agent()
    
    short_term_memory = ShortTermMemory(
        embedder_config={
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small"
            }
        }
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[],
        verbose=False,
        process=Process.sequential,
        memory=False,
        short_term_memory=short_term_memory
    )
    
    return crew, agent


# Ініціалізація session state
if "crew" not in st.session_state:
    st.session_state.crew, st.session_state.agent = initialize_crew()
    st.session_state.messages = []

# UI
st.title("🤖 AI Чат-бот")
st.caption("Чат-бот з короткостроковою пам'яттю")

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
    
    # Створити task та отримати відповідь
    task = Task(
        description=f"Відповісти на повідомлення користувача: {prompt}",
        agent=st.session_state.agent,
        expected_output="Природна та корисна відповідь на повідомлення користувача"
    )
    
    st.session_state.crew.tasks = [task]
    
    with st.chat_message("assistant"):
        with st.spinner("Думаю..."):
            response = st.session_state.crew.kickoff()
            st.write(response)
    
    # Додати відповідь бота в історію
    st.session_state.messages.append({"role": "assistant", "content": str(response)})
