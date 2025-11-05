import os
import streamlit as st
from dotenv import load_dotenv
from crewai import Task, Crew, Process
from crewai.memory import ShortTermMemory
from agents.conversation_agent import create_conversation_agent
from tools.rag_tool import create_rag_tool
from utils.pdf_processor import PDFProcessor
from utils.vector_store import VectorStoreManager

# Завантаження змінних оточення
load_dotenv()


def initialize_crew():
    """Ініціалізація агента та crew один раз для сесії"""
    # Ініціалізація RAG компонентів
    vector_store_manager = VectorStoreManager()
    rag_tool = create_rag_tool(vector_store_manager)

    # Створення агента з RAG tool
    agent = create_conversation_agent(tools=[rag_tool])

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

    return crew, agent, vector_store_manager


def process_uploaded_pdf(uploaded_file, vector_store_manager):
    """
    Обробка завантаженого PDF файлу

    Args:
        uploaded_file: Завантажений файл від Streamlit
        vector_store_manager: Менеджер векторного сховища

    Returns:
        dict: Інформація про обробку
    """
    try:
        # Читаємо байти файлу
        pdf_bytes = uploaded_file.read()

        # Обробка PDF
        processor = PDFProcessor()
        chunks = processor.load_pdf_from_bytes(pdf_bytes, uploaded_file.name)

        # Додаємо до векторної бази
        ids = vector_store_manager.add_documents(chunks)

        # Отримуємо статистику
        summary = processor.get_document_summary(chunks)

        return {
            "success": True,
            "filename": uploaded_file.name,
            "chunks_count": len(chunks),
            "document_ids": ids,
            "summary": summary
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "filename": uploaded_file.name
        }
    finally:
        # Перемотуємо файл на початок на всякий випадок
        uploaded_file.seek(0)


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
    st.session_state.crew, st.session_state.agent, st.session_state.vector_store = initialize_crew()
    st.session_state.messages = []
    st.session_state.uploaded_files = []

# UI
st.title("🤖 AI Чат-бот з RAG")
st.caption("Чат-бот з пам'яттю та можливістю роботи з PDF документами")

# Sidebar для управління документами
with st.sidebar:
    st.header("📚 Управління документами")

    # Завантаження PDF
    st.subheader("Завантажити PDF")
    uploaded_file = st.file_uploader(
        "Виберіть PDF файл",
        type=["pdf"],
        help="Завантажте PDF документ для аналізу"
    )

    if uploaded_file is not None:
        if st.button("📤 Завантажити та обробити"):
            with st.spinner(f"Обробка {uploaded_file.name}..."):
                result = process_uploaded_pdf(uploaded_file, st.session_state.vector_store)

                if result["success"]:
                    st.success(f"✓ Файл '{result['filename']}' успішно завантажено!")
                    st.info(f"Створено {result['chunks_count']} фрагментів тексту")

                    # Додаємо до списку завантажених файлів
                    if result["filename"] not in st.session_state.uploaded_files:
                        st.session_state.uploaded_files.append(result["filename"])
                else:
                    st.error(f"✗ Помилка: {result['error']}")

    # Відображення завантажених документів
    st.subheader("Завантажені документи")
    source_files = st.session_state.vector_store.get_all_source_files()

    if source_files:
        st.write(f"📄 Всього документів: {len(source_files)}")
        for filename in source_files:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(f"• {filename}")
            with col2:
                if st.button("🗑️", key=f"del_{filename}", help=f"Видалити {filename}"):
                    st.session_state.vector_store.delete_by_source_file(filename)
                    if filename in st.session_state.uploaded_files:
                        st.session_state.uploaded_files.remove(filename)
                    st.rerun()
    else:
        st.info("Немає завантажених документів")

    # Статистика
    doc_count = st.session_state.vector_store.get_collection_count()
    st.metric("Всього фрагментів", doc_count)

    st.divider()

    # Кнопки управління
    if st.button("🗑️ Очистити всі документи"):
        st.session_state.vector_store.delete_collection()
        st.session_state.uploaded_files = []
        st.rerun()

# Основна область чату
col1, col2 = st.columns([6, 1])
with col1:
    st.subheader("💬 Чат")
with col2:
    if st.button("🔄 Очистити чат"):
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
