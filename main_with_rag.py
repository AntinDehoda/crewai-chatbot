"""
Консольний інтерфейс для чат-бота з RAG підтримкою
"""
import os
from dotenv import load_dotenv
from crewai import Task, Crew, Process
from crewai.memory.short_term.short_term_memory import ShortTermMemory
from agents.conversation_agent import create_conversation_agent
from tools.rag_tool import create_rag_tool
from utils.vector_store import VectorStoreManager
from utils.pdf_processor import PDFProcessor
from utils.greeting_generator import generate_greeting_with_documents

# Завантаження змінних оточення
load_dotenv()


def print_menu():
    """Виводить меню команд"""
    print("\n" + "="*60)
    print("КОМАНДИ:")
    print("  chat           - Розмова з ботом")
    print("  upload <path>  - Завантажити PDF файл")
    print("  list           - Показати завантажені документи")
    print("  delete <name>  - Видалити документ")
    print("  clear-docs     - Очистити всі документи")
    print("  clear-chat     - Очистити історію чату")
    print("  stats          - Показати статистику")
    print("  help           - Показати це меню")
    print("  exit           - Вихід")
    print("="*60 + "\n")


def upload_pdf(file_path: str, processor: PDFProcessor, vector_store: VectorStoreManager):
    """
    Завантажує PDF файл до векторної бази

    Args:
        file_path: Шлях до PDF файлу
        processor: PDF процесор
        vector_store: Менеджер векторного сховища
    """
    try:
        print(f"\n📄 Завантаження файлу: {file_path}")

        # Перевірка існування файлу
        if not os.path.exists(file_path):
            print(f"✗ Файл не знайдено: {file_path}")
            return

        # Обробка PDF
        print("⏳ Обробка PDF...")
        chunks = processor.load_pdf(file_path)

        # Додавання до векторної бази
        print("⏳ Додавання до бази даних...")
        ids = vector_store.add_documents(chunks)

        # Статистика
        summary = processor.get_document_summary(chunks)

        print(f"\n✓ Файл успішно завантажено!")
        print(f"  📊 Статистика:")
        print(f"     - Фрагментів: {summary['total_chunks']}")
        print(f"     - Символів: {summary['total_characters']}")
        print(f"     - Середній розмір фрагмента: {summary['avg_chunk_size']}")
        print(f"     - ID документів: {len(ids)}")

    except Exception as e:
        print(f"✗ Помилка при завантаженні: {e}")


def list_documents(vector_store: VectorStoreManager):
    """Виводить список завантажених документів"""
    source_files = vector_store.get_all_source_files()

    if not source_files:
        print("\n📚 Немає завантажених документів")
        return

    print(f"\n📚 Завантажені документи ({len(source_files)}):")
    for i, filename in enumerate(source_files, 1):
        print(f"  {i}. {filename}")


def show_stats(vector_store: VectorStoreManager):
    """Показує статистику векторної бази"""
    doc_count = vector_store.get_collection_count()
    source_files = vector_store.get_all_source_files()

    print("\n📊 Статистика:")
    print(f"  - Всього фрагментів: {doc_count}")
    print(f"  - Всього документів: {len(source_files)}")
    print(f"  - Розташування: {vector_store.persist_directory}")


def main():
    """Головна функція"""
    print("\n" + "="*60)
    print("🤖 AI ЧАТ-БОТ З RAG ПІДТРИМКОЮ")
    print("="*60)

    # Ініціалізація компонентів
    print("\n⏳ Ініціалізація...")

    vector_store = VectorStoreManager()
    rag_tool = create_rag_tool(vector_store)
    agent = create_conversation_agent(tools=[rag_tool])
    pdf_processor = PDFProcessor()

    short_term_memory = ShortTermMemory(
        embedder_config={
            "provider": "openai",
            "model": "text-embedding-3-small",
        }
    )

    crew = Crew(
        agents=[agent],
        tasks=[],
        process=Process.sequential,
        verbose=True,
        memory=False,
        short_term_memory=short_term_memory
    )

    print("✓ Готово!\n")

    # Показуємо статистику при старті
    show_stats(vector_store)
    print_menu()

    messages = []
    chat_mode = False

    # Основний цикл
    while True:
        try:
            if chat_mode:
                user_input = input("\n💬 Ти (або 'exit' для виходу з чату): ").strip()

                if user_input.lower() in ['exit', 'quit', 'back']:
                    chat_mode = False
                    print("\n↩️  Повернення до меню команд")
                    print_menu()
                    continue

                if not user_input:
                    continue

                # Додаємо повідомлення до історії
                messages.append({"role": "user", "content": user_input})

                # Формуємо контекст
                context = ""
                if len(messages) > 1:
                    recent = messages[-6:-1]  # Останні 5 повідомлень (без поточного)
                    context = "Історія розмови:\n"
                    for msg in recent:
                        role = "Користувач" if msg["role"] == "user" else "Асистент"
                        context += f"{role}: {msg['content']}\n"
                    context += "\n"

                # Створюємо задачу
                task = Task(
                    description=f"""{context}Нове повідомлення користувача: {user_input}

Відповісти на повідомлення, враховуючи історію розмови.
Якщо питання стосується документів, використовуй інструмент 'Search PDF Documents'.""",
                    expected_output="Природна та корисна відповідь українською мовою",
                    agent=agent,
                )

                crew.tasks = [task]

                print("\n🤖 Бот: ", end="", flush=True)
                response = crew.kickoff()
                print(response)

                # Додаємо відповідь до історії
                messages.append({"role": "assistant", "content": str(response)})

            else:
                # Режим команд
                command = input("\n📝 Команда: ").strip()

                if not command:
                    continue

                parts = command.split(maxsplit=1)
                cmd = parts[0].lower()

                if cmd in ['exit', 'quit', 'вихід', 'вийти']:
                    print("\n👋 До побачення!")
                    break

                elif cmd == 'chat':
                    chat_mode = True
                    print("\n💬 Режим чату активовано (введіть 'exit' для виходу)\n")
                    # Показуємо привітання з інформацією про документи
                    greeting = generate_greeting_with_documents(vector_store, use_llm=False)
                    print(greeting)

                elif cmd == 'upload':
                    if len(parts) < 2:
                        print("✗ Використання: upload <шлях_до_pdf>")
                    else:
                        upload_pdf(parts[1], pdf_processor, vector_store)

                elif cmd == 'list':
                    list_documents(vector_store)

                elif cmd == 'delete':
                    if len(parts) < 2:
                        print("✗ Використання: delete <назва_файлу>")
                    else:
                        vector_store.delete_by_source_file(parts[1])

                elif cmd == 'clear-docs':
                    confirm = input("⚠️  Видалити всі документи? (yes/no): ")
                    if confirm.lower() in ['yes', 'y', 'так']:
                        vector_store.delete_collection()
                    else:
                        print("Операцію скасовано")

                elif cmd == 'clear-chat':
                    messages = []
                    print("✓ Історію чату очищено")

                elif cmd == 'stats':
                    show_stats(vector_store)

                elif cmd == 'help':
                    print_menu()

                else:
                    print(f"✗ Невідома команда: {cmd}")
                    print("Введіть 'help' для списку команд")

        except KeyboardInterrupt:
            print("\n\n👋 Вихід через Ctrl+C")
            break
        except Exception as e:
            print(f"\n✗ Помилка: {e}")


if __name__ == "__main__":
    main()
