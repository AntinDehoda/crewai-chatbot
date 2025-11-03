import os
from dotenv import load_dotenv
from crewai import Task, Crew
from agents.conversation_agent import create_conversation_agent

# Завантаження змінних оточення
load_dotenv()


def main():
    """
    Головна функція для запуску Conversation Agent
    """
    
    # Перевірка наявності API ключа
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Помилка: OPENAI_API_KEY не знайдено в .env файлі")
        print("Будь ласка, створи .env файл та додай свій API ключ")
        return
    
    print("🤖 Ініціалізація Conversation Agent...")
    
    # Створення агента
    conversation_agent = create_conversation_agent()
    
    print("✅ Агент готовий до роботи!\n")
    
    # Інтерактивний цикл розмови
    while True:
        # Отримання вводу від користувача
        user_input = input("Ти: ")
        
        # Перевірка на вихід
        if user_input.lower() in ['exit', 'quit', 'вихід', 'вийти']:
            print("👋 До побачення!")
            break
        
        if not user_input.strip():
            continue
        
        # Створення Task для агента
        task = Task(
            description=f"Відповісти на повідомлення користувача: {user_input}",
            agent=conversation_agent,
            expected_output="Природна та корисна відповідь на повідомлення користувача"
        )
        
        # Створення Crew та виконання
        crew = Crew(
            agents=[conversation_agent],
            tasks=[task],
            verbose=False
        )
        
        print("\n🤖 Агент: ", end="", flush=True)
        
        try:
            result = crew.kickoff()
            print(f"{result}\n")
        except Exception as e:
            print(f"❌ Помилка: {e}\n")


if __name__ == "__main__":
    main()