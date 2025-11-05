"""
Chat Crew - координує роботу агентів та завдань
"""
from crewai import Crew, Task, Process
from agents.conversation_agent import create_conversation_agent
from crewai.memory.short_term.short_term_memory import ShortTermMemory

class ChatCrew:
    """Клас для управління чат-ботом на базі CrewAI"""
    
    def __init__(self):
        """Ініціалізація crew з conversation agent"""
        self.conversation_agent = create_conversation_agent()
        self.crew = None
        self._initialize_crew()
    
    def _initialize_crew(self):
        """Створення crew з базовою конфігурацією"""
        short_term_memory = ShortTermMemory(
            embedder_config={
                "provider": "openai",
                "model": "text-embedding-3-small",
            }
        )
        self.crew = Crew(
            agents=[self.conversation_agent],
            tasks=[],  # Tasks будуть додаватися динамічно
            process=Process.sequential,
            verbose=True,
            memory=False,
            short_term_memory=short_term_memory
        )
    
    def chat(self, user_message: str) -> str:
        """
        Обробка повідомлення користувача
        
        Args:
            user_message: Повідомлення від користувача
            
        Returns:
            str: Відповідь від агента
        """
        # Створення задачі для агента
        task = Task(
            description=f"Відповісти на повідомлення користувача: {user_message}",
            expected_output="Природна та корисна відповідь українською мовою",
            agent=self.conversation_agent,
        )
        
        # Оновлення crew з новою задачею
        self.crew.tasks = [task]
        
        # Виконання задачі
        result = self.crew.kickoff()
        
        return result
    
    def reset_memory(self):
        """Очищення пам'яті crew та агента"""
        self._initialize_crew()
        print("✓ Пам'ять очищено")


if __name__ == "__main__":
    # Тест ChatCrew
    print("Ініціалізація ChatCrew...")
    chat_crew = ChatCrew()
    print("✓ ChatCrew готовий до роботи\n")
    
    # Тестова розмова
    test_messages = [
        "Привіт! Як справи?",
        "Розкажи мені про штучний інтелект",
        "Що ти запам'ятав з нашої розмови?"
    ]
    
    for msg in test_messages:
        print(f"\n👤 Користувач: {msg}")
        response = chat_crew.chat(msg)
        print(f"🤖 Бот: {response}")
