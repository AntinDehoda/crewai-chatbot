"""
RAG Tool - інструмент для пошуку інформації в завантажених PDF документах
"""
from crewai_tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from utils.vector_store import VectorStoreManager


class RAGSearchInput(BaseModel):
    """Input schema for RAG search tool."""
    query: str = Field(..., description="Запит для пошуку в документах")


class RAGSearchTool(BaseTool):
    name: str = "Search PDF Documents"
    description: str = (
        "Шукає інформацію в завантажених PDF документах. "
        "Використовуй цей інструмент, коли користувач запитує про вміст документів, "
        "або коли потрібна інформація з PDF файлів. "
        "Приклади запитів: 'що сказано про...', 'знайди інформацію про...', "
        "'які деталі про...', 'що написано в документі про...'"
    )
    args_schema: Type[BaseModel] = RAGSearchInput

    def __init__(self, vector_store_manager: VectorStoreManager = None):
        """
        Ініціалізація RAG tool

        Args:
            vector_store_manager: Менеджер векторного сховища
        """
        super().__init__()
        self.vector_store_manager = vector_store_manager or VectorStoreManager()

    def _run(self, query: str) -> str:
        """
        Виконує пошук в документах

        Args:
            query: Запит для пошуку

        Returns:
            str: Результати пошуку
        """
        # Перевіряємо, чи є документи в базі
        doc_count = self.vector_store_manager.get_collection_count()

        if doc_count == 0:
            return (
                "В базі даних немає завантажених документів. "
                "Будь ласка, завантажте PDF документи перед пошуком."
            )

        # Виконуємо пошук
        results = self.vector_store_manager.search_with_scores(query, k=4)

        if not results:
            return (
                f"Не знайдено релевантної інформації за запитом: '{query}'. "
                "Спробуйте переформулювати запит або перевірте чи завантажені потрібні документи."
            )

        # Форматуємо результати
        formatted_results = []
        formatted_results.append(f"Знайдено {len(results)} релевантних фрагментів:\n")

        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("source_file", "невідоме джерело")
            page = doc.metadata.get("page", "невідома сторінка")
            content = doc.page_content.strip()

            formatted_results.append(f"\n--- Фрагмент {i} (джерело: {source}, сторінка: {page}) ---")
            formatted_results.append(content)
            formatted_results.append(f"(релевантність: {score:.4f})")

        return "\n".join(formatted_results)


def create_rag_tool(vector_store_manager: VectorStoreManager = None) -> RAGSearchTool:
    """
    Створює RAG tool для агента

    Args:
        vector_store_manager: Менеджер векторного сховища

    Returns:
        RAGSearchTool: Інструмент для RAG пошуку
    """
    return RAGSearchTool(vector_store_manager=vector_store_manager)
