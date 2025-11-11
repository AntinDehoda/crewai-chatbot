"""
Base Vector Store - абстрактний базовий клас для векторних сховищ
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from langchain_core.documents import Document


class BaseVectorStore(ABC):
    """Абстрактний базовий клас для векторних сховищ"""

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Додає документи до векторної бази даних

        Args:
            documents: Список документів для додавання

        Returns:
            List[str]: Список ID доданих документів
        """
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        k: int = 4,
        filter_dict: Optional[dict] = None
    ) -> List[Document]:
        """
        Пошук релевантних документів

        Args:
            query: Запит для пошуку
            k: Кількість результатів
            filter_dict: Фільтр для метаданих (опціонально)

        Returns:
            List[Document]: Список релевантних документів
        """
        pass

    @abstractmethod
    def search_with_scores(
        self,
        query: str,
        k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        Пошук релевантних документів з оцінкою релевантності

        Args:
            query: Запит для пошуку
            k: Кількість результатів

        Returns:
            List[Tuple[Document, float]]: Список пар (документ, оцінка)
        """
        pass

    @abstractmethod
    def get_collection_count(self) -> int:
        """
        Отримує кількість документів у колекції

        Returns:
            int: Кількість документів
        """
        pass

    @abstractmethod
    def delete_collection(self):
        """Видаляє всю колекцію"""
        pass

    @abstractmethod
    def get_all_source_files(self) -> List[str]:
        """
        Отримує список всіх завантажених файлів

        Returns:
            List[str]: Список імен файлів
        """
        pass

    @abstractmethod
    def delete_by_source_file(self, source_file: str):
        """
        Видаляє документи за ім'ям файлу

        Args:
            source_file: Ім'я файлу для видалення
        """
        pass

    @property
    @abstractmethod
    def store_type(self) -> str:
        """
        Повертає тип векторного сховища

        Returns:
            str: Назва типу сховища (наприклад, "ChromaDB", "FAISS")
        """
        pass
