"""
Vector Store Manager - управління векторною базою даних для RAG (ChromaDB)
"""
import os
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from .base_vector_store import BaseVectorStore

load_dotenv()


class VectorStoreManager(BaseVectorStore):
    """Клас для управління векторною базою даних"""

    def __init__(
        self,
        collection_name: str = "pdf_documents",
        persist_directory: str = None
    ):
        """
        Ініціалізація Vector Store Manager

        Args:
            collection_name: Назва колекції в ChromaDB
            persist_directory: Директорія для збереження даних
        """
        self.collection_name = collection_name

        # Директорія за замовчуванням
        if persist_directory is None:
            persist_directory = os.path.join(
                os.path.expanduser("~"),
                ".local",
                "share",
                "crewai-chatbot",
                "rag_documents"
            )

        self.persist_directory = persist_directory

        # Створюємо директорію якщо не існує
        os.makedirs(self.persist_directory, exist_ok=True)

        # Ініціалізація embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        # Ініціалізація vector store
        self.vector_store = None
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        """Ініціалізує або завантажує існуючу векторну базу даних"""
        try:
            # Спробуємо завантажити існуючу базу
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
        except Exception as e:
            print(f"Створюємо нову векторну базу даних: {e}")
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Додає документи до векторної бази даних

        Args:
            documents: Список документів для додавання

        Returns:
            List[str]: Список ID доданих документів
        """
        if not documents:
            return []

        # Додаємо документи
        ids = self.vector_store.add_documents(documents)

        return ids

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
            filter_dict: Фільтр для метаданих

        Returns:
            List[Document]: Список релевантних документів
        """
        if filter_dict:
            results = self.vector_store.similarity_search(
                query,
                k=k,
                filter=filter_dict
            )
        else:
            results = self.vector_store.similarity_search(query, k=k)

        return results

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
            List[tuple[Document, float]]: Список пар (документ, оцінка)
        """
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results

    def get_collection_count(self) -> int:
        """
        Отримує кількість документів у колекції

        Returns:
            int: Кількість документів
        """
        try:
            collection = self.vector_store._collection
            return collection.count()
        except Exception:
            return 0

    def delete_collection(self):
        """Видаляє всю колекцію"""
        try:
            self.vector_store.delete_collection()
            self._initialize_vector_store()
            print(f"✓ Колекція '{self.collection_name}' видалена")
        except Exception as e:
            print(f"✗ Помилка при видаленні колекції: {e}")

    def get_all_source_files(self) -> List[str]:
        """
        Отримує список всіх завантажених файлів

        Returns:
            List[str]: Список імен файлів
        """
        try:
            # Отримуємо всі документи з колекції
            collection = self.vector_store._collection
            results = collection.get()

            # Витягуємо унікальні імена файлів з метаданих
            source_files = set()
            if results and "metadatas" in results:
                for metadata in results["metadatas"]:
                    if metadata and "source_file" in metadata:
                        source_files.add(metadata["source_file"])

            return sorted(list(source_files))
        except Exception as e:
            print(f"Помилка при отриманні списку файлів: {e}")
            return []

    def delete_by_source_file(self, source_file: str):
        """
        Видаляє документи за ім'ям файлу

        Args:
            source_file: Ім'я файлу для видалення
        """
        try:
            collection = self.vector_store._collection
            results = collection.get(
                where={"source_file": source_file}
            )

            if results and "ids" in results:
                collection.delete(ids=results["ids"])
                print(f"✓ Видалено документи з файлу: {source_file}")
            else:
                print(f"Документи з файлу '{source_file}' не знайдено")
        except Exception as e:
            print(f"✗ Помилка при видаленні документів: {e}")

    @property
    def store_type(self) -> str:
        """
        Повертає тип векторного сховища

        Returns:
            str: "ChromaDB"
        """
        return "ChromaDB"
