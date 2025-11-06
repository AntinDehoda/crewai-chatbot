"""
FAISS Vector Store Manager - управління векторною базою даних для RAG (FAISS)
"""
import os
import pickle
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from dotenv import load_dotenv
from .base_vector_store import BaseVectorStore

load_dotenv()


class FAISSVectorStoreManager(BaseVectorStore):
    """Клас для управління векторною базою даних з використанням FAISS"""

    def __init__(
        self,
        index_name: str = "pdf_documents",
        persist_directory: str = None
    ):
        """
        Ініціалізація FAISS Vector Store Manager

        Args:
            index_name: Назва індексу
            persist_directory: Директорія для збереження даних
        """
        self.index_name = index_name

        # Директорія за замовчуванням
        if persist_directory is None:
            persist_directory = os.path.join(
                os.path.expanduser("~"),
                ".local",
                "share",
                "crewai-chatbot",
                "rag_documents_faiss"
            )

        self.persist_directory = persist_directory
        self.index_path = os.path.join(self.persist_directory, f"{index_name}.faiss")
        self.metadata_path = os.path.join(self.persist_directory, f"{index_name}_metadata.pkl")

        # Створюємо директорію якщо не існує
        os.makedirs(self.persist_directory, exist_ok=True)

        # Ініціалізація embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        # Ініціалізація vector store
        self.vector_store = None
        self.documents_metadata = []  # Зберігаємо метадані окремо
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        """Ініціалізує або завантажує існуючий FAISS індекс"""
        try:
            if os.path.exists(self.index_path):
                # Завантажуємо існуючий індекс
                self.vector_store = FAISS.load_local(
                    self.persist_directory,
                    self.embeddings,
                    self.index_name,
                    allow_dangerous_deserialization=True
                )
                # Завантажуємо метадані
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path, 'rb') as f:
                        self.documents_metadata = pickle.load(f)
                print(f"✓ Завантажено FAISS індекс: {self.index_name}")
            else:
                # Створюємо новий порожній індекс
                # FAISS потребує хоча б один документ для ініціалізації
                self.vector_store = None
                self.documents_metadata = []
                print(f"✓ Створено новий FAISS індекс: {self.index_name}")
        except Exception as e:
            print(f"Помилка при ініціалізації FAISS: {e}")
            self.vector_store = None
            self.documents_metadata = []

    def _save_vector_store(self):
        """Зберігає FAISS індекс на диск"""
        if self.vector_store is not None:
            self.vector_store.save_local(self.persist_directory, self.index_name)
            # Зберігаємо метадані
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.documents_metadata, f)

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

        # Зберігаємо метадані документів
        start_id = len(self.documents_metadata)
        for i, doc in enumerate(documents):
            self.documents_metadata.append({
                "id": str(start_id + i),
                "source_file": doc.metadata.get("source_file", "unknown"),
                "page": doc.metadata.get("page", 0),
                "chunk_id": doc.metadata.get("chunk_id", 0)
            })

        # Додаємо документи
        if self.vector_store is None:
            # Створюємо новий індекс з першими документами
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            # Додаємо до існуючого індексу
            self.vector_store.add_documents(documents)

        # Зберігаємо на диск
        self._save_vector_store()

        # Повертаємо ID
        return [meta["id"] for meta in self.documents_metadata[start_id:]]

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
            filter_dict: Фільтр для метаданих (обмежена підтримка в FAISS)

        Returns:
            List[Document]: Список релевантних документів
        """
        if self.vector_store is None:
            return []

        results = self.vector_store.similarity_search(query, k=k)

        # FAISS не підтримує нативну фільтрацію, робимо постфільтрацію
        if filter_dict:
            filtered_results = []
            for doc in results:
                match = True
                for key, value in filter_dict.items():
                    if doc.metadata.get(key) != value:
                        match = False
                        break
                if match:
                    filtered_results.append(doc)
            return filtered_results[:k]

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
            List[Tuple[Document, float]]: Список пар (документ, оцінка)
        """
        if self.vector_store is None:
            return []

        results = self.vector_store.similarity_search_with_score(query, k=k)

        # FAISS повертає L2 distance (менше = краще)
        # Конвертуємо в similarity score (більше = краще)
        results_with_similarity = []
        for doc, distance in results:
            # Перетворюємо L2 distance в similarity (0 to 1)
            similarity = 1 / (1 + distance)
            results_with_similarity.append((doc, similarity))

        return results_with_similarity

    def get_collection_count(self) -> int:
        """
        Отримує кількість документів у колекції

        Returns:
            int: Кількість документів
        """
        if self.vector_store is None:
            return 0
        return len(self.documents_metadata)

    def delete_collection(self):
        """Видаляє всю колекцію"""
        try:
            # Видаляємо файли індексу
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
            if os.path.exists(self.metadata_path):
                os.remove(self.metadata_path)

            # Очищуємо в пам'яті
            self.vector_store = None
            self.documents_metadata = []

            print(f"✓ FAISS індекс '{self.index_name}' видалено")
        except Exception as e:
            print(f"✗ Помилка при видаленні індексу: {e}")

    def get_all_source_files(self) -> List[str]:
        """
        Отримує список всіх завантажених файлів

        Returns:
            List[str]: Список імен файлів
        """
        source_files = set()
        for metadata in self.documents_metadata:
            source_files.add(metadata["source_file"])
        return sorted(list(source_files))

    def delete_by_source_file(self, source_file: str):
        """
        Видаляє документи за ім'ям файлу

        Args:
            source_file: Ім'я файлу для видалення
        """
        # FAISS не підтримує видалення окремих документів ефективно
        # Потрібно пересоздати індекс без цих документів
        print("⚠️  FAISS не підтримує ефективне видалення окремих документів.")
        print("    Для видалення файлу використовуйте delete_collection() та завантажте документи заново.")

    @property
    def store_type(self) -> str:
        """
        Повертає тип векторного сховища

        Returns:
            str: "FAISS"
        """
        return "FAISS"
