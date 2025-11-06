"""
Utils package - утиліти для PDF RAG системи
"""
from .pdf_processor import PDFProcessor
from .vector_store import VectorStoreManager
from .faiss_vector_store import FAISSVectorStoreManager
from .base_vector_store import BaseVectorStore

def create_vector_store(store_type: str = "chromadb", **kwargs) -> BaseVectorStore:
    """
    Фабричний метод для створення векторного сховища

    Args:
        store_type: Тип сховища ("chromadb" або "faiss")
        **kwargs: Додаткові аргументи для конкретного сховища

    Returns:
        BaseVectorStore: Екземпляр векторного сховища

    Example:
        >>> vector_store = create_vector_store("chromadb")
        >>> vector_store = create_vector_store("faiss", index_name="my_index")
    """
    store_type = store_type.lower()

    if store_type == "chromadb" or store_type == "chroma":
        return VectorStoreManager(**kwargs)
    elif store_type == "faiss":
        return FAISSVectorStoreManager(**kwargs)
    else:
        raise ValueError(f"Невідомий тип vector store: {store_type}. Доступні: chromadb, faiss")

__all__ = [
    "PDFProcessor",
    "VectorStoreManager",
    "FAISSVectorStoreManager",
    "BaseVectorStore",
    "create_vector_store"
]
