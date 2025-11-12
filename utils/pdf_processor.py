"""
PDF Processor - обробка PDF документів для RAG системи
"""
import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class PDFProcessor:
    """Клас для завантаження та обробки PDF документів"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Ініціалізація PDF процесора

        Args:
            chunk_size: Розмір chunk'ів для поділу тексту
            chunk_overlap: Перекриття між chunk'ами
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Завантажує PDF файл та розбиває на chunk'и

        Args:
            file_path: Шлях до PDF файлу

        Returns:
            List[Document]: Список документів (chunk'ів)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF файл не знайдено: {file_path}")

        # Завантаження PDF
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        # Розбиття на chunk'и
        chunks = self.text_splitter.split_documents(pages)

        # Додавання метаданих
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": i,
                "source_file": os.path.basename(file_path),
                "total_chunks": len(chunks)
            })

        return chunks

    def load_pdf_from_bytes(self, file_bytes: bytes, filename: str) -> List[Document]:
        """
        Завантажує PDF з байтів (для web upload)

        Args:
            file_bytes: Байти PDF файлу
            filename: Ім'я файлу

        Returns:
            List[Document]: Список документів (chunk'ів)
        """
        # Створюємо тимчасовий файл
        temp_path = f"/tmp/{filename}"
        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        try:
            chunks = self.load_pdf(temp_path)
            return chunks
        finally:
            # Видаляємо тимчасовий файл
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def get_document_summary(self, chunks: List[Document]) -> dict:
        """
        Отримує статистику про документ

        Args:
            chunks: Список chunk'ів документа

        Returns:
            dict: Статистика документа
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_characters": 0,
                "source_file": "unknown"
            }

        total_chars = sum(len(chunk.page_content) for chunk in chunks)

        return {
            "total_chunks": len(chunks),
            "total_characters": total_chars,
            "source_file": chunks[0].metadata.get("source_file", "unknown"),
            "avg_chunk_size": total_chars // len(chunks) if chunks else 0
        }
