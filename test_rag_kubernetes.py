"""
RAG Kubernetes Test - порівняння ChromaDB vs FAISS на Kubernetes документації
"""
import os
import time
import json
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from utils import create_vector_store
from utils.pdf_processor import PDFProcessor
from utils.base_vector_store import BaseVectorStore

load_dotenv()


# ============================================================================
# CONFIGURATION
# ============================================================================

class TestConfig:
    """Конфігурація тестування"""

    # Параметри chunking
    CHUNK_SIZE = 1000  # Розмір chunk'ів для поділу документів
    CHUNK_OVERLAP = 200  # Перекриття між chunk'ами

    # Параметри пошуку
    TOP_K = 4  # Кількість документів для витягування
    ALPHA = 0.5  # MMR diversity (0.0 = тільки релевантність, 1.0 = тільки різноманітність)

    # Шляхи
    PDF_FOLDER = "data/pdf"  # Папка з PDF документами
    RESULTS_FOLDER = "test_results"  # Папка для результатів

    # LLM для генерації відповідей
    LLM_MODEL = "gpt-4o-mini"
    LLM_TEMPERATURE = 0.7


# ============================================================================
# KUBERNETES QUESTIONS
# ============================================================================

KUBERNETES_QUESTIONS = [
    "What is Kubernetes?",
    "What is a Pod in Kubernetes?",
    "What is the difference between a Pod and a Container?",
    "What is a Kubernetes cluster?",
    "What is a Node in Kubernetes?",
    "What is the role of the control plane?",
    "What is kubectl?",
    "What is a namespace in Kubernetes?",
    "What is the purpose of etcd in Kubernetes?",
    "What is a Kubernetes API server?",
    "What is a Deployment in Kubernetes?",
    "What is a ReplicaSet?",
    "How do you scale a Deployment?",
    "What is a StatefulSet?",
    "What is a DaemonSet?",
    "What is the difference between Deployment and StatefulSet?",
    "What is a Job in Kubernetes?",
    "What is a CronJob?",
    "How do you perform a rolling update?",
    "What is a rollback in Kubernetes?",
    "What is a Service in Kubernetes?",
    "What are the types of Kubernetes Services?",
    "What is a ClusterIP service?",
    "What is a NodePort service?",
    "What is a LoadBalancer service?",
    "What is an Ingress?",
    "What is the difference between Service and Ingress?",
    "How do Pods communicate with each other?",
    "What is a NetworkPolicy?",
    "What is DNS in Kubernetes?",
    "How do you implement zero-downtime deployments in Kubernetes?",
    "What are the best practices for managing secrets in Kubernetes production environments?",
    "How do you implement auto-scaling for applications in Kubernetes?",
    "What is the recommended approach for implementing health checks and readiness probes?",
    "How do you implement persistent storage for databases in Kubernetes?",
    "What are the strategies for implementing multi-tenancy in a Kubernetes cluster?",
    "How do you implement monitoring and logging in a Kubernetes production environment?",
    "What are the best practices for resource limits and requests configuration?",
    "How do you implement blue-green deployment strategy in Kubernetes?",
    "What are the security best practices for hardening a production Kubernetes cluster?",
]


# ============================================================================
# RAG TESTER CLASS
# ============================================================================

class RAGKubernetesTester:
    """Клас для тестування RAG систем на Kubernetes документації"""

    def __init__(self, config: TestConfig):
        self.config = config
        self.pdf_processor = PDFProcessor(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE
        )

        # Створюємо папки
        os.makedirs(config.PDF_FOLDER, exist_ok=True)
        os.makedirs(config.RESULTS_FOLDER, exist_ok=True)

        # Vector stores
        self.chromadb = None
        self.faiss = None

    def load_pdfs_to_stores(self, force_reload: bool = False):
        """
        Завантажує PDF документи до обох vector stores

        Args:
            force_reload: Примусово перезавантажити навіть якщо вже є документи
        """
        print("\n" + "="*80)
        print("📄 ЗАВАНТАЖЕННЯ PDF ДОКУМЕНТІВ")
        print("="*80 + "\n")

        # Ініціалізуємо vector stores
        print("Ініціалізація vector stores...")
        self.chromadb = create_vector_store("chromadb", collection_name="kubernetes_docs")
        self.faiss = create_vector_store("faiss", index_name="kubernetes_docs")
        print("✓ Vector stores ініціалізовано\n")

        # Перевіряємо чи вже є документи
        chromadb_count = self.chromadb.get_collection_count()
        faiss_count = self.faiss.get_collection_count()

        if not force_reload and chromadb_count > 0 and faiss_count > 0:
            print(f"✓ Документи вже завантажені:")
            print(f"  ChromaDB: {chromadb_count} документів")
            print(f"  FAISS: {faiss_count} документів")
            print("\n  Використовуємо існуючі дані. Для перезавантаження запустіть з force_reload=True\n")
            return

        # Знаходимо PDF файли
        pdf_folder = Path(self.config.PDF_FOLDER)
        pdf_files = list(pdf_folder.glob("*.pdf"))

        if not pdf_files:
            print(f"⚠️  УВАГА: Не знайдено PDF файлів у папці '{self.config.PDF_FOLDER}'")
            print(f"   Будь ласка, додайте Kubernetes PDF документи у цю папку\n")
            return

        print(f"Знайдено {len(pdf_files)} PDF файлів:\n")
        for pdf_file in pdf_files:
            print(f"  • {pdf_file.name}")
        print()

        # Завантажуємо та обробляємо PDF
        all_chunks = []
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"[{i}/{len(pdf_files)}] Обробка: {pdf_path.name}")
            try:
                chunks = self.pdf_processor.load_pdf(str(pdf_path))
                all_chunks.extend(chunks)
                print(f"   ✓ {len(chunks)} chunk'ів створено")
            except Exception as e:
                print(f"   ✗ Помилка: {e}")

        if not all_chunks:
            print("\n⚠️  Не вдалося обробити жоден документ\n")
            return

        print(f"\n📊 Всього chunk'ів створено: {len(all_chunks)}")
        print(f"   Chunk size: {self.config.CHUNK_SIZE}")
        print(f"   Chunk overlap: {self.config.CHUNK_OVERLAP}\n")

        # Завантажуємо до ChromaDB
        print("Завантаження до ChromaDB...")
        start_time = time.time()
        self.chromadb.add_documents(all_chunks)
        chromadb_time = time.time() - start_time
        print(f"✓ ChromaDB: {len(all_chunks)} документів за {chromadb_time:.2f}с\n")

        # Завантажуємо до FAISS
        print("Завантаження до FAISS...")
        start_time = time.time()
        self.faiss.add_documents(all_chunks)
        faiss_time = time.time() - start_time
        print(f"✓ FAISS: {len(all_chunks)} документів за {faiss_time:.2f}с\n")

        print("="*80)
        print(f"✓ ДОКУМЕНТИ УСПІШНО ЗАВАНТАЖЕНІ")
        print(f"  ChromaDB: {chromadb_time:.2f}с ({len(all_chunks)/chromadb_time:.1f} chunk/с)")
        print(f"  FAISS: {faiss_time:.2f}с ({len(all_chunks)/faiss_time:.1f} chunk/с)")
        print("="*80 + "\n")

    def retrieve_and_answer(
        self,
        vector_store: BaseVectorStore,
        question: str
    ) -> Dict[str, Any]:
        """
        Витягує контекст та генерує відповідь

        Args:
            vector_store: Vector store для пошуку
            question: Запитання

        Returns:
            Dict з результатами
        """
        # Пошук контекстів
        search_start = time.time()
        results_with_scores = vector_store.search_with_scores(question, k=self.config.TOP_K)
        search_time = time.time() - search_start

        # Витягуємо контексти та scores
        contexts = []
        scores = []
        for doc, score in results_with_scores:
            contexts.append(doc.page_content)
            scores.append(score)

        # Генеруємо відповідь
        context_text = "\n\n".join(contexts)
        prompt = f"""Based on the following context, answer the question.

Context:
{context_text}

Question: {question}

Answer:"""

        generation_start = time.time()
        response = self.llm.invoke(prompt)
        generation_time = time.time() - generation_start

        answer = response.content

        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "scores": scores,
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "search_time_ms": search_time * 1000,
            "generation_time_ms": generation_time * 1000,
            "total_time_ms": (search_time + generation_time) * 1000
        }

    def run_comparison_test(self, questions: List[str] = None) -> Dict[str, Any]:
        """
        Запускає порівняльний тест

        Args:
            questions: Список запитань (за замовчуванням KUBERNETES_QUESTIONS)

        Returns:
            Dict з результатами тестування
        """
        if questions is None:
            questions = KUBERNETES_QUESTIONS

        print("\n" + "="*80)
        print("🔬 ПОРІВНЯЛЬНЕ ТЕСТУВАННЯ RAG СИСТЕМ")
        print("="*80)
        print(f"\nПараметри:")
        print(f"  • Top-K: {self.config.TOP_K}")
        print(f"  • Alpha (MMR): {self.config.ALPHA}")
        print(f"  • Chunk size: {self.config.CHUNK_SIZE}")
        print(f"  • Питань: {len(questions)}")
        print(f"\nVector Stores:")
        print(f"  • ChromaDB: {self.chromadb.get_collection_count()} документів")
        print(f"  • FAISS: {self.faiss.get_collection_count()} документів")
        print("\n" + "="*80 + "\n")

        results = {
            "config": {
                "chunk_size": self.config.CHUNK_SIZE,
                "chunk_overlap": self.config.CHUNK_OVERLAP,
                "top_k": self.config.TOP_K,
                "alpha": self.config.ALPHA,
                "llm_model": self.config.LLM_MODEL,
                "timestamp": datetime.now().isoformat()
            },
            "questions": [],
            "summary": {
                "chromadb": {},
                "faiss": {},
                "comparison": {}
            }
        }

        chromadb_times = []
        faiss_times = []
        chromadb_scores = []
        faiss_scores = []

        # Тестуємо кожне питання
        for i, question in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] {question[:60]}...")

            # ChromaDB
            print("   ChromaDB...", end=" ", flush=True)
            chromadb_result = self.retrieve_and_answer(self.chromadb, question)
            print(f"{chromadb_result['total_time_ms']:.0f}ms (score: {chromadb_result['avg_score']:.4f})")

            # FAISS
            print("   FAISS...   ", end=" ", flush=True)
            faiss_result = self.retrieve_and_answer(self.faiss, question)
            print(f"{faiss_result['total_time_ms']:.0f}ms (score: {faiss_result['avg_score']:.4f})")

            # Зберігаємо результати
            results["questions"].append({
                "question": question,
                "chromadb": chromadb_result,
                "faiss": faiss_result
            })

            # Збираємо статистику
            chromadb_times.append(chromadb_result['total_time_ms'])
            faiss_times.append(faiss_result['total_time_ms'])
            chromadb_scores.append(chromadb_result['avg_score'])
            faiss_scores.append(faiss_result['avg_score'])

            print()

        # Обчислюємо статистику
        results["summary"]["chromadb"] = {
            "avg_time_ms": sum(chromadb_times) / len(chromadb_times),
            "avg_score": sum(chromadb_scores) / len(chromadb_scores),
            "total_time_s": sum(chromadb_times) / 1000
        }

        results["summary"]["faiss"] = {
            "avg_time_ms": sum(faiss_times) / len(faiss_times),
            "avg_score": sum(faiss_scores) / len(faiss_scores),
            "total_time_s": sum(faiss_times) / 1000
        }

        # Порівняння
        results["summary"]["comparison"] = {
            "speedup_factor": results["summary"]["chromadb"]["avg_time_ms"] / results["summary"]["faiss"]["avg_time_ms"],
            "score_difference": results["summary"]["chromadb"]["avg_score"] - results["summary"]["faiss"]["avg_score"],
            "faster_store": "faiss" if results["summary"]["faiss"]["avg_time_ms"] < results["summary"]["chromadb"]["avg_time_ms"] else "chromadb",
            "better_score": "chromadb" if results["summary"]["chromadb"]["avg_score"] > results["summary"]["faiss"]["avg_score"] else "faiss"
        }

        return results

    def print_summary(self, results: Dict[str, Any]):
        """Виводить зведення результатів"""
        print("\n" + "="*80)
        print("📊 ЗВЕДЕНІ РЕЗУЛЬТАТИ")
        print("="*80 + "\n")

        chromadb = results["summary"]["chromadb"]
        faiss = results["summary"]["faiss"]
        comp = results["summary"]["comparison"]

        print("CHROMADB:")
        print(f"  • Середній час відповіді: {chromadb['avg_time_ms']:.1f} ms")
        print(f"  • Середній similarity score: {chromadb['avg_score']:.4f}")
        print(f"  • Загальний час: {chromadb['total_time_s']:.2f} s")

        print("\nFAISS:")
        print(f"  • Середній час відповіді: {faiss['avg_time_ms']:.1f} ms")
        print(f"  • Середній similarity score: {faiss['avg_score']:.4f}")
        print(f"  • Загальний час: {faiss['total_time_s']:.2f} s")

        print("\nПОРІВНЯННЯ:")
        print(f"  • 🏆 Швидше: {comp['faster_store'].upper()} ({comp['speedup_factor']:.2f}x)")
        print(f"  • 🎯 Кращий score: {comp['better_score'].upper()}")
        print(f"  • 📊 Різниця у score: {abs(comp['score_difference']):.4f}")

        print("\n" + "="*80 + "\n")

    def save_results_to_json(self, results: Dict[str, Any]):
        """Зберігає детальні результати у JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"kubernetes_rag_test_{timestamp}.json"
        filepath = os.path.join(self.config.RESULTS_FOLDER, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"✓ Детальні результати збережено: {filepath}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Головна функція"""

    print("\n" + "="*80)
    print(" "*20 + "🧪 KUBERNETES RAG TESTING")
    print("="*80 + "\n")

    # Створюємо тестер
    config = TestConfig()
    tester = RAGKubernetesTester(config)

    # Завантажуємо PDF документи
    tester.load_pdfs_to_stores(force_reload=False)

    # Перевіряємо чи є документи
    if tester.chromadb.get_collection_count() == 0 or tester.faiss.get_collection_count() == 0:
        print("⚠️  Неможливо запустити тест без документів.")
        print(f"   Додайте Kubernetes PDF документи у папку '{config.PDF_FOLDER}' та запустіть знову.\n")
        return

    # Запускаємо порівняльний тест
    results = tester.run_comparison_test(KUBERNETES_QUESTIONS)

    # Виводимо зведення
    tester.print_summary(results)

    # Зберігаємо детальні результати
    tester.save_results_to_json(results)

    print("="*80)
    print(" "*25 + "✓ ТЕСТ ЗАВЕРШЕНО")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
