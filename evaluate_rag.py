"""
RAG Evaluation Script - оцінка якості RAG систем з використанням ragas
Використовує Kubernetes документацію з папки data/pdf/
"""
import os
import time
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset
import pandas as pd

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

from utils import create_vector_store
from utils.pdf_processor import PDFProcessor
from langchain_openai import ChatOpenAI

load_dotenv()


class RAGEvaluator:
    """Клас для оцінки якості RAG систем"""

    def __init__(self, vector_store_type: str = "chromadb"):
        """
        Ініціалізація RAG Evaluator

        Args:
            vector_store_type: Тип векторного сховища ("chromadb" або "faiss")
        """
        self.vector_store_type = vector_store_type
        self.vector_store = create_vector_store(vector_store_type)
        self.pdf_processor = PDFProcessor()

        # LLM для генерації відповідей
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7
        )

    def load_documents(self, pdf_paths: List[str]):
        """
        Завантажує PDF документи до векторного сховища

        Args:
            pdf_paths: Список шляхів до PDF файлів
        """
        print(f"\n📄 Завантаження документів до {self.vector_store_type}...")

        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                print(f"⚠️  Файл не знайдено: {pdf_path}")
                continue

            print(f"   Обробка: {pdf_path}")
            chunks = self.pdf_processor.load_pdf(pdf_path)
            self.vector_store.add_documents(chunks)
            print(f"   ✓ Додано {len(chunks)} chunk'ів")

        total_docs = self.vector_store.get_collection_count()
        print(f"\n✓ Всього документів у {self.vector_store_type}: {total_docs}\n")

    def retrieve_context(self, query: str, k: int = 4) -> List[str]:
        """
        Витягує релевантні контексти для запиту

        Args:
            query: Запит користувача
            k: Кількість контекстів

        Returns:
            List[str]: Список контекстів
        """
        results = self.vector_store.search(query, k=k)
        contexts = [doc.page_content for doc in results]
        return contexts

    def generate_answer(self, query: str, contexts: List[str]) -> str:
        """
        Генерує відповідь на основі запиту та контекстів

        Args:
            query: Запит користувача
            contexts: Релевантні контексти

        Returns:
            str: Згенерована відповідь
        """
        context_text = "\n\n".join(contexts)

        prompt = f"""На основі наступного контексту відповідь на запитання користувача.

Контекст:
{context_text}

Запитання: {query}

Відповідь:"""

        response = self.llm.invoke(prompt)
        return response.content

    def evaluate_rag(
        self,
        test_questions: List[Dict[str, Any]],
        metrics: List = None
    ) -> pd.DataFrame:
        """
        Оцінює якість RAG системи з використанням ragas

        Args:
            test_questions: Список тестових запитань
                Формат: [
                    {
                        "question": "питання",
                        "ground_truth": "еталонна відповідь" (опціонально)
                    },
                    ...
                ]
            metrics: Список метрик ragas для оцінки

        Returns:
            pd.DataFrame: Результати оцінки
        """
        if metrics is None:
            metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ]

        print(f"\n🔍 Оцінка RAG системи ({self.vector_store_type})...")
        print(f"   Кількість тестових запитань: {len(test_questions)}")
        print(f"   Метрики: {[m.name for m in metrics]}\n")

        # Підготовка даних для оцінки
        questions = []
        answers = []
        contexts_list = []
        ground_truths = []

        for i, test_item in enumerate(test_questions, 1):
            question = test_item["question"]
            print(f"   [{i}/{len(test_questions)}] Обробка: {question[:50]}...")

            # Отримуємо контексти
            contexts = self.retrieve_context(question, k=4)

            # Генеруємо відповідь
            answer = self.generate_answer(question, contexts)

            # Зберігаємо для оцінки
            questions.append(question)
            answers.append(answer)
            contexts_list.append(contexts)

            # Ground truth (якщо є)
            if "ground_truth" in test_item:
                ground_truths.append(test_item["ground_truth"])
            else:
                ground_truths.append(answer)  # Використовуємо згенеровану відповідь

        # Створюємо dataset для ragas
        eval_dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths
        })

        # Виконуємо оцінку
        print("\n   Виконання оцінки ragas...")
        start_time = time.time()

        result = evaluate(
            eval_dataset,
            metrics=metrics,
            llm=self.llm,
            embeddings=self.vector_store.embeddings
        )

        elapsed_time = time.time() - start_time

        print(f"   ✓ Оцінка завершена за {elapsed_time:.2f} секунд\n")

        # Конвертуємо результати в DataFrame
        results_df = result.to_pandas()

        # Додаємо інформацію про vector store
        results_df['vector_store'] = self.vector_store_type

        return results_df


def compare_vector_stores(
    pdf_paths: List[str],
    test_questions: List[Dict[str, Any]],
    store_types: List[str] = None
) -> pd.DataFrame:
    """
    Порівнює різні векторні сховища

    Args:
        pdf_paths: Список шляхів до PDF файлів
        test_questions: Тестові запитання
        store_types: Типи векторних сховищ для порівняння

    Returns:
        pd.DataFrame: Зведені результати порівняння
    """
    if store_types is None:
        store_types = ["chromadb", "faiss"]

    all_results = []

    for store_type in store_types:
        print(f"\n{'='*60}")
        print(f"ОЦІНКА: {store_type.upper()}")
        print(f"{'='*60}")

        evaluator = RAGEvaluator(vector_store_type=store_type)

        # Завантажуємо документи
        evaluator.load_documents(pdf_paths)

        # Оцінюємо
        results = evaluator.evaluate_rag(test_questions)
        all_results.append(results)

    # Об'єднуємо результати
    combined_results = pd.concat(all_results, ignore_index=True)

    return combined_results


def print_comparison_summary(results_df: pd.DataFrame):
    """
    Виводить зведену таблицю порівняння

    Args:
        results_df: DataFrame з результатами оцінки
    """
    print("\n" + "="*60)
    print("ЗВЕДЕНІ РЕЗУЛЬТАТИ ПОРІВНЯННЯ")
    print("="*60 + "\n")

    # Вибираємо тільки числові колонки (метрики)
    numeric_cols = results_df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()

    # Виключаємо vector_store якщо вона числова (не повинна бути)
    metric_columns = [col for col in numeric_cols if col != 'vector_store']

    if not metric_columns:
        print("⚠️  Не знайдено числових метрик для порівняння")
        print(f"   Доступні колонки: {results_df.columns.tolist()}\n")
        return

    # Групуємо по vector_store та обчислюємо середні значення
    summary = results_df.groupby('vector_store')[metric_columns].mean()

    print(summary.to_string())
    print("\n" + "="*60 + "\n")

    # Визначаємо переможця по кожній метриці
    print("🏆 ПЕРЕМОЖЦІ ПО МЕТРИКАМ:")
    for metric in metric_columns:
        winner = summary[metric].idxmax()
        winner_score = summary.loc[winner, metric]
        print(f"   {metric}: {winner} ({winner_score:.4f})")

    print("\n" + "="*60 + "\n")


def save_summary_to_txt(results_df: pd.DataFrame, output_path: str):
    """
    Зберігає аналітичне summary у TXT файл (без повних питань та відповідей)

    Args:
        results_df: DataFrame з результатами оцінки
        output_path: Шлях до вихідного TXT файлу
    """
    # Вибираємо тільки числові колонки (метрики)
    numeric_cols = results_df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()
    metric_columns = [col for col in numeric_cols if col != 'vector_store']

    if not metric_columns:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("⚠️  Не знайдено числових метрик для порівняння\n")
        return

    # Групуємо по vector_store та обчислюємо середні значення
    summary = results_df.groupby('vector_store')[metric_columns].mean()

    # Формуємо текст звіту
    lines = []
    lines.append("="*80)
    lines.append(" "*20 + "RAG EVALUATION SUMMARY (RAGAS METRICS)")
    lines.append("="*80)
    lines.append(f"\nЧас генерації: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Кількість питань оцінено: {len(results_df) // len(summary)}")
    lines.append("\n" + "-"*80)
    lines.append("СЕРЕДНІ ЗНАЧЕННЯ МЕТРИК")
    lines.append("-"*80 + "\n")

    # Виводимо метрики для кожного vector store
    for store in summary.index:
        lines.append(f"{store.upper()}:")
        for metric in metric_columns:
            value = summary.loc[store, metric]
            lines.append(f"  • {metric:25s}: {value:.4f}")
        lines.append("")

    # Визначаємо переможців
    lines.append("-"*80)
    lines.append("🏆 ПЕРЕМОЖЦІ ПО МЕТРИКАМ")
    lines.append("-"*80 + "\n")

    for metric in metric_columns:
        winner = summary[metric].idxmax()
        winner_score = summary.loc[winner, metric]
        loser_score = summary[metric].min()
        diff = winner_score - loser_score
        diff_percent = (diff / loser_score * 100) if loser_score > 0 else 0

        lines.append(f"{metric}:")
        lines.append(f"  Winner: {winner.upper()} ({winner_score:.4f})")
        lines.append(f"  Перевага: +{diff:.4f} (+{diff_percent:.1f}%)")
        lines.append("")

    # Загальні висновки
    lines.append("-"*80)
    lines.append("ВИСНОВКИ")
    lines.append("-"*80 + "\n")

    # Підраховуємо скільки метрик виграв кожен store
    wins = {}
    for store in summary.index:
        wins[store] = sum(1 for metric in metric_columns if summary[metric].idxmax() == store)

    overall_winner = max(wins.items(), key=lambda x: x[1])
    lines.append(f"Загальний переможець: {overall_winner[0].upper()}")
    lines.append(f"  Виграно метрик: {overall_winner[1]}/{len(metric_columns)}")
    lines.append("")

    for store in summary.index:
        lines.append(f"{store}:")
        lines.append(f"  Виграно метрик: {wins[store]}/{len(metric_columns)}")

    lines.append("\n" + "="*80)
    lines.append("ОПИС МЕТРИК:")
    lines.append("="*80)
    lines.append("• faithfulness       - Наскільки відповідь базується на контексті (без галюцинацій)")
    lines.append("• answer_relevancy   - Наскільки відповідь релевантна до запитання")
    lines.append("• context_precision  - Точність витягнутого контексту")
    lines.append("• context_recall     - Повнота витягнутого контексту")
    lines.append("="*80 + "\n")

    # Записуємо у файл
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"✓ Аналітичне summary збережено в {output_path}")


# Приклад використання
if __name__ == "__main__":
    # Kubernetes тестові запитання з ground truth
    test_questions = [
        {
            "question": "What is Kubernetes?",
            "ground_truth": "Kubernetes is an open-source container orchestration platform for automating deployment, scaling, and management of containerized applications."
        },
        {
            "question": "What is a Pod in Kubernetes?",
            "ground_truth": "A Pod is the smallest deployable unit in Kubernetes, representing one or more containers that share network and storage resources."
        },
        {
            "question": "What is the difference between a Pod and a Container?",
            "ground_truth": "A Container is a single application instance, while a Pod can contain one or more tightly coupled containers that share resources and run together on the same node."
        },
        {
            "question": "What is a Deployment in Kubernetes?",
            "ground_truth": "A Deployment is a Kubernetes resource that manages ReplicaSets and provides declarative updates for Pods, enabling rolling updates and rollbacks."
        },
        {
            "question": "What is a Service in Kubernetes?",
            "ground_truth": "A Service is an abstraction that defines a logical set of Pods and a policy for accessing them, providing stable network endpoints for dynamic Pod sets."
        },
        {
            "question": "What are the types of Kubernetes Services?",
            "ground_truth": "The main types are ClusterIP (internal), NodePort (exposes on node port), LoadBalancer (external load balancer), and ExternalName (DNS alias)."
        },
        {
            "question": "What is an Ingress?",
            "ground_truth": "Ingress is a Kubernetes resource that manages external HTTP/HTTPS access to services, providing routing rules, SSL termination, and name-based virtual hosting."
        },
        {
            "question": "How do you perform a rolling update?",
            "ground_truth": "Rolling updates are performed by updating the Deployment specification, which gradually replaces old Pods with new ones while maintaining availability."
        },
        {
            "question": "What are the best practices for managing secrets in Kubernetes?",
            "ground_truth": "Best practices include using Secrets resources, encrypting data at rest, using RBAC for access control, rotating secrets regularly, and considering external secret management tools."
        },
        {
            "question": "How do you implement auto-scaling in Kubernetes?",
            "ground_truth": "Auto-scaling can be implemented using Horizontal Pod Autoscaler (HPA) for scaling Pods based on metrics, and Cluster Autoscaler for scaling nodes."
        },
    ]

    # Автоматично завантажуємо всі PDF з папки data/pdf/
    pdf_folder = Path("data/pdf")
    if not pdf_folder.exists():
        print("\n" + "="*60)
        print("⚠️  ПАПКА data/pdf/ НЕ ЗНАЙДЕНА")
        print("="*60)
        print("\nСтворіть папку data/pdf/ та додайте туди Kubernetes PDF документи")
        print("\nПриклад структури:")
        print("   data/pdf/")
        print("       ├── kubernetes-basics.pdf")
        print("       ├── kubernetes-networking.pdf")
        print("       └── kubernetes-storage.pdf\n")
        exit(1)

    pdf_paths = list(pdf_folder.glob("*.pdf"))

    if not pdf_paths:
        print("\n" + "="*60)
        print("⚠️  PDF ФАЙЛИ НЕ ЗНАЙДЕНО")
        print("="*60)
        print("\nДодайте Kubernetes PDF документи в папку data/pdf/")
        print("\nРекомендовані джерела:")
        print("   - Official Kubernetes documentation exports")
        print("   - Kubernetes in Action (book)")
        print("   - Kubernetes patterns documentation\n")
        exit(1)

    print(f"\n📚 Знайдено {len(pdf_paths)} PDF файл(ів) в data/pdf/:")
    for pdf_path in pdf_paths:
        print(f"   • {pdf_path.name}")
    print()

    # Конвертуємо Path об'єкти в рядки
    pdf_paths = [str(p) for p in pdf_paths]

    # Порівнюємо векторні сховища
    results = compare_vector_stores(pdf_paths, test_questions)

    # Виводимо зведені результати
    print_comparison_summary(results)

    # Зберігаємо аналітичне summary у TXT
    os.makedirs("test_results", exist_ok=True)
    save_summary_to_txt(results, "test_results/rag_evaluation_summary.txt")
