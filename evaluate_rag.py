"""
RAG Evaluation Script - оцінка якості RAG систем з використанням ragas
"""
import os
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from datasets import Dataset
import pandas as pd

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_relevancy
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
                context_recall,
                context_relevancy
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

    # Групуємо по vector_store та обчислюємо середні значення
    metric_columns = [col for col in results_df.columns
                     if col not in ['question', 'answer', 'contexts', 'ground_truth', 'vector_store']]

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


# Приклад використання
if __name__ == "__main__":
    # Тестові запитання
    test_questions = [
        {
            "question": "Що таке RAG?",
            "ground_truth": "RAG (Retrieval-Augmented Generation) - це техніка, яка поєднує пошук релевантних документів з генерацією відповідей."
        },
        {
            "question": "Які переваги використання векторних баз даних?",
            "ground_truth": "Векторні бази даних дозволяють ефективно шукати семантично подібні документи."
        },
        {
            "question": "Як працює семантичний пошук?",
            "ground_truth": "Семантичний пошук використовує embeddings для знаходження документів з подібним значенням."
        }
    ]

    # Шляхи до PDF файлів
    pdf_paths = [
        # Додайте шляхи до ваших PDF файлів
        # "path/to/document1.pdf",
        # "path/to/document2.pdf",
    ]

    if not pdf_paths:
        print("⚠️  Додайте шляхи до PDF файлів у змінну pdf_paths")
        print("   Приклад: pdf_paths = ['document.pdf']")
    else:
        # Порівнюємо векторні сховища
        results = compare_vector_stores(pdf_paths, test_questions)

        # Виводимо зведені результати
        print_comparison_summary(results)

        # Зберігаємо детальні результати
        results.to_csv("rag_evaluation_results.csv", index=False)
        print("✓ Детальні результати збережено в rag_evaluation_results.csv")
