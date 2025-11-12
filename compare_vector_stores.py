"""
Quick Comparison Tool - швидке порівняння ChromaDB vs FAISS
Використовує Kubernetes документацію з папки data/pdf/
"""
import os
import time
from datetime import datetime
from typing import List, Dict
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

from utils import create_vector_store
from utils.pdf_processor import PDFProcessor

load_dotenv()


class VectorStoreComparator:
    """Клас для порівняння векторних сховищ"""

    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.results = []

    def benchmark_vector_store(
        self,
        store_type: str,
        pdf_paths: List[str],
        test_queries: List[str],
        k: int = 4
    ) -> Dict:
        """
        Тестує один тип векторного сховища

        Args:
            store_type: Тип векторного сховища
            pdf_paths: Шляхи до PDF файлів
            test_queries: Тестові запити
            k: Кількість результатів

        Returns:
            Dict: Метрики продуктивності
        """
        print(f"\n{'='*60}")
        print(f"BENCHMARK: {store_type.upper()}")
        print(f"{'='*60}\n")

        vector_store = create_vector_store(store_type)

        # 1. Вимірюємо час завантаження документів
        print("📄 Завантаження документів...")
        load_start = time.time()

        total_chunks = 0
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                print(f"⚠️  Файл не знайдено: {pdf_path}")
                continue

            chunks = self.pdf_processor.load_pdf(pdf_path)
            vector_store.add_documents(chunks)
            total_chunks += len(chunks)
            print(f"   ✓ {os.path.basename(pdf_path)}: {len(chunks)} chunk'ів")

        load_time = time.time() - load_start
        print(f"\n   Час завантаження: {load_time:.3f}с")
        print(f"   Всього chunk'ів: {total_chunks}\n")

        # 2. Вимірюємо швидкість пошуку
        print("🔍 Тестування пошуку...")
        search_times = []
        retrieved_docs = []

        for i, query in enumerate(test_queries, 1):
            print(f"   [{i}/{len(test_queries)}] {query[:50]}...")

            search_start = time.time()
            results = vector_store.search(query, k=k)
            search_time = time.time() - search_start

            search_times.append(search_time)
            retrieved_docs.append(len(results))

            print(f"      Час: {search_time*1000:.2f}ms, Знайдено: {len(results)} док.")

        avg_search_time = sum(search_times) / len(search_times)
        print(f"\n   Середній час пошуку: {avg_search_time*1000:.2f}ms\n")

        # 3. Тест на точність (якісна оцінка)
        print("📊 Аналіз релевантності...")
        relevance_scores = []

        for query in test_queries[:3]:  # Беремо перші 3 для швидкості
            results_with_scores = vector_store.search_with_scores(query, k=k)

            if results_with_scores:
                avg_score = sum(score for _, score in results_with_scores) / len(results_with_scores)
                relevance_scores.append(avg_score)

        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        print(f"   Середня релевантність: {avg_relevance:.4f}\n")

        # 4. Пам'ять та розмір
        doc_count = vector_store.get_collection_count()
        print(f"   Документів у базі: {doc_count}")

        # Збираємо метрики
        metrics = {
            "vector_store": store_type,
            "total_chunks": total_chunks,
            "load_time_sec": load_time,
            "avg_search_time_ms": avg_search_time * 1000,
            "avg_relevance_score": avg_relevance,
            "documents_count": doc_count,
            "chunks_per_second": total_chunks / load_time if load_time > 0 else 0
        }

        return metrics

    def compare_stores(
        self,
        pdf_paths: List[str],
        test_queries: List[str],
        store_types: List[str] = None
    ) -> pd.DataFrame:
        """
        Порівнює векторні сховища

        Args:
            pdf_paths: Шляхи до PDF файлів
            test_queries: Тестові запити
            store_types: Типи векторних сховищ

        Returns:
            pd.DataFrame: Таблиця порівняння
        """
        if store_types is None:
            store_types = ["chromadb", "faiss"]

        results = []

        for store_type in store_types:
            metrics = self.benchmark_vector_store(store_type, pdf_paths, test_queries)
            results.append(metrics)

        return pd.DataFrame(results)


def print_comparison_table(df: pd.DataFrame):
    """Виводить красиву таблицю порівняння"""

    print("\n" + "="*80)
    print(" " * 25 + "ПОРІВНЯЛЬНА ТАБЛИЦЯ")
    print("="*80 + "\n")

    # Форматуємо таблицю
    comparison_data = []

    metrics = [
        ("Векторне сховище", "vector_store", "{}"),
        ("Завантажено chunk'ів", "total_chunks", "{:.0f}"),
        ("Час завантаження (с)", "load_time_sec", "{:.3f}"),
        ("Швидкість завант. (chunk/s)", "chunks_per_second", "{:.1f}"),
        ("Середній час пошуку (ms)", "avg_search_time_ms", "{:.2f}"),
        ("Середня релевантність", "avg_relevance_score", "{:.4f}"),
    ]

    for label, key, fmt in metrics:
        row = {"Метрика": label}
        for _, store_data in df.iterrows():
            store_name = store_data["vector_store"]
            value = store_data[key]
            row[store_name] = fmt.format(value)
        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))

    print("\n" + "="*80 + "\n")

    # Визначаємо переможців
    print("🏆 ПЕРЕМОЖЦІ:\n")

    # Швидкість завантаження
    fastest_load = df.loc[df['chunks_per_second'].idxmax()]
    print(f"   Найшвидше завантаження: {fastest_load['vector_store'].upper()}")
    print(f"      ({fastest_load['chunks_per_second']:.1f} chunk/s)\n")

    # Швидкість пошуку
    fastest_search = df.loc[df['avg_search_time_ms'].idxmin()]
    print(f"   Найшвидший пошук: {fastest_search['vector_store'].upper()}")
    print(f"      ({fastest_search['avg_search_time_ms']:.2f} ms)\n")

    # Релевантність
    best_relevance = df.loc[df['avg_relevance_score'].idxmax()]
    print(f"   Найкраща релевантність: {best_relevance['vector_store'].upper()}")
    print(f"      ({best_relevance['avg_relevance_score']:.4f})\n")

    print("="*80 + "\n")


def save_summary_to_txt(df: pd.DataFrame, output_folder: str = "test_results") -> str:
    """
    Зберігає аналітичне summary у TXT файл з timestamp

    Args:
        df: DataFrame з результатами порівняння
        output_folder: Папка для збереження результатів

    Returns:
        str: Шлях до створеного файлу
    """
    # Створюємо папку якщо не існує
    os.makedirs(output_folder, exist_ok=True)

    # Генеруємо назву файлу з timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"vector_store_comparison_{timestamp}.txt"
    filepath = os.path.join(output_folder, filename)

    # Формуємо текст звіту
    lines = []
    lines.append("="*80)
    lines.append(" "*15 + "QUICK PERFORMANCE BENCHMARK SUMMARY")
    lines.append("="*80)
    lines.append(f"\nЧас генерації: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Кількість vector stores: {len(df)}")
    lines.append("\n" + "-"*80)
    lines.append("ПОРІВНЯЛЬНА ТАБЛИЦЯ")
    lines.append("-"*80 + "\n")

    # Таблиця метрик
    metrics = [
        ("Векторне сховище", "vector_store", "{}"),
        ("Завантажено chunk'ів", "total_chunks", "{:.0f}"),
        ("Час завантаження (с)", "load_time_sec", "{:.3f}"),
        ("Швидкість завант. (chunk/s)", "chunks_per_second", "{:.1f}"),
        ("Середній час пошуку (ms)", "avg_search_time_ms", "{:.2f}"),
        ("Середня релевантність", "avg_relevance_score", "{:.4f}"),
    ]

    # Виводимо дані для кожного vector store
    for _, store_data in df.iterrows():
        store_name = store_data["vector_store"]
        lines.append(f"{store_name.upper()}:")
        for label, key, fmt in metrics[1:]:  # Пропускаємо перший (vector_store)
            value = store_data[key]
            lines.append(f"  • {label:30s}: {fmt.format(value)}")
        lines.append("")

    # Визначаємо переможців
    lines.append("-"*80)
    lines.append("🏆 ПЕРЕМОЖЦІ")
    lines.append("-"*80 + "\n")

    # Швидкість завантаження
    fastest_load = df.loc[df['chunks_per_second'].idxmax()]
    slowest_load = df.loc[df['chunks_per_second'].idxmin()]
    speedup_load = fastest_load['chunks_per_second'] / slowest_load['chunks_per_second']

    lines.append("Найшвидше завантаження:")
    lines.append(f"  Winner: {fastest_load['vector_store'].upper()}")
    lines.append(f"  Швидкість: {fastest_load['chunks_per_second']:.1f} chunk/s")
    lines.append(f"  Перевага: {speedup_load:.2f}x швидше\n")

    # Швидкість пошуку
    fastest_search = df.loc[df['avg_search_time_ms'].idxmin()]
    slowest_search = df.loc[df['avg_search_time_ms'].idxmax()]
    speedup_search = slowest_search['avg_search_time_ms'] / fastest_search['avg_search_time_ms']

    lines.append("Найшвидший пошук:")
    lines.append(f"  Winner: {fastest_search['vector_store'].upper()}")
    lines.append(f"  Час пошуку: {fastest_search['avg_search_time_ms']:.2f} ms")
    lines.append(f"  Перевага: {speedup_search:.2f}x швидше\n")

    # Релевантність
    best_relevance = df.loc[df['avg_relevance_score'].idxmax()]
    worst_relevance = df.loc[df['avg_relevance_score'].idxmin()]
    diff_relevance = best_relevance['avg_relevance_score'] - worst_relevance['avg_relevance_score']
    diff_percent = (diff_relevance / worst_relevance['avg_relevance_score'] * 100) if worst_relevance['avg_relevance_score'] > 0 else 0

    lines.append("Найкраща релевантність:")
    lines.append(f"  Winner: {best_relevance['vector_store'].upper()}")
    lines.append(f"  Score: {best_relevance['avg_relevance_score']:.4f}")
    lines.append(f"  Перевага: +{diff_relevance:.4f} (+{diff_percent:.1f}%)\n")

    # Висновки
    lines.append("-"*80)
    lines.append("ВИСНОВКИ")
    lines.append("-"*80 + "\n")

    # Рахуємо переможців
    wins = {}
    for _, store_data in df.iterrows():
        store = store_data['vector_store']
        wins[store] = 0

    wins[fastest_load['vector_store']] += 1
    wins[fastest_search['vector_store']] += 1
    wins[best_relevance['vector_store']] += 1

    for store in wins:
        lines.append(f"{store.upper()}:")
        lines.append(f"  Виграно категорій: {wins[store]}/3")

    lines.append("\n" + "="*80)
    lines.append("ОПИС МЕТРИК:")
    lines.append("="*80)
    lines.append("• Швидкість завантаження - Скільки chunk'ів на секунду може завантажити")
    lines.append("• Швидкість пошуку       - Середній час пошуку релевантних документів")
    lines.append("• Релевантність          - Якість знайдених документів (similarity score)")
    lines.append("="*80 + "\n")

    # Записуємо у файл
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"✓ Аналітичне summary збережено в {filepath}")

    return filepath


def main():
    """Головна функція"""

    # Kubernetes тестові запити
    test_queries = [
        "What is Kubernetes?",
        "What is a Pod in Kubernetes?",
        "What is the difference between a Pod and a Container?",
        "What is a Deployment in Kubernetes?",
        "What is a Service in Kubernetes?",
        "What are the types of Kubernetes Services?",
        "What is an Ingress?",
        "How do you perform a rolling update?",
        "What are the best practices for managing secrets in Kubernetes?",
        "How do you implement auto-scaling in Kubernetes?",
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
        return

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
        return

    print(f"\n📚 Знайдено {len(pdf_paths)} PDF файл(ів) в data/pdf/:")
    for pdf_path in pdf_paths:
        print(f"   • {pdf_path.name}")
    print()

    # Конвертуємо Path об'єкти в рядки
    pdf_paths = [str(p) for p in pdf_paths]

    # Створюємо компаратор
    comparator = VectorStoreComparator()

    # Порівнюємо векторні сховища
    results_df = comparator.compare_stores(pdf_paths, test_queries)

    # Виводимо результати
    print_comparison_table(results_df)

    # Зберігаємо аналітичне summary у TXT з timestamp
    save_summary_to_txt(results_df)


if __name__ == "__main__":
    main()
