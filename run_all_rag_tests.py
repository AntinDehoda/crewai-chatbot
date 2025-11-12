"""
RAG Complete Test Runner - запускає всі 3 аналітичні інструменти
Генерує summary про роботу всіх інструментів з посиланнями на детальні результати
"""
import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import json


class RAGTestRunner:
    """Клас для запуску всіх RAG тестів"""

    def __init__(self):
        self.results_folder = Path("test_results")
        self.results_folder.mkdir(exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.summary = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "results_files": {},
            "overall_summary": {}
        }

    def print_header(self, title: str):
        """Виводить заголовок"""
        print("\n" + "="*80)
        print(f" {title:^78} ")
        print("="*80 + "\n")

    def run_script(self, script_name: str, description: str) -> tuple[bool, float, str]:
        """
        Запускає Python скрипт

        Args:
            script_name: Назва скрипта
            description: Опис скрипта

        Returns:
            Tuple (success, duration, error_message)
        """
        self.print_header(f"🚀 {description}")

        print(f"Запуск: python {script_name}")
        print(f"Початок: {datetime.now().strftime('%H:%M:%S')}\n")

        start_time = time.time()

        try:
            # Запускаємо скрипт
            result = subprocess.run(
                [sys.executable, script_name],
                capture_output=True,
                text=True,
                timeout=600  # 10 хвилин timeout
            )

            duration = time.time() - start_time

            # Виводимо stdout
            if result.stdout:
                print(result.stdout)

            # Перевіряємо чи успішно
            if result.returncode == 0:
                print(f"✓ Скрипт завершився успішно за {duration:.2f} секунд")
                return True, duration, ""
            else:
                print(f"✗ Скрипт завершився з помилкою (код {result.returncode})")
                if result.stderr:
                    print(f"Помилка: {result.stderr}")
                return False, duration, result.stderr

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            error_msg = f"Timeout after {duration:.0f} seconds"
            print(f"✗ {error_msg}")
            return False, duration, error_msg

        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            print(f"✗ Помилка виконання: {error_msg}")
            return False, duration, error_msg

    def read_comparison_results(self) -> Optional[Dict]:
        """Читає результати compare_vector_stores.py з TXT файлу"""
        # Знаходимо найновіший файл результатів
        txt_files = list(self.results_folder.glob("vector_store_comparison_*.txt"))

        if not txt_files:
            return None

        # Беремо найновіший файл
        latest_file = max(txt_files, key=lambda p: p.stat().st_mtime)

        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                content = f.read()

            summary = {
                "chromadb": {},
                "faiss": {},
                "winners": {},
                "file": latest_file.name
            }

            # Парсимо метрики для кожного vector store
            current_store = None
            for line in content.split('\n'):
                line = line.strip()

                # Визначаємо поточний store
                if line.startswith('CHROMADB:'):
                    current_store = 'chromadb'
                elif line.startswith('FAISS:'):
                    current_store = 'faiss'
                # Парсимо метрики
                elif current_store and line.startswith('•'):
                    # Формат: • Завантажено chunk'ів            : 500
                    parts = line.split(':')
                    if len(parts) == 2:
                        metric_name = parts[0].replace('•', '').strip()
                        metric_value = parts[1].strip()
                        try:
                            # Мапінг українських назв на ключі
                            metric_map = {
                                "Завантажено chunk'ів": "total_chunks",
                                "Час завантаження (с)": "load_time_sec",
                                "Швидкість завант. (chunk/s)": "chunks_per_second",
                                "Середній час пошуку (ms)": "avg_search_time_ms",
                                "Середня релевантність": "avg_relevance_score"
                            }
                            if metric_name in metric_map:
                                key = metric_map[metric_name]
                                summary[current_store][key] = float(metric_value)
                        except ValueError:
                            pass

            # Парсимо переможців
            lines = content.split('\n')
            in_winners_section = False
            current_category = None

            for i, line in enumerate(lines):
                line = line.strip()

                if '🏆 ПЕРЕМОЖЦІ' in line:
                    in_winners_section = True
                    continue

                if in_winners_section:
                    # Виходимо з секції переможців
                    if 'ВИСНОВКИ' in line:
                        break

                    # Категорія переможця
                    if line in ['Найшвидше завантаження:', 'Найшвидший пошук:', 'Найкраща релевантність:']:
                        current_category = line.rstrip(':')
                    # Winner: FAISS
                    elif current_category and line.startswith('Winner:'):
                        winner_store = line.split(':')[1].strip().lower()

                        # Мапінг категорій на ключі
                        category_map = {
                            'Найшвидше завантаження': 'fastest_loading',
                            'Найшвидший пошук': 'fastest_search',
                            'Найкраща релевантність': 'best_relevance'
                        }
                        if current_category in category_map:
                            summary['winners'][category_map[current_category]] = winner_store

            return summary

        except Exception as e:
            print(f"⚠️  Помилка читання {latest_file}: {e}")
            return None

    def read_evaluation_results(self) -> Optional[Dict]:
        """Читає результати evaluate_rag.py з TXT файлу"""
        # Знаходимо найновіший файл результатів
        txt_files = list(self.results_folder.glob("rag_evaluation_summary_*.txt"))

        if not txt_files:
            return None

        # Беремо найновіший файл
        latest_file = max(txt_files, key=lambda p: p.stat().st_mtime)

        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                content = f.read()

            summary = {
                "chromadb": {},
                "faiss": {},
                "winners": {},
                "file": latest_file.name
            }

            # Парсимо метрики для кожного vector store
            current_store = None
            for line in content.split('\n'):
                line = line.strip()

                # Визначаємо поточний store
                if line.startswith('CHROMADB:'):
                    current_store = 'chromadb'
                elif line.startswith('FAISS:'):
                    current_store = 'faiss'
                # Парсимо метрики
                elif current_store and line.startswith('•'):
                    # Формат: • faithfulness       : 0.8456
                    parts = line.split(':')
                    if len(parts) == 2:
                        metric_name = parts[0].replace('•', '').strip()
                        metric_value = parts[1].strip()
                        try:
                            summary[current_store][metric_name] = float(metric_value)
                        except ValueError:
                            pass

            # Парсимо переможців
            lines = content.split('\n')
            in_winners_section = False
            current_metric = None

            for i, line in enumerate(lines):
                line = line.strip()

                if '🏆 ПЕРЕМОЖЦІ ПО МЕТРИКАМ' in line:
                    in_winners_section = True
                    continue

                if in_winners_section:
                    # Виходимо з секції переможців
                    if line.startswith('-'*10) or line.startswith('='*10):
                        if 'ВИСНОВКИ' in line or i > 0 and 'ВИСНОВКИ' in lines[i-1]:
                            break

                    # Метрика (faithfulness:, answer_relevancy:, etc.)
                    if line and ':' in line and not line.startswith('Winner:') and not line.startswith('Перевага:'):
                        current_metric = line.rstrip(':').strip()
                    # Winner: CHROMADB (0.8456)
                    elif current_metric and line.startswith('Winner:'):
                        winner_part = line.split(':')[1].strip()
                        winner_store = winner_part.split('(')[0].strip().lower()
                        summary['winners'][current_metric] = winner_store

            return summary

        except Exception as e:
            print(f"⚠️  Помилка читання {latest_file}: {e}")
            return None

    def read_kubernetes_results(self) -> Optional[Dict]:
        """Читає результати test_rag_kubernetes.py"""
        # Знаходимо найновіший файл результатів
        json_files = list(self.results_folder.glob("kubernetes_rag_test_*.json"))

        if not json_files:
            return None

        # Беремо найновіший файл
        latest_file = max(json_files, key=lambda p: p.stat().st_mtime)

        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            summary_data = data.get('summary', {})

            summary = {
                "chromadb": summary_data.get('chromadb', {}),
                "faiss": summary_data.get('faiss', {}),
                "comparison": summary_data.get('comparison', {}),
                "questions_tested": len(data.get('questions', [])),
                "file": latest_file.name
            }

            return summary

        except Exception as e:
            print(f"⚠️  Помилка читання {latest_file}: {e}")
            return None

    def generate_summary_report(self):
        """Генерує підсумковий звіт"""
        self.print_header("📊 ПІДСУМКОВИЙ ЗВІТ")

        report_lines = []
        report_lines.append("="*80)
        report_lines.append(f"{'RAG TESTING SUMMARY REPORT':^80}")
        report_lines.append("="*80)
        report_lines.append(f"\nЧас виконання: {self.summary['timestamp']}")
        report_lines.append("\n" + "-"*80)

        # 1. Quick Performance Benchmark
        if 'compare_vector_stores' in self.summary['tests']:
            test_info = self.summary['tests']['compare_vector_stores']
            report_lines.append("\n1. QUICK PERFORMANCE BENCHMARK (compare_vector_stores.py)")
            report_lines.append("-"*80)

            if test_info['success']:
                report_lines.append(f"   Статус: ✓ Успішно завершено за {test_info['duration']:.2f}с")
                report_lines.append(f"   Результати: {self.summary['results_files'].get('comparison', 'N/A')}")

                results = self.summary.get('comparison_results')
                if results:
                    report_lines.append("\n   Основні метрики:")

                    chromadb = results.get('chromadb', {})
                    faiss = results.get('faiss', {})
                    winners = results.get('winners', {})

                    if chromadb and faiss:
                        report_lines.append(f"\n   ChromaDB:")
                        report_lines.append(f"      • Швидкість завантаження: {chromadb.get('chunks_per_second', 0):.1f} chunk/s")
                        report_lines.append(f"      • Час пошуку: {chromadb.get('avg_search_time_ms', 0):.2f} ms")
                        report_lines.append(f"      • Релевантність: {chromadb.get('avg_relevance_score', 0):.4f}")

                        report_lines.append(f"\n   FAISS:")
                        report_lines.append(f"      • Швидкість завантаження: {faiss.get('chunks_per_second', 0):.1f} chunk/s")
                        report_lines.append(f"      • Час пошуку: {faiss.get('avg_search_time_ms', 0):.2f} ms")
                        report_lines.append(f"      • Релевантність: {faiss.get('avg_relevance_score', 0):.4f}")

                        report_lines.append(f"\n   🏆 Переможці:")
                        report_lines.append(f"      • Найшвидше завантаження: {winners.get('fastest_loading', 'N/A').upper()}")
                        report_lines.append(f"      • Найшвидший пошук: {winners.get('fastest_search', 'N/A').upper()}")
                        report_lines.append(f"      • Найкраща релевантність: {winners.get('best_relevance', 'N/A').upper()}")
            else:
                report_lines.append(f"   Статус: ✗ Помилка")
                report_lines.append(f"   Помилка: {test_info.get('error', 'Unknown error')}")

        # 2. RAGAS Evaluation
        if 'evaluate_rag' in self.summary['tests']:
            test_info = self.summary['tests']['evaluate_rag']
            report_lines.append("\n\n2. RAGAS METRICS EVALUATION (evaluate_rag.py)")
            report_lines.append("-"*80)

            if test_info['success']:
                report_lines.append(f"   Статус: ✓ Успішно завершено за {test_info['duration']:.2f}с")
                report_lines.append(f"   Результати: {self.summary['results_files'].get('evaluation', 'N/A')}")

                results = self.summary.get('evaluation_results')
                if results:
                    report_lines.append("\n   RAGAS метрики:")

                    chromadb = results.get('chromadb', {})
                    faiss = results.get('faiss', {})
                    winners = results.get('winners', {})

                    if chromadb and faiss:
                        report_lines.append(f"\n   ChromaDB:")
                        for metric, value in chromadb.items():
                            if metric != 'winners':
                                report_lines.append(f"      • {metric}: {value:.4f}")

                        report_lines.append(f"\n   FAISS:")
                        for metric, value in faiss.items():
                            if metric != 'winners':
                                report_lines.append(f"      • {metric}: {value:.4f}")

                        if winners:
                            report_lines.append(f"\n   🏆 Переможці по метрикам:")
                            for metric, winner in winners.items():
                                report_lines.append(f"      • {metric}: {winner.upper()}")
            else:
                report_lines.append(f"   Статус: ✗ Помилка")
                report_lines.append(f"   Помилка: {test_info.get('error', 'Unknown error')}")

        # 3. Kubernetes Comprehensive Test
        if 'test_rag_kubernetes' in self.summary['tests']:
            test_info = self.summary['tests']['test_rag_kubernetes']
            report_lines.append("\n\n3. KUBERNETES COMPREHENSIVE TEST (test_rag_kubernetes.py)")
            report_lines.append("-"*80)

            if test_info['success']:
                report_lines.append(f"   Статус: ✓ Успішно завершено за {test_info['duration']:.2f}с")
                report_lines.append(f"   Результати: {self.summary['results_files'].get('kubernetes', 'N/A')}")

                results = self.summary.get('kubernetes_results')
                if results:
                    report_lines.append(f"\n   Тестів виконано: {results.get('questions_tested', 0)} питань")

                    chromadb = results.get('chromadb', {})
                    faiss = results.get('faiss', {})
                    comparison = results.get('comparison', {})

                    if chromadb and faiss:
                        report_lines.append(f"\n   ChromaDB:")
                        report_lines.append(f"      • Середній час відповіді: {chromadb.get('avg_time_ms', 0):.1f} ms")
                        report_lines.append(f"      • Середній similarity score: {chromadb.get('avg_score', 0):.4f}")
                        report_lines.append(f"      • Загальний час: {chromadb.get('total_time_s', 0):.2f} s")

                        report_lines.append(f"\n   FAISS:")
                        report_lines.append(f"      • Середній час відповіді: {faiss.get('avg_time_ms', 0):.1f} ms")
                        report_lines.append(f"      • Середній similarity score: {faiss.get('avg_score', 0):.4f}")
                        report_lines.append(f"      • Загальний час: {faiss.get('total_time_s', 0):.2f} s")

                        if comparison:
                            report_lines.append(f"\n   🏆 Порівняння:")
                            report_lines.append(f"      • Швидше: {comparison.get('faster_store', 'N/A').upper()} ({comparison.get('speedup_factor', 0):.2f}x)")
                            report_lines.append(f"      • Кращий score: {comparison.get('better_score', 'N/A').upper()}")
            else:
                report_lines.append(f"   Статус: ✗ Помилка")
                report_lines.append(f"   Помилка: {test_info.get('error', 'Unknown error')}")

        # Загальний підсумок
        report_lines.append("\n\n" + "="*80)
        report_lines.append("ЗАГАЛЬНИЙ ПІДСУМОК")
        report_lines.append("="*80)

        total_tests = len(self.summary['tests'])
        successful_tests = sum(1 for t in self.summary['tests'].values() if t['success'])

        report_lines.append(f"\nВсього тестів: {total_tests}")
        report_lines.append(f"Успішно: {successful_tests}")
        report_lines.append(f"Помилок: {total_tests - successful_tests}")

        # Загальна тривалість
        total_duration = sum(t['duration'] for t in self.summary['tests'].values())
        report_lines.append(f"\nЗагальний час виконання: {total_duration:.2f} секунд ({total_duration/60:.1f} хвилин)")

        # Рекомендації
        report_lines.append("\n" + "-"*80)
        report_lines.append("РЕКОМЕНДАЦІЇ:")
        report_lines.append("-"*80)

        # Аналізуємо результати та даємо рекомендації
        comp_results = self.summary.get('comparison_results', {})
        eval_results = self.summary.get('evaluation_results', {})
        k8s_results = self.summary.get('kubernetes_results', {})

        recommendations = []

        # На основі швидкості пошуку
        if comp_results and comp_results.get('winners', {}).get('fastest_search') == 'faiss':
            faiss_speed = comp_results.get('faiss', {}).get('avg_search_time_ms', 0)
            chromadb_speed = comp_results.get('chromadb', {}).get('avg_search_time_ms', 0)
            if chromadb_speed > 0:
                speedup = chromadb_speed / faiss_speed
                recommendations.append(
                    f"• FAISS швидше на {speedup:.1f}x - рекомендується для додатків де критична швидкість пошуку"
                )

        # На основі якості
        if comp_results and comp_results.get('winners', {}).get('best_relevance') == 'chromadb':
            recommendations.append(
                "• ChromaDB показує кращу релевантність - рекомендується коли важлива якість результатів"
            )

        # На основі RAGAS метрик
        if eval_results:
            chromadb_wins = sum(1 for w in eval_results.get('winners', {}).values() if w == 'chromadb')
            total_metrics = len(eval_results.get('winners', {}))
            if chromadb_wins > total_metrics / 2:
                recommendations.append(
                    f"• ChromaDB виграв у {chromadb_wins}/{total_metrics} RAGAS метрик - краща якість RAG відповідей"
                )

        if not recommendations:
            recommendations.append("• Обидва vector stores показують схожі результати")
            recommendations.append("• Вибір залежить від конкретних вимог вашого додатку")

        for rec in recommendations:
            report_lines.append(rec)

        report_lines.append("\n" + "="*80)
        report_lines.append(f"Звіт згенеровано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("="*80 + "\n")

        # Виводимо звіт
        report_text = "\n".join(report_lines)
        print(report_text)

        # Зберігаємо звіт
        report_path = self.results_folder / f"rag_tests_summary_{self.timestamp}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f"\n✓ Підсумковий звіт збережено: {report_path}")

        # Зберігаємо JSON з усіма даними
        json_path = self.results_folder / f"rag_tests_summary_{self.timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.summary, f, indent=2, ensure_ascii=False)

        print(f"✓ JSON дані збережено: {json_path}\n")

    def run_all_tests(self):
        """Запускає всі тести"""
        self.print_header("🧪 RAG COMPLETE TEST SUITE")

        print("Цей скрипт запустить всі 3 аналітичні інструменти:")
        print("  1. compare_vector_stores.py - швидке benchmark тестування")
        print("  2. evaluate_rag.py - RAGAS метрики якості")
        print("  3. test_rag_kubernetes.py - комплексне тестування на 40 питаннях")
        print("\nПримітка: Виконання може зайняти 10-20 хвилин залежно від кількості документів\n")

        total_start = time.time()

        # 1. Quick Performance Benchmark
        success, duration, error = self.run_script(
            "compare_vector_stores.py",
            "TEST 1/3: Quick Performance Benchmark"
        )
        self.summary['tests']['compare_vector_stores'] = {
            'success': success,
            'duration': duration,
            'error': error
        }
        if success:
            comp_results = self.read_comparison_results()
            if comp_results:
                self.summary['results_files']['comparison'] = f"test_results/{comp_results.get('file', 'vector_store_comparison.txt')}"
                self.summary['comparison_results'] = comp_results

        # 2. RAGAS Evaluation
        success, duration, error = self.run_script(
            "evaluate_rag.py",
            "TEST 2/3: RAGAS Metrics Evaluation"
        )
        self.summary['tests']['evaluate_rag'] = {
            'success': success,
            'duration': duration,
            'error': error
        }
        if success:
            eval_results = self.read_evaluation_results()
            if eval_results:
                self.summary['results_files']['evaluation'] = f"test_results/{eval_results.get('file', 'rag_evaluation_summary.txt')}"
                self.summary['evaluation_results'] = eval_results

        # 3. Kubernetes Comprehensive Test
        success, duration, error = self.run_script(
            "test_rag_kubernetes.py",
            "TEST 3/3: Kubernetes Comprehensive Test"
        )
        self.summary['tests']['test_rag_kubernetes'] = {
            'success': success,
            'duration': duration,
            'error': error
        }
        if success:
            k8s_results = self.read_kubernetes_results()
            if k8s_results:
                self.summary['results_files']['kubernetes'] = f"test_results/{k8s_results['file']}"
                self.summary['kubernetes_results'] = k8s_results

        # Загальна тривалість
        total_duration = time.time() - total_start
        self.summary['overall_summary']['total_duration'] = total_duration

        # Генеруємо підсумковий звіт
        self.generate_summary_report()

        self.print_header("✓ ВСІ ТЕСТИ ЗАВЕРШЕНО")
        print(f"Загальний час виконання: {total_duration:.2f} секунд ({total_duration/60:.1f} хвилин)\n")


def main():
    """Головна функція"""
    print("\n" + "="*80)
    print(f"{'🔬 RAG COMPLETE TEST SUITE':^80}")
    print("="*80)
    print(f"\nЗапуск: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nПеревірка наявності PDF документів...")

    # Перевіряємо чи є PDF файли
    pdf_folder = Path("data/pdf")
    if not pdf_folder.exists() or not list(pdf_folder.glob("*.pdf")):
        print("\n" + "="*80)
        print("⚠️  УВАГА: PDF ФАЙЛИ НЕ ЗНАЙДЕНО")
        print("="*80)
        print(f"\nСтворіть папку '{pdf_folder}' та додайте туди Kubernetes PDF документи")
        print("\nПриклад структури:")
        print("   data/pdf/")
        print("       ├── kubernetes-basics.pdf")
        print("       ├── kubernetes-networking.pdf")
        print("       └── kubernetes-storage.pdf")
        print("\nРекомендовані джерела:")
        print("   - Official Kubernetes documentation exports")
        print("   - Kubernetes in Action (book)")
        print("   - Kubernetes patterns documentation")
        print("\n" + "="*80 + "\n")
        return

    pdf_files = list(pdf_folder.glob("*.pdf"))
    print(f"✓ Знайдено {len(pdf_files)} PDF файл(ів):\n")
    for pdf in pdf_files:
        print(f"  • {pdf.name}")
    print()

    # Запускаємо тести
    runner = RAGTestRunner()
    runner.run_all_tests()


if __name__ == "__main__":
    main()
