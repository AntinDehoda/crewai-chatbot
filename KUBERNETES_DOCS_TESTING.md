# 🧪 Kubernetes docs RAG Testing Framework

## Overview

This testing framework allows you to compare **ChromaDB vs FAISS** vector stores on Kubernetes documentation with 40 standardized questions.

## Features

✅ **Automated PDF Loading** - Load all PDFs from `data/pdf/` folder
✅ **Dual Vector Store Testing** - Tests both ChromaDB and FAISS simultaneously
✅ **Configurable Parameters** - Adjust chunk_size, top_k, and alpha (MMR diversity)
✅ **40 Kubernetes Questions** - From basic to advanced production scenarios
✅ **Skip Already-Loaded Docs** - Efficient testing without reloading
✅ **Console Summary** - Response times, similarity scores, comparison
✅ **JSON Export** - Detailed results with answers and contexts

---

## Quick Start

### 1. Prepare Kubernetes PDFs

Add your Kubernetes PDF documentation to the `data/pdf/` folder:

```bash
data/pdf/
├── kubernetes-basics.pdf
├── kubernetes-networking.pdf
├── kubernetes-storage.pdf
└── ...
```

**Recommended PDFs:**
- Official Kubernetes documentation exports
- Kubernetes in Action (book)
- Kubernetes patterns documentation
- Production best practices guides

### 2. Run the Test

**Basic usage (uses existing data if available):**
```bash
python test_rag_kubernetes.py
```

**Command-line options:**
```bash
# Clear both databases before testing
python test_rag_kubernetes.py --clear-stores

# Force reload PDFs even if data exists
python test_rag_kubernetes.py --force-reload

# Clear databases AND force reload
python test_rag_kubernetes.py --clear-stores --force-reload

# Show help
python test_rag_kubernetes.py --help
```

**When to use what:**
- No flags: Fastest - uses existing data if available
- `--clear-stores`: Fresh start - clears ChromaDB and FAISS before loading
- `--force-reload`: Reload PDFs even if documents exist in stores
- Both flags: Complete clean slate - clear everything and reload from PDFs

### 3. View Results

**Console Output:**
- Progress for each question
- Response times and similarity scores
- Summary comparison

**JSON File:**
- Saved in `test_results/kubernetes_rag_test_YYYYMMDD_HHMMSS.json`
- Full answers from both stores
- Retrieved contexts
- Detailed metrics

---

## Configuration

Edit the `TestConfig` class in `test_rag_kubernetes.py`:

```python
class TestConfig:
    # Chunking parameters
    CHUNK_SIZE = 1000        # Size of document chunks
    CHUNK_OVERLAP = 200      # Overlap between chunks

    # Retrieval parameters
    TOP_K = 4                # Number of documents to retrieve
    ALPHA = 0.5              # MMR diversity (0.0 = relevance only, 1.0 = diversity only)

    # Paths
    PDF_FOLDER = "data/pdf"              # Where to find PDFs
    RESULTS_FOLDER = "test_results"      # Where to save results

    # LLM settings
    LLM_MODEL = "gpt-4o-mini"
    LLM_TEMPERATURE = 0.7
```

### Parameter Explanations

**chunk_size**
- Smaller (500): More precise, but may miss context
- Medium (1000): Balanced (recommended)
- Larger (1500): More context, but less precise

**top_k**
- 3: Fast, minimal context
- 4: Balanced (recommended)
- 5+: More context, slower

**alpha** (MMR Diversity)
- 0.0: Pure relevance ranking
- 0.5: Balance relevance and diversity (recommended)
- 1.0: Maximum diversity (may sacrifice relevance)

---

## Test Questions

The script includes 40 Kubernetes questions covering:

### Basic Concepts (10 questions)
- What is Kubernetes?
- What is a Pod in Kubernetes?
- What is the difference between a Pod and a Container?
- What is a Kubernetes cluster?
- What is a Node in Kubernetes?
- What is the role of the control plane?
- What is kubectl?
- What is a namespace in Kubernetes?
- What is the purpose of etcd in Kubernetes?
- What is a Kubernetes API server?

### Workload Management (10 questions)
- What is a Deployment in Kubernetes?
- What is a ReplicaSet?
- How do you scale a Deployment?
- What is a StatefulSet?
- What is a DaemonSet?
- What is the difference between Deployment and StatefulSet?
- What is a Job in Kubernetes?
- What is a CronJob?
- How do you perform a rolling update?
- What is a rollback in Kubernetes?

### Networking (10 questions)
- What is a Service in Kubernetes?
- What are the types of Kubernetes Services?
- What is a ClusterIP service?
- What is a NodePort service?
- What is a LoadBalancer service?
- What is an Ingress?
- What is the difference between Service and Ingress?
- How do Pods communicate with each other?
- What is a NetworkPolicy?
- What is DNS in Kubernetes?

### Production Best Practices (10 questions)
- How do you implement zero-downtime deployments?
- What are the best practices for managing secrets?
- How do you implement auto-scaling for applications?
- What is the recommended approach for health checks and readiness probes?
- How do you implement persistent storage for databases?
- What are the strategies for implementing multi-tenancy?
- How do you implement monitoring and logging?
- What are the best practices for resource limits and requests?
- How do you implement blue-green deployment strategy?
- What are the security best practices for hardening?

---

## Output Format

### Console Output Example

```
================================================================================
📄 ЗАВАНТАЖЕННЯ PDF ДОКУМЕНТІВ
================================================================================

Знайдено 5 PDF файлів:

  • kubernetes-basics.pdf
  • kubernetes-networking.pdf
  • kubernetes-storage.pdf
  • kubernetes-security.pdf
  • kubernetes-production.pdf

[1/5] Обробка: kubernetes-basics.pdf
   ✓ 87 chunk'ів створено
[2/5] Обробка: kubernetes-networking.pdf
   ✓ 124 chunk'ів створено
...

📊 Всього chunk'ів створено: 523
   Chunk size: 1000
   Chunk overlap: 200

✓ ДОКУМЕНТИ УСПІШНО ЗАВАНТАЖЕНІ
  ChromaDB: 12.45с (42.0 chunk/с)
  FAISS: 8.23с (63.5 chunk/с)

================================================================================
🔬 ПОРІВНЯЛЬНЕ ТЕСТУВАННЯ RAG СИСТЕМ
================================================================================

Параметри:
  • Top-K: 4
  • Alpha (MMR): 0.5
  • Chunk size: 1000
  • Питань: 40

Vector Stores:
  • ChromaDB: 523 документів
  • FAISS: 523 документів

================================================================================

[1/40] What is Kubernetes?...
   ChromaDB... 1234ms (score: 0.8456)
   FAISS...    892ms (score: 0.8312)

[2/40] What is a Pod in Kubernetes?...
   ChromaDB... 1156ms (score: 0.8901)
   FAISS...    823ms (score: 0.8834)

...

================================================================================
📊 ЗВЕДЕНІ РЕЗУЛЬТАТИ
================================================================================

CHROMADB:
  • Середній час відповіді: 1189.3 ms
  • Середній similarity score: 0.8456
  • Загальний час: 47.57 s

FAISS:
  • Середній час відповіді: 856.7 ms
  • Середній similarity score: 0.8412
  • Загальний час: 34.27 s

ПОРІВНЯННЯ:
  • 🏆 Швидше: FAISS (1.39x)
  • 🎯 Кращий score: CHROMADB
  • 📊 Різниця у score: 0.0044

✓ Детальні результати збережено: test_results/kubernetes_rag_test_20250107_143022.json
```

### JSON Output Structure

```json
{
  "config": {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "top_k": 4,
    "alpha": 0.5,
    "llm_model": "gpt-4o-mini",
    "timestamp": "2025-01-07T14:30:22"
  },
  "questions": [
    {
      "question": "What is Kubernetes?",
      "chromadb": {
        "question": "What is Kubernetes?",
        "answer": "Kubernetes is an open-source container orchestration platform...",
        "contexts": [
          "Kubernetes is a portable, extensible platform...",
          "The name Kubernetes originates from Greek...",
          "..."
        ],
        "scores": [0.8456, 0.8123, 0.7892, 0.7654],
        "avg_score": 0.8031,
        "search_time_ms": 45.23,
        "generation_time_ms": 1189.12,
        "total_time_ms": 1234.35
      },
      "faiss": {
        "question": "What is Kubernetes?",
        "answer": "Kubernetes is a container orchestration system...",
        "contexts": [...],
        "scores": [0.8312, 0.8045, 0.7823, 0.7601],
        "avg_score": 0.7945,
        "search_time_ms": 12.87,
        "generation_time_ms": 879.34,
        "total_time_ms": 892.21
      }
    },
    ...
  ],
  "summary": {
    "chromadb": {
      "avg_time_ms": 1189.3,
      "avg_score": 0.8456,
      "total_time_s": 47.57
    },
    "faiss": {
      "avg_time_ms": 856.7,
      "avg_score": 0.8412,
      "total_time_s": 34.27
    },
    "comparison": {
      "speedup_factor": 1.39,
      "score_difference": 0.0044,
      "faster_store": "faiss",
      "better_score": "chromadb"
    }
  }
}
```

---

## Use Cases

### 1. Benchmarking Vector Stores

Compare ChromaDB vs FAISS performance on your specific documentation:

```bash
# Run with default settings
python test_rag_kubernetes.py

# Analyze results
cat test_results/kubernetes_rag_test_*.json | jq '.summary'
```

### 2. Optimizing Chunk Size

Test different chunk sizes to find optimal configuration:

```python
# Edit TestConfig
CHUNK_SIZE = 500   # Test 1: Small chunks
# Run test

CHUNK_SIZE = 1000  # Test 2: Medium chunks
# Run test

CHUNK_SIZE = 1500  # Test 3: Large chunks
# Run test

# Compare JSON results
```

### 3. Tuning Retrieval Parameters

Experiment with top_k and alpha:

```python
# More context, slower
TOP_K = 6
ALPHA = 0.3  # Favor relevance

# Fewer context, faster
TOP_K = 3
ALPHA = 0.7  # Favor diversity
```

### 4. Quality Assurance

Validate RAG system before production deployment:

```bash
# Test with production-like data
# Review answers for accuracy
# Check response times meet SLA
```

---

## Advanced Usage

### Testing Custom Questions

Edit `KUBERNETES_QUESTIONS` in the script:

```python
KUBERNETES_QUESTIONS = [
    "Your custom question 1?",
    "Your custom question 2?",
    # ...
]
```

### Clearing Databases

Clear existing ChromaDB and FAISS databases before testing:

```bash
python test_rag_kubernetes.py --clear-stores
```

**When to clear databases:**
- After changing chunk_size or chunk_overlap
- When PDFs have been significantly updated
- To benchmark from scratch
- When troubleshooting inconsistent results

### Force Reload Documents

Force reload PDFs even if documents already exist:

```bash
python test_rag_kubernetes.py --force-reload
```

**Difference between flags:**
- `--clear-stores`: Deletes all documents from both databases, then loads PDFs
- `--force-reload`: Skips the "already loaded" check and reloads anyway (adds to existing)
- Both together: Complete reset - clear databases then load fresh

### Programmatic Usage

```python
from test_rag_kubernetes import RAGKubernetesTester, TestConfig

# Custom configuration
config = TestConfig()
config.CHUNK_SIZE = 800
config.TOP_K = 5

# Initialize tester
tester = RAGKubernetesTester(config)
tester.load_pdfs_to_stores()

# Run custom questions
custom_questions = ["What is a Pod?", "What is a Service?"]
results = tester.run_comparison_test(custom_questions)

# Access results
print(results["summary"]["comparison"]["faster_store"])
```

---

## Troubleshooting

### No PDF Files Found

```
⚠️  УВАГА: Не знайдено PDF файлів у папці 'data/pdf'
```

**Solution:** Add PDF files to `data/pdf/` folder

### Documents Not Loading

Check if vector stores have permissions:

```bash
# Check ChromaDB directory
ls -la ~/.local/share/crewai-chatbot/

# Check FAISS directory
ls -la ~/.local/share/crewai-chatbot/rag_documents_faiss/
```

### Slow Performance

- Reduce `TOP_K` (e.g., from 4 to 3)
- Use smaller `CHUNK_SIZE` (e.g., 800 instead of 1000)
- Ensure PDFs are already loaded (skip reload)

### Out of Memory

- Use FAISS instead of ChromaDB (lower memory)
- Reduce `CHUNK_SIZE`
- Process fewer PDFs at once

---

## Tips for Best Results

1. **Quality PDFs**: Use text-based PDFs, not scanned images
2. **Relevant Docs**: Include comprehensive Kubernetes documentation
3. **Consistent Format**: Use similar formatting across PDFs
4. **Multiple Runs**: Run tests multiple times to average results
5. **Production Simulation**: Use production-sized datasets

---

## Comparison with Other Tools

| Tool | Purpose | Speed | Detail |
|------|---------|-------|--------|
| `test_rag_kubernetes.py` | Domain-specific testing (Kubernetes) | Fast | High |
| `compare_vector_stores.py` | Quick benchmark | Very Fast | Medium |
| `evaluate_rag.py` | Full ragas evaluation | Slow | Very High |

**When to use what:**
- **Quick check**: `compare_vector_stores.py`
- **Domain testing**: `test_rag_kubernetes.py` (this tool)
- **Full evaluation**: `evaluate_rag.py`

---

## Future Enhancements

- [ ] Ground truth answers for accuracy metrics
- [ ] Automated chunk size optimization
- [ ] Multi-run averaging
- [ ] Visualization dashboard
- [ ] Export to Excel/CSV
- [ ] Custom question sets per domain
- [ ] Hybrid search testing

---

## References

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [RAG Processing Guide](RAG_PROCESSING.md)
- [Vector Stores Comparison](VECTOR_STORES_COMPARISON.md)
