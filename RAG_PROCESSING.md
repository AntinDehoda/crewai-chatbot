# 🔍 RAG Processing Documentation

## Overview

This document describes the Retrieval-Augmented Generation (RAG) implementation in the CrewAI Chatbot. The system uses **Dense Vector Semantic RAG** with OpenAI embeddings and ChromaDB for efficient document retrieval.

## RAG Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      RAG PIPELINE                                │
└─────────────────────────────────────────────────────────────────┘

1. INGESTION PHASE
   PDF Document → PDFProcessor → Text Chunks
                                     ↓
                          OpenAI Embeddings (1536-dim)
                                     ↓
                          ChromaDB Vector Store

2. RETRIEVAL PHASE
   User Query → OpenAI Embedding → Cosine Similarity Search
                                           ↓
                                 Top-K Relevant Chunks

3. GENERATION PHASE
   Retrieved Chunks + Query → CrewAI Agent → Response
```

---

## Components

### 1. PDF Processor (`utils/pdf_processor.py`)

Handles PDF loading, text extraction, and chunking.

**Key Features:**
- Loads PDF files using `PyPDFLoader`
- Splits text into manageable chunks (1000 characters with 200 overlap)
- Preserves metadata (source file, page number, chunk ID)

**Code Example:**

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class PDFProcessor:
    """Клас для завантаження та обробки PDF документів"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Ініціалізація PDF процесора

        Args:
            chunk_size: Розмір chunk'ів для поділу тексту (default: 1000)
            chunk_overlap: Перекриття між chunk'ами (default: 200)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]  # Hierarchical splitting
        )

    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Завантажує PDF файл та розбиває на chunk'и

        Returns:
            List[Document]: Список документів (chunk'ів) з метаданими
        """
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
```

**Why Chunking Matters:**

| Aspect | Value | Reason |
|--------|-------|--------|
| **Chunk Size** | 1000 chars | Balance between context and precision |
| **Overlap** | 200 chars | Prevents information loss at boundaries |
| **Separator Hierarchy** | `\n\n` → `\n` → ` ` | Respects document structure |

**Example Output:**

```
Input PDF: contract.pdf (50 pages, 25,000 words)
↓
Output: 120 chunks
├── Chunk 0: "This agreement is entered..." (page 1)
├── Chunk 1: "The parties agree to the..." (page 1-2)
├── Chunk 2: "Section 2: Payment Terms..." (page 2)
└── ...
```

---

### 2. Vector Store Manager (`utils/vector_store.py`)

Manages embeddings and vector database operations.

**Key Features:**
- Uses OpenAI `text-embedding-3-small` (1536 dimensions)
- ChromaDB for efficient vector storage and retrieval
- Persistent storage across sessions
- Metadata filtering support

**Code Example:**

```python
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

class VectorStoreManager:
    """Клас для управління векторною базою даних"""

    def __init__(self, collection_name: str = "pdf_documents"):
        # Persistent storage location
        persist_directory = os.path.join(
            os.path.expanduser("~"),
            ".local/share/crewai-chatbot/rag_documents"
        )

        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",  # 1536 dimensions
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        # Initialize ChromaDB vector store
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Додає документи до векторної бази даних

        Process:
        1. Text → Embedding (via OpenAI API)
        2. Vector + Metadata → ChromaDB
        3. Automatic indexing for fast retrieval
        """
        ids = self.vector_store.add_documents(documents)
        return ids

    def search(self, query: str, k: int = 4) -> List[Document]:
        """
        Пошук релевантних документів

        Algorithm:
        1. Query → Embedding (1536-dim vector)
        2. Cosine similarity with all stored vectors
        3. Return top-K most similar chunks
        """
        results = self.vector_store.similarity_search(query, k=k)
        return results

    def search_with_scores(self, query: str, k: int = 4):
        """
        Пошук з оцінками релевантності

        Returns:
            List[tuple[Document, float]]: (document, similarity_score)
            Scores range from 0.0 (dissimilar) to 1.0 (identical)
        """
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results
```

**Embedding Process:**

```
Text Chunk: "The warranty period is 2 years from purchase date."
           ↓
OpenAI text-embedding-3-small
           ↓
Vector: [0.023, -0.145, 0.892, ..., 0.456]  # 1536 dimensions
           ↓
ChromaDB Storage + Indexing
```

**Similarity Search Example:**

```python
# Query
query = "What's the warranty duration?"

# Query Embedding
query_vector = [0.031, -0.152, 0.885, ..., 0.448]  # Similar to document

# Cosine Similarity Calculation
doc1: "warranty period is 2 years"     → similarity: 0.8542 ✅ High match
doc2: "payment terms net 30 days"      → similarity: 0.2341 ❌ Low match
doc3: "guarantee covers 24 months"     → similarity: 0.7923 ✅ Good match
doc4: "shipping within 5 business days" → similarity: 0.1892 ❌ Low match

# Return top-4 results (sorted by similarity)
```

---

### 3. RAG Search Tool (`tools/rag_tool.py`)

CrewAI tool that the agent uses to search documents.

**Key Features:**
- Integrates with CrewAI's tool system
- Formats results with source citations
- Handles empty database gracefully

**Code Example:**

```python
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

class RAGSearchInput(BaseModel):
    """Input schema for RAG search tool."""
    query: str = Field(..., description="Запит для пошуку в документах")


class RAGSearchTool(BaseTool):
    name: str = "Search PDF Documents"
    description: str = (
        "Шукає інформацію в завантажених PDF документах. "
        "Використовуй цей інструмент, коли користувач запитує про вміст документів."
    )
    args_schema: Type[BaseModel] = RAGSearchInput
    _vector_store_manager: Optional[VectorStoreManager] = PrivateAttr(default=None)

    def _run(self, query: str) -> str:
        """
        Виконує пошук в документах

        Returns:
            str: Formatted results with sources and relevance scores
        """
        # Check if documents exist
        doc_count = self._vector_store_manager.get_collection_count()
        if doc_count == 0:
            return "В базі даних немає завантажених документів."

        # Perform semantic search
        results = self._vector_store_manager.search_with_scores(query, k=4)

        # Format results with metadata
        formatted_results = [f"Знайдено {len(results)} релевантних фрагментів:\n"]

        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("source_file", "невідоме джерело")
            page = doc.metadata.get("page", "невідома сторінка")
            content = doc.page_content.strip()

            formatted_results.append(
                f"\n--- Фрагмент {i} (джерело: {source}, сторінка: {page}) ---"
            )
            formatted_results.append(content)
            formatted_results.append(f"(релевантність: {score:.4f})")

        return "\n".join(formatted_results)
```

**Tool Output Example:**

```
User Query: "What is the payment schedule?"

Tool Output:
Знайдено 4 релевантних фрагментів:

--- Фрагмент 1 (джерело: contract.pdf, сторінка: 3) ---
Payment Schedule: The client agrees to pay in three installments:
- 30% upon contract signing
- 40% upon project milestone completion
- 30% upon final delivery
(релевантність: 0.8734)

--- Фрагмент 2 (джерело: contract.pdf, сторінка: 4) ---
All payments must be made within 15 days of invoice date via wire transfer
or certified check.
(релевантність: 0.7621)

--- Фрагмент 3 (джерело: terms.pdf, сторінка: 2) ---
Late payments will incur a 1.5% monthly interest charge on outstanding balances.
(релевантність: 0.6892)

--- Фрагмент 4 (джерело: contract.pdf, сторінка: 3) ---
Payment terms are NET-15 from invoice date unless otherwise agreed in writing.
(релевантність: 0.6543)
```

---

### 4. Conversation Agent Integration (`agents/conversation_agent.py`)

The agent is equipped with the RAG tool and decides when to use it.

**Code Example:**

```python
from crewai import Agent
from langchain_openai import ChatOpenAI

def create_conversation_agent(tools: Optional[List] = None):
    """
    Створює агента з RAG підтримкою
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7
    )

    backstory = """Ти - дружній та компетентний асистент, який допомагає
    користувачам у різних питаннях.

    У тебе є доступ до інструменту пошуку в завантажених PDF документах.
    Коли користувач запитує про вміст документів або конкретну інформацію,
    яка може бути в PDF файлах, використовуй інструмент 'Search PDF Documents'
    для пошуку релевантної інформації.

    Якщо інформації немає в документах або документи не завантажені,
    відповідай на основі своїх загальних знань."""

    agent = Agent(
        role="Асистент для розмови",
        goal="Підтримувати розмову та допомагати знаходити інформацію",
        backstory=backstory,
        verbose=True,
        memory=True,  # ShortTermMemory for conversation context
        llm=llm,
        tools=tools,  # Includes RAG tool
        allow_delegation=False
    )

    return agent
```

**Agent Decision Making:**

```
User: "Привіт! Як справи?"
Agent: [No document search needed] → General response

User: "Що написано про гарантію в документі?"
Agent: [Detects document query] → Uses RAG tool → Responds with findings

User: "А яка погода сьогодні?"
Agent: [No document search needed] → General knowledge response

User: "Поясни деталі з контракту про оплату"
Agent: [Document-specific query] → Uses RAG tool → Provides extracted info
```

---

## Complete RAG Flow

### Example: User Asks About Document Content

```python
# 1. USER UPLOADS PDF
pdf_file = "technical_specs.pdf"
processor = PDFProcessor()
chunks = processor.load_pdf(pdf_file)
# Result: 87 chunks created

# 2. CHUNKS ARE EMBEDDED AND STORED
vector_store = VectorStoreManager()
ids = vector_store.add_documents(chunks)
# Result: 87 vectors stored in ChromaDB

# 3. USER ASKS QUESTION
user_query = "What are the system requirements?"

# 4. AGENT DECIDES TO USE RAG TOOL
# Agent thinks: "This is asking about document content, I should search"

# 5. RAG TOOL SEARCHES
rag_tool = RAGSearchTool(vector_store_manager=vector_store)
search_results = rag_tool._run(user_query)
# Result: Top 4 relevant chunks retrieved

# 6. AGENT RECEIVES SEARCH RESULTS
# Tool returns formatted text with sources:
"""
Знайдено 4 релевантних фрагментів:

--- Фрагмент 1 (джерело: technical_specs.pdf, сторінка: 12) ---
System Requirements:
- CPU: Intel i5 or equivalent
- RAM: 8GB minimum, 16GB recommended
- Storage: 250GB SSD
(релевантність: 0.9123)
...
"""

# 7. AGENT GENERATES FINAL RESPONSE
# Combines retrieved information with conversational context
agent_response = """
Згідно з технічними специфікаціями (сторінка 12), системні вимоги наступні:

📌 Процесор: Intel i5 або еквівалент
💾 Оперативна пам'ять: мінімум 8GB, рекомендовано 16GB
💿 Накопичувач: 250GB SSD

Ці вимоги забезпечать оптимальну роботу системи.
"""

# 8. USER RECEIVES RESPONSE
```

---

## Technical Specifications

### Embedding Model

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Model** | `text-embedding-3-small` | OpenAI's latest embedding model |
| **Dimensions** | 1536 | Vector size |
| **Max Input** | 8191 tokens | ~30,000 characters |
| **Cost** | $0.02 / 1M tokens | Very economical |
| **Performance** | MTEB score: 62.3% | Industry-leading |

### Vector Database

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Database** | ChromaDB | Open-source vector DB |
| **Distance Metric** | Cosine Similarity | Measures angle between vectors |
| **Index Type** | HNSW | Hierarchical Navigable Small World |
| **Storage** | Persistent | Survives restarts |
| **Location** | `~/.local/share/crewai-chatbot/rag_documents/` | User home directory |

### Chunking Strategy

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Chunk Size** | 1000 characters | ~200 words, good context window |
| **Overlap** | 200 characters | Prevents boundary information loss |
| **Splitters** | `\n\n`, `\n`, ` ` | Respects document structure |
| **Method** | Recursive | Tries larger separators first |

### Retrieval Parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| **Top-K** | 4 | Balance between recall and context length |
| **Similarity Threshold** | None | Returns all K results |
| **Reranking** | No | Simple pipeline for now |
| **MMR** | No | Could add for diversity |

---

## Performance Characteristics

### Time Complexity

```
Operation                Time Complexity    Notes
─────────────────────────────────────────────────────
PDF Loading              O(n)               n = file size
Text Chunking            O(n)               Linear with text length
Embedding Generation     O(c)               c = number of chunks (API call)
Vector Storage           O(c log c)         ChromaDB indexing
Similarity Search        O(log N)           HNSW index, N = total vectors
Response Generation      O(k * m)           k = chunks, m = token generation
```

### Scalability

```
Document Count    Chunks    Storage    Search Time
────────────────────────────────────────────────────
1-10 PDFs         ~500      ~50MB      <100ms
10-100 PDFs       ~5,000    ~500MB     <200ms
100-1000 PDFs     ~50,000   ~5GB       <500ms
1000+ PDFs        ~500,000  ~50GB      <1s
```

---

## RAG Type Classification

### This Implementation

✅ **Dense Retrieval RAG**
- Uses neural embeddings (not sparse TF-IDF)
- Semantic understanding of meaning
- Industry standard approach

✅ **Naive RAG Architecture**
- Simple pipeline: Index → Retrieve → Generate
- No pre/post processing
- Effective for most use cases

❌ **Not TF-IDF**
- TF-IDF uses sparse keyword vectors
- Our solution uses dense semantic embeddings
- Much more sophisticated

### Comparison to Other RAG Types

```
                        This Solution    Advanced RAG    TF-IDF RAG
──────────────────────────────────────────────────────────────────────
Semantic Understanding  ✅ Yes           ✅ Yes          ❌ No
Query Expansion         ❌ No            ✅ Yes          ❌ No
Reranking              ❌ No            ✅ Yes          ❌ No
Hybrid Search          ❌ No            ✅ Yes          ❌ No
Production Ready       ✅ Yes           ✅ Yes          ⚠️  Limited
Implementation Time     Fast             Slow            Fast
Maintenance            Easy             Complex         Easy
Cost                   Medium           High            Low
```

---

## Example Use Cases

### 1. Legal Document Analysis

```python
# Upload contract
processor.load_pdf("contract.pdf")

# Query
"What are the termination clauses?"

# Result: Finds all relevant sections about contract termination
# across multiple pages, even if worded differently
```

### 2. Technical Documentation Search

```python
# Upload manual
processor.load_pdf("user_manual.pdf")

# Query
"How do I reset the device?"

# Result: Finds reset procedures, troubleshooting sections,
# and related maintenance info
```

### 3. Research Paper Analysis

```python
# Upload multiple papers
processor.load_pdf("paper1.pdf")
processor.load_pdf("paper2.pdf")

# Query
"What methodologies were used for data collection?"

# Result: Extracts methodology sections from both papers
# with source citations
```

---

## Advantages of This Approach

### 1. Semantic Understanding
```
Query: "vehicle maintenance schedule"
Matches:
✅ "car service intervals"
✅ "automobile upkeep timeline"
✅ "maintenance schedule for vehicles"
✅ "recommended service periods"
```

### 2. Multilingual Support
```
Works with Ukrainian, English, and 100+ languages
OpenAI embeddings understand cross-lingual semantics
```

### 3. Source Attribution
```
Every result includes:
- Source filename
- Page number
- Relevance score
- Full context
```

### 4. Persistent Storage
```
Documents stay indexed across sessions
No need to re-upload
Fast startup time
```

---

## Limitations & Future Improvements

### Current Limitations

1. **No Query Expansion**
   - Single query embedding
   - Could miss related concepts

2. **No Reranking**
   - Simple cosine similarity
   - Could improve precision with cross-encoder

3. **Fixed Chunk Size**
   - 1000 characters for all documents
   - Some documents might benefit from different sizes

4. **No Hybrid Search**
   - Only semantic search
   - Missing keyword matching benefits

### Potential Improvements

```python
# 1. Add Reranking
from sentence_transformers import CrossEncoder
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# 2. Hybrid Search (Dense + Sparse)
from langchain.retrievers import EnsembleRetriever
hybrid_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.7, 0.3]
)

# 3. Query Expansion
def expand_query(query):
    return [query, generate_synonym_query(query), generate_related_query(query)]

# 4. Contextual Chunk Embeddings
def add_context_to_chunk(chunk, document_summary):
    return f"Document: {document_summary}\n\nChunk: {chunk}"
```

---

## Code Usage Examples

### Example 1: Simple PDF Upload and Search

```python
from utils.pdf_processor import PDFProcessor
from utils.vector_store import VectorStoreManager
from tools.rag_tool import create_rag_tool

# Initialize components
processor = PDFProcessor()
vector_store = VectorStoreManager()
rag_tool = create_rag_tool(vector_store)

# Upload PDF
chunks = processor.load_pdf("document.pdf")
vector_store.add_documents(chunks)

# Search
results = rag_tool._run("What is the main topic?")
print(results)
```

### Example 2: Batch Upload Multiple PDFs

```python
import glob

# Find all PDFs
pdf_files = glob.glob("documents/*.pdf")

# Process and upload
for pdf_path in pdf_files:
    print(f"Processing {pdf_path}...")
    chunks = processor.load_pdf(pdf_path)
    ids = vector_store.add_documents(chunks)
    print(f"✓ Added {len(ids)} chunks from {pdf_path}")
```

### Example 3: Search with Metadata Filtering

```python
# Search only in specific document
results = vector_store.search(
    query="payment terms",
    k=4,
    filter_dict={"source_file": "contract.pdf"}
)
```

### Example 4: Get Statistics

```python
# Check database stats
total_chunks = vector_store.get_collection_count()
source_files = vector_store.get_all_source_files()

print(f"Total chunks: {total_chunks}")
print(f"Source files: {source_files}")
```

---

## Conclusion

This RAG implementation uses **modern dense vector embeddings** with OpenAI's `text-embedding-3-small` model and ChromaDB for efficient semantic search. It's **NOT naive TF-IDF**, but rather a production-ready semantic RAG system that understands meaning and context.

The architecture follows the **"Naive RAG" pattern** (simple pipeline), but uses **state-of-the-art technology** (neural embeddings, vector databases) that powers most modern RAG applications.

**Key Strengths:**
- ✅ Semantic understanding of queries
- ✅ Fast similarity search with HNSW indexing
- ✅ Persistent storage across sessions
- ✅ Source attribution with page numbers
- ✅ Integration with CrewAI agent system
- ✅ Production-ready and scalable

**Perfect for:**
- Document Q&A systems
- Knowledge base search
- Contract analysis
- Technical documentation
- Research paper exploration

---

## References

- [OpenAI Embeddings Documentation](https://platform.openai.com/docs/guides/embeddings)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
- [CrewAI Tools Documentation](https://docs.crewai.com/core-concepts/Tools/)
