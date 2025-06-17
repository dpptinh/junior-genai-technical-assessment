---
title: rag_and_reasoning_frameworks_tutorial
---

# 🧠 RAG and Reasoning Frameworks Tutorial

---
## 🔷 What is RAG (Retrieval-Augmented Generation)?

<details - open>
<summary>Definition and Purpose of RAG</summary>

---

- **RAG** is a technique that enhances the performance of Large Language Models (LLMs) by retrieving relevant information from external sources and incorporating it into the model's input at inference time.
- This helps the model produce **more accurate, up-to-date, and context-aware** responses.



### ⚙️ Core Components


- **Retrieval**
  - Search for relevant data or documents using semantic similarity.
- **Augmentation**
  - Inject retrieved content into the prompt with clear structure and instructions for the LLM.
  - You can also include re-ranking, filtering, or summarization steps if needed.
- **Generation**
  - The augmented prompt is passed to an LLM to generate the final response.

---
</details>

---

## 🔄 RAG Workflow Explained

<details - open>
<summary>Complete RAG Workflow from Input to Output</summary>

---
### 🖼️ System Architecture

- The diagram below illustrates the interaction between user input, retriever, context injection, and LLM output:
- ![RAG Architecture](RAG_Architecture.png)


### 🧭 RAG Step-by-Step Flow
- **Step 1: 📥 Receive a User Query**
  - The system gets a user input (question or prompt).
  - For example:
    - _"What are the latest advancements in robotics?"_
- **Step 2: 🔍 Search for Relevant Information**
  - The retriever looks for relevant content from external data sources — such as vector databases, knowledge graph, or APIs.
- **Step 3: 📚 Retrieve the Best Matching Content**
  - Extract the most relevant chunks or passages based on the query.
- **Step 4: 🧩 Augment the Prompt with Context**
  - Build a new prompt combining:
    - Instructions
    - Retrieved context
    - User question
- **Step 5: 🧠 Generate a Response with the LLM**
  - The LLM uses the combined prompt to generate a response that’s more informed and reliable.
- **Step 6: ✅ Return the Answer to the User**
  - The final output is accurate, context-rich, and ideally better than the LLM’s default behavior without retrieval.

---

</details>

---
## 🧪 Simple RAG Implementation in Python

<details - open>
<summary>Step-by-step Python Code for RAG</summary>

---

- **Install the required libraries**:
  ```bash
  pip install chromadb sentence-transformers langchain-openai
  ```
- **Python Code**:
  ```python
  import os
  from dotenv import load_dotenv
  from sentence_transformers import SentenceTransformer
  import chromadb
  from chromadb.config import Settings
  from langchain_openai import OpenAI

  # Load API key từ biến môi trường
  load_dotenv()
  os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

  # 1. Load embedding model
  embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

  # 2. Initialize Chroma in-memory DB
  chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
  collection = chroma_client.create_collection(name="rag_demo")

  # 3. Sample documents
  documents = [
      "This is the first document chunk.",
      "This is the second document chunk.",
      "More information in the third chunk."
  ]
  doc_ids = [f"doc_{i}" for i in range(len(documents))]

  # 4. Generate and insert embeddings
  embeddings = embedding_model.encode(documents).tolist()
  collection.add(documents=documents, ids=doc_ids, embeddings=embeddings)

  # 5. Retrieval function
  def retrieve(query, k=2):
      query_embedding = embedding_model.encode([query])[0].tolist()
      results = collection.query(query_embeddings=[query_embedding], n_results=k)
      return results['documents'][0]

  # 6. Prompt augmentation
  def create_prompt(query, contexts):
      context_text = "\n".join(contexts)
      return f"""You are a helpful assistant. Use the following context to answer the question.
  If you don't know the answer, say you don't know.

  Context:
  {context_text}

  Question:
  {query}

  Answer:"""

  # 7. Generate answer from OpenAI
  def generate_answer(prompt):
      llm = OpenAI()
      response = llm.invoke(prompt)
      return response.content # Corrected based on likely intent; original had response.content not response

  # 8. Full RAG pipeline
  def rag_pipeline(query):
      contexts = retrieve(query)
      prompt = create_prompt(query, contexts)
      return generate_answer(prompt)

  # 9. Example usage
  if __name__ == "__main__":
      user_query = "Tell me about the second document."
      answer = rag_pipeline(user_query)
      print("Answer:", answer)
  ```

---

</details>

---
## 💾 Vector Databases
<details open>
<summary>Introduction to Vector DBs</summary>

---
### 💡 Introduction to Vector DBs
- Vector databases are essential for enabling efficient and intelligent document retrieval in RAG systems.
- Unlike traditional databases that rely on exact keyword matches, vector DBs use dense numerical representations (embeddings) of documents and queries, allowing for semantic search.
- This means that even if a user's query doesn't contain the exact same words as the document, the system can still identify relevant content based on meaning and context.
- This dramatically improves the quality of retrieved data.
- Key capabilities include:
  - High-dimensional storage: Supports millions of embedding vectors.
  - Fast similarity search: Efficiently finds the top-k most similar vectors.
  - Scalability: Designed for high-volume, low-latency retrievals.
  - Metadata filtering: Combine vector similarity with structured filters (e.g., by document type or date).



### 📊 Comparative Analysis of Vector Databases
<details open>
<summary>Comparative analysis of Pinecone, Weaviate, and Chroma</summary>

---

#### Overall Comparison
- Overall comparison table:
  | Criteria                 | Pinecone                                | Weaviate                              | Chroma                                |
  |--------------------------|-----------------------------------------|-------------------------------------|--------------------------------------|
  | **Deployment Model**     | Fully managed cloud service             | Self-hosted or cloud, supports Docker/K8s | Self-hosted, embedded, native Python |
  | **Query Performance (p95)**| Very low (`10-100 ms`)                  | Low (`50-200 ms`)                     | Varies (`100-500 ms`)                |
  | **Scalability**          | Billions of vectors, auto-scaling       | Hundreds of millions of vectors, well-distributed | Tens of millions of vectors          |
  | **Enterprise Features**  | Comprehensive: HA, DR, access control, monitoring, encryption, SLA | Good: HA, DR (cloud), access control, monitoring, encryption | Basic: Limited HA, manual DR, simple access control |
  | **API & SDK Integration**| Simple REST API, integrates with popular embedding models | GraphQL, REST, supports multiple vectorizers, hybrid search | Simple API, Python-native, supports multiple embedding providers |
  | **Cost**                 | High                                    | Free open-source (self-hosted), pay-as-you-go cloud | Free, development cost to scale      |
  | **Operational Complexity**| Low, no infrastructure management needed| Medium to high, requires expertise for optimization | Low, easy setup, suitable for dev, prototyping |
  | **Suitable for**         | Large enterprises, needing SLAs, high security, production scale | Enterprises needing flexibility, hybrid search, multi-modal, open-source | Startups, individual developers, rapid prototyping, limited budget |

---

#### Detailed Pros and Cons

**Pinecone**
- **Advantages**
  - High performance, very low query latency, handles billions of vectors and large concurrent queries.
  - Fully managed cloud service, minimizing operational overhead.
  - Meets enterprise security standards (SOC 2 Type 2), offers SLAs, private cloud, and multi-region replication.
  - Simple REST API, easy integration with popular embedding models.
- **Disadvantages**
  - High operational cost, not suitable for small projects or individuals.
  - Less infrastructure customization, dependent on the service provider.
  - No self-hosted version, limited control over the entire stack.


**Weaviate**
- **Advantages**
  - Open-source, flexible deployment (self-hosted or cloud), supports Docker/Kubernetes.
  - Supports vector search combined with structured queries (GraphQL, BM25), multi-modal (text, image, audio).
  - Modular architecture, integrates multiple vectorizers, dynamic data updates.
  - Good enterprise features: HA, DR (cloud), access control, monitoring, data encryption.
- **Disadvantages**
  - Configuration and operation can be complex, requiring high expertise.
  - Self-hosting incurs infrastructure overhead, needs a strong technical team.
  - Some advanced features are only available in the paid cloud version.

**Chroma**
- **Advantages**
  - Simple API, easy setup, native Python, developer-friendly.
  - Supports in-memory and file-based storage, lightweight, easy integration with multiple embedding providers.
  - Free, suitable for startups, individual developers, limited budgets.
  - Fast indexing speed for small to medium-sized data.
- **Disadvantages**
  - Lacks enterprise features like advanced monitoring, security, HA.
  - Higher query latency, limited capability for handling concurrent queries.
  - Less scalability and capacity for handling large datasets.
  - Manual disaster recovery, limited security and monitoring.

---

#### Recommendations Based on Production Needs
- Recommendations based on production needs:
  | Deployment Need                                     | Suitable Platform                               | Key Reason                                      |
  |-----------------------------------------------------|-------------------------------------------------|-------------------------------------------------|
  | Large enterprise, needs SLA, security, production scale | **Pinecone**                                     | High performance, managed service, good security |
  | Flexible, multi-purpose, hybrid search, open-source   | **Weaviate**                                     | Multi-modal support, self-host or cloud option  |
  | Startup, individual dev, prototyping, low budget      | **Chroma**                                       | Developer-friendly, easy deployment, low cost   |

---

#### Considerations for Production Deployment
- Design the embedding model and data schema appropriately from the start.
- Monitor queries, recall rate, and resources to ensure performance.
- Plan for scaling in accordance with data volume and query load.
- Security: authentication, authorization, data encryption.
- Effective data backup and recovery, especially with self-hosted or cloud platforms offering DR.

---

#### Conclusion
- Choosing the right vector database platform is a critical factor for the success and scalability of an enterprise RAG system.
- **Pinecone** is a top choice for large enterprises needing high stability and security with a comprehensive managed service.
- **Weaviate** is suitable for organizations requiring a flexible, multi-purpose solution, with options for self-hosting or cloud, prioritizing open-source.
- **Chroma** is an ideal choice for startups, individual developers, or small projects needing rapid development, low cost, and Python-friendliness.

---

</details>

---
## Chunking strategies
<details open>
<summary>Introduction and chunking strategies</summary>

---

In Retrieval-Augmented Generation (RAG) architecture, **chunking** (text segmentation) is a key data preprocessing stage. This process involves dividing source documents into smaller, more manageable units of information called "chunks". The main goals of chunking are:

1.  **Compatibility with LLM's Context Window:** Each chunk must be appropriately sized to fit the input token limit of the Large Language Model (LLM).
2.  **Preserving Semantic Integrity:** Chunks need to maintain the core meaning and context of the original text. Arbitrary splitting can disrupt the logical flow and degrade information quality.
3.  **Optimization for Information Retrieval:** In RAG, chunks are vectorized (via embedding models) and indexed. An effective chunking strategy will create chunks with high semantic cohesion, helping the system accurately retrieve the most relevant text segments for the user's query.

The choice of an appropriate chunking strategy directly impacts the performance, accuracy, and computational efficiency of the entire RAG system. This document will detail common and advanced chunking methods.

---

### Common Chunking Strategies

<details open>
<summary>Details on chunking strategies</summary>

---
#### Fixed-Size Chunking

**Overview and Mechanism:**
-   The most basic method, dividing text into chunks of a fixed length (number of characters or tokens).
-   Often uses a `chunk_overlap` parameter to maintain some context between consecutive chunks, minimizing information loss at split points.

**Advantages:**
-   Simple to implement.
-   Low computational cost, fast processing speed.

**Disadvantages:**
-   High risk of cutting across semantic units (sentences, clauses, ideas), reducing coherence.
-   Chunks can become difficult to understand or lack necessary context for the LLM.

**Use Cases:**
-   Unstructured texts or when structure is not a priority.
-   Initial prototyping tasks or applications requiring low cost and high speed.
-   When absolute control over chunk size is needed.

**Implementation with LangChain:**
-   Use `CharacterTextSplitter` (splits by character) or `TokenTextSplitter` (splits by token).
  ```python
  from langchain_text_splitters import CharacterTextSplitter

  text_content = """GPT-4.1 stands out with its superior context processing capabilities, allowing the model to receive and analyze up to 1 million tokens in a single query – eight times the 128,000 token limit of the previous GPT-4o. This makes the model ideal for fields requiring the processing of large amounts of data, such as legal document analysis, finance, or complex programming. OpenAI asserts that GPT‑4.1 has been trained to better identify important information and minimize “noise” from irrelevant data in both short and long contexts.""" # Replace with actual content

  text_splitter = CharacterTextSplitter(
      chunk_size = 256,    # Maximum chunk size (unit: characters)
      chunk_overlap  = 30, # Number of overlapping characters between chunks
  )
  chunks = text_splitter.split_text(text_content)
  # chunks is a list of strings
  ```

---

#### Recursive Chunking

**Overview and Mechanism:**
-   This method splits text based on a list of separators arranged in order of priority (e.g., `["\n\n", "\n", ". ", " ", ""]`).
-   Initially, it tries to split the text using the highest priority separator (often indicators of large structural breaks like double newlines). If the resulting chunk still exceeds the target `chunk_size`, the process recurses, applying the next separator in the list until the desired chunk size is achieved or the list of separators is exhausted.

**Advantages:**
-   Provides a good balance between maintaining semantic coherence and controlling chunk size.
-   Adapts well to various types of document structures, especially natural text.
-   Often the recommended default choice for unstructured or semi-structured text.

**Disadvantages:**
-   Requires careful configuration of the separator list and their order to optimize for specific data types.

**Use Cases:**
-   Most types of unstructured or semi-structured text (articles, documents, web content).
-   When the goal is to keep related semantic units as close together as possible while still ensuring chunk size.

**Implementation with LangChain:**
-   Use `RecursiveCharacterTextSplitter`.
  ```python
  from langchain_text_splitters import RecursiveCharacterTextSplitter

  text_content = "Your complex document with paragraphs, lines, and sentences..." # Replace

  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size = 500,        # Target chunk size (unit: characters)
      chunk_overlap = 50,       # Number of overlapping characters
      separators=["\n\n", "\n", ". ", " ", ""], # Priority order of separators
  )
  chunks = text_splitter.split_text(text_content)
  ```

---

#### Content-Aware / Semantic Chunking

**Overview and Mechanism:**
-   Uses Natural Language Processing (NLP) techniques to identify natural breakpoints based on changes in topic or meaning within the text.
-   Instead of relying on fixed sizes or syntactic separators, this method analyzes semantic content.
-   Often involves using embedding models to calculate semantic similarity between adjacent sentences or paragraphs. A significant drop in similarity can indicate a suitable split point, signaling a topic shift.

**Advantages:**
-   Creates chunks with very high semantic coherence, closely aligning with the document's thematic structure.
-   Often leads to better retrieval quality as chunks focus on a specific topic or idea.

**Disadvantages:**
-   Significantly higher computational cost compared to rule-based methods due to the need for embedding calculations and similarity comparisons.
-   Requires access to embedding models and more complex processing logic.
-   Processing speed can be slower, especially with large datasets.

**Use Cases:**
-   Complex documents where semantic integrity and thematic relevance are paramount.
-   Applications requiring high retrieval accuracy, such as in-depth Q&A systems or knowledge base construction.

**Implementation with LangChain:**
-   `LangChain` provides `SemanticChunker` in the `langchain_experimental.text_splitter` module.
  ```python
  # Requires installation: pip install langchain_experimental langchain_openai sentence-transformers
  import os
  from langchain_experimental.text_splitter import SemanticChunker
  from langchain_openai import OpenAIEmbeddings # Or another embedding model
  text_content = "Topic A sentence 1. Topic A sentence 2. Topic B sentence 1. Topic B sentence 2."
  embeddings_model = OpenAIEmbeddings(api_key = os.getenv("OPENAI_API_KEY")) # can be replaced with another embedding model

  semantic_splitter = SemanticChunker(
      embeddings_model,
      breakpoint_threshold_type="percentile"
  )
  # SemanticChunker returns a list of Document objects
  chunk_documents = semantic_splitter.create_documents([text_content])
  # chunks = [doc.page_content for doc in chunk_documents]
  ```

---

#### Document-Based / Structure-Aware Chunking

**Overview and Mechanism:**
-   Splits text based on the inherent logical structure or format of the document, e.g., chapters, sections, headers, lists, tables, or specific tags in formats like Markdown, HTML, XML.
-   For source code, can split by functions, classes, or logical blocks.

**Advantages:**
-   Maintains the logical flow and coherence according to the structure designed by the author.
-   Chunks are often naturally meaningful and contextually relevant as they follow the logical divisions of the original document.
-   Very effective for well-formatted documents.

**Disadvantages:**
-   Less effective for unstructured or inconsistently formatted documents.
-   Depends on the formatting quality of the input document.

**Use Cases:**
-   Clearly structured documents like Markdown files (technical documentation, READMEs), HTML (web pages), JSON, XML, API documentation, source code.
-   When wanting to leverage the existing semantic structure within the document.

**Implementation with LangChain:**
  ```python
  from langchain_text_splitters import MarkdownHeaderTextSplitter

  markdown_doc_content = """
  # Title
  ## Section 1
  Content of section 1.
  ### Subsection 1.1
  Content of subsection 1.1
  ## Section 2
  Content of section 2.
  """ # Replace with actual Markdown content

  headers_to_split_on = [
      ("#", "Header 1"),
      ("##", "Header 2"),
      ("###", "Header 3"),
  ]
  markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
  # split_text returns a list of Documents, each Document has page_content and metadata
  document_based_chunks = markdown_splitter.split_text(markdown_doc_content)
  ```
---

#### Agentic Chunking (or Propositional Chunking) - Advanced

**Overview and Mechanism:**
-   An advanced strategy where an LLM agent actively participates in the chunking process.
-   The agent analyzes the document content, possibly considering potential queries, to determine optimal chunk boundaries.
-   One approach is **Propositional Chunking**, which focuses on dividing text into "propositions" – the smallest, semantically independent units of information or assertions. Each proposition can then be considered a chunk.

**Advantages:**
-  Most adaptive and contextually relevant, leveraging the LLM's deep understanding and reasoning capabilities.
-   Creates very granular and focused chunks, which can significantly improve tasks requiring high precision like event extraction or logical reasoning.

**Disadvantages:**
-   High cost due to requiring multiple LLM calls.
-   Long execution time.

**Use Cases:**
-   Tasks requiring extremely high precision information extraction, logical reasoning, or in advanced research applications.
-   When computational cost is not a primary constraint and high-detail chunk quality is a top priority.

**Implementation with LangChain (Conceptual Illustration):**
```python
# Requires installation: pip install langchain-openai
import os
from langchain_openai import OpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Ensure OPENAI_API_KEY is set in the environment or passed directly
llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0)

text_to_chunk = """Artificial intelligence (AI) is rapidly transforming various sectors. Machine learning, a subset of AI, enables systems to learn from data. Deep neural networks are a popular technique within machine learning."""

system_prompt = "You are an AI assistant tasked with segmenting the provided text into distinct, semantically coherent propositions or minimal units of information. Each proposition should ideally be a self-contained statement. Return each proposition on a new line, separated by two newline characters (\n\n)."
human_prompt = f"Please segment the following text into semantic propositions:\n\n{text_to_chunk}"

messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content=human_prompt)
]
response = llm.invoke(messages)
raw_chunks = response.content.split("\n\n")
agentic_chunks = [chunk.strip() for chunk in raw_chunks if chunk.strip()]

```

---

#### Late Chunking - Emerging Concept

**Overview and Mechanism:**
-   An emerging technique where detailed chunking does not occur entirely in the initial preprocessing step.
-   A key aspect to achieve its effectiveness is the ability to generate embeddings for sub-chunks while retaining the context of the entire document. This is often achieved by embedding the entire large text first, then extracting or mapping parts of that embedding to the sub-chunks.

**Advantages:**
-   **Keeps context:** Late chunking ensures each sub-chunk maintains overall context by first embedding the entire document (or super-chunk). This preserves cross-references and relationships across the text within the sub-chunk embeddings.
-   **Better retrieval:** Chunk embeddings generated through late chunking become semantically richer and more accurate, thereby improving retrieval results in RAG systems due to the model's deeper understanding of the original document's semantics.
-   **Handles long texts:** This method is particularly useful for very long documents that traditional models cannot process entirely in one go due to token limits.
-   **Relevance Maximization:** Capable of maximizing chunk relevance for specific queries, as chunking is contextually adjusted.

**Disadvantages:**
-   More complex to implement than traditional chunking methods.
-   Longer processing time.

**Use Cases:**
-   When user queries are highly diverse and require dynamically adjusted context windows.
-   For adaptive retrieval systems.
-   In cases where pre-chunking an entire large document corpus is not feasible or cost/storage-inefficient.

**Implementation:**
*The code snippet below illustrates how to create embeddings for small chunks such that they retain contextual information from the entire document. This is an important technique that can be applied in a Late Chunking system.*
```python
# Requires installation: pip install sentence-transformers torch
from sentence_transformers import SentenceTransformer
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Use a popular embedding model, replace if needed
embedding_model_name = 'hiieu/halong_embedding'
try:
    embedding_model = SentenceTransformer(embedding_model_name, device=device)
except Exception as e:
    print(f"Error loading model {embedding_model_name}: {e}")
    exit()

document = """OpenAI has just officially launched its latest AI model named GPT-4.1, marking a major step forward in the field of artificial intelligence. This event opens up many new potential applications in various industries. Notably, OpenAI decided to delay the launch of GPT-5 to focus on improving and perfecting current products.

GPT-4.1 stands out with its superior context processing capabilities, allowing the model to receive and analyze up to 1 million tokens in a single query – eight times the 128,000 token limit of the previous GPT-4o. This makes the model ideal for fields requiring the processing of large amounts of data, such as legal document analysis, finance, or complex programming. OpenAI asserts that GPT‑4.1 has been trained to better identify important information and minimize “noise” from irrelevant data in both short and long contexts.
Not only stopping at expanding the context limit, GPT-4.1 has also been significantly improved in terms of programming ability and instruction following. According to OpenAI's announcement, this model performs 21% better than GPT-4o in programming tests, and also outperforms GPT-4.5 by 27%. This is particularly important for software developers and businesses looking for more effective AI solutions.
Another noteworthy point is that the cost of using GPT-4.1 has decreased by 26% compared to GPT-4o, helping businesses and developers save significantly when deploying large-scale AI applications. This is an important competitive factor as the AI market is witnessing the emergence of new competitors like DeepSeek – known for its highly efficient AI model."""

# Step 1: Split the document into base units (e.g., sentences) - "base chunks". Other chunking methods can be used.
base_chunks_text = [s.strip() for s in document.split(".") if s.strip()]
base_chunks_text = [s + "." if not s.endswith(".") else s for s in base_chunks_text]


# Step 2: Tokenize the entire document to get token embeddings with global context
# SentenceTransformer usually handles this internally when calling .encode() or through its modules.
# To illustrate getting token embeddings, we access the transformer module inside.
transformer_module = embedding_model._first_module().auto_model
tokenizer = embedding_model.tokenizer

doc_tokens = tokenizer(document, return_tensors='pt', padding=True, truncation=True).to(device)
with torch.no_grad():
    doc_outputs = transformer_module(**doc_tokens, output_hidden_states=False)
    # last_hidden_state contains token embeddings for the entire document
    # Size: (batch_size=1, sequence_length, hidden_size)
    doc_token_embeddings = doc_outputs.last_hidden_state.squeeze(0) # Remove batch dimension

# Step 3: Get embedding for each sentence with mean pooling and interaction
base_chunks_embedding = []  # Initialize a list to store sentence embeddings
curr_token_idx = 1  # Current token index, starting from 1 to skip the special [CLS] token
for base_chunk_text in base_chunks_text:  # Iterate through each sentence in the document
    base_chunk_tokens = embedding_model.tokenizer(base_chunk_text, return_tensors='pt').to(device)  # Tokenize the sentence
    length = base_chunk_tokens['input_ids'].shape[1] - 2  # Calculate sentence length after subtracting [CLS] and [SEP] tokens
    # Calculate sentence embedding by averaging its token embeddings from the document's token embeddings
    base_chunk_embedding = doc_token_embeddings[curr_token_idx:curr_token_idx+length].mean(dim=0) # Squeeze(0) removed as doc_token_embeddings is already 2D
    base_chunks_embedding.append(base_chunk_embedding)  # Add sentence embedding to the list
    curr_token_idx += length  # Update current token index

base_chunks_embedding = torch.stack(base_chunks_embedding)  # Convert list of embeddings to a stacked tensor
```
**Technical Explanation of the Code Snippet:**
The core idea is to create embedding vectors for each `base_chunk` (e.g., each sentence) such that they not only represent the semantics of that sentence alone but also reflect its position and role within the context of the entire document. This is achieved by:
1.  **Global Embedding:** Processing the entire document to obtain embedding vectors for each token within that document (`doc_token_embeddings`). Each of these token embeddings has been "informed" (contextualized) by its surrounding tokens in the entire document.
2.  **Mapping and Pooling:** For each `base_chunk`, identify the corresponding tokens in `doc_token_embeddings`. Then, extract these token vectors and apply a pooling operation (e.g., mean pooling) to create a single embedding vector for that `base_chunk`. This resulting vector is expected to be semantically richer than embedding each chunk independently.

---
</details>


### Guidance on Choosing an Optimal Chunking Strategy
<details open>
<summary>Optimal Chunking Strategy</summary>

---

There is no "one-size-fits-all" chunking strategy. The choice depends on the following factors:

**Dataset Characteristics:**

  - **Structured Documents (Markdown, HTML, Code):** `Structure-Aware Chunking` is often the top choice.
  - **Unstructured Documents (Plain text, extracted PDF):** `Recursive Chunking` is a strong starting point. `Semantic Chunking` can be considered if higher semantic quality is required and computational cost is acceptable.
  - **Short or disjointed texts:** Simpler methods like `Fixed-Size Chunking` may be sufficient.

**Application Goals (RAG):**

  - Prioritizing Retrieval Accuracy: `Semantic Chunking`, `Agentic Chunking` (if feasible), or techniques within `Late Chunking` can be beneficial.
  - Prioritizing Speed and Low Cost: `Fixed-Size Chunking` or `Recursive Chunking` with a simple configuration.


**Computational Resources & Development Time:**
  - `Semantic Chunking`, `Agentic Chunking`, and `Late chunking` demand more resources and processing time.
  - Simpler methods are generally faster and easier to implement.

**`chunk_overlap` Parameter:**
  - A reasonable overlap value (e.g., 10-20% of chunk size) is often useful for maintaining semantic continuity.
  - Too much overlap can unnecessarily increase the number of chunks and storage/computational costs, and may also introduce noise into the retrieval process.

**Practical Recommendations:**
-   **Start with a Balanced Solution:** `RecursiveCharacterTextSplitter` is often a good starting choice due to its flexibility and effectiveness.
-   **Experimentation and Evaluation:** The best way to determine the optimal strategy is to experiment with multiple methods on your specific dataset and evaluate the results (e.g., via RAG evaluation metrics like context precision, context recall, answer faithfulness).
-   **Leverage Metadata:** When chunking, try to preserve and associate important metadata (e.g., document source, page number, chapter/section title) with each chunk. This metadata is very valuable for filtering retrieval results, citing sources, and debugging.
---
</details>


### Conclusion

Text chunking is a fundamental component with a significant impact on the performance of Retrieval-Augmented Generation systems. Understanding the nature, pros, and cons of each strategy, along with the ability to implement them, allows developers to optimize the data processing pipeline for LLMs.