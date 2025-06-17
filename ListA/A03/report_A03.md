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

---

</details>

### ⚙️ Core Components
<details - open>
<summary>Major Stages in the RAG Process</summary>

---

- **Retrieval**  
  → Search for relevant data or documents using semantic similarity.

- **Augmentation**  
  → Inject retrieved content into the prompt with clear structure and instructions for the LLM. You can also include re-ranking, filtering, or summarization steps if needed.

- **Generation**  
  → The augmented prompt is passed to an LLM to generate the final response.

---

</details>

---

## 🔄 RAG Workflow Explained
---

### 🧭 RAG Step-by-Step Flow
<details - open>
<summary>Complete RAG Workflow from Input to Output</summary>

---

- **Step 1: 📥 Receive a User Query**  
  The system gets a user input (question or prompt). For example:  
  _"What are the latest advancements in robotics?"_

- **Step 2: 🔍 Search for Relevant Information**  
  The retriever looks for relevant content from external data sources — such as vector databases, documents, or APIs.

- **Step 3: 📚 Retrieve the Best Matching Content**  
  Extract the most relevant chunks or passages based on the query.

- **Step 4: 🧩 Augment the Prompt with Context**  
  Build a new prompt combining:
  - Instructions  
  - Retrieved context  
  - User question

- **Step 5: 🧠 Generate a Response with the LLM**  
  The LLM uses the combined prompt to generate a response that’s more informed and reliable.

- **Step 6: ✅ Return the Answer to the User**  
  The final output is accurate, context-rich, and ideally better than the LLM’s default behavior without retrieval.

---

</details>

---

## 📊 RAG Architecture Diagram
---

### 🖼️ System Architecture
<details - open>
<summary>Visual Overview of RAG System</summary>

---

- The diagram below illustrates the interaction between user input, retriever, context injection, and LLM output:

  ![RAG Architecture](RAG_Architecture.png)

---

</details>

---

## 🧪 Simple RAG Implementation in Python
---

### 🧰 Setup and Code Walkthrough
<details - open>
<summary>Step-by-step Python Code for RAG</summary>

---

- **Install the required libraries**:
```bash
  pip install chromadb sentence-transformers langchain-openai
```
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
      return response.content

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



