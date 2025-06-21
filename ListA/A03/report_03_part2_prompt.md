---
title: a03_prompt_design_and_analysis
---

---
## Prompt Design and Analysis for RAG Systems
---

### 📝 Prompt Template for Naive RAG
<details - open>
<summary>Analysis of the structure and strategy for the Naive RAG prompt</summary>

---

#### 🎯 Overview and Objectives

- **Strategy**: This is a "zero-shot" prompt designed for a simple RAG pipeline (Naive RAG).
  - **"Zero-shot"**: The model must answer the question immediately without any examples.
- **Primary Objectives**:
  - **Enhance Grounding**: Force the Large Language Model (LLM) to rely **strictly** on the provided documents (`context`) to generate an answer.
  - **Mitigate Hallucinations**: Prevent the LLM from inventing information or using unverified external knowledge.
  - **Ensure Consistent Output Formatting**: Provide clear instructions on how to respond, especially when no relevant information is found.

---

#### 🧱 Prompt Structure

- **Code**:
  ```python
  prompt_template = """<role>
  You are an AI assistant specialized in extracting information from documents. Your task is to answer the question based STRICTLY on the provided content.
  </role>

  <input>
      <documents>
      {context}
      </documents>

      <question>
      {question}
      </question>
  </input>

  <task>
  Based on the documents above, your task is to answer the question.
  </task>

  <instructions>
  1. ONLY use information found in the documents
  2. If no relevant information is available, answer exactly: "I don't know"
  3. Return ONLY the answer — nothing else
  </instructions>
  """
  ```

---

#### 🧩 Detailed Component Analysis

- **Rationale for XML Tags (`<role>`, `<input>`, etc.)**:
  - Models like Claude are trained to recognize this structure, which helps them clearly separate the role, input data, and instructions.
- **`<role>`**:
  - **Purpose**: To set the "persona" and context for the LLM. It specifies the AI's role as a specialized information extractor, not a creative conversational assistant.
- **`<input>`**:
  - **Purpose**: To encapsulate all input data, helping the LLM distinguish between background information (`documents`) and the request to be processed (`question`).
  - **`{context}`**: A placeholder for the text chunks retrieved from the vector database. This is the "Augmented" component in RAG.
  - **`{question}`**: A placeholder for the user's query.
- **`<task>`**:
  - **Purpose**: To issue a direct, concise command that reinforces the primary mission, thereby increasing the LLM's focus.
- **`<instructions>`**:
  - **Purpose**: This is the most critical section for controlling the LLM's behavior.
  - **`1. ONLY use information...`**: The core directive for enforcing grounding.
  - **`2. If no relevant information...`**: A critical "guardrail." It provides a safe exit path, preventing the LLM from guessing when data is unavailable.
  - **`3. Return ONLY the answer...`**: Ensures a clean, machine-parsable output, eliminating conversational filler like "Based on the documents you provided...".

---

#### ✅ Advantages and Limitations

- **Advantages**:
  - **High Reliability**: Very effective for factual question-answering (Q&A) tasks.
  - **Easy to Control**: The rigid structure helps minimize undesirable behaviors.
  - **Simplicity**: Straightforward to implement and debug.
- **Limitations**:
  - **Less Flexible**: Not suitable for tasks requiring complex reasoning, information synthesis from multiple sources, or extended conversations.
  - **Dependent on Retrieval Quality**: If the retrieval stage fails, this prompt has no self-correction mechanism.

---
</details>

### 🧠 Prompt Template for the ReAct Framework
<details - open>
<summary>Analysis of the structure and logic for an Agent using the ReAct framework</summary>

---

#### 🎯 Overview and Objectives

- **Strategy**: This prompt is not just for answering a question, but for empowering the LLM to become an **autonomous Agent**.
- **Operating Model**: It teaches the LLM to follow the **Thought -> Action -> Observation** cycle.
- **Primary Objectives**:
  - **Complex Problem Solving**: Allows the LLM to break down a large problem into smaller, sequential steps.
  - **Tool Use**: Grants the LLM the ability to proactively use external tools (e.g., `hybrid_retrieve`, `web_search`) to gather necessary information.
  - **Increase Transparency**: The "Thought" stream reveals the LLM's reasoning process, making it easier for humans to debug and understand its logic.

---

#### 🧱 Prompt Structure

- **Code**:
  ```python
  prompt_template = '''Answer the following questions as best you can. You have access to the following tools:

  {tools}

  Use the following format:

  Question: the input question you must answer
  Thought: you should always think about what to do
  Action: the action to take, should be one of [{tool_names}]
  Action Input: the input to the action
  Observation: the result of the action
  ... (this Thought/Action/Action Input/Observation can repeat N times)
  Thought: I now know the final answer
  Final Answer: the final answer to the original input question. If no relevant information is available, answer exactly: "I don't know". Return ONLY the answer — nothing else. If question is yes/no question, only return 'Yes' or 'No'.

  Begin!

  Question: {input}
  Thought:{agent_scratchpad}'''
  ```

---

#### 🧩 Detailed Component Analysis

- **`{tools}` and `{tool_names}`**:
  - **Purpose**: These are placeholders that a framework (e.g., LangChain) automatically populates.
  - **`{tools}`**: Contains a detailed description of each tool, helping the LLM understand what each tool does.
  - **`{tool_names}`**: Lists the names of the tools the LLM is allowed to choose in the `Action` step.
- **`Use the following format:`**:
  - **Purpose**: This is a "meta-instruction" that defines the entire iterative structure of the reasoning process.
- **`Thought:`**:
  - **Purpose**: Forces the LLM to "think out loud." This is where the LLM analyzes the situation and plans its next action. This is the **"Reasoning"** part of ReAct.
- **`Action:` and `Action Input:`**:
  - **Purpose**: The LLM makes an executive decision: it selects a tool and provides the input for it. This is the **"Acting"** part of ReAct.
- **`Observation:`**:
  - **Purpose**: A placeholder for the result returned from the tool. This information will be used in the next `Thought` step.
- **`... (this ... can repeat N times)`**:
  - **Purpose**: Explicitly indicates that this cycle can be repeated, enabling multi-hop reasoning.
- **`{agent_scratchpad}`**:
  - **Purpose**: A special variable that acts as the agent's "short-term memory." The framework automatically populates this with the history of previous `Thought/Action/Observation` cycles, helping the agent track its progress and avoid repeating mistakes.

---

#### ✅ Strategic Comparison with Naive RAG

- **Information Flow**:
  - **Naive RAG**: A one-way, static flow. Context is provided once.
    - `Retrieve -> Augment -> Generate`
  - **ReAct**: A dynamic and iterative flow. The agent actively seeks context as needed.
    - `(Thought -> Action -> Observation) -> (Thought -> ...) -> Final Answer`
- **Capability**:
  - **Naive RAG**: Best for **Question Answering** based on a fixed set of documents.
  - **ReAct**: Best for **Problem Solving** that requires lookups, synthesis, and complex reasoning.
- **Level of Control**:
  - **Naive RAG**: Provides tight control over the LLM's behavior.
  - **ReAct**: Grants the LLM higher autonomy, which is more powerful but can also lead to unexpected loops or inefficient actions if not designed carefully.

---
</details>

---