# LangChain + LCEL Beginner Notes (No Magic Version)

This file explains what your RAG code is doing, line by line in plain language.
Assume zero prior knowledge.

---

## 1) Big Picture: What You Already Built

You have 2 stages:

1. **Indexing stage** (offline/prep)
   - Load text
   - Split into chunks
   - Create embeddings
   - Store in vector DB (FAISS/Chroma)

2. **Query stage** (runtime/chat)
   - User asks question
   - Retriever fetches relevant chunks
   - LLM answers using those chunks

You said you are clear till embeddings. Great. We focus on stage 2.

---

## 2) What Is a Retriever?

When you do:

```python
retriever = vectorstore.as_retriever()
```

you are creating a wrapper object with a standard method (`invoke` / `get_relevant_documents`) so LangChain can call it uniformly.

Internally, at runtime, this wrapper calls vector store search functions like:
- similarity search
- top-k selection

So yes, your intuition is correct:
`as_retriever()` means "convert vector DB into a retrieval interface".

---

## 3) Why `create_history_aware_retriever` Needs 3 Args

```python
history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    condense_prompt
)
```

It needs 3 things because it builds a mini pipeline:

1. **`llm`**
   - used to rewrite follow-up question into standalone search query

2. **`retriever`**
   - used to fetch documents after rewrite

3. **`condense_prompt`**
   - instructions for rewrite behavior
   - example: "Given history + question, output one standalone search query."

### Under the hood conceptually

It creates something like:

```text
(chat_history + input question)
        |
        v
  condense_prompt + llm
        |
        v
standalone query string
        |
        v
    retriever.invoke(query)
        |
        v
   relevant documents
```

So it is not black magic, just a pre-chain + retriever.

---

## 4) LCEL in Simple Words

LCEL = LangChain Expression Language.
It lets you compose blocks using `|` (pipe), like Unix pipelines.

Example:

```python
chain = prompt | llm | StrOutputParser()
```

means:
1. Fill prompt
2. Send to LLM
3. Parse result as string

For dict-style LCEL:

```python
{
  "context": some_chain,
  "question": RunnableLambda(...)
} | rag_prompt | llm | parser
```

means:
- Build a dictionary first (each key can come from different subchains),
- then feed that dictionary into `rag_prompt`.

---

## 5) End-to-End Runtime Flow (With History-Aware Retriever)

Use this mental model for each user message:

```text
User question
   |
   v
Input dict arrives:
{"question": "...", "chat_history": [...]}
   |
   +--> retriever_input_adapter
   |      (renames question -> input)
   |      output: {"input": "...", "chat_history": [...]}
   |
   +--> history_aware_retriever
   |      step A: condense_prompt + llm => standalone query
   |      step B: retriever => top-k docs
   |
   +--> format_docs => one context string
   |
   v
rag_prompt gets:
- context
- chat_history
- question
   |
   v
llm answers
   |
   v
string output
```

---

## 6) "How Is chat_history Filled?"

Important: `create_history_aware_retriever` does **not** auto-store history by itself.

There are 2 patterns:

### A) Manual history (beginner simple)

You maintain a Python list.

```python
from langchain_core.messages import HumanMessage, AIMessage

chat_history = []

q1 = "What is RAG?"
a1 = rag_chain.invoke({"question": q1, "chat_history": chat_history})
chat_history.extend([HumanMessage(content=q1), AIMessage(content=a1)])

q2 = "How does it help?"
a2 = rag_chain.invoke({"question": q2, "chat_history": chat_history})
```

Here **you** fill it after every turn.

### B) Automatic history (better app design)

Use `RunnableWithMessageHistory` + session id.
Then wrapper stores user/AI messages per session.

---

## 7) Why Follow-Up Questions Fail Without History-Aware Retrieval

Conversation:
- Q1: "What is RAG?"
- Q2: "How does it help?"

If retriever sees only Q2, it searches for "how does it help" (ambiguous).
So chunks may be wrong.

History-aware rewrite converts Q2 into:
- "How does RAG help?"

Now retrieval quality improves.

---

## 8) Behind-the-Scenes Call Order in Your RAG Chain

Given your chain structure, this is the practical call order:

1. User calls `rag_chain.invoke({...})`
2. LCEL evaluates dict block keys:
   - `context` branch executes first:
     - adapter -> history-aware retriever -> format_docs
   - `question` branch copies question
   - `chat_history` branch passes history
3. Combined dict goes into `rag_prompt`
4. Prompt goes to `llm`
5. Output parser returns plain string

That is the full execution path.

---

## 9) Minimal Mental Model (Memorize This)

RAG chat with history = 3 mini pipelines:

1. **Rewrite pipeline**: history + question -> standalone query  
2. **Retrieve pipeline**: standalone query -> top-k docs  
3. **Answer pipeline**: docs + history + question -> final answer

If you remember only this, you can explain it to anyone.

---

## 10) Practical Debug Tips

When things feel magical, print intermediate outputs:

- Print rewritten standalone query (from condense step)
- Print retrieved docs before answer
- Print final prompt text (or prompt variables)

If each step looks correct, system will be correct.

---

## 11) One Clean Example (Pseudo-LCEL)

```python
# 1) Rewrite follow-up question
standalone_query = (condense_prompt | llm | StrOutputParser()).invoke(
    {"chat_history": chat_history, "input": question}
)

# 2) Retrieve docs using rewritten query
docs = retriever.invoke(standalone_query)
context = "\n\n".join(d.page_content for d in docs)

# 3) Answer with context + history
answer = (rag_prompt | llm | StrOutputParser()).invoke(
    {
        "context": context,
        "chat_history": chat_history,
        "question": question,
    }
)
```

This is almost exactly what your LCEL chain is doing, just expanded manually.

---

## 12) Final Answer to Your Core Doubt

You are not missing fundamentals. You are at the exact stage where abstraction feels "magical".
The solution is to view LangChain as **small pipes connected together**, not as one giant black box.

Once you can explain:
- rewrite -> retrieve -> answer
- and who stores chat history,

you already understand the core architecture.

