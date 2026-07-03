"""
Ask RAG — Interactive AWS Security Q&A
-----------------------------------------
This script is the "use" step of the RAG pipeline. It does NOT re-embed
anything — it reconnects to the vector database already built by
build_vectorstore.py, and lets you ask questions interactively from the
terminal.

For each question, it:
  1. Embeds your question and searches the vector store for the most
     similar chunks from the AWS Security Pillar whitepaper
  2. Passes those chunks to the LLM as context
  3. The LLM answers using ONLY that retrieved context (not its own
     general training knowledge), and says "I don't know" if the
     context doesn't cover the question

Note: this version has no memory between questions — each question is
answered independently. A natural next step is adding conversation
history so follow-up questions like "explain that more" work.
"""

from dotenv import load_dotenv
load_dotenv()  # loads OPENAI_API_KEY from .env

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

# ---------------------------------------------------------------------------
# Reconnect to the EXISTING vector store (built earlier by build_vectorstore.py)
# ---------------------------------------------------------------------------
# Important: this does NOT re-run embeddings or cost any API calls by itself.
# It just opens the ./chroma_db folder that already has the embedded chunks
# saved on disk.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")

# The LLM used to generate the final natural-language answer
llm = ChatOpenAI(model="gpt-5.4-mini")


def ask(question: str) -> str:
    """
    Given a question, retrieve the most relevant chunks from the AWS
    Security whitepaper and ask the LLM to answer using only that context.
    """
    # Step 1: Similarity search — find the top 3 chunks most related in
    # MEANING to the question (not just matching keywords).
    results = vectorstore.similarity_search(question, k=3)

    # Step 2: Combine the retrieved chunks into a single block of text
    # to hand to the model as context.
    context = "\n\n".join([doc.page_content for doc in results])

    # Step 3: Build the prompt. The two key instructions here are what
    # make this "RAG" rather than just a regular chat prompt:
    #   - "using only the context below" forces the model to rely on the
    #     retrieved document content instead of its own general knowledge
    #   - "say you don't know" reduces hallucination when the retrieved
    #     chunks don't actually contain the answer
    prompt = f"""Answer the question using only the context below. 
If the context doesn't contain the answer, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

    # Step 4: Send the prompt to the LLM and return its answer text.
    response = llm.invoke(prompt)
    return response.content


# ---------------------------------------------------------------------------
# Interactive terminal loop
# ---------------------------------------------------------------------------
print("AWS Security RAG Assistant — ask a question, or type 'exit' to quit.\n")

while True:
    question = input("You: ").strip()

    # Let the user exit cleanly instead of forcing Ctrl+C
    if question.lower() in ("exit", "quit"):
        print("Goodbye!")
        break

    # Skip empty input (e.g. pressing Enter with nothing typed)
    if not question:
        continue

    answer = ask(question)
    print(f"\nAssistant: {answer}\n")
