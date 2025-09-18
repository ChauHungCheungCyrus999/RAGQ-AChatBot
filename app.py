import os
import re
import sys
import langdetect
from flask import render_template, Flask, request, jsonify, Response, stream_with_context
from ollama import Client
from langchain.docstore.document import Document
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from train_vector_db import DB_SAVE_PATH


app = Flask(__name__)
session_state = {}  # Dictionary to maintain session-level state (prompt history, vectordb, etc.)


# ====== Vector DB Loader ======
def load_vectordb():
    # Loads the FAISS vector DB from disk using the embedding model.
    if os.path.exists(DB_SAVE_PATH):
        return FAISS.load_local(DB_SAVE_PATH, embed_model, allow_dangerous_deserialization=True)
    else:
        # Raise error if FAISS database is missing, forcing user to generate it first.
        raise RuntimeError(
            f"FAISS vector DB not found at {DB_SAVE_PATH}. Please run train_vector_db.py once before starting the Flask app."
        )


# ====== Utility to Clean Model Output ======
def clean_response(response_text):
    # Remove <think>...</think> blocks (internal reasoning from model, if leaked)
    no_think = re.sub(r"<think>.*?</think>\n?", "", response_text, flags=re.DOTALL)
    # Remove everything up to and including the first "A:" marker
    cleaned = re.sub(r'^.*?A:\s*', '', no_think, flags=re.DOTALL)
    return cleaned.strip()


# ====== Model Configuration ======
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")  # Ollama server endpoint
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen3:32b")        # Model for answering queries
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest")  # Model for embedding


print(f"""
OLLAMA_URL: {OLLAMA_URL}
OLLAMA_LLM_MODEL: {OLLAMA_LLM_MODEL}
OLLAMA_EMBEDDING_MODEL: {OLLAMA_EMBEDDING_MODEL}
""")


# Instantiate LLM model and embedding model objects
llm = OllamaLLM(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_URL)
embed_model = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_URL)


# ====== Load Vector DB at Startup ======
vectordb = load_vectordb()  # Load FAISS index into memory
session_state["vectordb"] = vectordb

"""# Add all documents into initial prompt for richer context
initial_prompt = [{
    "role": "system",
    "content": "initialize",
}]
    Contetstr = doc.page_content.encode('utf-8', errors='ignore').decode('utf-8')
    print(f"doc_id{doc_id}: {Contetstr}")
    initial_prompt.append({"role": "system", "content": Contetstr})
    
session_state["prompt"] = initial_prompt  # Save conversation initialization
"""

# ====== System Prompt Template ======
with open("system-prompt.txt", "r", encoding="utf-8") as f:
    system_message = f.read()

sys.stdout.reconfigure(encoding='utf-8')  # ensure UTF-8 encoding for console output

# Access FAISS document store for debugging / prompt context
docs = vectordb.docstore._dict  

# Add all documents into initial prompt for richer context
"""
for doc_id, doc in docs.items():
    Contetstr = doc.page_content.encode('utf-8', errors='ignore').decode('utf-8')
    print(f"doc_id{doc_id}: {Contetstr}")
    initial_prompt.append({"role": "system", "content": Contetstr}) """

# ====== Ensure Prompt History Is Available ======
if "prompt" not in session_state:
    session_state["prompt"] = []
prompt = session_state.get("prompt")


# ====== Routes / API ======
@app.route('/')
def home():
    # Home route to render web UI (index.html must exist)
    return render_template('index.html')


@app.route('/rag-query', methods=['POST'])
def rag_query():
    # Endpoint for answering RAG queries
    data = request.get_json()
    question = data.get("question")
    if not question:
        return jsonify({"error": "No question found"}), 400
    
    # Shortcut replies for simple greetings or test cases
    if len(question.strip()) == 1 or question.lower() in ['hi', 'hello', 'test']:
        def stream_greeting():
            yield "您好！如有任何關於判頭通訊網的使用查詢，歡迎隨時向我提問。"
        return Response(stream_with_context(stream_greeting()), mimetype='text/plain')
        

    vectordb = session_state.get("vectordb", None)
    if not vectordb:
        return jsonify({"error": "No vector database found"}), 500
    print(question)

    # Perform similarity search against FAISS vector DB (retrieve top-1 most relevant doc)
    search_results = vectordb.similarity_search_with_relevance_scores(query=question, k=30)
    question_results = []
    for doc, score in search_results:
        question_results.append((doc, score))
        print(doc)

    TopK = 10
    top_questions = question_results[:TopK]

    docs_by_id_type = {}
    for doc in vectordb.docstore._dict.values():
        type_ = doc.metadata.get("type")
        id_ = doc.metadata.get("id")
        lang_ = doc.metadata.get("lang")
        
        key = (type_, id_, lang_)
        docs_by_id_type[key] = doc

    initial_prompt = [{
        "role": "system",
        "content": "initialize",
    }]

    # Detect question language
    if(langdetect.detect(question) == "en"):
        use_lang = "en"
    else:
        use_lang = "cn"

    for i in range(1, TopK + 1):
        initial_prompt.append({"role": "system", "content": "initial"})


    full_qa_texts = []
    count = 1
    for doc, score in top_questions:
        qid = doc.metadata.get("id")
        print(f"vro THE LANUAGE U ARE USING IS: {use_lang}")
        q_text = docs_by_id_type.get(( "question", qid, use_lang), Document(page_content="")).page_content
        a_text = docs_by_id_type.get(("answer", qid, use_lang), Document(page_content="")).page_content


        print(f"Q: {q_text}\nA: {a_text}")
        full_qa_texts.append(f"Q: {q_text} A: {a_text}")

        initial_prompt[count] = {"role": "system", "content": f"Q: {q_text} A: {a_text}"}
        session_state["prompt"] = initial_prompt  # Save conversation initialization
        count += 1
    
    
    session_state["prompt"] = initial_prompt  # Save conversation initialization
    retrieved_answers_text = "\n".join(full_qa_texts)
    print(f"retrieved_answers_text: {retrieved_answers_text}")

    
    # session_state["prompt"] = initial_prompt  # Save conversation initialization
    # Update system role prompt dynamically with retrieved context
    prompt = session_state.get("prompt", [])
    if prompt and len(prompt) > 0:
        prompt[0] = {
            "role": "system",
            "content": system_message.format(retrieved_answers_text=retrieved_answers_text)
        }
    else:
        prompt = [{
            "role": "system",
            "content": system_message.format(retrieved_answers_text=retrieved_answers_text)
        }]
    session_state["prompt"] = prompt

    # Append current user query into conversation history
    prompt.append({"role": "user", "content": question})

    # Debug: Print out current conversation state
    print("==================================== Debug(Prompt Checking) ====================================")
    for i, entry in enumerate(prompt):
        print(f"Entry {i}: Role = {entry.get('role')}, Content = {entry.get('content')}")

    # Streaming generator for model response
    def generate_stream():
        result = ""
        ollama = Client(host=OLLAMA_URL)
        # Use chat API with stream=True to yield partial responses
        for chunk in ollama.chat(model=OLLAMA_LLM_MODEL, messages=prompt, stream=True):
            text = chunk.get("message", "")
            if text:
                piece = str(text["content"])
                result += piece
                yield piece

        # Append final assistant response into history
        # prompt.append({"role": "assistant", "content": result})
        session_state["prompt"] = prompt

    # Return streaming response to client
    return Response(stream_with_context(generate_stream()), mimetype='text/plain')


# ====== Entry Point ======
if __name__ == "__main__":
    app.run(debug=True, port=5000)  # Start Flask app on port 5000 in debug mode