import os
import mysql.connector
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.docstore.document import Document
from flask import Flask
from langchain.text_splitter import RecursiveCharacterTextSplitter

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest")
embed_model = OllamaEmbeddings(
    model=OLLAMA_EMBEDDING_MODEL,
    base_url=OLLAMA_URL
)
DB_SAVE_PATH = "faq_faiss_index"

app = Flask(__name__)

def get_db_connection():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="vendor-portal-faq-bot"
    )
    print("DB connect sucessfully!")
    return conn

def fetch_text_from_db():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT HEADER_ID, QUESTION_CN, QUESTION_EN, ANSWER_CN, ANSWER_EN FROM `faq`")
    rows = cursor.fetchall()
    docs = []
    for row in rows:
        # Create separate documents for questions and answers in both languages
        q_cn = Document(page_content=row['QUESTION_CN'], metadata={"type": "question", "id": row['HEADER_ID'], "lang": "cn"})
        a_cn = Document(page_content=row['ANSWER_CN'], metadata={"type": "answer", "id": row['HEADER_ID'], "lang": "cn"})
        q_en = Document(page_content=row['QUESTION_EN'], metadata={"type": "question", "id": row['HEADER_ID'], "lang": "en"})
        a_en = Document(page_content=row['ANSWER_EN'], metadata={"type": "answer", "id": row['HEADER_ID'], "lang": "en"})
        docs.extend([q_cn, a_cn, q_en, a_en])
    """    for row in rows:
            content = f"Q: {row['QUESTION_CN']} + A: {row['ANSWER_CN']}. Q: {row['QUESTION_EN']} +  A: {row['ANSWER_EN']}"
            docs.append(Document(
                page_content=content,
                metadata={"id": row['HEADER_ID']}
            )) """
    cursor.close()
    conn.close()
    return docs

def chunk_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=128
    )
    chunked = []
    for doc in docs:
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            chunked.append(
                Document(
                    page_content=chunk,
                    metadata={**doc.metadata, "chunk": i}
                )
            )
    return chunked

def build_and_save_faiss():
    print("Fetching data from MySQL...")
    docs = fetch_text_from_db()
    print(f"Fetched {len(docs)} FAQ items.")
    #chunked_docs = chunk_docs(docs)  # Uncomment to use chunking
    print("Building FAISS index...")
    db = FAISS.from_documents(docs, embed_model)
    db.save_local(DB_SAVE_PATH)
    print(f"Vector DB saved to: {DB_SAVE_PATH}")

if __name__ == "__main__":
    build_and_save_faiss()