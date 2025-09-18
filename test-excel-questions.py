import pandas as pd
import requests
import re
import time
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import OllamaEmbeddings

# === Configuration ===
EXCEL_PATH = "./test-cases.xlsx"
API_URL = "http://localhost:5000/rag-query"
QUESTION_COLUMN = "Question"
ACTUAL_ANSWER_COLUMN = "Actual Answer"
GENERATED_ANSWER_COLUMN = "Generated Answer"
SIMILARITY_COLUMN = "Similarity Score"
ACCURATE_COLUMN = "Accurate"
START_ROW = 0
SIMILARITY_THRESHOLD = 0.7

# === Embedding Model ===
OLLAMA_URL = "http://localhost:11434"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text:latest"
embed_model = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_URL)

# === Utility: Clean Model Output ===
def clean_response(text):
    return re.sub(r"<think>.*?</think>\n?", "", text, flags=re.DOTALL).strip()

# === Utility: Compute Cosine Similarity ===
def compute_similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    embeddings = embed_model.embed_documents([text1, text2])
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return round(float(score), 4)

# === Load Excel File ===
df = pd.read_excel(EXCEL_PATH)

# Ensure required columns exist
for col in [ACTUAL_ANSWER_COLUMN, GENERATED_ANSWER_COLUMN, SIMILARITY_COLUMN, ACCURATE_COLUMN]:
    if col not in df.columns:
        df[col] = ""

# === Before the loop ===
accuracy_count = 0

# === Iterate Through Questions and Compare Answers ===
for idx in df.index[START_ROW:]:
    question = str(df.at[idx, QUESTION_COLUMN]).strip()
    if not question:
        continue

    print(f"\n🔍 Asking (Row {idx}): {question}")

    try:
        response = requests.post(API_URL, json={"question": question}, timeout=60)
        if response.status_code == 200:
            raw_answer = response.text.strip()
            cleaned_answer = clean_response(raw_answer)
            df.at[idx, GENERATED_ANSWER_COLUMN] = cleaned_answer

            if cleaned_answer == "我沒找到相關資料，能換個方式問嗎？":
                df.at[idx, SIMILARITY_COLUMN] = 0.0
                df.at[idx, ACCURATE_COLUMN] = "FALSE"
                print(f"⚠️ No relevant data found. Marked as inaccurate.")
            else:
                actual_answer = str(df.at[idx, ACTUAL_ANSWER_COLUMN]).strip()
                score = compute_similarity(cleaned_answer, actual_answer)
                df.at[idx, SIMILARITY_COLUMN] = score
                df.at[idx, ACCURATE_COLUMN] = "TRUE" if score > SIMILARITY_THRESHOLD else "FALSE"

                # === Accuracy count increment ===
                if df.at[idx, ACCURATE_COLUMN] == "TRUE":
                    accuracy_count += 1

                print(f"🧠 Generated: {cleaned_answer}")
                print(f"📌 Actual:    {actual_answer}")
                print(f"📊 Score:     {score}")
                print(f"✅ Accurate:  {df.at[idx, ACCURATE_COLUMN]}")
        else:
            error_msg = f"Error {response.status_code}"
            df.at[idx, GENERATED_ANSWER_COLUMN] = error_msg
            df.at[idx, SIMILARITY_COLUMN] = 0.0
            df.at[idx, ACCURATE_COLUMN] = "FALSE"
            print(f"❌ {error_msg}")
    except Exception as e:
        exception_msg = f"Exception: {str(e)}"
        df.at[idx, GENERATED_ANSWER_COLUMN] = exception_msg
        df.at[idx, SIMILARITY_COLUMN] = 0.0
        df.at[idx, ACCURATE_COLUMN] = "FALSE"
        print(f"⚠️ {exception_msg}")

    # Save after each response
    df.to_excel(EXCEL_PATH, index=False)
    time.sleep(0.5)

print(f"\n✅ All responses and similarity scores saved to {EXCEL_PATH}")
print(f"🔢 Accurate count: {accuracy_count} / {len(df.index[START_ROW:])}")