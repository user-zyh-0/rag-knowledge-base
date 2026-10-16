import os
from openai import OpenAI
from dotenv import load_dotenv

import json
import sqlite3

DB_PATH = "./rag.db"

load_dotenv(override=True)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
LLM_MODEL = os.getenv("LLM_MODEL")

emb_client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url=os.getenv("SILICONFLOW_BASE_URL")
)
llm_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)


def cosine(a,b):
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x,y in zip(a,b):
        dot += x*y
        na += x*x
        nb += y*y
    return dot / ((na ** 0.5)*(nb ** 0.5))


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_text(text, size, overlap):
    return [text[i : i + size] for i in range(0, len(text), size - overlap)]


def embed(texts):
    resp = emb_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [item.embedding for item in resp.data]


def build_index(chunks,vecs):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS chunks")
    cursor.execute("CREATE TABLE chunks ("
                   "id TEXT PRIMARY KEY , "
                   "content TEXT NOT NULL , "
                   "vector BLOB NOT NULL )")
    rows = [
        (f"chunk_{i}", chunk, json.dumps(vec).encode("utf-8"))
        for i, (chunk, vec) in enumerate(zip(chunks, vecs))
    ]

    conn.executemany(
        "INSERT INTO chunks (id, content, vector) VALUES (?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    return len(rows)


def retrieve(question, k = 3):
    qvec = embed([question])[0]

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    rows = cursor.execute("SELECT id, content, vector FROM chunks").fetchall()
    conn.close()

    scored = []
    for cid, content, blob in rows:
        vec = json.loads(blob.decode("utf-8"))
        scored.append((cosine(qvec, vec), cid, content))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]


def ask(question, k=3):
    hits= retrieve(question, k)
    print("  [命中的块]", [cid for _, cid, _ in hits])

    context = ""
    for i, (score, cid, content) in enumerate(hits, 1):
        context += f"【资料{i}】{content}\n"

    prompt = f"""请根据以下资料回答问题，只使用资料中的信息，不要编造。
    如果资料中没有相关内容，请回答"资料中没有提到"。

    资料：
    {context}
    
    问题：{question}
    """

    resp = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content


if __name__ == '__main__':
    # print(read_file("data/test1.txt"))
    # chunks = split_text(read_file("data/test1.txt"), 500, 50)
    # for i, c in enumerate(chunks, 1):
    #     print(f"[{i}] 长度={len(c)}")
    #     print(c)
    #     print("-" * 40)

    # vecs = embed(["番茄炒蛋要大火快炒", "红烧肉要先焯水"])
    # print(vecs[0],vecs[1],vecs[2])

    # text = read_file("data/test1.txt")
    # chunks = split_text(text, 500, 50)
    # vecs = embed(chunks)
    # n = build_index(chunks, vecs)
    # print("库里条数 =", n)
    # print("块数=", len(chunks))

    # for q in ["切块长度和重叠分别设为多少？",
    #           "相似度阈值设的是多少？",
    #           "ZQ41-RAG-DEMO-2026 这个编号在哪？",
    #           "每天建议喝多少水？"]:
    #     print(f"\n【问】{q}")
    #     for score, cid, content in retrieve(q, k=3):
    #         print(f"  {score:.4f}  {cid}  {content[:35]}")

    print("\n" + "=" * 60)
    print("【ask 测试】")
    print("=" * 60)
    for q in ["切块长度和重叠分别设为多少？",
              "相似度阈值设的是多少？",
              "红烧肉怎么做？"]:
        print(f"\n【问】{q}")
        print(f"【答】{ask(q)}")




