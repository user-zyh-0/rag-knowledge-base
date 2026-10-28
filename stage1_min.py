import os
from openai import OpenAI
from dotenv import load_dotenv

from sqlalchemy import create_engine, Column, String, Text, LargeBinary,Integer,DateTime,ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base,relationship

from datetime import datetime
import json

import jieba
import math
from collections import Counter

load_dotenv(override=True)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
LLM_MODEL = os.getenv("LLM_MODEL")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))


emb_client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url=os.getenv("SILICONFLOW_BASE_URL")
)
llm_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)
MYSQL_URL = os.getenv("MYSQL_URL")
engine = create_engine(MYSQL_URL, echo=False)

Base = declarative_base()

class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(String(64), primary_key=True)
    doc_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"),
                    nullable=False, index=True)
    content = Column(Text, nullable=False)
    vector = Column(LargeBinary, nullable=False)

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False, unique=True)
    size_bytes = Column(Integer, nullable=False)
    chunk_count = Column(Integer, nullable=False, default=0)
    created_at  = Column(DateTime, nullable=False, default=datetime.now)

    chunks = relationship("Chunk", cascade="all, delete-orphan", passive_deletes=True)

Session = sessionmaker(bind=engine)


def cosine(a,b):
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x,y in zip(a,b):
        dot += x*y
        na += x*x
        nb += y*y
    return dot / ((na ** 0.5)*(nb ** 0.5))


def add_document(filename, text):
    Base.metadata.create_all(engine)

    # 查重
    with Session() as session:
        if session.query(Document).filter(Document.filename == filename).first():
            return None, f"文档已存在: {filename}"

    chunks = split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    vecs = embed(chunks)
    size_bytes = len(text.encode("utf-8"))
    created_at = datetime.now()

    with Session() as session:
        doc = Document(
            filename=filename,
            size_bytes=size_bytes,
            chunk_count=len(chunks),
            created_at=created_at,
        )
        session.add(doc)
        session.flush()
        doc_id = doc.id

        session.add_all([
            Chunk(
                id=f"{doc_id}_chunk_{i}",
                doc_id=doc_id,
                content = c,
                vector=json.dumps(v).encode("utf-8")
            )
            for i, (c, v) in enumerate(zip(chunks, vecs))
        ])
        session.commit()
        return {
            "id": doc_id,
            "filename": filename,
            "size_bytes": size_bytes,
            "chunk_count": len(chunks),
            "created_at": created_at,
        }, None


def list_documents():
    with Session() as session:
        rows = session.query(Document).order_by(Document.id).all()
        return [{
            "id": r.id,
            "filename": r.filename,
            "size_bytes": r.size_bytes,
            "chunk_count": r.chunk_count,
            "created_at": r.created_at,
        } for r in rows]


def delete_document(doc_id):
    with Session() as session:
        doc = session.get(Document, doc_id)
        if doc is None:
            return None, f"文档不存在: id={doc_id}"

        info = {
            "id": doc.id,
            "filename": doc.filename,
            "chunk_count": doc.chunk_count,
        }
        session.delete(doc)
        session.commit()
        return info, None



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


def retrieve(question, k = 3):
    qvec = embed([question])[0]

    with Session() as session:
        rows = session.query(Chunk).all()

        scored = []
        for c in rows:
            vec = json.loads(c.vector.decode("utf-8"))
            scored.append((cosine(qvec, vec), c.id, c.content))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]


def _tokenize(text):
    text = text.replace("_", " ").replace("-", " ")
    return [w for w in jieba.lcut_for_search(text) if w.strip()]


def bm25_retrieve(question, k = 3):
    k1, b = 1.5,0.75
    q_tokens = _tokenize(question)

    with Session() as session:
        rows = session.query(Chunk).all()
        corpus = [(_tokenize(r.content),r.id,r.content) for r in rows]
    total_docs = len(corpus)
    if total_docs == 0:
        return []

    avgdl = sum(len(toks) for toks, _, _ in corpus) / total_docs

    idf = {}
    for qt in set(q_tokens):
        df = sum(1 for toks, _, _ in corpus if qt in toks)
        idf[qt] = math.log((total_docs - df + 0.5) / (df + 0.5) + 1)

    results = []
    for toks, cid, content in corpus:
        tf = Counter(toks)
        dl = len(toks)
        score = 0.0

        for qt in q_tokens:
            f = tf.get(qt, 0)
            if f == 0:
                continue
            score += idf[qt] * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))

        if score > 0:
            results.append((score, cid, content))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:k]


def hybrid_retrieve(question, k = 3, pool=10, rrf_k=60):
    # pool 应远小于语料规模（生产 50~100）；本项目仅 10 块，pool=全量，
    # 融合退化为名次平均，区分度有限——机制正确，收益需更大语料才能体现

    vec_hits = retrieve(question, k=pool)
    bm25_hits = bm25_retrieve(question, k=pool)
    scores = {}
    contents = {}
    for rank, (_, cid, content) in enumerate(vec_hits, 1):
        scores[cid] = scores.get(cid, 0) + 1 / (rrf_k + rank)
        contents[cid] = content
    for rank, (_, cid, content) in enumerate(bm25_hits, 1):
        scores[cid] = scores.get(cid, 0) + 1 / (rrf_k + rank)
        contents[cid] = content
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, cid, contents[cid]) for cid, score in ranked[:k]]

def ask(question, k=5):
    hits= hybrid_retrieve(question, k)
    print("  [命中的块]", [cid for _, cid, _ in hits])

    context = ""
    for i, (score, cid, content) in enumerate(hits, 1):
        context += f"【资料{i}】{content}\n"

    prompt = f"""请根据以下资料回答问题，只使用资料中的信息，不要编造。
如果资料中没有相关内容，请回答"资料中没有提到"。
如果资料中涉及多个不同的含义或场景，请分别说明，不要只答一个。

资料：
{context}

问题：{question}
"""

    resp = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content


def ask_stream(question, k=5):
    hits = hybrid_retrieve(question, k)
    print("  [命中的块]", [cid for _, cid, _ in hits])

    context = ""
    for i, (score, cid, content) in enumerate(hits, 1):
        context += f"【资料{i}】{content}\n"

    prompt = f"""请根据以下资料回答问题，只使用资料中的信息，不要编造。
如果资料中没有相关内容，请回答"资料中没有提到"。
如果资料中涉及多个不同的含义或场景，请分别说明，不要只答一个。

资料：
{context}

问题：{question}
"""

    stream = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if getattr(delta, "reasoning_content", None):
            yield delta.reasoning_content
        if delta.content:
            yield delta.content


if __name__ == '__main__':
    print(ask("400 是什么意思"))
    print(ask("怎么防止模型乱编"))
    print(ask("top_k 取多少"))
