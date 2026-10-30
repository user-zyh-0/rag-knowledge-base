import os
import re

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


def add_document(filename, text, chunker=None):
    Base.metadata.create_all(engine)

    # 查重
    with Session() as session:
        if session.query(Document).filter(Document.filename == filename).first():
            return None, f"文档已存在: {filename}"

    # chunks = split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    chunks = chunker(text) if chunker else split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
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


def split_by_heading(text, max_size=800):
    parts = re.split(r"(?=^#{1,3} )", text, flags=re.M)
    chunks = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) <= max_size:
            chunks.append(p)
        else:
            for i in range(0, len(p), max_size):
                chunks.append(p[i : i + max_size])
    return chunks


def _is_empty_heading(chunk):
    """判断是不是只有标题和分隔线、没有正文的空壳块"""
    body = re.sub(r"^#{1,6} .*$", "", chunk, flags=re.M)   # 去掉标题行
    body = re.sub(r"^=+$", "", body, flags=re.M)           # 去掉 ==== 分隔线
    return len(body.strip()) < 20


def merge_empty_headings(chunks):
    """把空壳块合并到下一块，保留章节上下文"""
    merged = []
    carry = ""
    for c in chunks:
        if _is_empty_heading(c):
            carry += c + "\n"          # 攒起来，等下一块
        else:
            merged.append((carry + c).strip())
            carry = ""
    if carry:                          # 结尾还有没合并的
        merged.append(carry.strip())
    return merged


def split_smart(text, size=500, overlap=50):
    if re.search(r"^#{1,6} ", text, flags=re.M):
        return merge_empty_headings(split_by_heading(text, max_size=size * 2))
    return split_text(text, size, overlap)


def embed(texts):
    resp = emb_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [item.embedding for item in resp.data]


def retrieve(question, k=3, doc_id=None):
    qvec = embed([question])[0]

    with Session() as session:
        # rows = session.query(Chunk).all()
        query = session.query(Chunk)
        if doc_id is not None:
            query = query.filter(Chunk.doc_id == doc_id)
        rows = query.all()

        scored = []
        for c in rows:
            vec = json.loads(c.vector.decode("utf-8"))
            scored.append((cosine(qvec, vec), c.id, c.content))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]


def _tokenize(text):
    text = text.replace("_", " ").replace("-", " ")
    return [w for w in jieba.lcut_for_search(text) if w.strip()]


def bm25_retrieve(question, k=3, doc_id=None):
    k1, b = 1.5,0.75
    q_tokens = _tokenize(question)

    with Session() as session:
        query = session.query(Chunk)
        if doc_id is not None:
            query = query.filter(Chunk.doc_id == doc_id)
        rows = query.all()

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


def hybrid_retrieve(question, k=3, pool=10, rrf_k=60, doc_id=None):
    # pool 应远小于语料规模（生产 50~100）；本项目仅 10 块，pool=全量，
    # 融合退化为名次平均，区分度有限——机制正确，收益需更大语料才能体现

    vec_hits = retrieve(question, k=pool, doc_id=doc_id)
    bm25_hits = bm25_retrieve(question, k=pool, doc_id=doc_id)
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


CORPUS_HINT = """资料是一份后端与 AI 学习笔记，涉及：HTTP 状态码、数据库表结构、SQLite 类型严格性、
RAG 完整链路、切分参数（chunk_size / chunk_overlap）、检索参数（top_k / 相似度阈值）、
Python 可变默认参数、算法练习、开发工具记录。"""


def rewrite_query(question,n=2):
    prompt = f"""资料主题：{CORPUS_HINT}

把下面的用户问题改写成 {n} 个**用于检索的关键词组合**，要求：
1. 输出关键词或短语，**不要写成完整问句**，去掉"多少/怎么/如何/咋/吗"这类疑问词
2. **必须使用资料主题里出现过的术语**，不要用同义词替换
3. 每行一个，不要编号、不要解释

用户问题：{question}
"""


    resp = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    text = resp.choices[0].message.content
    return [line.strip() for line in text.split("\n") if line.strip()][:n]


def multi_retrieve(question, k=5, n = 2):
    queries = [question] + rewrite_query(question,n)

    scores = {}
    contents = {}
    for q in queries:
        hits = hybrid_retrieve(q, k=k)
        for rank, (_, cid, content) in enumerate(hits, 1):
            scores[cid] = scores.get(cid, 0) + 1 / (60 + rank)
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


CASES = [
    ("切片切多大合适",       "chunk_size"),
    ("top_k 取多少",         "top_k"),
    ("相似度阈值设多少",     "0.58"),
    ("数据库主键用什么类型", "INTEGER"),
    ("怎么防止模型乱编",     "编造"),
    ("RAG 的完整链路是什么", "向量化"),
]


def hit_rate(doc_id, k=3):
    """命中率：top-k 结果里有没有含答案关键词"""
    hits = []
    for q, kw in CASES:
        results = hybrid_retrieve(q, k=k, doc_id=doc_id)
        ok = any(kw in c for _, _, c in results)
        hits.append(ok)
    return sum(hits) / len(hits), hits


if __name__ == "__main__":
    text = read_file("data/test1.txt")
    info, err = add_document("test1_heading.txt", text, chunker=split_smart)
    print("入库:", info, err)

    for name, did in [("固定长度", 4), ("按标题", info["id"] if info else None)]:
        if did is None:
            continue
        rate, detail = hit_rate(did, k=3)
        print(f"{name:8s} doc_id={did}  命中率={rate:.0%}  {detail}")
