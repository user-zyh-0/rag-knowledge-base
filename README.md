# RAG 知识库问答系统

基于 **FastAPI + MySQL** 的最小可用 RAG（检索增强生成）实现。上传文档后自动切分、向量化入库；提问时检索最相关的片段交给大模型生成答案，并强制模型只依据检索到的资料作答。

> 已完成文档管理（上传 / 列表 / 删除）与问答接口（完整答案 / 流式逐字输出）。

## 功能

- **文档管理**：上传（自动切分 + 向量化）、列表、删除
- **语义检索**：余弦相似度全表扫描，返回最相关的 Top-K 片段
- **防幻觉问答**：Prompt 约束模型只使用检索结果，资料不足时明确回答"资料中没有提到"
- **流式输出**：`/chat/stream` 逐字返回，首字延迟约 1 秒（见「设计说明」）
- **数据一致性**：删除文档时通过数据库外键级联，自动清除其全部向量块

## 技术栈

| 层 | 选型 |
|---|---|
| Web 框架 | FastAPI 0.142 + Pydantic 2 |
| ORM / 数据库 | SQLAlchemy 2.1 + MySQL 8（utf8mb4） |
| Embedding | 硅基流动 `BAAI/bge-m3`（1024 维，免费） |
| LLM | DeepSeek `deepseek-flash` |

## 快速开始

### 1. 准备环境

- Python 3.10+（开发环境为 3.13）
- MySQL 8.0（需提前创建数据库）

```sql
CREATE DATABASE rag_db DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

### 2. 安装依赖

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. 配置环境变量

复制 `.env.example` 为 `.env`，填入自己的 Key：

```ini
MYSQL_URL=mysql+pymysql://root:你的密码@localhost:3306/rag_db?charset=utf8mb4

DEEPSEEK_API_KEY=sk-xxx
SILICONFLOW_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1

LLM_MODEL=deepseek-flash
EMBEDDING_MODEL=BAAI/bge-m3
```

- DeepSeek Key：https://platform.deepseek.com/api_keys
- 硅基流动 Key：https://cloud.siliconflow.cn/account/ak

> `MYSQL_URL` 中的 `+pymysql` 与 `?charset=utf8mb4` 都不能省略。密码含 `@ : / # %` 时需 URL 编码（如 `@` → `%40`）。

### 4. 启动

```bash
python main.py
```

打开 http://127.0.0.1:8000/docs ，在自动生成的 Swagger 页面里直接调用接口。

## 接口

| 方法 | 路径 | 说明 | 状态码 |
|---|---|---|---|
| `POST` | `/documents` | 上传文档（JSON：`filename` + `content`） | 201 / 409 重名 / 422 参数非法 |
| `GET` | `/documents` | 列出全部文档 | 200 |
| `DELETE` | `/documents/{doc_id}` | 删除文档及其全部向量块 | 200 / 404 不存在 / 422 非整数 |
| `POST` | `/chat` | 问答，返回完整答案 | 200 / 422 参数非法 |
| `POST` | `/chat/stream` | 问答，流式逐字返回 | 200 / 422 参数非法 |

调用示例：

```bash
curl -X POST http://127.0.0.1:8000/documents \
  -H "Content-Type: application/json" \
  -d '{"filename": "计算机网络.txt", "content": "第一章：计算机网络体系结构……"}'
```

流式问答（加 `-N` 关闭 curl 缓冲才能看到逐字效果）：

```bash
curl -N -X POST http://127.0.0.1:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "切块长度和重叠分别设为多少？"}'
```

## 项目结构

```
RAGKnowledgeBase/
├── main.py              # FastAPI 接口层：路由 + 请求校验 + 错误码映射
├── stage1_min.py        # 核心逻辑：切分 / 向量化 / 存储 / 检索 / 生成
├── requirements.txt
├── .env.example         # 环境变量样板（不含真实密钥）
├── README.md
└── data/
    └── test1.txt        # 示例文档
```

`main.py` 只负责把 HTTP 请求翻译成对 `stage1_min` 的调用，业务逻辑全部集中在 `stage1_min.py`。

## 数据模型

```
documents (1) ──< chunks (N)
```

- `documents`：文档元信息（文件名、字节数、块数、创建时间），`filename` 唯一
- `chunks`：文本块 + 1024 维向量（`BLOB` 存 JSON 序列化后的浮点数组）
- 外键 `chunks.doc_id → documents.id`，带 `ON DELETE CASCADE`

删除采用 `cascade="all, delete-orphan"` + `passive_deletes=True` + 数据库外键级联三者配合：ORM 不加载子对象，只发一条 `DELETE`，由数据库完成级联。相比默认行为（先 `SELECT` 所有子行再逐条 `DELETE`），文档块数越多差距越大。

## 设计说明

**为什么暂时不用向量数据库**

当前 10 块规模下，纯 Python 余弦全表扫描单次约 0.9 ms；按线性外推，1 万块约 0.7 s、10 万块约 7 s。万级以下引入 ANN 索引收益为负，且实测 10 块时 numpy 向量化反而更慢（数组创建开销大于计算收益）。计划在块数超过 10 万时再引入索引层，届时只需替换 `retrieve()` 内部实现，对外接口不变。

**为什么向量直接存 MySQL**

`bge-m3` 输出已归一化（模长实测 1.000000），余弦相似度等价于点积，无需额外索引结构。每块向量约 22 KB，`BLOB` 上限 64 KB，当前占用约 35%。若改用更高维的 Embedding 模型（如 3072 维），需同步升级为 `MEDIUMBLOB`。

**防幻觉策略**

- Prompt 显式约束"只使用资料中的信息，不要编造"
- 资料不足时要求回答"资料中没有提到"
- 检索结果按分数排序后取 Top-3，附带来源块 ID 便于排查

**流式输出：推理模型的 `reasoning_content` 不能丢**

`deepseek-flash` 是推理模型，流式 delta 里有两个字段：`reasoning_content`（思考过程）和 `content`（正式回答）。实测发现，**思考阶段 `content` 恒为 `None`** —— 若只转发 `content`，首字延迟就等于整个思考时长（实测 17s 以上，期间屏幕一片空白，看起来像卡死）。

把思考过程一并 `yield` 出去之后，首字延迟降到约 1s，且思考内容本身会滚动十几秒，流式效果一眼可见。

> **TTFT（首字延迟）才是流式体验的核心指标，不是总耗时。** 用户感知的是"多久开始出字"，而不是"多久全部出完"。

## 进度

- [x] 阶段一：最小闭环（切分 → 向量化 → 存储 → 检索 → 生成）
- [x] 阶段二：文档管理（MySQL + SQLAlchemy + FastAPI CRUD）
- [x] 阶段三：流式输出（`/chat` 完整答案 + `/chat/stream` 逐字输出 + 推理内容透传）
- [ ] 阶段四：检索优化（BM25 + RRF 混合检索、Query 改写、切分策略对比）
- [ ] 阶段五：多轮对话（`session_id` + 对话历史）
- [ ] 阶段六：前端界面
