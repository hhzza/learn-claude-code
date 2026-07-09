# s021 RAG 知识库系统

基于 s20 Comprehensive Agent 扩展的混合检索增强生成（RAG）系统。

## 架构总览

```
                          s021_rag/code.py (agent 主循环)
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
              index_documents  search_knowledge   KNOWLEDGE.md
              (索引文档)        (检索知识)       (系统提示词清单)
                    │             │
                    ▼             ▼
              ┌─────────────────────────────────┐
              │         rag/retriever.py         │
              │         RRF 混合检索             │
              │     ┌──────────┬──────────┐      │
              │     │  BM25    │  向量    │      │
              │     │ (稀疏)   │ (稠密)   │      │
              │     └────┬─────┴────┬─────┘      │
              └──────────┼──────────┼────────────┘
                         │          │
              ┌──────────┼──────────┼────────────┐
              │  bm25_index.py      │            │
              │  (手写算法)   vector_store.py     │
              │               (numpy + 余弦)      │
              └──────────────────────────────────┘
                                    │
                              embedder.py
                         (百炼 text-embedding-v4)

  加载 & 切分:
  ┌─────────────────────────────────────────────┐
  │  loader.py          │    chunker.py          │
  │  Docling PDF 解析   │    分格式切分策略      │
  │  + MD/PY/YAML/TXT   │    PDF/MD/PY/Config    │
  └─────────────────────────────────────────────┘
```

## 新增模块

### `rag/__init__.py`
两个数据类：
```python
@dataclass
class Document:
    path: Path           # 源文件绝对路径
    format: str          # "pdf" | "markdown" | "python" | "text" | "json" | "yaml"
    text: str            # 提取的纯文本
    metadata: dict       # 格式相关元数据 (sections, tables, headings...)

@dataclass
class Chunk:
    id: str              # "paper.pdf#c3"
    text: str            # chunk 完整文本
    source: Path         # 来源文件路径
    chunk_index: int     # 在源文件中的序号
    start_line: int      # 起始行号
    end_line: int        # 结束行号
    heading: str         # 最近的标题
    metadata: dict       # 格式标签
```

### `rag/loader.py`

文档加载 + 格式检测 + 文本提取。

**支持的格式：**

| 格式 | 提取方式 | 元数据 |
|------|---------|--------|
| `.pdf` | **Docling** — IBM 文档理解工具 | sections, tables, language |
| `.md` | 原样保留 | headings 数量 |
| `.py` | 保留完整源码 | 函数/类列表 |
| `.json` | key-value 展平 | keys 数量 |
| `.yaml` | 同上 | keys 数量 |
| `.txt` | 原样 | 行数 |

**Docling PDF 解析流程：**
```
PDF → 渲染为图片 → docling-layout-heron (RT-DETR-v2, 164MB)
                 → 检测 17 类区域 (title/section_header/text/table/...)
                 → 计算阅读顺序 (跨双栏)
                 → table → tableformer (342MB) → 表格结构提取
                 → 文字层直接提取 (无需 OCR)
                 → doc.export_to_markdown()
```

**缓存机制：**
- 解析结果存为 `paper.pdf.cache.json`，放在 PDF 旁边
- 以 PDF 的 `mtime` 作为版本标记，修改后自动失效重建
- 首次处理 ~4 分钟，缓存命中 ~毫秒

**关键函数：**
```python
load_file(path, do_ocr=False, do_table_structure=True) -> Document | None
load_directory(root, patterns=["*.pdf","*.md"], recursive=True) -> list[Document]
```

### `rag/chunker.py`

分格式文档切分策略，目标每 chunk ~2000 字符（~500 tokens）。

**切分策略：**

| 格式 | 策略 | 合并 | 拆分 |
|------|------|------|------|
| **pdf** | 使用 Docling 预解析的 sections 作为边界 | 短节 (<600 chars) 向前合并 | 大节 (>4000 chars) 按段落拆 |
| **markdown** | 按 `##`/`###` 标题切分 | 同上 | 先按 `###` 子标题拆，再按段落 |
| **python** | AST 解析 → 按函数/类分组 | 小函数打包 (≤2000 chars) | 大类的按方法拆 |
| **json/yaml** | 每个顶层 key 一个 chunk | — | — |
| **text** | 段落感知固定窗口 | — | 窗口 2000 chars + 重叠 400 chars |

**关键函数：**
```python
chunk_one(doc: Document) -> list[Chunk]
chunk_documents(docs: list[Document]) -> list[Chunk]
```

### `rag/embedder.py`

百炼 `text-embedding-v4` 向量化封装。

- **模型**: `text-embedding-v4`，维度 1024
- **调用方式**: OpenAI 兼容模式 (`https://dashscope.aliyuncs.com/compatible-mode/v1`)
- **批量**: 每批 ≤10 条（API 限制），自动分批
- **重试**: 3 次指数退避

```python
embedder = Embedder(api_key="sk-xxx")
vec = embedder.embed_query("什么是知识蒸馏")     # (1024,) ndarray
vecs = embedder.embed(["文本1", "文本2"])        # (N, 1024) ndarray
vecs = embedder.embed_chunks(chunks)             # 直接处理 Chunk 对象
```

### `rag/bm25_index.py`

纯 Python 手写 BM25 稀疏检索算法，零依赖。

**核心公式：**
```
BM25(q, d) = Σ IDF(tᵢ) × ────────────────────────────────
                           k₁·(1-b + b·|d|/avgdl) + TF(tᵢ,d)

IDF(t) = log((N - n(t) + 0.5) / (n(t) + 0.5) + 1)
```

- `k₁ = 1.5`：词频饱和系数
- `b = 0.75`：文档长度归一化
- 分词：英文按空格 + 小写化，CJK 按单字符

```python
bm25 = BM25Index()
bm25.add_chunks(chunks)
results = bm25.search("multi-scale distillation", k=10)
# → [{id, text, heading, score}, ...]
```

### `rag/vector_store.py`

Numpy 向量存储 + 余弦相似度搜索。

**存储结构：**
- `xxx.json`：文本元数据 (JSON)
- `xxx.json.vectors.npy`：向量矩阵 `(N, 1024)` float32 (NumPy binary)

**搜索：** 归一化后矩阵乘法 `np.dot(v_norm, q.T)`，100% recall（暴力搜索）。

```python
store = VectorStore(uri="./vectors.json")
store.add_chunks(chunks, embedder)          # embed + store
results = store.search(query_vec, k=5)      # 余弦搜索
results = store.search_by_text("query", embedder, k=5)  # 文本搜
```

### `rag/retriever.py`

RRF 混合检索编排。

**RRF 公式：**
```
RRF(doc) = Σ 1 / (k + rankᵢ(doc))
```
其中 `k=60`，`rankᵢ` 是文档在第 i 个检索器中的排名（1-indexed）。

融合流程：
```
query
  ├──→ BM25.search(k=20)    → [(id, bm25_score), ...]
  ├──→ VectorStore.search(k=20) → [(id, cosine_score), ...]
  │
  └──→ RRF 融合 ──→ 排序 ──→ top-K
```

```python
retriever = HybridRetriever(bm25, vector_store, embedder)
results = retriever.search("query", k=5)
# → [{id, text, heading, source, rrf_score}, ...]
```

## Agent 集成 (`s021_rag/code.py`)

### 新增工具

**`search_knowledge`** — 混合检索知识库
```
输入: query (搜索查询), top_k (返回数量, 默认 5)
输出: top-K chunks 的原文片段 + 来源标注
     每个结果 600 字符截断，带 RRF 分数和源文件名
```

**`index_documents`** — 索引文档到知识库
```
输入: path (目录路径, 默认 "."), patterns (glob 模式, 默认 "*.pdf,*.md")
输出: 索引统计 (文档数 → chunk 数 → 向量数 → BM25 文档数)
```

### System Prompt 感知

每个 turn 的 system prompt 自动注入知识库清单：

```
Knowledge base (use search_knowledge to retrieve from these):
- [2502.15016v3.pdf] TimeDistill: Efficient Long-Term Time Series
  Forecasting with MLP via Cross-Architecture Distillation (46 chunks, pdf)
```

agent 看到文档标题后，可以自行判断用户问题是否涉及已索引的知识，按需调用 `search_knowledge`。

### 知识库持久化

```
learn-claude-code/
  .rag_vectors.json             ← 文本元数据 (chunk id, text, heading, source)
  .rag_vectors.json.vectors.npy ← 向量矩阵 (N × 1024 float32)
  .rag_index/
    KNOWLEDGE.md                ← 文档清单 (注入 system prompt)
```

重启时 `init_rag()` 自动：
1. 从 `.rag_vectors.json` 加载向量 + 元数据
2. 从元数据中的 `text` 字段重建 BM25 索引（无需重新 embedding）
3. 从 `KNOWLEDGE.md` 读取文档清单注入 system prompt

## 环境配置 (.env)

```env
# LLM (Anthropic-compatible API)
ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
ANTHROPIC_API_KEY=sk-your-deepseek-key
MODEL_ID=deepseek-v4-pro

# Embedding (百炼)
DASHSCOPE_API_KEY=sk-your-bailian-key
```

## 实测数据

**测试文档**: `2502.15016v3.pdf`（TimeDistill 论文，2847 KB）

| 指标 | 数值 |
|------|------|
| Docling 解析时间 | ~4 分钟 (CPU) |
| 缓存加载时间 | 瞬时 |
| 提取文本量 | 173,656 字符 |
| 识别 sections | 50 个 |
| 最终 chunks | 46 个 (avg 3,800 chars) |
| 向量维度 | 1024 |
| BM25 唯一词数 | 2,241 |
| 检索延迟 | < 100ms (混合检索) |

**检索效果示例：**
```
"knowledge distillation"
  BM25 #1: 2.2 Knolwedge Distillation (2.19)
  向量 #1: 2.2 Knolwedge Distillation (0.790)

"multi scale multi period"
  BM25 #1: 4.2 Multi-Period Distillation (4.53)
  向量 #1: E Implementation Details (0.636)

RRF 融合后:
  #1: 2.2 Knolwedge Distillation (RRF=0.0328, 双榜第一)
  #2: 4 Methodology (RRF=0.0320, BM25#2 + 向量#3)
  #3: M Comparison with KD (RRF=0.0308, 某方高排被捞起)
```

## 启动方式

```powershell
cd "d:/deskface/claude code/learn-claude-code"
python s021_rag/code.py
```

启动后：
```
s021 >> index_documents(path="D:/papers", patterns="*.pdf")
s021 >> 这篇论文的核心方法是什么？    ← agent 自动调 search_knowledge
```

## 依赖

```
anthropic          # Claude API client
python-dotenv      # .env 加载
pyyaml             # YAML 解析
openai             # 百炼 embedding (OpenAI 兼容模式)
numpy              # 向量存储 + 计算
docling            # PDF 解析 (需要 HuggingFace 模型下载 ~500MB)
```
