# Interview Notes

## Elevator Pitch

这是一个面向咨询研究和政策分析场景的端到端 RAG 项目。我没有把重点放在 agent 编排上，而是先把更基础但更重要的部分做扎实：可复现的语料构建、可解释的检索、可追溯的引用，以及一个最小但可信的 benchmark。

## 为什么这个项目有价值

- 咨询和研究场景最先需要的不是“会写漂亮答案”，而是“能找到对的证据”。
- 很多个人 RAG 项目只展示 UI 或大模型调用，这个项目更强调工程边界、复现性和检索误差分析。
- 对面试官来说，这个项目能同时展示数据处理、检索系统、评估意识和产品化表达。

## 技术挑战与修复

### 1. 原型能跑，但不可复现

- 修复 ingestion 死循环
- 去掉在线 NLTK 依赖
- 增加离线 embedding fallback
- 把构建结果收敛为 `chunks.jsonl`、`documents.jsonl`、`ingest_manifest.json` 和 `index/manifest.json`

### 2. 语料产物不可信

- 从源 PDF 重新构建语料
- 加强 metadata，包括标题、来源、日期、页码和 section
- 记录跳过文件和失败页，而不是静默丢弃

### 3. 检索能命中，但 top-k 不干净

- 增加同页 / 同 section 去重
- 增加文档多样性惩罚
- 增加轻量 query term coverage bonus
- 保留所有 heuristic 权重为显式配置，便于解释与调参

## 我学到的东西

- 对小语料咨询研究场景，最先决定体验的通常不是模型大小，而是 ingestion 质量和检索排序。
- 没有 benchmark 的 `hybrid`、compression、reranking 讨论都容易流于主观印象。
- 一个作品集项目不需要把系统做得很大，但必须让别人看得懂、跑得通、讲得清。

## 为什么 hybrid 在一些问题上更好

- `hybrid` 同时利用向量相似度和关键词匹配。
- 当 query 包含明确的政策名称、国家实体、监管措辞或机构术语时，关键词信号非常有价值。
- 例如 `AI action plan`, `pro-innovation regulatory approach`, `AI Policy Observatory` 这类问题，`hybrid` 往往比纯向量更容易把正确文档排到前列。
- 但对 `labour markets` 这类宽主题 query，`hybrid` 也可能因为词面重合而把偏题 chunk 提前，所以它不是对所有问题都绝对更优。

## 5 分钟面试讲法

1. 先用一句话定义项目：这是一个为咨询研究设计的、可复现的轻量 RAG 系统。
2. 用架构图说明链路：ingest -> index -> retrieve -> compress -> answer -> evaluate。
3. 展示 `ingest_manifest.json` 和 `index/manifest.json`，说明这个项目不是一次性 notebook。
4. 跑一条 `hybrid` query，展示结构化回答和引用。
5. 跑 benchmark，对比 `vector` 和 `hybrid`。
6. 主动指出一个失败案例，说明你知道系统当前边界在哪里。

## 适合主动说出的 trade-off

- 我优先做了可复现和可解释，而不是急着接更大的模型。
- 生成层保持轻量 template，是为了先把 retrieval 和 citation 基线做可靠。
- 当前 benchmark 很小，所以我只把它当作仓库内对比工具，不把它包装成“通用 SOTA 结果”。
