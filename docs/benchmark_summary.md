# Benchmark Summary

## Benchmark 目标

这个 benchmark 不是为了宣称“最优检索效果”，而是为了回答两个更实际的问题：

1. 这个仓库当前的 `vector` 和 `hybrid` 检索是否有可解释的差异？
2. 项目是否已经具备一个最小但可信的评估闭环？

## 设置

- 语料：3 份 PDF 报告 / 政策文件
- 评估问题：12 条人工设计 query
- 评估粒度：文档级相关性
- top-k：5
- embedding backend：`local-hash-v1`
- modes：`vector` vs `hybrid`

问题类别分布：

| Category | Queries | 说明 |
|---|---:|---|
| `policy_regulation` | 4 | 政策名称、监管措辞、国家行动计划 |
| `governance_standards` | 5 | 国际治理、标准、可信 AI、互操作、信任 |
| `strategy_impact` | 3 | 劳动力影响、竞争力、国家战略、安全应用 |

## 总体结果

| Mode | hit@5 | MRR | nDCG@5 | avg_unique_docs@5 | avg_front_matter_hits@5 |
|---|---:|---:|---:|---:|---:|
| `vector` | 1.0000 | 0.8333 | 0.8822 | 2.5000 | 0.1667 |
| `hybrid` | 1.0000 | 0.9583 | 0.9559 | 2.6667 | 0.0833 |

读法：

- `hit@5` 在这个小 benchmark 上都为 1.0，说明两个模式都能把正确文档放进前五。
- 更有区分度的是 `MRR` 和 `nDCG@5`，它们反映“正确文档是否排得更靠前”。
- `hybrid` 在整体上更稳，且前言/目录类噪声更少。

## 分类别结果

| Category | Mode | hit@5 | MRR | nDCG@5 |
|---|---|---:|---:|---:|
| `policy_regulation` | `vector` | 1.0000 | 0.7500 | 0.8155 |
| `policy_regulation` | `hybrid` | 1.0000 | 1.0000 | 1.0000 |
| `governance_standards` | `vector` | 1.0000 | 0.9000 | 0.9262 |
| `governance_standards` | `hybrid` | 1.0000 | 1.0000 | 0.9839 |
| `strategy_impact` | `vector` | 1.0000 | 0.8333 | 0.8978 |
| `strategy_impact` | `hybrid` | 1.0000 | 0.8333 | 0.8502 |

结论：

- `hybrid` 对 `policy_regulation` 提升最明显，说明词面锚定较强的问题更适合结合 keyword 信号。
- `governance_standards` 里 `hybrid` 也更稳，尤其在机构名词和治理术语上更容易把正确文档排到前列。
- `strategy_impact` 是当前薄弱项。对更宽泛、更抽象的问题，`hybrid` 不一定优于 `vector`。

## 代表性案例

### Hybrid 明显更好

Query: `Which source provides an AI action plan for the United States?`

- `vector`: 正确文档排在第 2
- `hybrid`: 正确文档排在第 1

原因：这是典型的政策名称 / 国家实体问题，BM25 风格的词面匹配能明显帮助排序。

### 仍然存在的失败模式

Query: `What are the risks and policy concerns around AI and labour markets?`

- `vector`: 正确文档排在第 1
- `hybrid`: 正确文档排在第 2

原因：这是宽主题 query，词面上与多个文档的“风险”“policy”“impact”表述都相近，`hybrid` 可能把局部词面更强但主题不完全对口的 chunk 提前。

## 误差分析

- 小 benchmark 仍显示出 front matter 噪声。虽然检索清洁度已提升，但前言、摘要、目录页还没有被完全消除。
- broad thematic query 仍容易混入跨主题 chunk，说明当前 lightweight reranking 更适合“政策名词”和“治理术语”，而不是抽象概念对齐。
- 评估粒度是文档级，不是 chunk 级或答案级，因此它更适合比较检索模式，不适合证明引用精度已经充分解决。

## 这个 benchmark 能证明什么

- 仓库已经具备可重复的检索评估闭环
- `hybrid` 相比 `vector` 有可解释、可复现的优势
- 项目不是只会“跑通 demo”，而是有最小限度的测量与误差意识

## 这个 benchmark 不能证明什么

- 不能证明系统在其他领域也有效
- 不能证明答案生成已经达到生产级事实性
- 不能证明当前 heuristic 就是最优配置

## 复现命令

```bash
./.venv/bin/python -m src.evaluate \
  --index_dir index \
  --questions eval/retrieval_benchmark.jsonl \
  --modes vector hybrid \
  --top_k 5
```
