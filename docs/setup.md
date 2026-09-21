# RAG 向量检索 - 环境配置清单

> 当前正式入口是 `python agent_main.py`，默认 `AGENT_EXECUTION_PROFILE=enforce`。
> 澄清问题默认使用确定性原文，不额外调用 DeepSeek；只有显式设置
> `AGENT_LLM_CLARIFICATION=1` 才启用模型润色，并计入模型额度。
> `main.py` 只用于 generation 化的入库/维护；正式库禁止 legacy 原地重建。
> 不要通过删除 `vector_db` 或固定 ES 索引来修复问题，这会绕过 ACTIVE generation 合同。

代码已全部就位（main.py / embedder.py / vector_store.py / es_manager.py），
你只需配置以下环境。

## ✅ 已完成（无需操作）

| 项 | 状态 |
|----|------|
| chromadb 1.5.9 | 已装入 venv |
| Elasticsearch + IK | docker compose up -d 即可（9200 端口） |
| 代码 | 语法验证通过 |
| Windows 版 Ollama + bge-m3 | 已装好（宿主 IP 连通验证通过，向量 1024 维） |

## WSL 访问 Windows Ollama（已自动处理）

WSL2 默认 NAT 网络下 `localhost` 访问不到 Windows 上的 Ollama（这是 WSL 网络机制，不是配置错）。
embedder 已内置自动探测：localhost 不通时自动解析 Windows 宿主 IP（默认网关）并切换，
无需任何配置。实测已自动切到 `http://192.168.144.1:11434`。
如特殊环境需要手动指定：设置环境变量 `OLLAMA_URL=http://<地址>:11434`。

## ⬜ 待配置：Ollama + bge-m3（唯一要做的）

**方式 A（推荐，最简单）—— Windows 版 Ollama：**
1. 下载安装：https://ollama.com/download/windows
   （官网慢可用网盘/镜像；装完 Ollama 常驻后台，WSL 里 localhost:11434 自动通）
2. 拉模型：`ollama pull bge-m3`（约 1.2GB，支持中文，1024 维）
3. 验证：`ollama list` 能看到 bge-m3

**方式 B —— WSL 版（项目里已备好脚本）：**
```bash
bash /mnt/f/A_ShiXi/Project/STUDY/install_ollama.sh   # 从 GitHub 镜像下载安装
~/ollama/bin/ollama pull bge-m3                       # 拉模型
~/ollama/bin/ollama serve                             # 前台启动(或后台 nohup)
```
注意：ollama serve 需常驻；WSL 关终端会停，可用 `nohup ~/ollama/bin/ollama serve &`。

## 运行

```bash
cd /mnt/f/A_ShiXi/Project/STUDY
source venv/bin/activate
python agent_main.py
```

## 功能菜单（v4：建库/查询/测试分离）

```
1. 建立正式知识库   (pdfs/          → 正式ES索引 pdf_documents       + vector_db/)
2. 建立测试知识库   (tests/pdfs/    → 测试ES索引 pdf_documents_test  + tests/vector_db/)
3. 正式库检索       (es / vec / hybrid)
4. 测试库检索
5. 检索准确性测试   (tests/test_retrieval.py, 人工标注集)
6. 自动化召回率测试 (tests/auto_retrieval_test.py, 零标注)
0. 退出
```

**建库一次即可**，之后每次启动直接选 3/4 查询，不用重建。
正式库与测试库完全隔离（PDF目录/ES索引/向量库目录/输出目录均不同）。

## 目录结构（测试全部集中在 tests/）

```
STUDY/
├── main.py / es_manager.py / embedder.py / vector_store.py / pdf_parser.py / config.py
├── pdfs/  vector_db/  output/        # 正式库数据
├── tests/                            # 一切测试相关
│   ├── test_retrieval.py             # 手动标注评测
│   ├── auto_retrieval_test.py        # 自动化评测(零标注)
│   ├── run_all.sh                    # 一键: 建测试库+自动化测试
│   ├── data/                         # 测试集
│   │   ├── queries.json              # 测试库手动标注集
│   │   ├── queries_main.json         # 正式库手动标注集
│   │   ├── auto_queries.json         # 测试库自动生成集(42条)
│   │   ├── auto_queries_main.json    # 正式库自动生成集
│   │   └── queries.example.json      # 格式示例
│   ├── pdfs/                         # 测试 PDF
│   ├── vector_db/                    # 测试向量库
│   └── output/                       # 测试解析输出
└── docker-compose.yaml / requirements.txt / ...
```

## 准确性测试

**自动化（零标注，一键）：**
```bash
bash tests/run_all.sh              # 建测试库+跑评测(每篇3条查询)
bash tests/run_all.sh --brief     # 只看汇总
python tests/auto_retrieval_test.py --db main   # 测正式库
```

**手动标注（更接近真实场景）：**
1. 编辑 `tests/data/queries.json`（格式见 queries.example.json）
2. `python tests/test_retrieval.py`（--db main 测正式库, --brief 只看汇总）
3. 输出：Recall@1/3/5/10 + MRR@10，三种模式对比 + 每条命中/误召回明细

## 检索模式（运行时切换）

```
mode         查看当前模式
mode es      ES BM25（关键词精确匹配）
mode vec     仅向量检索（语义相似，如"关于误差补偿的方法"）
mode hybrid  多路召回（ES + 向量，RRF 融合，默认）
```

## 架构

```
           ┌─→ ES BM25 召回 top20 ─┐
查询 ──────┤                       ├─→ RRF 融合 ─→ 文档排序 top10
           └─→ bge-m3 向量 top20 ─┘

RRF: score(d) = Σ 1/(60 + rank_i(d))   # 双路命中排名靠前的文档排最前
```

## 数据存放（分开独立）

| 数据 | 位置 |
|------|------|
| ES 全文索引 | Docker volume `es_data`（容器内） |
| 向量库 | 项目根 `vector_db/`（ChromaDB 持久化目录） |
| 解析结果 | `output/*.json` |

## 故障降级（已内置）

- Ollama 没启动 → 程序警告但不退出，`es` 模式可用，`hybrid` 自动降为纯 ES
- 向量库为空 → `vec`/`hybrid` 提示重新索引
- 数据不一致 → 先运行只读诊断；如需重建，创建并验证新的 PREPARED generation，确认后再 CAS 激活。
