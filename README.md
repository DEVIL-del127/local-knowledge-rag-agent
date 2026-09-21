# STUDY RAG 私人知识库 Agent

STUDY 是一个面向本地 PDF 知识库的 RAG 系统。它组合 Elasticsearch 关键词检索、Chroma
向量检索、结构化 NLU、证据校验、会话记忆和 DeepSeek 回答，默认以只读方式查询已激活的
知识库代际，避免运行时意外重建或覆盖正式数据。

## 主要能力

- PDF 解析、结构化切块和可追踪的分代入库
- Elasticsearch + Chroma 混合检索
- 契约优先的 `nlu_v2` 查询理解与逻辑计划
- 基于证据的回答生成和引用校验
- 会话记忆、限流、熔断、预算与检查点恢复
- 正式库与测试库隔离
- V1-A 统一低成本语义路由、上下文绑定和失败关闭
- 后端权威 conversation/request 标识与跨会话隔离

## 运行环境

- Python 3.10+
- Elasticsearch 8.15
- Ollama 与 `bge-m3`
- DeepSeek API（生成回答时需要）
- 推荐在 WSL/Linux 中运行；当前项目虚拟环境目录统一为 `venv/`

## 快速开始

```bash
git clone <your-repository-url>
cd STUDY
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[ingestion]"
python -m pip install pytest
cp .env.example .env
```

编辑 `.env`，至少配置 `DEEPSEEK_API_KEY`。启动本地依赖：

```bash
docker compose up -d
ollama pull bge-m3
```

正式问答入口：

```bash
python agent_main.py
# 安装后也可以使用：study
```

入库和维护入口：

```bash
python main.py
```

## 测试

```bash
python -m pytest -q
```

需要 Elasticsearch、Ollama 或外部模型的集成测试，应在相应服务可用时单独执行。测试输出、
向量库、模型、日志、PDF 和本地状态均不进入 Git。

## 项目结构

```text
agent_main.py       正式 Agent CLI 入口
main.py             PDF 入库、维护和底层检索入口
agent/              Agent 编排、运行时、模型调用和工具
core/               Elasticsearch、Chroma、PDF 与检索网关
ingestion/          分代入库流水线
kb-agent/nlu_v2/    当前结构化查询理解实现
memory/             会话与长期记忆
telemetry/          日志、追踪和审计事件
tests/              正式测试套件与稳定 fixtures
scripts/            运维、迁移、诊断和验收脚本
config/             MCP 等静态配置
docs/               架构、安装和历史说明
```

完整架构与请求链路见 [项目详细讲解](docs/项目详细讲解.md)，环境配置见
[安装与运行说明](docs/setup.md)。

## 本地数据

以下内容默认被 Git 忽略，需要在目标机器重新创建或自行迁移：

- `.env`：密钥和本地配置
- `pdfs/`：原始知识库文档
- `vector_db/`：Chroma 数据
- `data/`、`agent_state/`：代际注册表、缓存、检查点和记忆
- `kb-agent/models/`：本地模型文件
- `logs/`、`output/`：运行与测试输出

不要提交真实密钥、私人 PDF、模型权重或运行数据库。

## 版本

当前稳定发布版本为 `v2`（包版本 `2.0.0`），对应已冻结并验收通过的 V1-A。
V1-B 动态学习/热更新不包含在本稳定版本内。版本变化记录见 [CHANGELOG.md](CHANGELOG.md)，
V1-A v2 的验收边界见 [v2 发布说明](docs/v2-release-notes.md)。
GitHub 建库与推送步骤见 [GitHub 发布说明](docs/github-release.md)。
