#!/bin/bash
# 一键: 建测试知识库 + 自动化召回率测试
# 用法: bash tests/run_all.sh [--per-doc N] [--brief]
cd "$(dirname "$0")/.." || exit 1

PER_DOC=3
BRIEF=""
for arg in "$@"; do
  case $arg in
    --per-doc=*) PER_DOC="${arg#*=}" ;;
    --brief) BRIEF="--brief" ;;
  esac
done

echo "=== 1/2 建立测试知识库 (tests/pdfs -> 测试ES索引 + tests/vector_db) ==="
./venv/bin/python - <<'PYEOF'
from main import PDFSearchSystem
s = PDFSearchSystem(db='test')
s.process_pdfs()
print('测试库向量块数:', s.vector_store.count())
PYEOF

echo "=== 2/2 自动化召回率测试 (每篇 ${PER_DOC} 条查询) ==="
./venv/bin/python tests/auto_retrieval_test.py --db test --per-doc "$PER_DOC" $BRIEF
