# main.py - v4 (菜单式: 建库/查询/测试 功能分离, 正式库与测试库完全隔离)
# 功能:
#   1. 建立正式知识库   (pdfs/      -> ES索引 pdf_documents        + vector_db/)
#   2. 建立测试知识库   (test_pdfs/ -> ES索引 pdf_documents_test   + vector_db_test/)
#   3. 正式库检索       (es/vec/hybrid 三模式)
#   4. 测试库检索
#   5. 检索准确性测试   (独立脚本 tests/test_retrieval.py, 测试集 test_data/queries.json)
# 建库一次, 之后查询无需重建; 正式/测试数据完全分开存放。
import os
import glob
import json
import sys
import subprocess
from datetime import datetime
from typing import List, Dict
import logging
from tqdm import tqdm

from core.pdf_parser import PDFParser
from core.es_manager import ESManager
from core.embedder import OllamaEmbedder
from core.vector_store import VectorStore

# --- UTF-8 加固 ---
def _setup_logging():
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    try:
        from telemetry.log_setup import setup_app_logging
        setup_app_logging(log_dir=os.environ.get("APP_LOG_DIR", "logs/app"), console=True)
    except Exception:
        logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

RRF_K = 60
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


class PDFSearchSystem:
    """检索系统: db='main' 正式库 / db='test' 测试库, 数据完全隔离"""

    DB_CONFIG = {
        'main': {
            'label': '正式',
            'pdf_dir': 'pdfs',
            'output_dir': 'output',
            'vector_dir': 'vector_db',
            'es_index': 'pdf_documents',
            'vector_collection': 'pdf_chunks',
        },
        'test': {
            'label': '测试',
            'pdf_dir': os.path.join('tests', 'pdfs'),
            'output_dir': os.path.join('tests', 'output'),
            'vector_dir': os.path.join('tests', 'vector_db'),
            'es_index': 'pdf_documents_test',
            'vector_collection': 'pdf_chunks_test',
        },
    }

    def __init__(self, es_host='localhost', es_port=9200, db='main',
                 ollama_url='http://localhost:11434', embed_model='bge-m3',
                 read_only=False, project_root=None):
        self.read_only = read_only
        if db not in self.DB_CONFIG:
            raise ValueError(f"db 必须是 main/test, 得到: {db}")
        cfg = self.DB_CONFIG[db]
        self.db = db
        self.label = cfg['label']
        self.parser = PDFParser()

        # 运行环境判断
        if os.environ.get('RUN_IN_DOCKER') == '1' or os.path.exists('/app'):
            base_dir = '/app'
        else:
            base_dir = str(project_root or PROJECT_ROOT)

        self.pdf_dir = os.path.join(base_dir, cfg['pdf_dir'])
        self.output_dir = os.path.join(base_dir, cfg['output_dir'])
        self.vector_dir = os.path.join(base_dir, cfg['vector_dir'])

        if not read_only:
            os.makedirs(self.pdf_dir, exist_ok=True)
            os.makedirs(self.output_dir, exist_ok=True)

        # 组件: ES(索引名隔离) + 向量库(目录/集合隔离)
        self.es_manager = ESManager(host=es_host, port=es_port, index_name=cfg['es_index'])
        self.embedder = OllamaEmbedder(base_url=ollama_url, model=embed_model)
        self.vector_store = VectorStore(
            persist_dir=self.vector_dir,
            collection_name=cfg['vector_collection'],
            embedder=self.embedder,
            read_only=read_only,
        )
        self.search_mode = 'hybrid'  # es | vec | hybrid

        logger.info(f"[{self.label}库] PDF目录: {self.pdf_dir}")
        logger.info(f"[{self.label}库] 向量库: {self.vector_dir} | ES索引: {cfg['es_index']}")

    # ============ 建库 ============
    def process_pdfs(self):
        if self.read_only:
            raise PermissionError("agent runtime cannot ingest or rebuild the knowledge base")
        mode = os.environ.get("INGESTION_PIPELINE_MODE", "staged").strip().lower()
        if mode != "legacy":
            return self._process_pdfs_staged()

        if self.db == "main":
            raise RuntimeError(
                "正式库禁止 legacy 原地重建；请使用 staged generation 流程。"
            )

        logger.warning(
            "INGESTION_PIPELINE_MODE=legacy 会原地重建索引，仅用于显式兼容；"
            "推荐移除该变量使用 staged pipeline。"
        )
        pdf_files = glob.glob(os.path.join(self.pdf_dir, '*.pdf'))

        if not pdf_files:
            logger.warning(f"[{self.label}库] 未找到PDF文件: {self.pdf_dir}")
            return

        logger.info(f"[{self.label}库] 找到 {len(pdf_files)} 个PDF文件, 开始重建双索引...")

        # ES 与向量库同步重建(索引名/目录各自独立, 不影响另一个库)
        self.es_manager.create_index()
        self.vector_store.clear()

        documents = []
        for pdf_path in tqdm(pdf_files, desc=f"处理[{self.label}库]PDF"):
            try:
                parsed = self.parser.parse_pdf(pdf_path)
                doc_id = os.path.basename(pdf_path).replace('.pdf', '')

                doc = {
                    'id': doc_id,
                    'filename': parsed['filename'],
                    'content': parsed['text'],
                    'content_length': len(parsed['text']),
                    'page_count': parsed.get('total_pages', 0),
                    'created_date': datetime.now().isoformat(),
                    'metadata': parsed.get('metadata', {})
                }
                documents.append(doc)

                # 向量索引: 按页分块 + bge-m3 嵌入(失败不阻断 ES 索引)
                try:
                    n_chunks = self.vector_store.add_pages(
                        doc_id, parsed['filename'], parsed.get('pages', [])
                    )
                    if n_chunks:
                        logger.info(f"{parsed['filename']}: 向量化 {n_chunks} 个文本块")
                except Exception as e:
                    logger.warning(f"{parsed['filename']} 向量化失败(已跳过, 不影响ES索引): {e}")

                output_path = os.path.join(self.output_dir, f"{doc_id}.json")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(parsed, f, ensure_ascii=False, indent=2)

            except Exception as e:
                logger.exception(f"处理PDF失败 {pdf_path}: {e}")

        if documents:
            success, failed = self.es_manager.bulk_index(documents)
            logger.info(f"[{self.label}库] ES索引完成: {success}个成功, {failed}个失败")
            logger.info(f"[{self.label}库] 向量库完成: 共 {self.vector_store.count()} 个块")
        else:
            logger.warning(f"[{self.label}库] 没有成功解析任何PDF文档")

    def _process_pdfs_staged(self):
        """Build a prepared generation without changing the active generation."""
        pdf_files = glob.glob(os.path.join(self.pdf_dir, "*.pdf"))
        if not pdf_files:
            logger.warning("[%s库] 未找到PDF文件: %s", self.label, self.pdf_dir)
            return None
        from ingestion.parsers.docling_parser import DoclingParser, ParserUnavailable
        from ingestion.pipeline.coordinator import IngestionCoordinator, IngestionRejected
        from ingestion.pipeline.generation_registry import GenerationRegistry
        from ingestion.pipeline.stage import ChromaStagingSink, ElasticsearchStagingSink

        registry_path = os.environ.get(
            "INGESTION_REGISTRY_PATH",
            os.path.join(PROJECT_ROOT, "data", "ingestion", "generation_registry.sqlite3"),
        )
        manifest_dir = os.environ.get(
            "INGESTION_MANIFEST_DIR",
            os.path.join(PROJECT_ROOT, "data", "ingestion", "manifests"),
        )
        registry = GenerationRegistry(registry_path)
        # Fail before an expensive parse if the staging backend is unavailable.
        self.es_manager.es.info()
        coordinator = IngestionCoordinator(
            parser=DoclingParser(cache_dir=os.path.join(
                PROJECT_ROOT, "data", "ingestion", "parse_cache"
            )),
            registry=registry,
            es_sink_factory=lambda name: ElasticsearchStagingSink(self.es_manager.es, name),
            vector_sink_factory=lambda name: ChromaStagingSink(
                self.vector_store.client, name, self.embedder,
                persist_dir=str(self.vector_store.persist_dir),
            ),
            manifest_dir=manifest_dir,
            embedding_model=getattr(self.embedder, "model", "bge-m3"),
            embedding_dimension=1024,
        )
        try:
            manifest = coordinator.stage(pdf_files)
        except ParserUnavailable as exc:
            logger.error("Docling 未就绪，未修改当前知识库: %s", exc)
            return None
        except IngestionRejected as exc:
            logger.error("入库质量门未通过，当前知识库保持不变: %s", exc)
            return None
        logger.info(
            "新代际 %s 已 PREPARED，当前尚未激活。请先审查 manifest 后显式执行激活。",
            manifest.generation_id,
        )
        return manifest.to_dict()

    def activate_generation(self, generation_id: str) -> int:
        """Explicit promotion entry; callers must obtain user approval first."""
        from ingestion.parsers.docling_parser import DoclingParser
        from ingestion.pipeline.coordinator import IngestionCoordinator
        from ingestion.pipeline.generation_registry import GenerationRegistry
        from ingestion.pipeline.stage import ChromaStagingSink, ElasticsearchStagingSink

        registry_path = os.environ.get(
            "INGESTION_REGISTRY_PATH",
            os.path.join(PROJECT_ROOT, "data", "ingestion", "generation_registry.sqlite3"),
        )
        manifest_dir = os.environ.get(
            "INGESTION_MANIFEST_DIR",
            os.path.join(PROJECT_ROOT, "data", "ingestion", "manifests"),
        )
        registry = GenerationRegistry(registry_path, manifest_root=manifest_dir)
        coordinator = IngestionCoordinator(
            parser=DoclingParser(), registry=registry,
            es_sink_factory=lambda name: ElasticsearchStagingSink(self.es_manager.es, name),
            vector_sink_factory=lambda name: ChromaStagingSink(
                self.vector_store.client, name, self.embedder,
                persist_dir=str(self.vector_store.persist_dir),
            ),
            manifest_dir=manifest_dir,
            embedding_model=getattr(self.embedder, "model", "bge-m3"),
            embedding_dimension=1024,
        )
        return coordinator.activate(generation_id)

    # ============ 三种检索 ============
    def es_search(self, query: str, size: int = 10) -> List[Dict]:
        return self.es_manager.search_bm25(query, size)

    def vec_search(self, query: str, top_k: int = 10) -> List[Dict]:
        return self.vector_store.search(query, top_k=top_k)

    def hybrid_search(self, query: str, top_n: int = 10, include_chunks: bool = False):
        """多路召回: ES + 向量, RRF 融合(文档级聚合)
        include_chunks=True 时返回 (文档级结果, 向量块级结果), 供 Agent 层一次取全
        """
        es_results = self.es_manager.search_bm25(query, size=top_n * 2) or []
        try:
            vec_results = self.vector_store.search(query, top_k=top_n * 2) or []
        except Exception as e:
            logger.warning(f"向量检索失败, 本次降级为纯ES: {e}")
            vec_results = []

        fused = {}
        for rank, r in enumerate(es_results):
            fn = r['filename']
            e = fused.setdefault(fn, {
                'filename': fn, 'es': False, 'vec': False, 'rrf': 0.0,
                'es_score': None, 'vec_score': None,
                'highlights': [], 'vec_text': None,
            })
            e['es'] = True
            e['rrf'] += 1.0 / (RRF_K + rank + 1)
            e['es_score'] = r['score']
            e['highlights'] = r.get('highlights', [])

        # 向量侧: 标准 RRF —— 每文档只按最佳排名贡献一次分
        # (避免同文档多 chunk 重复加分扭曲融合结果)
        best_vec = {}
        for rank, r in enumerate(vec_results):
            fn = r['filename']
            if fn not in best_vec:  # 首次出现即最佳排名
                best_vec[fn] = (rank, r)

        for fn, (rank, r) in best_vec.items():
            e = fused.setdefault(fn, {
                'filename': fn, 'es': False, 'vec': False, 'rrf': 0.0,
                'es_score': None, 'vec_score': None,
                'highlights': [], 'vec_text': None,
            })
            e['vec'] = True
            e['rrf'] += 1.0 / (RRF_K + rank + 1)
            e['vec_score'] = r['score']
            e['vec_text'] = r['text'][:150]

        ranked = sorted(fused.values(), key=lambda x: -x['rrf'])[:top_n]
        if include_chunks:
            return ranked, vec_results
        return ranked

    # ============ 检索入口 + 展示 ============
    def search(self, query: str, size: int = 10):
        logger.info(f"[{self.label}库] 执行检索[{self.search_mode}]: {query}")
        print(f"\n=== [{self.label}库] 检索结果 [{self.search_mode}]: {query} ===\n")

        if self.search_mode == 'es':
            results = self.es_search(query, size)
            if not results:
                print("未找到相关结果")
                return
            for i, r in enumerate(results, 1):
                print(f"结果 {i} (得分: {r['score']:.3f})")
                print(f"文件: {r['filename']}")
                if r['highlights']:
                    print("高亮内容:")
                    for h in r['highlights']:
                        print(f" ...{h}...")
                else:
                    print(f"预览: {r['content'][:200]}...")
                print("-" * 50)

        elif self.search_mode == 'vec':
            results = self.vec_search(query, size)
            if not results:
                print("未找到相关结果(向量库可能为空, 请先建库)")
                return
            for i, r in enumerate(results, 1):
                print(f"结果 {i} (相似度: {r['score']:.4f})")
                print(f"文件: {r['filename']} (第 {r['page']} 页)")
                print(f"片段: {r['text'][:200]}...")
                print("-" * 50)

        else:  # hybrid
            results = self.hybrid_search(query, size)
            if not results:
                print("未找到相关结果")
                return
            for i, r in enumerate(results, 1):
                tag = ('[ES+向量]' if r['es'] and r['vec']
                       else '[ES]' if r['es'] else '[向量]')
                print(f"结果 {i} (RRF: {r['rrf']:.4f}) {tag}")
                print(f"文件: {r['filename']}")
                score_line = []
                if r['es_score'] is not None:
                    score_line.append(f"ES:{r['es_score']:.3f}")
                if r['vec_score'] is not None:
                    score_line.append(f"向量:{r['vec_score']:.4f}")
                if score_line:
                    print("  得分: " + " | ".join(score_line))
                if r['highlights']:
                    print("  [ES] 高亮:")
                    for h in r['highlights'][:2]:
                        print(f"    ...{h}...")
                if r['vec_text']:
                    print(f"  [向量] 片段: {r['vec_text']}...")
                print("-" * 50)


# ============ 功能模块 ============
def build_knowledge_base(system: PDFSearchSystem):
    """建库(独立功能, 建一次即可, 之后查询无需重建)"""
    logger.info(f"开始建立[{system.label}库]知识库...")
    result = system.process_pdfs()
    if os.environ.get("INGESTION_PIPELINE_MODE", "staged").strip().lower() != "legacy":
        if not result:
            logger.error(f"[{system.label}库]知识库建立失败；当前 ACTIVE 代际未改变")
            return None
        logger.info(
            "[%s库]新代际已完成构建并等待激活: %s",
            system.label, result.get("generation_id"),
        )
        return result
    logger.info(f"[{system.label}库]知识库建立完成")
    return result


def run_search_loop(system: PDFSearchSystem):
    """进入某库的检索交互"""
    print("\n" + "=" * 50)
    print(f"[{system.label}库] 检索模式已启动 (当前: {system.search_mode})")
    print("命令: menu 返回主菜单 | mode 查看/切换模式 | help 帮助")
    print("=" * 50)

    while True:
        try:
            query = input("\n请输入检索关键词: ").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit', 'q', 'menu', 'back']:
                print("返回主菜单")
                break

            if query.lower() == 'help':
                print("\n使用说明:")
                print(" - 输入关键词进行检索(支持中英文)")
                print(" - mode         查看当前检索模式")
                print(" - mode es      仅ES BM25检索(关键词精确)")
                print(" - mode vec     仅向量检索(语义相似)")
                print(" - mode hybrid  多路召回(ES+向量, RRF融合, 默认)")
                print(" - menu         返回主菜单")
                continue

            if query.lower().startswith('mode'):
                parts = query.split()
                if len(parts) == 1:
                    print(f"当前检索模式: {system.search_mode}")
                elif parts[1] in ('es', 'vec', 'hybrid'):
                    system.search_mode = parts[1]
                    print(f"已切换为 {parts[1]} 模式")
                else:
                    print("模式可选: es / vec / hybrid")
                continue

            system.search(query)

        except KeyboardInterrupt:
            print("\n\n返回主菜单")
            break
        except UnicodeDecodeError as e:
            logger.error(f"输入编码异常: {e}")
        except Exception as e:
            logger.exception(f"运行时错误: {e}")


def run_retrieval_test(db: str = 'test'):
    """调用独立测试脚本评测检索准确性(人工标注集)"""
    script = os.path.join(PROJECT_ROOT, 'tests', 'test_retrieval.py')
    if not os.path.exists(script):
        logger.error(f"测试脚本不存在: {script}")
        return
    subprocess.run([sys.executable, script, '--db', db], cwd=PROJECT_ROOT)


def run_auto_test(db: str = 'test'):
    """调用自动化测试脚本(零标注, 自动生成查询看召回率)"""
    script = os.path.join(PROJECT_ROOT, 'tests', 'auto_retrieval_test.py')
    if not os.path.exists(script):
        logger.error(f"测试脚本不存在: {script}")
        return
    subprocess.run([sys.executable, script, '--db', db], cwd=PROJECT_ROOT)


# ============ 主菜单 ============
def main():
    _setup_logging()
    es_host = os.environ.get('ES_HOST', 'localhost')
    es_port = int(os.environ.get('ES_PORT', 9200))
    ollama_url = os.environ.get('OLLAMA_URL', 'http://localhost:11434')
    embed_model = os.environ.get('EMBED_MODEL', 'bge-m3')

    # 环境预检(只检查一次, 不阻断)
    probe = PDFSearchSystem(es_host=es_host, es_port=es_port, db='main',
                            ollama_url=ollama_url, embed_model=embed_model)
    logger.info("检查Elasticsearch连接...")
    try:
        probe.es_manager.es.info()
        logger.info("Elasticsearch连接成功!")
    except Exception as e:
        logger.warning(f"Elasticsearch连接失败: {e}")
        logger.info("请先启动: cd 项目目录 && docker compose up -d")
    if not probe.embedder.health_check():
        logger.warning("Ollama 不可用: 向量检索/建库会降级, 请先启动 Ollama")

    while True:
        print("\n" + "=" * 56)
        print("  PDF 检索系统 (ES BM25 + bge-m3 向量 多路召回)")
        print("=" * 56)
        print("  1. 建立正式知识库   (pdfs/      → 正式ES索引 + vector_db/)")
        print("  2. 建立测试知识库   (test_pdfs/ → 测试ES索引 + vector_db_test/)")
        print("  3. 正式库检索       (es / vec / hybrid)")
        print("  4. 测试库检索       (es / vec / hybrid)")
        print("  5. 检索准确性测试   (tests/test_retrieval.py, 针对测试库)")
        print("  6. 自动化召回率测试 (自动生成查询, 无需标注, 库未建会自动建)")
        print("  0. 退出")
        print("=" * 56)

        try:
            choice = input("请选择功能: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见!")
            break

        try:
            if choice == '1':
                build_knowledge_base(PDFSearchSystem(es_host, es_port, db='main',
                                                     ollama_url=ollama_url, embed_model=embed_model))
            elif choice == '2':
                build_knowledge_base(PDFSearchSystem(es_host, es_port, db='test',
                                                     ollama_url=ollama_url, embed_model=embed_model))
            elif choice == '3':
                run_search_loop(PDFSearchSystem(es_host, es_port, db='main',
                                                ollama_url=ollama_url, embed_model=embed_model))
            elif choice == '4':
                run_search_loop(PDFSearchSystem(es_host, es_port, db='test',
                                                ollama_url=ollama_url, embed_model=embed_model))
            elif choice == '5':
                run_retrieval_test('test')
            elif choice == '6':
                run_auto_test('test')
            elif choice == '0':
                print("再见!")
                break
            else:
                print("无效选择, 请输入 0-6")
        except Exception as e:
            logger.exception(f"功能执行失败: {e}")


if __name__ == "__main__":
    main()
