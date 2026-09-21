# config.py
import os

class Config:
    # 基础目录
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # 数据目录
    PDF_DIR = os.path.join(BASE_DIR, 'pdfs')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

    # Elasticsearch配置
    ES_HOST = os.environ.get('ES_HOST', 'localhost')
    ES_PORT = int(os.environ.get('ES_PORT', 9200))
    ES_INDEX = 'pdf_documents'

    # PDF解析配置
    OCR_LANG = 'chi_sim+eng'
    OCR_DPI = 300

    # 检索配置
    DEFAULT_SEARCH_SIZE = 10

    @classmethod
    def ensure_directories(cls):
        """确保必要目录存在"""
        os.makedirs(cls.PDF_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        return cls.PDF_DIR, cls.OUTPUT_DIR

# 使用示例
if __name__ == "__main__":
    pdf_dir, output_dir = Config.ensure_directories()
    print(f"PDF目录: {pdf_dir}")
    print(f"输出目录: {output_dir}")