# pdf_parser.py - 修复版
# 修复点:
# 1. _ocr_page 硬编码 /tmp 改为 tempfile 临时目录 (Windows 本地也能跑)
# 2. 其余逻辑保持
import os
import re
import tempfile
from typing import List, Dict
import pdfplumber
import PyPDF2
import pytesseract
from PIL import Image
try:
    import magic
except ImportError:  # python-magic requires an external libmagic DLL on Windows
    magic = None
import logging

logger = logging.getLogger(__name__)


class PDFParser:
    def __init__(self, ocr_lang='chi_sim+eng'):
        self.ocr_lang = ocr_lang

    def parse_pdf(self, pdf_path: str) -> Dict[str, any]:
        result = {
            'filename': os.path.basename(pdf_path),
            'text': '',
            'pages': [],
            'metadata': {}
        }

        try:
            if magic is not None:
                mime = magic.from_file(pdf_path, mime=True)
                if mime != 'application/pdf':
                    raise ValueError(f"不是PDF文件: {mime}")
            else:
                with open(pdf_path, "rb") as stream:
                    if stream.read(5) != b"%PDF-":
                        raise ValueError("不是PDF文件: invalid PDF signature")

            with pdfplumber.open(pdf_path) as pdf:
                result['metadata'] = pdf.metadata
                result['total_pages'] = len(pdf.pages)

                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text() or ''

                    if len(text.strip()) < 50:
                        logger.info(f"第{page_num}页文本较少，尝试OCR识别")
                        text = self._ocr_page(page, pdf_path, page_num)

                    result['pages'].append({
                        'page_num': page_num,
                        'text': text
                    })
                    result['text'] += text + '\n'

        except Exception as e:
            logger.error(f"解析PDF失败 {pdf_path}: {e}")
            result['text'] = self._parse_with_pypdf2(pdf_path)

        result['text'] = self._clean_text(result['text'])
        return result

    def _ocr_page(self, page, pdf_path: str, page_num: int) -> str:
        try:
            img = page.to_image(resolution=300)
            # 用系统临时目录, 兼容 Windows/Linux/Docker
            img_path = os.path.join(
                tempfile.gettempdir(),
                f"pdf_ocr_{os.path.basename(pdf_path)}_{page_num}.png"
            )
            img.save(img_path)
            text = pytesseract.image_to_string(
                Image.open(img_path),
                lang=self.ocr_lang,
                config='--psm 6'
            )
            os.remove(img_path)
            return text
        except Exception as e:
            logger.error(f"OCR失败: {e}")
            return ''

    def _parse_with_pypdf2(self, pdf_path: str) -> str:
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ''
                for page in reader.pages:
                    text += page.extract_text() or ''
                return text
        except Exception as e:
            logger.error(f"PyPDF2解析失败: {e}")
            return ''

    def _clean_text(self, text: str) -> str:
        """清洗文本"""
        if not text:
            return ""

        # 处理编码问题: bytes 按 UTF-8 兜底解码, 损坏字节替换而非崩溃
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='replace')

        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)

        # 保留中文、英文、数字和常用标点
        # 使用更宽松的规则，保留更多字符
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\.,;:!?()\-_\s\uFF01-\uFF5E\u3000-\u303F]', ' ', text)

        # 移除多余空格
        text = re.sub(r'\s+', ' ', text)

        return text.strip()
