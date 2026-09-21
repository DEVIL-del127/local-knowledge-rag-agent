# embedder.py - Ollama bge-m3 向量化客户端
# 调用 Ollama /api/embed 接口, 批量文本 -> embedding
# 自动探测: WSL2 NAT 模式下 localhost 访问不到 Windows 上的 Ollama,
# 会自动解析 Windows 宿主 IP(默认网关) 并切换, 无需手动配置
import os
import re
import subprocess
import time
import requests
import logging

logger = logging.getLogger(__name__)


def clean_embed_text(text: str) -> str:
    """清理无法向量化的字符(数学斜体/特殊Unicode/零宽字符), 保留中文/字母/数字
    背景: bge-m3 对全特殊符号文本输出 NaN, 导致 Ollama 500
    """
    # 去掉零宽字符/不可见格式符(PDF提取常见残留, 会导致 tokenizer 输出 NaN)
    t = re.sub(r'[\u200b\u200c\u200d\u2060\ufeff]', '', text)
    t = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


class OllamaEmbedder:
    def __init__(self, base_url='http://localhost:11434', model='bge-m3',
                 batch_size=16, timeout=180):
        # 环境变量 OLLAMA_URL 优先
        env_url = os.environ.get('OLLAMA_URL')
        if env_url:
            base_url = env_url
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self._probed = False
        self._session = requests.Session()

    # ---------- 地址探测 ----------
    def _try_url(self, url, timeout=3) -> bool:
        try:
            r = self._session.get(f"{url}/api/tags", timeout=timeout)
            return r.status_code == 200
        except Exception:
            return False

    def _host_ip(self):
        """WSL 下获取 Windows 宿主 IP(默认网关); 非 WSL 返回 None"""
        try:
            out = subprocess.run(
                ['ip', 'route', 'show', 'default'],
                capture_output=True, text=True, timeout=3,
            ).stdout
            m = re.search(r'default via (\S+)', out)
            if m:
                return m.group(1)
        except Exception:
            pass
        # 兜底: resolv.conf 的 nameserver 通常也是宿主 DNS
        try:
            with open('/etc/resolv.conf') as f:
                for line in f:
                    if line.startswith('nameserver'):
                        return line.split()[1]
        except Exception:
            pass
        return None

    def _ensure_available(self) -> bool:
        """确保 self.base_url 可用, localhost 不通时自动切换宿主 IP"""
        if self._try_url(self.base_url):
            self._probed = True
            return True

        if 'localhost' in self.base_url or '127.0.0.1' in self.base_url:
            host = self._host_ip()
            if host:
                alt = f"http://{host}:11434"
                if self._try_url(alt):
                    logger.info(f"localhost 不可达, 已自动切换为 Windows 宿主地址: {alt}")
                    self.base_url = alt
                    self._probed = True
                    return True
        return False

    def health_check(self) -> bool:
        """检查 Ollama 是否可用、模型是否已拉取"""
        if not self._ensure_available():
            logger.error(f"Ollama 连接失败({self.base_url})")
            logger.error("请确认 Ollama 已启动: Windows 托盘/命令行 ollama serve")
            logger.error("或手动指定地址: 设置环境变量 OLLAMA_URL=http://<host>:11434")
            return False

        try:
            resp = self._session.get(f"{self.base_url}/api/tags", timeout=5)
            models = [m['name'] for m in resp.json().get('models', [])]
            ok = any(self.model in m for m in models)
            if not ok:
                logger.warning(f"模型 {self.model} 未拉取, 请执行: ollama pull {self.model}")
            else:
                logger.info(f"Ollama 就绪: {self.base_url} (模型 {self.model} ✓)")
            return ok
        except Exception as e:
            logger.error(f"Ollama 检查失败: {e}")
            return False

    # ---------- 嵌入 ----------
    def _post(self, payload, retries=3):
        """POST /api/embed 带重试(瞬态 500/连接失败自动重试)"""
        last_err = None
        for attempt in range(retries):
            try:
                resp = self._session.post(
                    f"{self.base_url}/api/embed",
                    json=payload,
                    timeout=self.timeout,
                )
                if resp.status_code == 500 and attempt < retries - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                resp.raise_for_status()
                return resp.json()
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
                last_err = e
                if attempt < retries - 1:
                    time.sleep(2 * (attempt + 1))
        raise last_err

    def _embed_one_safe(self, text):
        """单条降级嵌入: 原文失败 -> 清洗后再试 -> 仍失败返回 None"""
        candidates = [text]
        cleaned = clean_embed_text(text)
        if cleaned and cleaned != text:
            candidates.append(cleaned)
        for c in candidates:
            try:
                data = self._post({"model": self.model, "input": [c]})
                return data["embeddings"][0]
            except Exception:
                continue
        logger.warning(f"文本无法向量化(可能全是特殊符号): {text[:60]}...")
        return None

    def embed(self, texts) -> list:
        """批量嵌入, 返回 list[list[float]|None], 自动分批; 坏文本返回 None 不崩溃"""
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return []

        if not self._probed:
            self._ensure_available()

        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            try:
                data = self._post({"model": self.model, "input": batch})
                results.extend(data["embeddings"])
            except Exception as e:
                logger.warning(f"第{i}批批量嵌入失败({e}), 逐条降级重试...")
                for t in batch:
                    results.append(self._embed_one_safe(t))
        return results

    def embed_query(self, text: str) -> list:
        """单条查询向量"""
        return self.embed([text])[0]
