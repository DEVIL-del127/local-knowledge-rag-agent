# -*- coding: utf-8 -*-
"""模糊时间窗口模型推理封装：query → (window_class, confidence)

接入点：nlu.py FUZZY_DEFAULTS 接口（_h_fuzzy_default 可选调用）
降级链：模型可用 → 模型预测；模型缺失/低置信 → 默认表（FUZZY_DEFAULTS 语义）
"""
import os
import sys
import json

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "models", "fuzzy_window")

# 窗口类 → (n, unit)；None = 高度模糊 → 澄清
CLASS_WINDOWS = {
    "none": None,
    "recent_1week": (1, "week"),
    "recent_1month": (1, "month"),
    "recent_3month": (3, "month"),
    "recent_6month": (6, "month"),
    "recent_1year": (12, "month"),
    "recent_2year": (24, "month"),
    "recent_3year": (36, "month"),
    "year_start": None, "year_mid": None, "year_end": None,
    "half_first": None, "half_second": None,
    "fuzzy_vague": None,
}

# 模型缺失时的默认表（与 FUZZY_DEFAULTS 语义一致，规则层兜底；长词优先匹配）
DEFAULT_MAP = {
    "最近一段时间": ("recent_3month", 1.0),
    "最近一两个月": ("recent_1month", 1.0),
    "最近半年": ("recent_6month", 1.0),
    "这半年": ("recent_6month", 1.0),
    "最近一年": ("recent_1year", 1.0),
    "近一年": ("recent_1year", 1.0),
    "这一年来": ("recent_1year", 1.0),
    "最近两年": ("recent_2year", 1.0),
    "近两年": ("recent_2year", 1.0),
    "最近三年": ("recent_3year", 1.0),
    "近三年": ("recent_3year", 1.0),
    "刚发布那会儿": ("fuzzy_vague", 1.0),
    "刚毕业那阵": ("fuzzy_vague", 1.0),
    "刚入行那会儿": ("fuzzy_vague", 1.0),
    "刚接触那阵子": ("fuzzy_vague", 1.0),
    "最近": ("recent_3month", 1.0),
    "近期": ("recent_3month", 1.0),
    "前阵子": ("recent_1month", 1.0),
    "那几年": ("fuzzy_vague", 1.0),
    "毕业后": ("fuzzy_vague", 1.0),
}

CONF_THRESHOLD = 0.60  # 低于此 → 视为低置信，调用方应转澄清
_DEFAULT_MODEL = None


class FuzzyWindowModel:
    """懒加载模型；不可用时自动降级默认表"""

    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or MODEL_DIR
        self._model = None
        self._tokenizer = None
        self._classes = None
        self._failed = None

    def _load(self):
        if self._model is not None or self._failed:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            if not os.path.exists(os.path.join(self.model_dir, "config.json")):
                self._failed = f"模型不存在: {self.model_dir}"
                return
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            self._model.eval()
            with open(os.path.join(self.model_dir, "classes.json"), encoding="utf-8") as f:
                self._classes = json.load(f)
            self._torch = torch
        except Exception as e:
            self._failed = str(e)

    @property
    def available(self) -> bool:
        self._load()
        return self._model is not None

    def predict(self, query: str):
        """返回 (window_class, confidence)；模型不可用 → 默认表；未命中 → None"""
        self._load()
        if self._model is not None:
            try:
                enc = self._tokenizer(query, return_tensors="pt",
                                      padding="max_length", truncation=True, max_length=64)
                with self._torch.no_grad():
                    logits = self._model(**enc).logits
                probs = self._torch.softmax(logits, dim=-1)[0]
                idx = int(probs.argmax())
                conf = float(probs[idx])
                cls = self._classes[idx]
                if cls == "none":
                    return None, conf
                return cls, conf
            except Exception:
                pass
        # 默认表兜底（规则层语义；长词优先，避免"最近"先吞"最近一年"）
        for word, (cls, conf) in sorted(DEFAULT_MAP.items(), key=lambda x: -len(x[0])):
            if word in query:
                return cls, conf
        return None, 0.0

    def window_for(self, query: str):
        """→ (n, unit) 或 None（高度模糊/低置信）"""
        cls, conf = self.predict(query)
        if cls is None:
            return None
        if conf < CONF_THRESHOLD:
            return None
        return CLASS_WINDOWS.get(cls)


def get_default_model() -> FuzzyWindowModel:
    """返回进程内共享模型，避免每个子句重复加载 Transformer 权重。"""
    global _DEFAULT_MODEL
    if _DEFAULT_MODEL is None:
        _DEFAULT_MODEL = FuzzyWindowModel()
    return _DEFAULT_MODEL


def reset_default_model() -> None:
    """测试/热更新用：下次调用重新创建共享实例。"""
    global _DEFAULT_MODEL
    _DEFAULT_MODEL = None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", help="查询句")
    args = ap.parse_args()
    m = FuzzyWindowModel()
    print(f"模型可用: {m.available}" + (f"（{m._failed}）" if m._failed else ""))
    if args.query:
        cls, conf = m.predict(args.query)
        print(f"Q: {args.query}")
        print(f"  → class={cls} conf={conf:.3f} window={m.window_for(args.query)}")
    else:
        for q in ["帮我找最近半年的论文", "最近一年关于贝叶斯的", "那几年的文献", "近期GAN研究"]:
            cls, conf = m.predict(q)
            print(f"Q: {q} → {cls} conf={conf:.3f} window={m.window_for(q)}")


if __name__ == "__main__":
    main()
