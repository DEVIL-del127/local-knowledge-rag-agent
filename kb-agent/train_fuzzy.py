# -*- coding: utf-8 -*-
"""模糊时间窗口分类模型训练：bert-base-chinese + CLS 分类头

数据：data/fuzzy_seed.jsonl（种子）+ data/annotations.jsonl（人工标注，可选合并）
输出：models/fuzzy_window/（transformers 格式）

用法：
  python train_fuzzy.py                 # 默认：种子 + 标注合并训练
  python train_fuzzy.py --epochs 5      # 调轮数
  python train_fuzzy.py --no-annot      # 只用种子
"""
import os
import sys
import json
import argparse
import random

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.abspath(__file__))
SEED_PATH = os.path.join(ROOT, "data", "fuzzy_seed.jsonl")
ANN_PATH = os.path.join(ROOT, "data", "annotations.jsonl")
OUT_DIR = os.path.join(ROOT, "models", "fuzzy_window")

# 类目 → 稳定顺序（训练/推理共用，禁止乱序）
CLASSES = ["none", "recent_1week", "recent_1month", "recent_3month", "recent_6month",
           "recent_1year", "recent_2year", "recent_3year",
           "year_start", "year_mid", "year_end",
           "half_first", "half_second", "fuzzy_vague"]
CLASS2ID = {c: i for i, c in enumerate(CLASSES)}


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    return rows


def build_dataset(use_annot=True, max_annot_per_class=30):
    """种子 + 标注合并；标注的 time_expected 转成窗口类样本（人工监督信号）"""
    rows = load_jsonl(SEED_PATH)
    samples = [{"query": r["query"], "label": r["window_class"]} for r in rows]
    if use_annot:
        anns = load_jsonl(ANN_PATH)
        added = 0
        per_class = {}
        for r in anns:
            te = r.get("time_expected") or {}
            word, w, unit = te.get("fuzzy_word"), te.get("window"), te.get("unit")
            if not word or not w:
                continue
            cls = _window_to_class(w, unit)
            if cls is None:
                continue
            if per_class.get(cls, 0) >= max_annot_per_class:
                continue
            per_class[cls] = per_class.get(cls, 0) + 1
            samples.append({"query": r["query"], "label": cls, "src": "annot"})
            added += 1
        if added:
            print(f"合并标注样本: {added} 条")
    # 校验类目合法性
    samples = [s for s in samples if s["label"] in CLASS2ID]
    if not samples:
        print("无有效样本")
        return [], []
    random.shuffle(samples)
    split = int(len(samples) * 0.9)
    return samples[:split], samples[split:]


def _window_to_class(n, unit):
    """(n, unit) → 窗口类。只映射训练支持的类目。"""
    if unit == "week" and n == 1:
        return "recent_1week"
    if unit == "month":
        return {1: "recent_1month", 3: "recent_3month", 6: "recent_6month",
                12: "recent_1year", 24: "recent_2year", 36: "recent_3year"}.get(n)
    return None


def train(epochs, batch_size, use_annot):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

    train_s, eval_s = build_dataset(use_annot=use_annot)
    if not train_s:
        print("训练集为空，退出")
        return
    print(f"训练集 {len(train_s)} / 验证集 {len(eval_s)}")

    model_name = "bert-base-chinese"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(CLASSES))

    def tok(examples):
        return tokenizer([e["query"] for e in examples],
                         padding="max_length", truncation=True, max_length=64)

    class DS(torch.utils.data.Dataset):
        def __init__(self, samples):
            enc = tok(samples)
            self.input_ids = enc["input_ids"]
            self.attn = enc["attention_mask"]
            self.labels = [CLASS2ID[s["label"]] for s in samples]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            return {"input_ids": torch.tensor(self.input_ids[i]),
                    "attention_mask": torch.tensor(self.attn[i]),
                    "labels": torch.tensor(self.labels[i])}

    os.makedirs(OUT_DIR, exist_ok=True)
    args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=1,
        report_to=[],
        fp16=False,
    )
    trainer = Trainer(model=model, args=args,
                      train_dataset=DS(train_s), eval_dataset=DS(eval_s))
    trainer.train()
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    # 类目顺序写入（推理封装依赖）
    with open(os.path.join(OUT_DIR, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(CLASSES, f, ensure_ascii=False)
    print(f"模型已保存: {OUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--no-annot", action="store_true", help="只用种子数据")
    args = ap.parse_args()
    train(args.epochs, args.batch_size, not args.no_annot)
